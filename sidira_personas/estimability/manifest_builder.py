"""
manifest_builder.py — the build-time precompute core (router's heavy lift).

Runs at persona-pack build (in sidira-persona-worker, per the architecture decision).
Joins what the worker already has — feature-store ColumnProfiles + the crawler's
Capabilities — into `estimability_manifest.json`, shipped beside persona_pack.json in R2.
At hypothesis runtime the router just MATCHES + RETRIEVES from this; no Snowflake re-query,
no model refit for the common case.

This module is REPO-AGNOSTIC on purpose (mirrors causal_estimator.py): it takes plain
dicts + a `load_frame` callable, so the worker supplies real ColumnProfiles / Capabilities
/ a Snowflake query fn, and tests supply fixtures. The capability->column linkage here is
rule-based (token overlap); in prod it's the LLM-as-translator MAP step — same output shape.

Vendored from research/empirical-uplift-spike/manifest_builder.py with Phase-1 extensions:
negative-control verdicts attached to each supportable effect, a grid cap with explicit
truncation accounting, and a frame-only convenience entry (`build_manifest_from_frame`).

Negative controls are NOT auto-derived from sibling outcomes: a lever frequently moves
several metrics together (e.g. retention AND conversion), so a real co-effect would
false-trip the guard. A valid negative control is a metric the treatment must NOT cause
yet which shares the confounding — nominating one is a domain/LLM judgment (the LLM-as-
translator MAP step), supplied via `neg_control_map` (Y column -> [vouched NC columns]).
This is a VALIDATED limit of the method, not a TODO: absent a caller/LLM-vouched control
the effect ships as `guard_confidence="unverified"` and the router routes it to the LLM.

No hard dependency on the product. Deps: numpy, pandas, scikit-learn (via causal_estimator).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import math
import re
import pandas as pd
import pandas.api.types as ptypes

from .causal_estimator import support_check, estimate, negative_control_guard

OUTCOME_HINTS = ("order", "retention", "retain", "convert", "conversion", "visit", "spend",
                 "revenue", "churn", "engage", "active", "purchase", "signup", "subscribe")
_TOKEN = re.compile(r"[a-z0-9]+")

# The negative-control guard thresholds the manifest is built with. Kept as named constants (not
# magic literals at the call site) so the value recorded on each CandidateEffect.guard_floor /
# guard_ratio is provably the SAME value passed into negative_control_guard — the manifest and the
# guard can never silently drift. These mirror the guard's own validated defaults (floor=0.05,
# ratio=1.5; campaign iter1-iter5).
NC_GUARD_FLOOR = 0.05
NC_GUARD_RATIO = 1.5

# Max candidate-control diagnostics computed per supportable effect (bounds build-time compute).
MAX_NC_DIAGNOSTICS = 20

# --- Coverage manifest + per-lever grounding-completeness (sidira-simulator#564) -------------
# A column whose null_fraction is at/above this (but below 1.0) is "behavioural"-missing: the
# column EXISTS and is tracked, users just mostly didn't do the thing. At/above 1.0 (or absent
# from the store entirely) it is "structural": the company doesn't track it at all, and any
# lever that needs it must route to the LLM — never be silently zero-imputed.
BEHAVIOURAL_NULL_THRESHOLD = 0.5

# Minimum adjustment covariates for a lever to count as grounding-complete. One numeric varying
# covariate is the floor for the propensity/standardization fitters to adjust for ANYTHING; with
# zero the "adjusted" estimate degenerates to the naive difference.
MIN_GROUNDING_COVARIATES = 1

# FIXED reason vocabulary for CandidateEffect.grounding_gaps (the consumer schema enumerates
# these verbatim). Order here is the canonical emission order within any one gaps list.
GROUNDING_GAP_TOKENS = (
    "no_treatment_column",
    "no_outcome_column",
    "insufficient_covariates",
    "no_vouched_negative_control",
    "structurally_missing_column",
)


def _nan_to_none(x: float | None) -> float | None:
    """NaN/inf are not valid strict JSON. Emit ``None`` instead so ``json.dumps(..., allow_nan=
    False)`` and the Pydantic consumer never choke. The vendored estimator returns
    ``neg_control_effect=float('nan')`` when no mapped control survives support-checking (a real
    abstain verdict); this is the emission-boundary sanitizer that keeps the manifest JSON-strict
    without touching the verbatim estimator."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _sanitize_effects(effects: dict[str, float]) -> dict[str, float]:
    """Drop NaN/inf residuals from the per-control effects dict (same JSON-strictness reason)."""
    return {k: v for k, v in effects.items()
            if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}


def _tokens(s: str) -> set[str]:
    return set(_TOKEN.findall(s.lower()))


@dataclass
class ColumnEntry:
    name: str
    dtype: str
    role: str                      # treatment_candidate | outcome_candidate | covariate
    distinct_count: int | None
    null_fraction: float
    has_variation: bool


@dataclass
class CapabilityLink:
    capability_name: str
    surfaces: list[str]
    treatment_column: str | None   # None -> not represented in the data -> LLM route
    linked: bool


@dataclass
class CoverageEntry:
    """One row of the pack-level coverage manifest (`Manifest.coverage`, #564).

    `missingness` is the three-way honesty verdict:
      - "present":     the column exists and is populated (null_fraction < BEHAVIOURAL_NULL_THRESHOLD)
      - "behavioural": the column exists but is sparsely populated — users just didn't do the
                       thing (BEHAVIOURAL_NULL_THRESHOLD <= null_fraction < 1.0)
      - "structural":  the company doesn't track it — the column is absent from the source
                       entirely, or present but 100% null

    `observed` is False for a column the config EXPECTED (a declared outcome/treatment or a
    mapped negative control) that the source doesn't have at all; its null_fraction is then
    unknown (None).
    """
    name: str
    missingness: str               # "present" | "structural" | "behavioural"
    null_fraction: float | None    # None when the column is absent from the source (unknown)
    observed: bool                 # True -> profiled from the source; False -> expected-but-absent


@dataclass
class CandidateEffect:
    treatment: str
    outcome: str
    covariates: list[str]
    supportable: bool
    support_reasons: list[str]
    n_treat: int
    n_control: int
    naive_ate: float | None
    t_learner_ate: float | None
    ipw_ate: float | None
    overlap_fraction: float | None
    naive_adjusted_gap: float | None
    sign_adjusted: str | None
    # --- Phase-1 negative-control verdict (the honesty gate before the empirical estimate ships)
    neg_controls: list[str]
    neg_control_effect: float | None
    guard_trips: bool | None
    guard_confidence: str          # "high" | "low/abstain" | "unverified"
    guard_reason: str | None
    # --- Full guard verdict fields (the exact keys the sidira-simulator consumer schema reads,
    #     backend/sidira_simulator/schemas/estimability.py::CandidateEffect). Additive/defaulted so
    #     the unsupportable / unverified rows and any direct constructor call stay valid.
    neg_control_effects: dict[str, float] = field(default_factory=dict)
    guard_floor: float | None = None      # from GuardedEstimate.floor (the |residual| trip threshold)
    guard_ratio: float | None = None      # the indistinguishability ratio the guard was run with
    # Per-candidate-control diagnostic pool for the reviewable lever card (#565): the treatment's
    # adjusted effect on EACH candidate control column (siblings AND covariate-role columns), keyed
    # by control name. A large |value| on a control the treatment cannot cause = confounding a human
    # reviewer should see. Default empty so every existing positional/keyword construction stays valid
    # and the default-off delta is additive-only (a single empty ``{}`` per candidate, absorbed by the
    # ``extra="ignore"`` consumer); only populated when build_manifest(nc_diagnostics=True) and only
    # for the supportable branch. Strictly dict[str, float] (JSON-safe, no NaN/None).
    nc_diagnostics: dict[str, float] = field(default_factory=dict)
    # --- Per-lever grounding-completeness verdict (#564). A lever is grounding-complete iff it
    # has (a) a treatment/proxy column, (b) an outcome column, (c) >= MIN_GROUNDING_COVARIATES
    # adjustment covariates, and (d) >= 1 usable vouched negative control (consistent with the
    # guard_confidence="unverified" semantics — no vouched NC is simply that gap, never a crash).
    # `grounding_gaps` draws ONLY from GROUNDING_GAP_TOKENS and is empty iff grounding_complete.
    # ORTHOGONAL to `supportable` (statistical estimability): a row can be grounding-complete yet
    # unsupportable (thin arms) and vice versa; the router requires BOTH before trusting the
    # empirical estimate. Additive/defaulted so existing constructor calls stay valid.
    grounding_complete: bool = False
    grounding_gaps: list[str] = field(default_factory=list)


@dataclass
class Manifest:
    manifest_version: str
    customer_id: str
    setup_id: str
    pack_version: str
    feature_store: str
    columns: list[ColumnEntry] = field(default_factory=list)
    capability_links: list[CapabilityLink] = field(default_factory=list)
    candidate_effects: list[CandidateEffect] = field(default_factory=list)
    truncated: dict = field(default_factory=dict)   # {"treatments_dropped": n, "outcomes_dropped": m}
    # Pack-level coverage manifest (#564): every profiled column plus every config-expected-but-
    # absent column, each with a structural/behavioural/present missingness verdict. Placed after
    # `truncated` so the existing positional construction stays valid.
    coverage: list[CoverageEntry] = field(default_factory=list)
    built_at: str | None = None    # stamped by the caller (worker), kept out for determinism


def _classify(profiles: list[dict],
              explicit_outcomes: set[str] | None = None,
              extra_outcome_hints: set[str] | None = None) -> list[ColumnEntry]:
    """Tag outcomes vs covariates. Treatment status is decided later, by LINKAGE — a
    covariate must never be demoted just because its name overlaps a capability term.

    Outcome discovery is config-driven (non-commerce verticals starve the hardcoded
    commerce `OUTCOME_HINTS`): a column is an outcome candidate if its name is in the
    customer-declared `explicit_outcomes` OR its tokens overlap the effective hint set
    (`OUTCOME_HINTS` plus any `extra_outcome_hints`). Both params default None -> the
    behavior is BYTE-IDENTICAL to the hint-only classifier."""
    hints = set(OUTCOME_HINTS) | (extra_outcome_hints or set())
    declared = explicit_outcomes or set()
    out = []
    for p in profiles:
        name, dc = p["name"], p.get("distinct_count")
        is_outcome = name in declared or bool(_tokens(name) & hints)
        role = "outcome_candidate" if is_outcome else "covariate"
        out.append(ColumnEntry(name=name, dtype=p.get("dtype", "unknown"), role=role,
                               distinct_count=dc, null_fraction=float(p.get("null_fraction", 0.0)),
                               has_variation=(dc is None or dc >= 2)))
    return out


def _link(capabilities: list[dict], cols: list[ColumnEntry]) -> list[CapabilityLink]:
    # eligible treatment columns = non-outcome, low-cardinality, varying (a lever, not a metric)
    treat_cols = [c for c in cols if c.role != "outcome_candidate" and c.has_variation
                  and c.distinct_count is not None and c.distinct_count <= 10]
    links = []
    for cap in capabilities:
        ctoks = _tokens(cap["capability_name"])
        best, best_score = None, 0
        for c in treat_cols:
            score = len(_tokens(c.name) & ctoks)
            if score > best_score:
                best, best_score = c.name, score
        links.append(CapabilityLink(capability_name=cap["capability_name"],
                                     surfaces=list(cap.get("surfaces", [])),
                                     treatment_column=best, linked=best is not None))
    return links


def _lowcard_treatments(cols: list[ColumnEntry]) -> list[str]:
    """Capability-free lever discovery: non-outcome, varying, BINARY (distinct==2) columns.

    Used when no crawler Capabilities are available (e.g. a frame-only build), where the
    token-overlap linkage yields nothing. The router still needs candidate levers, so we
    fall back to data shape: a binary non-metric column IS a plausible treatment.

    Cardinality is restricted to exactly 2 because the empirical estimator assumes a 0/1
    treatment (T==1 vs T==0). The {0,1}-VALUE check needs the frame and is applied via
    `binary_treatment_cols` in `build_manifest`; this prunes by cardinality alone. Multi-level
    / non-binary levers are out of scope for the empirical route (they go to the LLM) — a
    validated scope boundary, not a TODO.
    """
    return sorted({c.name for c in cols
                   if c.role != "outcome_candidate" and c.has_variation
                   and c.distinct_count == 2})


def _build_coverage(cols: list[ColumnEntry], expected: set[str]) -> list[CoverageEntry]:
    """Classify every profiled column — plus every config-EXPECTED name the source lacks — into
    the three-way missingness verdict (#564). `expected` is the union of declared outcomes/
    treatments and mapped negative-control names; an expected name absent from the profiles is
    the purest structural miss (the company doesn't have the column at all)."""
    entries: list[CoverageEntry] = []
    for c in cols:
        nf = _nan_to_none(float(c.null_fraction))
        if nf is not None and nf >= 1.0:
            miss = "structural"        # column exists but is 100% null -> effectively untracked
        elif nf is not None and nf >= BEHAVIOURAL_NULL_THRESHOLD:
            miss = "behavioural"       # tracked, sparsely populated -> users didn't do the thing
        else:
            miss = "present"           # populated (or null_fraction unknown -> assume present)
        entries.append(CoverageEntry(name=c.name, missingness=miss, null_fraction=nf,
                                     observed=True))
    known = {c.name for c in cols}
    for name in sorted(expected - known):
        entries.append(CoverageEntry(name=name, missingness="structural", null_fraction=None,
                                     observed=False))
    return entries


def _grounding_verdict(t_present: bool, y_present: bool, covariates: list[str],
                       usable_ncs: list[str], referenced: list[str],
                       structural_cols: set[str]) -> tuple[bool, list[str]]:
    """The per-lever grounding-completeness check (#564): (complete, gaps).

    Gaps are drawn from the FIXED ``GROUNDING_GAP_TOKENS`` vocabulary, emitted in canonical
    order; an empty list means grounding-complete. ``referenced`` is every column the lever's
    empirical route would touch (T, Y, adjustment covariates, mapped NC names) — any of them
    being structurally missing is a gap, so a structurally-absent column feeds THIS check
    instead of being silently zero-imputed downstream.
    """
    gaps: list[str] = []
    if not t_present:
        gaps.append("no_treatment_column")
    if not y_present:
        gaps.append("no_outcome_column")
    if len(covariates) < MIN_GROUNDING_COVARIATES:
        gaps.append("insufficient_covariates")
    if not usable_ncs:
        gaps.append("no_vouched_negative_control")
    if any(c in structural_cols for c in referenced):
        gaps.append("structurally_missing_column")
    return (not gaps, gaps)


def build_manifest(customer_id: str, setup_id: str, pack_version: str, feature_store: str,
                   column_profiles: list[dict], capabilities: list[dict],
                   load_frame: Callable[[list[str]], pd.DataFrame],
                   max_treatments: int = 8, max_outcomes: int = 8,
                   neg_control_map: dict[str, list[str]] | None = None,
                   binary_treatment_cols: set[str] | None = None,
                   explicit_outcomes: set[str] | None = None,
                   extra_outcome_hints: set[str] | None = None,
                   explicit_treatments: set[str] | None = None,
                   nc_diagnostics: bool = False) -> Manifest:
    """Join columns x capabilities and PRE-FIT the candidate (T->Y) effect grid.

    Each SUPPORTABLE effect carries a negative-control verdict ONLY when the caller/LLM has
    vouched for one via `neg_control_map` (outcome column Y -> [valid NC columns]); sibling
    outcomes are NOT used as controls automatically (a co-moved metric would false-trip the
    guard). Absent a vouched control, the effect is computed but left `guard_confidence=
    "unverified"` so the router routes it to the LLM. The grid is capped at
    `max_treatments` x `max_outcomes` and any excess is recorded on `Manifest.truncated`
    rather than silently dropped.

    The empirical estimator assumes a 0/1 treatment, so a discovered lever must also carry
    values exactly {0,1}. Column profiles alone can't confirm that, so `binary_treatment_cols`
    (when supplied — e.g. by `build_manifest_from_frame`, which has the frame) restricts
    treatment promotion to those verified-binary columns. Non-binary levers are out of scope
    for the empirical route and fall through to the LLM; `support_check` also rejects any that
    slip past here, so the non-frame path stays protected (defense in depth).

    Config-driven discovery (all three params default None -> byte-identical to the hint-only
    path): `explicit_outcomes`/`extra_outcome_hints` widen outcome recognition beyond the
    hardcoded commerce hints (threaded into `_classify`), and `explicit_treatments` restricts
    the discovered levers to the customer-declared set. HONESTY: a declared outcome still must
    be numeric to enter the empirical grid, and a declared treatment still must be binary
    {0,1} (verified via `binary_treatment_cols`); non-conforming declarations are RECORDED on
    `Manifest.truncated` (`outcomes_declared_not_numeric`, `outcomes_declared_not_found`,
    `treatments_declared_not_binary`), never silently dropped.
    """
    cols = _classify(column_profiles, explicit_outcomes, extra_outcome_hints)
    links = _link(capabilities, cols)
    if capabilities:
        treatments = sorted({lk.treatment_column for lk in links if lk.treatment_column})
    else:
        treatments = _lowcard_treatments(cols)
    if binary_treatment_cols is not None:
        treatments = [t for t in treatments if t in binary_treatment_cols]

    # Truncation/accounting ledger (grid cap + config-declaration accounting). NEVER silently drop.
    truncated: dict = {}

    # Config-declared levers (customer's real KPIs). Restrict discovery to the declared set, then
    # record any declared lever that did NOT survive — with an ACCURATE reason. Only name a lever
    # `not_binary` when we can PROVE it (it isn't in the verified `binary_treatment_cols`); a lever
    # that IS binary but was still dropped (never linked to a capability in the capability path, or
    # it's an outcome-role column) is recorded generically as `dropped`, never mislabeled.
    if explicit_treatments is not None:
        treatments = [t for t in treatments if t in explicit_treatments]
        dropped = explicit_treatments - set(treatments)
        if dropped:
            if binary_treatment_cols is not None:
                not_binary = sorted(d for d in dropped if d not in binary_treatment_cols)
                other = sorted(d for d in dropped if d in binary_treatment_cols)
                if not_binary:
                    truncated["treatments_declared_not_binary"] = not_binary
                if other:
                    truncated["treatments_declared_dropped"] = other
            else:
                # Binariness unverifiable (no frame) — don't claim a reason we can't prove.
                truncated["treatments_declared_dropped"] = sorted(dropped)

    # Outcomes must be numeric for the empirical estimator (it does Y.to_numpy(float)). A
    # hint-named but non-numeric column (e.g. a `signup_date` matched by the "signup" hint) is
    # NOT empirically estimable and must never enter the grid — route it to the LLM. Use an
    # ALLOWLIST of numeric dtype labels (the `_dtype_of` vocabulary emits "number"/"bool"/
    # "string"), so any non-numeric label a generic caller supplies ("object", "date",
    # "datetime", "category", "unknown", ...) is excluded, not just "string"/"text". Mirrors
    # the covariate dtype filter below.
    outcomes = sorted(c.name for c in cols if c.role == "outcome_candidate" and c.has_variation
                      and c.dtype in ("number", "bool"))

    # Config-declared outcomes that failed to enter the grid — record with an ACCURATE reason,
    # don't silently drop. A declared name present but excluded is EITHER non-numeric OR present-
    # but-non-varying (a constant column can't be an outcome); these are distinct and recorded
    # under separate keys so an operator isn't misled. A declared name ABSENT from the frame is a
    # third case.
    if explicit_outcomes:
        by_name = {c.name: c for c in cols}
        excluded = [n for n in explicit_outcomes if n in by_name and n not in outcomes]
        not_numeric = sorted(n for n in excluded if by_name[n].dtype not in ("number", "bool"))
        non_varying = sorted(n for n in excluded
                             if by_name[n].dtype in ("number", "bool") and not by_name[n].has_variation)
        not_found = sorted(n for n in explicit_outcomes if n not in by_name)
        if not_numeric:
            truncated["outcomes_declared_not_numeric"] = not_numeric
        if non_varying:
            truncated["outcomes_declared_non_varying"] = non_varying
        if not_found:
            truncated["outcomes_declared_not_found"] = not_found

    # Cap the grid — keep the first N (sorted), record the overflow. NEVER silently drop.
    if len(treatments) > max_treatments:
        truncated["treatments_dropped"] = len(treatments) - max_treatments
        treatments = treatments[:max_treatments]
    if len(outcomes) > max_outcomes:
        truncated["outcomes_dropped"] = len(outcomes) - max_outcomes
        outcomes = outcomes[:max_outcomes]

    # promote kept treatment columns to treatment_candidate in the catalog (after linkage/cap)
    treat_set = set(treatments)
    for c in cols:
        if c.name in treat_set:
            c.role = "treatment_candidate"

    # Covariates feed the propensity/standardization fitters (X.to_numpy(float)), so they too
    # must be numeric — ALLOWLIST numeric dtype labels (see the outcome filter above) rather
    # than denylisting only "string"/"text", which would leak "object"/"datetime"/"category".
    covars = [c.name for c in cols if c.role == "covariate" and c.has_variation
              and c.dtype in ("number", "bool")]

    # The store's known column names (from the profiles). A mapped negative control that names a
    # column NOT in the store (a stale/typo'd map entry) must never reach `load_frame(need)` — for
    # the frame-only `load_frame=lambda cols: df[cols]` an absent name raises KeyError and would
    # abort the ENTIRE manifest (total abstain). Filter such names out here so the candidate simply
    # degrades to the no-available-NC "unverified" path instead.
    known_cols = {c.name for c in cols}

    # --- Coverage manifest (#564): classify every profiled column, plus every config-EXPECTED
    # column the source lacks (declared outcomes/treatments + mapped NC names). Structural
    # columns feed the per-lever grounding-completeness check below — a lever needing an
    # untracked column is marked ungroundable and routes to the LLM, never zero-imputed.
    nc_map = neg_control_map or {}
    expected = (set(explicit_outcomes or ()) | set(explicit_treatments or ())
                | set(nc_map.keys()) | {c for ncs_ in nc_map.values() for c in ncs_})
    coverage = _build_coverage(cols, expected)
    structural_cols = {e.name for e in coverage if e.missingness == "structural"}

    # Candidate-control pool for the reviewable lever-card diagnostics (#565), computed only when
    # `nc_diagnostics` is on. It is EVERY numeric+varying column (spanning sibling-outcome AND
    # covariate-role columns) — the card shows the treatment's adjusted effect on all of them so a
    # human can see which ones move. Sorted for deterministic ordering + truncation. Per (T,Y) we
    # additionally drop C==T and C==Y at the estimate site.
    #   * Built FROM `cols` (the profiled column set), so nc_diag_pool ⊆ known_cols by construction —
    #     the extended `need` load below can never ask load_frame for a column the store didn't
    #     profile (same guarantee the existing X/vouched load already relies on); the post-load
    #     `c in df.columns` re-intersection is belt-and-suspenders for mid-build schema drift.
    #   * Empty when the flag is off, so the default path loads no extra columns and does no extra
    #     compute. The ONLY default-off manifest delta is an additive empty ``nc_diagnostics: {}`` per
    #     candidate (like the already-shipped ``neg_control_effects: {}`` / ``guard_floor`` siblings);
    #     the consumer parses with ``extra="ignore"`` so it is non-breaking, and there is no
    #     content-hash gate that the extra key could spuriously trip.
    nc_diag_pool = (sorted(c.name for c in cols if c.has_variation and c.dtype in ("number", "bool"))
                    if nc_diagnostics else [])

    effects: list[CandidateEffect] = []
    for T in treatments:
        for Y in outcomes:
            # Caller/LLM-vouched negative controls for this outcome (the MAP step); no
            # sibling-outcome auto-NC — a co-moved sibling would false-trip the guard. Restrict to
            # names known to the store BEFORE building `need`, so an absent NC never crashes the
            # load; re-intersect with df.columns after load as defense for real (Snowflake)
            # load_frames that could drop a column.
            vouched = [c for c in nc_map.get(Y, []) if c in known_cols and c != Y and c != T]
            # The negative controls must NOT also be adjustment covariates — adjusting for a column
            # while testing it as an outcome-side control neuters its residual (~0) and FALSELY
            # passes the guard. Exclude them from X per-outcome so the recorded covariates match
            # what is actually adjusted for, and the guard tests genuine controls (the guard also
            # strips them defensively; this keeps CandidateEffect.covariates honest).
            X = [c for c in covars if c != T and c not in vouched]
            # When nc_diagnostics is on, also load the whole numeric+varying pool so the per-control
            # diagnostics below can be estimated from this SAME single (T,Y) frame — still one query.
            # dict.fromkeys dedupes, so adding pool names that already appear in X/vouched is free.
            need = list(dict.fromkeys([T, Y, *X, *vouched, *nc_diag_pool]))   # one query, deduped
            df = load_frame(need)                               # the ONE query, at build time
            ncs = [c for c in vouched if c in df.columns]
            # Grounding-completeness verdict (#564), independent of the statistical support
            # check below. T/Y exist by grid construction; the NC criterion counts only USABLE
            # vouched controls (a 100%-null vouched control is structurally missing, not usable);
            # `referenced` includes the RAW mapped NC names (pre known_cols filter) so a mapped
            # control absent from the store surfaces as `structurally_missing_column` too.
            mapped_raw = [c for c in nc_map.get(Y, []) if c != Y and c != T]
            usable_ncs = [c for c in ncs if c not in structural_cols]
            grounding_complete, grounding_gaps = _grounding_verdict(
                True, True, X, usable_ncs, [T, Y, *X, *mapped_raw], structural_cols)
            sup = support_check(df, T, Y, X)
            if not sup.supportable:
                effects.append(CandidateEffect(
                    T, Y, X, False, sup.reasons, sup.n_treat, sup.n_control,
                    None, None, None, None, None, None,
                    neg_controls=[], neg_control_effect=None, guard_trips=None,
                    guard_confidence="unverified", guard_reason=None,
                    neg_control_effects={}, guard_floor=None, guard_ratio=None,
                    grounding_complete=grounding_complete, grounding_gaps=grounding_gaps))
                continue
            if ncs:
                g = negative_control_guard(df, T, Y, ncs, X,
                                           floor=NC_GUARD_FLOOR, ratio=NC_GUARD_RATIO)
                est = g.estimate
                # Sanitize NaN/inf at the emission boundary (JSON-strictness) — the guard returns
                # neg_control_effect=NaN when no mapped control survives support-checking; keep the
                # rest of that real abstain verdict (trips=True, low/abstain, reason, floor/ratio).
                neg_control_effect, guard_trips = _nan_to_none(g.neg_control_effect), g.trips
                guard_confidence, guard_reason = g.confidence, g.reason
                neg_control_effects = _sanitize_effects(g.neg_control_effects)
                guard_floor, guard_ratio = g.floor, NC_GUARD_RATIO
            else:
                est = estimate(df, T, Y, X)
                neg_control_effect, guard_trips = None, None
                guard_confidence, guard_reason = "unverified", None
                neg_control_effects = {}
                guard_floor, guard_ratio = None, None
            # Per-candidate-control diagnostics for the reviewable lever card (#565): the treatment's
            # adjusted effect on each candidate control. Computed only when the flag is on and only in
            # this supportable branch (the estimate below needs a supportable T->Y arm). The pool is
            # re-intersected with df.columns (KeyError-safety for a Snowflake load that drops a column,
            # mirroring the `ncs` handling) and capped at MAX_NC_DIAGNOSTICS.
            diag: dict[str, float] = {}
            if nc_diagnostics:
                pool = [c for c in nc_diag_pool if c != T and c != Y and c in df.columns]
                if len(pool) > MAX_NC_DIAGNOSTICS:
                    # PER-CANDIDATE max, not a grid total: the key name says so explicitly so an
                    # operator can't misread it as the summed count of skipped diagnostics. Every
                    # supportable candidate diagnoses the same-size pool, so this max is the number of
                    # controls dropped from the WORST (largest) card — the honest "up to N per card".
                    dropped_n = len(pool) - MAX_NC_DIAGNOSTICS
                    truncated["nc_diagnostics_max_dropped_per_candidate"] = max(
                        truncated.get("nc_diagnostics_max_dropped_per_candidate", 0), dropped_n)
                    pool = pool[:MAX_NC_DIAGNOSTICS]
                for C in pool:
                    # The control must NOT be in its own adjustment set — adjusting for it while
                    # testing it neuters the residual to ~0 (the same self-neuter bug fixed for the
                    # guard). Estimate on the covariates MINUS C.
                    X_c = [x for x in X if x != C]
                    if not support_check(df, T, C, X_c).supportable:
                        continue                                # degenerate/thin arm -> no diagnostic
                    val = _nan_to_none(estimate(df, T, C, X_c).t_learner_ate)
                    if val is not None:                         # keep the dict strictly float-valued
                        diag[C] = float(val)
            effects.append(CandidateEffect(
                T, Y, X, True, [], sup.n_treat, sup.n_control,
                est.naive_ate, est.t_learner_ate, est.ipw_ate,
                est.overlap_fraction, est.naive_adjusted_gap, est.sign_adjusted,
                neg_controls=ncs, neg_control_effect=neg_control_effect,
                guard_trips=guard_trips, guard_confidence=guard_confidence,
                guard_reason=guard_reason, neg_control_effects=neg_control_effects,
                guard_floor=guard_floor, guard_ratio=guard_ratio, nc_diagnostics=diag,
                grounding_complete=grounding_complete, grounding_gaps=grounding_gaps))

    # --- Ungroundable stubs for DECLARED levers/outcomes that are structurally missing (#564).
    # Today a declared column absent from the store (or 100% null) only lands in the `truncated`
    # accounting ledger; the router never sees it as a lever. Emit an explicit unsupportable
    # candidate row for each so the consumer routes it to the LLM WITH a reason — the honest
    # alternative to silent zero-imputation. Zero extra compute (no load_frame call). A declared
    # structural treatment is paired with each surviving outcome (or, when none survived, each
    # declared structural outcome); a declared structural outcome is paired with each surviving
    # treatment. A structural declaration with no partner at all emits no row (there is nothing
    # to route) and remains visible in `coverage` + `truncated`.
    def _ungroundable_stub(t: str, y: str) -> CandidateEffect:
        mapped_raw = [c for c in nc_map.get(y, []) if c != y and c != t]
        vouched = [c for c in mapped_raw if c in known_cols]
        usable = [c for c in vouched if c not in structural_cols]
        # Mirror the grid's X: candidate covariates minus the lever and its vouched controls.
        x_stub = [c for c in covars if c != t and c not in vouched]
        complete, gaps = _grounding_verdict(
            t not in structural_cols, y not in structural_cols,
            x_stub, usable, [t, y, *x_stub, *mapped_raw], structural_cols)
        return CandidateEffect(
            t, y, x_stub, False,
            ["declared column structurally missing (ungroundable; see grounding_gaps)"],
            0, 0, None, None, None, None, None, None,
            neg_controls=[], neg_control_effect=None, guard_trips=None,
            guard_confidence="unverified", guard_reason=None,
            neg_control_effects={}, guard_floor=None, guard_ratio=None,
            grounding_complete=complete, grounding_gaps=gaps)

    structural_treatments = sorted(t for t in (explicit_treatments or ()) if t in structural_cols)
    structural_outcomes = sorted(y for y in (explicit_outcomes or ()) if y in structural_cols)
    for t in structural_treatments:
        for y in (outcomes or structural_outcomes):
            effects.append(_ungroundable_stub(t, y))
    for y in structural_outcomes:
        for t in treatments:                      # disjoint from the loop above (kept levers only)
            effects.append(_ungroundable_stub(t, y))

    return Manifest("1.0", customer_id, setup_id, pack_version, feature_store,
                    cols, links, effects, truncated, coverage)


def _dtype_of(s: pd.Series) -> str:
    """Map a pandas dtype to the manifest's coarse dtype vocabulary."""
    if ptypes.is_bool_dtype(s):
        return "bool"
    if ptypes.is_numeric_dtype(s):
        return "number"
    return "string"


def _safe_nunique(s: pd.Series) -> int | None:
    """Distinct count that tolerates unhashable cell values.

    JSON/dict/list-valued columns (e.g. the synthetic fixture's `json_counts` /
    `json_ratios` when materialised as dict or list cells) raise
    ``TypeError: unhashable type`` inside ``nunique()``. Return ``None`` for those rather
    than crashing the whole manifest: a ``None`` distinct_count is never promoted to a
    treatment (``_lowcard_treatments`` requires ``distinct_count == 2``) and such columns
    are non-numeric, so the dtype filter keeps them out of covariates and outcomes too."""
    try:
        return int(s.nunique())
    except TypeError:
        return None


def _is_binary_01(s: pd.Series) -> bool:
    """True iff the column's non-null values are exactly ``{0, 1}``.

    Returns ``False`` on ANY exception (e.g. unhashable dict/list cells raising inside
    ``pd.unique``), so an unhashable column is simply never a treatment candidate."""
    try:
        return set(pd.unique(s.dropna())) == {0, 1}
    except Exception:
        return False


def default_neg_control_map(column_profiles: list[dict],
                            *, max_controls: int = 5,
                            explicit_outcomes: set[str] | None = None,
                            extra_outcome_hints: set[str] | None = None) -> dict[str, list[str]]:
    """Build-time FALLBACK negative-control map (the negative-control-OUTCOME design, auto-nominated).

    For each outcome-candidate column ``Y``, nominate up to ``max_controls`` OTHER
    outcome-candidate columns (name != Y, has_variation, numeric) as its negative controls —
    "metrics the treatment shouldn't move." Returns ``{Y: [nc, ...]}`` for every eligible outcome
    that has at least one sibling; outcomes with no sibling are omitted.

    WHY this exists: without ANY negative-control map the manifest's guard never runs, so every
    candidate ships ``guard_trips=None`` and the downstream watchdog can only answer CANT_VALIDATE.
    This auto-map lets the guard actually run on a frame-only persona build, so candidates get a
    real ``guard_trips`` True/False and the watchdog can CONFIRM/CONTRADICT.

    HONEST CAVEAT — this is a HEURISTIC, not a vouched control. A lever frequently moves several
    metrics together (e.g. retention AND conversion), so some "other outcome" nominated here may be
    LEGITIMATELY affected by the treatment; using it as a negative control can then FALSE-TRIP the
    guard (a spurious abstain). That is deliberately conservative — a false trip only routes the
    effect to the LLM, it never ships a wrong empirical estimate. The reviewable-card / LLM
    nomination step (#565) later refines this into domain/LLM-vouched controls; this auto-map is
    the fallback, not the final word. It is used only when the caller supplies no explicit
    ``neg_control_map``.

    Only numeric ("number"/"bool"), varying outcome candidates are nominated — the exact set that
    can enter the empirical grid, so no nominated control is a dead entry the guard would skip.

    `explicit_outcomes`/`extra_outcome_hints` (default None -> unchanged) thread the same
    config-driven outcome discovery into the classifier, so config-declared (non-hint) outcomes
    are recognized as siblings and can be auto-nominated as each other's negative controls.
    """
    outcomes = [c for c in _classify(column_profiles, explicit_outcomes, extra_outcome_hints)
                if c.role == "outcome_candidate" and c.has_variation
                and c.dtype in ("number", "bool")]
    names = [c.name for c in outcomes]
    nc_map: dict[str, list[str]] = {}
    for y in names:
        controls = [n for n in names if n != y][:max_controls]
        if controls:
            nc_map[y] = controls
    return nc_map


def build_manifest_from_frame(df: pd.DataFrame, customer_id: str, setup_id: str,
                              pack_version: str, feature_store: str,
                              max_treatments: int = 8, max_outcomes: int = 8,
                              neg_control_map: dict[str, list[str]] | None = None,
                              max_rows: int = 50_000,
                              auto_neg_controls: bool = False,
                              outcomes: list[str] | None = None,
                              extra_outcome_hints: list[str] | None = None,
                              treatments: list[str] | None = None,
                              nc_diagnostics: bool = False) -> Manifest:
    """Frame-only convenience: derive column profiles from `df` itself and build the manifest.

    No crawler Capabilities are available in this repo, so treatment candidates are discovered
    from data shape (binary non-outcome columns); see `_lowcard_treatments`. The whole frame is
    in memory, so `load_frame` is a trivial column selector and we can confirm the {0,1}-VALUE
    constraint directly: only columns whose non-null values are exactly {0,1} are eligible
    levers (a 2-level {30,40} or a 3-level {0,1,2} column is NOT a 0/1 treatment and is out of
    scope for the empirical route). `neg_control_map` (outcome -> [vouched NC columns]) is
    passed through unchanged.

    `auto_neg_controls` (default False, so this convenience stays byte-compatible with callers
    that expect the conservative no-auto-NC contract) opts IN to the build-time FALLBACK: when it
    is set AND no explicit `neg_control_map` is supplied, `default_neg_control_map` nominates
    sibling outcomes as auto negative controls so the guard actually RUNS and candidates carry a
    real `guard_trips` True/False rather than None (the downstream CANT_VALIDATE). It is a
    heuristic that can false-trip — see `default_neg_control_map` for the honest caveat; the
    persona build (`run_pipeline`) opts in, direct callers get the conservative default.

    BOUNDED PRECOMPUTE: this build is synchronous before the pack returns, so its cost is
    capped. When `len(df) > max_rows` the effect grid is fit on a deterministic random sample
    of `max_rows` rows (`df.sample(n=max_rows, random_state=0)`) taken BEFORE profiling/
    load_frame, so BOTH the column profiles AND every `load_frame` slice come from the same
    sample. Enabling the manifest therefore adds only bounded build time (<= max_rows rows x
    the capped `max_treatments` x `max_outcomes` grid), never unbounded. The column catalog's
    `null_fraction`/`distinct_count` are computed on the sample, which is fine.

    Config-driven discovery (all default None -> unchanged): `outcomes` declares the customer's
    real KPI columns (recognized even when their names match no commerce hint),
    `extra_outcome_hints` widens the token hint set, and `treatments` restricts the discovered
    levers to the declared set. Declared outcomes still must be numeric and declared treatments
    still must be binary {0,1} to enter the empirical grid; non-conforming declarations are
    recorded on `Manifest.truncated`, never silently dropped (see `build_manifest`).

    `nc_diagnostics` (default False) opts IN to computing, per supportable candidate, the
    treatment's adjusted effect on every numeric+varying candidate control
    (`CandidateEffect.nc_diagnostics`) — the data a reviewable lever card (#565) shows so a human
    can vouch which controls actually reveal confounding. Bounded: capped at MAX_NC_DIAGNOSTICS.
    Default-off is behaviorally unchanged (no extra columns loaded, no extra compute); the only
    manifest delta is an additive empty ``nc_diagnostics: {}`` per candidate, which the
    ``extra="ignore"`` consumer absorbs (no content-hash gate exists to trip on it).
    """
    # Cap the estimability sample (the pack/clustering upstream already used the full df).
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0)
    column_profiles = [{
        "name": c,
        "dtype": _dtype_of(df[c]),
        "distinct_count": _safe_nunique(df[c]),
        "null_fraction": float(df[c].isna().mean()),
    } for c in df.columns]
    binary_treatment_cols = {c for c in df.columns if _is_binary_01(df[c])}
    # Config-driven discovery: convert the list params to sets (or None) once, threaded into both
    # the auto-NC map (so declared outcomes are recognized as siblings) and the builder. A bare
    # YAML scalar (`outcomes: completion_rate`) arrives as a str — wrap it so it becomes a single
    # column name, NOT `set("completion_rate")` (a set of characters that silently matches nothing).
    def _as_set(v):
        if not v:
            return None
        if isinstance(v, str):
            return {v}
        return set(v)
    explicit_outcomes = _as_set(outcomes)
    extra_hints = _as_set(extra_outcome_hints)
    explicit_treatments = _as_set(treatments)
    # Opt-in fallback: with no caller/LLM-vouched map, use the auto sibling-outcome negative
    # controls so the guard runs (else every candidate ships guard_trips=None). Heuristic +
    # conservative; see default_neg_control_map for the false-trip caveat.
    if neg_control_map is None and auto_neg_controls:
        neg_control_map = default_neg_control_map(
            column_profiles, explicit_outcomes=explicit_outcomes, extra_outcome_hints=extra_hints)
    return build_manifest(customer_id, setup_id, pack_version, feature_store,
                          column_profiles, capabilities=[],
                          load_frame=lambda cols: df[cols],
                          max_treatments=max_treatments, max_outcomes=max_outcomes,
                          neg_control_map=neg_control_map,
                          binary_treatment_cols=binary_treatment_cols,
                          explicit_outcomes=explicit_outcomes,
                          extra_outcome_hints=extra_hints,
                          explicit_treatments=explicit_treatments,
                          nc_diagnostics=nc_diagnostics)
