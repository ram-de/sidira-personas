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
import re
import pandas as pd
import pandas.api.types as ptypes

from .causal_estimator import support_check, estimate, negative_control_guard

OUTCOME_HINTS = ("order", "retention", "retain", "convert", "conversion", "visit", "spend",
                 "revenue", "churn", "engage", "active", "purchase", "signup", "subscribe")
_TOKEN = re.compile(r"[a-z0-9]+")


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
    built_at: str | None = None    # stamped by the caller (worker), kept out for determinism


def _classify(profiles: list[dict]) -> list[ColumnEntry]:
    """Tag outcomes vs covariates. Treatment status is decided later, by LINKAGE — a
    covariate must never be demoted just because its name overlaps a capability term."""
    out = []
    for p in profiles:
        name, dc = p["name"], p.get("distinct_count")
        role = "outcome_candidate" if (_tokens(name) & set(OUTCOME_HINTS)) else "covariate"
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


def build_manifest(customer_id: str, setup_id: str, pack_version: str, feature_store: str,
                   column_profiles: list[dict], capabilities: list[dict],
                   load_frame: Callable[[list[str]], pd.DataFrame],
                   max_treatments: int = 8, max_outcomes: int = 8,
                   neg_control_map: dict[str, list[str]] | None = None,
                   binary_treatment_cols: set[str] | None = None) -> Manifest:
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
    """
    cols = _classify(column_profiles)
    links = _link(capabilities, cols)
    if capabilities:
        treatments = sorted({lk.treatment_column for lk in links if lk.treatment_column})
    else:
        treatments = _lowcard_treatments(cols)
    if binary_treatment_cols is not None:
        treatments = [t for t in treatments if t in binary_treatment_cols]

    # Outcomes must be numeric for the empirical estimator (it does Y.to_numpy(float)). A
    # hint-named but non-numeric column (e.g. a `signup_date` string matched by the "signup"
    # hint) is NOT empirically estimable and must never enter the grid — route it to the LLM.
    # Mirrors the covariate dtype filter below.
    outcomes = sorted(c.name for c in cols if c.role == "outcome_candidate" and c.has_variation
                      and c.dtype not in ("string", "text"))

    # Cap the grid — keep the first N (sorted), record the overflow. NEVER silently drop.
    truncated: dict = {}
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

    covars = [c.name for c in cols if c.role == "covariate" and c.has_variation
              and c.dtype not in ("string", "text")]

    nc_map = neg_control_map or {}
    effects: list[CandidateEffect] = []
    for T in treatments:
        X = [c for c in covars if c != T]
        for Y in outcomes:
            need = list(dict.fromkeys([T, Y, *X, *nc_map.get(Y, [])]))  # one query, deduped
            df = load_frame(need)                               # the ONE query, at build time
            # Caller/LLM-vouched negative controls for this outcome (the MAP step); no
            # sibling-outcome auto-NC — a co-moved sibling would false-trip the guard.
            ncs = [c for c in nc_map.get(Y, []) if c in df.columns and c != Y and c != T]
            sup = support_check(df, T, Y, X)
            if not sup.supportable:
                effects.append(CandidateEffect(
                    T, Y, X, False, sup.reasons, sup.n_treat, sup.n_control,
                    None, None, None, None, None, None,
                    neg_controls=[], neg_control_effect=None, guard_trips=None,
                    guard_confidence="unverified", guard_reason=None))
                continue
            if ncs:
                g = negative_control_guard(df, T, Y, ncs, X)
                est = g.estimate
                neg_control_effect, guard_trips = g.neg_control_effect, g.trips
                guard_confidence, guard_reason = g.confidence, g.reason
            else:
                est = estimate(df, T, Y, X)
                neg_control_effect, guard_trips = None, None
                guard_confidence, guard_reason = "unverified", None
            effects.append(CandidateEffect(
                T, Y, X, True, [], sup.n_treat, sup.n_control,
                est.naive_ate, est.t_learner_ate, est.ipw_ate,
                est.overlap_fraction, est.naive_adjusted_gap, est.sign_adjusted,
                neg_controls=ncs, neg_control_effect=neg_control_effect,
                guard_trips=guard_trips, guard_confidence=guard_confidence,
                guard_reason=guard_reason))
    return Manifest("1.0", customer_id, setup_id, pack_version, feature_store,
                    cols, links, effects, truncated)


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


def build_manifest_from_frame(df: pd.DataFrame, customer_id: str, setup_id: str,
                              pack_version: str, feature_store: str,
                              max_treatments: int = 8, max_outcomes: int = 8,
                              neg_control_map: dict[str, list[str]] | None = None) -> Manifest:
    """Frame-only convenience: derive column profiles from `df` itself and build the manifest.

    No crawler Capabilities are available in this repo, so treatment candidates are discovered
    from data shape (binary non-outcome columns); see `_lowcard_treatments`. The whole frame is
    in memory, so `load_frame` is a trivial column selector and we can confirm the {0,1}-VALUE
    constraint directly: only columns whose non-null values are exactly {0,1} are eligible
    levers (a 2-level {30,40} or a 3-level {0,1,2} column is NOT a 0/1 treatment and is out of
    scope for the empirical route). `neg_control_map` (outcome -> [vouched NC columns]) is
    passed through unchanged.
    """
    column_profiles = [{
        "name": c,
        "dtype": _dtype_of(df[c]),
        "distinct_count": _safe_nunique(df[c]),
        "null_fraction": float(df[c].isna().mean()),
    } for c in df.columns]
    binary_treatment_cols = {c for c in df.columns if _is_binary_01(df[c])}
    return build_manifest(customer_id, setup_id, pack_version, feature_store,
                          column_profiles, capabilities=[],
                          load_frame=lambda cols: df[cols],
                          max_treatments=max_treatments, max_outcomes=max_outcomes,
                          neg_control_map=neg_control_map,
                          binary_treatment_cols=binary_treatment_cols)
