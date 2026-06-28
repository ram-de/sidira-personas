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
    """Capability-free lever discovery: non-outcome, varying, low-cardinality (2..10) columns.

    Used when no crawler Capabilities are available (e.g. a frame-only build), where the
    token-overlap linkage yields nothing. The router still needs candidate levers, so we
    fall back to data shape: a small-cardinality non-metric column IS a plausible treatment.
    """
    return sorted({c.name for c in cols
                   if c.role != "outcome_candidate" and c.has_variation
                   and c.distinct_count is not None and 2 <= c.distinct_count <= 10})


def build_manifest(customer_id: str, setup_id: str, pack_version: str, feature_store: str,
                   column_profiles: list[dict], capabilities: list[dict],
                   load_frame: Callable[[list[str]], pd.DataFrame],
                   max_treatments: int = 8, max_outcomes: int = 8) -> Manifest:
    """Join columns x capabilities and PRE-FIT the candidate (T->Y) effect grid.

    Each SUPPORTABLE effect carries a negative-control verdict (using the OTHER outcome
    candidates as controls); the grid is capped at `max_treatments` x `max_outcomes` and any
    excess is recorded on `Manifest.truncated` rather than silently dropped.
    """
    cols = _classify(column_profiles)
    links = _link(capabilities, cols)
    if capabilities:
        treatments = sorted({lk.treatment_column for lk in links if lk.treatment_column})
    else:
        treatments = _lowcard_treatments(cols)

    outcomes = sorted(c.name for c in cols if c.role == "outcome_candidate" and c.has_variation)

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

    effects: list[CandidateEffect] = []
    for T in treatments:
        X = [c for c in covars if c != T]
        for Y in outcomes:
            ncs = [o for o in outcomes if o != Y]               # other outcome candidates = controls
            need = list(dict.fromkeys([T, Y, *X, *ncs]))        # one feature-store query, deduped
            df = load_frame(need)                               # the ONE query, at build time
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


def build_manifest_from_frame(df: pd.DataFrame, customer_id: str, setup_id: str,
                              pack_version: str, feature_store: str,
                              max_treatments: int = 8, max_outcomes: int = 8) -> Manifest:
    """Frame-only convenience: derive column profiles from `df` itself and build the manifest.

    No crawler Capabilities are available in this repo, so treatment candidates are discovered
    from data shape (low-cardinality non-outcome columns); see `_lowcard_treatments`. The whole
    frame is in memory, so `load_frame` is a trivial column selector.
    """
    column_profiles = [{
        "name": c,
        "dtype": _dtype_of(df[c]),
        "distinct_count": int(df[c].nunique()),
        "null_fraction": float(df[c].isna().mean()),
    } for c in df.columns]
    return build_manifest(customer_id, setup_id, pack_version, feature_store,
                          column_profiles, capabilities=[],
                          load_frame=lambda cols: df[cols],
                          max_treatments=max_treatments, max_outcomes=max_outcomes)
