"""Coverage-manifest + per-lever grounding-completeness contract tests (sidira-simulator#564).

Two additive producer contracts pinned here, both consumed verbatim by the sidira-simulator
schema:

1. `Manifest.coverage` — a top-level `coverage` section classifying every profiled column (plus
   every config-expected-but-absent column) as "present" / "behavioural" / "structural" under the
   field name `missingness`, with `null_fraction` where known.
2. `CandidateEffect.grounding_complete` / `.grounding_gaps` — the per-lever grounding check:
   complete iff treatment column + outcome column + sufficient covariates + >=1 usable vouched
   negative control; gaps drawn ONLY from the fixed GROUNDING_GAP_TOKENS vocabulary. A declared
   lever/outcome whose column is structurally missing yields an explicit ungroundable stub row
   (LLM route with a reason), never a silent zero-impute.

Fixtures are seeded numpy — deterministic, no external CSVs.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd

from sidira_personas.estimability import (
    BEHAVIOURAL_NULL_THRESHOLD,
    GROUNDING_GAP_TOKENS,
    build_manifest_from_frame,
)

# The exact keys the consumer schema reads off each coverage entry / candidate effect.
COVERAGE_ENTRY_KEYS = {"name", "missingness", "null_fraction", "observed"}
GROUNDING_KEYS = {"grounding_complete", "grounding_gaps"}


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _store(seed: int = 101, n: int = 4000, with_covariate: bool = True) -> pd.DataFrame:
    """Binary lever confounded by a latent factor, a hint-named outcome, and a valid NC."""
    rng = np.random.default_rng(seed)
    conf = rng.standard_normal(n)
    flag = (rng.random(n) < _sig(1.3 * conf)).astype(int)
    retention = (rng.random(n) < _sig(2.0 * conf - 1.0 * flag)).astype(float)
    pre_signup = (rng.random(n) < _sig(1.8 * conf)).astype(float)   # NC (outcome-hinted name)
    cols = {"treatment_flag": flag, "retention": retention, "pre_signup": pre_signup}
    if with_covariate:
        cols["other"] = rng.standard_normal(n)
    return pd.DataFrame(cols)


# --- 1. the complete lever ----------------------------------------------------------------

def test_complete_lever_is_grounding_complete_with_empty_gaps():
    m = build_manifest_from_frame(_store(), "acme", "default", "v_1", "fixture://complete",
                                  neg_control_map={"retention": ["pre_signup"]})
    ce = {(e.treatment, e.outcome): e for e in m.candidate_effects}[
        ("treatment_flag", "retention")]
    assert ce.supportable is True
    assert ce.grounding_complete is True
    assert ce.grounding_gaps == []                 # empty list IS the complete signal


# --- 2. each gap token fires from a real build path ----------------------------------------

def test_gap_no_vouched_negative_control():
    """No NC map (and auto-NC off, the direct-caller default) -> the only gap is the NC one —
    consistent with the existing guard_confidence="unverified" semantics."""
    m = build_manifest_from_frame(_store(), "acme", "default", "v_1", "fixture://no-nc")
    ce = {(e.treatment, e.outcome): e for e in m.candidate_effects}[
        ("treatment_flag", "retention")]
    assert ce.supportable is True
    assert ce.guard_confidence == "unverified"     # unchanged NC-vouching semantics
    assert ce.grounding_complete is False
    assert ce.grounding_gaps == ["no_vouched_negative_control"]


def test_gap_insufficient_covariates():
    """A store with no numeric covariate for adjustment -> insufficient_covariates."""
    m = build_manifest_from_frame(_store(with_covariate=False), "acme", "default", "v_1",
                                  "fixture://no-covars",
                                  neg_control_map={"retention": ["pre_signup"]})
    ce = {(e.treatment, e.outcome): e for e in m.candidate_effects}[
        ("treatment_flag", "retention")]
    assert ce.covariates == []
    assert ce.grounding_complete is False
    assert ce.grounding_gaps == ["insufficient_covariates"]


def test_gap_no_treatment_column_via_declared_absent_lever():
    """A config-declared lever whose column the store simply doesn't have -> an explicit
    ungroundable stub row (LLM route with a reason), not just a truncated-ledger entry."""
    m = build_manifest_from_frame(_store(), "acme", "default", "v_1", "fixture://ghost-lever",
                                  neg_control_map={"retention": ["pre_signup"]},
                                  treatments=["ghost_flag"])
    stubs = [e for e in m.candidate_effects if e.treatment == "ghost_flag"]
    assert stubs, "expected an ungroundable stub row for the declared-but-absent lever"
    ce = {e.outcome: e for e in stubs}["retention"]
    assert ce.supportable is False
    assert ce.grounding_complete is False
    assert ce.grounding_gaps == ["no_treatment_column", "structurally_missing_column"]
    # the accounting ledger still records the drop (unchanged)
    assert "ghost_flag" in m.truncated.get("treatments_declared_not_binary", [])


def test_gap_no_outcome_column_via_declared_absent_outcome():
    rng = np.random.default_rng(7)
    n = 3000
    df = pd.DataFrame({
        "variant_flag": (rng.random(n) < 0.5).astype(int),      # binary lever (kept)
        "score_a": rng.standard_normal(n),                      # covariate
        "score_b": rng.standard_normal(n),                      # vouched NC for the ghost KPI
    })
    m = build_manifest_from_frame(df, "acme", "default", "v_1", "fixture://ghost-kpi",
                                  outcomes=["ghost_metric"],
                                  neg_control_map={"ghost_metric": ["score_b"]})
    ce = {(e.treatment, e.outcome): e for e in m.candidate_effects}[
        ("variant_flag", "ghost_metric")]
    assert ce.supportable is False
    assert ce.grounding_complete is False
    assert ce.grounding_gaps == ["no_outcome_column", "structurally_missing_column"]
    assert m.truncated.get("outcomes_declared_not_found") == ["ghost_metric"]


def test_gap_structurally_missing_vouched_nc():
    """A vouched NC column that EXISTS but is 100% null is structurally missing -> not usable:
    both the NC gap and the structural gap fire, and the guard's own abstain is untouched."""
    df = _store()
    df["pre_null"] = np.nan                        # tracked-in-schema, never populated
    m = build_manifest_from_frame(df, "acme", "default", "v_1", "fixture://null-nc",
                                  neg_control_map={"retention": ["pre_null"]})
    ce = {(e.treatment, e.outcome): e for e in m.candidate_effects}[
        ("treatment_flag", "retention")]
    assert ce.supportable is True
    assert ce.grounding_complete is False
    assert ce.grounding_gaps == ["no_vouched_negative_control", "structurally_missing_column"]
    # NC-vouching semantics unchanged: the guard still ran on the mapped control and abstained
    assert ce.guard_trips is True
    assert ce.guard_confidence == "low/abstain"


def test_gap_tokens_are_always_from_the_fixed_vocabulary():
    df = _store()
    df["pre_null"] = np.nan
    m = build_manifest_from_frame(df, "acme", "default", "v_1", "fixture://vocab",
                                  neg_control_map={"retention": ["pre_null"]},
                                  treatments=["treatment_flag", "ghost_flag"],
                                  outcomes=["retention", "ghost_metric"])
    assert m.candidate_effects
    for ce in m.candidate_effects:
        assert set(ce.grounding_gaps) <= set(GROUNDING_GAP_TOKENS)
        assert ce.grounding_complete is (ce.grounding_gaps == [])


# --- 3. coverage classification ------------------------------------------------------------

def test_coverage_classifies_present_behavioural_structural_and_absent():
    n = 10
    df = pd.DataFrame({
        "full_col": np.arange(n, dtype=float),                            # 0% null
        "mild_col": [np.nan] * 4 + list(np.arange(6, dtype=float)),       # 40% null -> present
        "sparse_col": [np.nan] * 6 + list(np.arange(4, dtype=float)),     # 60% null -> behavioural
        "edge_col": [np.nan] * 5 + list(np.arange(5, dtype=float)),       # exactly at threshold
        "empty_col": [np.nan] * n,                                        # 100% null -> structural
    })
    m = build_manifest_from_frame(df, "acme", "default", "v_1", "fixture://coverage",
                                  treatments=["ghost_flag"])              # declared, absent
    cov = {e.name: e for e in m.coverage}
    assert cov["full_col"].missingness == "present"
    assert cov["full_col"].null_fraction == 0.0
    assert cov["mild_col"].missingness == "present"
    assert cov["sparse_col"].missingness == "behavioural"
    assert abs(cov["sparse_col"].null_fraction - 0.6) < 1e-12
    assert cov["edge_col"].missingness == "behavioural"                   # >= threshold
    assert BEHAVIOURAL_NULL_THRESHOLD == 0.5
    # 100% null: the schema has the column but the company effectively doesn't track it
    assert cov["empty_col"].missingness == "structural"
    assert cov["empty_col"].null_fraction == 1.0
    assert cov["empty_col"].observed is True
    # declared-but-absent: the purest structural miss; null_fraction unknown
    assert cov["ghost_flag"].missingness == "structural"
    assert cov["ghost_flag"].null_fraction is None
    assert cov["ghost_flag"].observed is False
    # every observed source column is covered
    assert set(df.columns) <= set(cov)


# --- 4. emitted JSON contract --------------------------------------------------------------

def test_manifest_json_contract_coverage_and_grounding_keys_strict_json():
    """asdict(manifest) -> strict JSON: top-level `coverage` with the exact entry shape, both
    grounding keys on EVERY candidate (grid rows and stubs), and no NaN/Infinity anywhere."""
    df = _store()
    df["pre_null"] = np.nan
    m = build_manifest_from_frame(df, "acme", "default", "v_1", "fixture://contract",
                                  neg_control_map={"retention": ["pre_signup"]},
                                  treatments=["treatment_flag", "ghost_flag"],
                                  outcomes=["retention"])
    payload = dataclasses.asdict(m)
    text = json.dumps(payload, allow_nan=False)    # raises on any NaN/Infinity
    reloaded = json.loads(text)

    assert reloaded["coverage"], "expected a non-empty top-level coverage section"
    for entry in reloaded["coverage"]:
        assert set(entry) == COVERAGE_ENTRY_KEYS
        assert entry["missingness"] in ("present", "structural", "behavioural")

    assert reloaded["candidate_effects"]
    treatments_seen = set()
    for ce in reloaded["candidate_effects"]:
        assert GROUNDING_KEYS <= set(ce)
        assert isinstance(ce["grounding_complete"], bool)
        assert isinstance(ce["grounding_gaps"], list)
        treatments_seen.add(ce["treatment"])
    assert "ghost_flag" in treatments_seen         # the stub row rides the same contract
