"""Guard-field contract tests for the estimability producer (M1, sidira-simulator#572).

The sidira-simulator consumer schema (backend/sidira_simulator/schemas/estimability.py::
CandidateEffect) reads an EXACT set of guard_* keys off each candidate effect. These tests pin
that the producer emits every one of them via `dataclasses.asdict`, that they are populated when a
caller/LLM-vouched negative control runs the guard, and that the honest CAN'T-VALIDATE path (no
available NC) leaves them None/empty. Fixtures are seeded numpy — deterministic, no external CSVs.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd

from sidira_personas.estimability import build_manifest_from_frame
from sidira_personas.estimability.manifest_builder import NC_GUARD_FLOOR, NC_GUARD_RATIO

# The exact guard-related keys the consumer schema reads off each candidate_effect dict.
REQUIRED_GUARD_KEYS = {
    "neg_control_effect",
    "neg_control_effects",
    "guard_floor",
    "guard_ratio",
    "guard_trips",
    "guard_confidence",
    "guard_reason",
}


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _confounded_store_with_nc(seed: int = 101, n: int = 4000) -> pd.DataFrame:
    """A binary lever confounded by `conf`, plus a valid negative control (`pre_signup`) that
    shares `conf` but is NOT caused by the lever — so the guard reveals residual confounding."""
    rng = np.random.default_rng(seed)
    conf = rng.standard_normal(n)                                 # shared confounder (a covariate)
    other = rng.standard_normal(n)                                # observed covariate (not conf)
    flag = (rng.random(n) < _sig(1.3 * conf)).astype(int)         # binary lever -> treatment
    retention = (rng.random(n) < _sig(2.0 * conf - 1.0 * flag)).astype(float)
    pre_signup = (rng.random(n) < _sig(1.8 * conf)).astype(float)  # NC: shares conf, not caused by flag
    return pd.DataFrame({"treatment_flag": flag, "other": other,
                         "retention": retention, "pre_signup": pre_signup})


def test_supportable_candidate_has_guard_fields_populated():
    """(a) at least one supportable candidate has guard_trips + the full guard verdict populated."""
    store = _confounded_store_with_nc()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://guard-fields",
                                  neg_control_map={"retention": ["pre_signup"]})

    ce = {e.outcome: e for e in m.candidate_effects}["retention"]
    assert ce.supportable is True
    # guard ran -> every verdict field is populated (not the None/empty CAN'T-VALIDATE sentinel)
    assert ce.guard_trips is not None
    assert ce.neg_control_effect is not None
    assert ce.neg_control_effects == {"pre_signup": ce.neg_control_effect}
    assert ce.guard_floor == NC_GUARD_FLOOR
    assert ce.guard_ratio == NC_GUARD_RATIO
    assert ce.guard_confidence in ("high", "low/abstain")
    assert ce.guard_reason
    # the NC shares the confounder the guard can't neutralise -> hidden confounding -> TRIP
    assert ce.guard_trips is True
    assert ce.guard_confidence == "low/abstain"


def test_asdict_is_json_serializable_with_exact_guard_keys():
    """(b) asdict(manifest) round-trips through json and every candidate carries the exact keys."""
    store = _confounded_store_with_nc()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://json",
                                  neg_control_map={"retention": ["pre_signup"]})

    payload = dataclasses.asdict(m)
    # serializes without error (the consumer ships this beside persona_pack.json in R2)
    text = json.dumps(payload)
    reloaded = json.loads(text)

    assert reloaded["candidate_effects"], "expected at least one candidate effect"
    for ce in reloaded["candidate_effects"]:
        assert REQUIRED_GUARD_KEYS <= set(ce), (
            f"missing guard keys: {REQUIRED_GUARD_KEYS - set(ce)}"
        )
        # neg_control_effects survives the json round-trip as a dict keyed by control name
        assert isinstance(ce["neg_control_effects"], dict)


def test_candidate_without_available_nc_has_none_guard_fields():
    """(c) no vouched NC for the outcome -> honest CAN'T-VALIDATE: guard fields None/empty."""
    store = _confounded_store_with_nc()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://no-nc")  # no map

    ce = {e.outcome: e for e in m.candidate_effects}["retention"]
    assert ce.supportable is True
    assert ce.guard_trips is None                 # downstream watchdog reads this as CAN'T-VALIDATE
    assert ce.neg_control_effect is None
    assert ce.neg_control_effects == {}
    assert ce.guard_floor is None
    assert ce.guard_ratio is None
    assert ce.guard_confidence == "unverified"


def test_absent_neg_control_column_degrades_not_aborts():
    """Fix 2: a neg_control_map naming a column ABSENT from the store must NOT abort the build
    (KeyError inside load_frame). The candidate degrades to the unverified path and the manifest
    still carries every OTHER candidate."""
    rng = np.random.default_rng(77)
    n = 4000
    conf = rng.standard_normal(n)
    flag = (rng.random(n) < _sig(1.3 * conf)).astype(int)
    store = pd.DataFrame({
        "treatment_flag": flag,
        "conf": conf,
        "retention": (rng.random(n) < _sig(2.0 * conf - flag)).astype(float),
        "conversion": (rng.random(n) < _sig(1.5 * conf - 0.5 * flag)).astype(float),
    })
    # 'ghost_metric' does not exist in the store; 'conversion' is a real sibling outcome
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://ghost",
                                  neg_control_map={"retention": ["ghost_metric"]})

    by_outcome = {e.outcome: e for e in m.candidate_effects}
    # build did not abort — both candidates present
    assert {"retention", "conversion"} <= set(by_outcome)
    ce = by_outcome["retention"]
    assert ce.supportable is True
    # stale NC name filtered out -> no available NC -> honest CAN'T-VALIDATE
    assert ce.neg_controls == []
    assert ce.guard_trips is None
    assert ce.guard_confidence == "unverified"
    assert ce.neg_control_effect is None
    assert ce.neg_control_effects == {}


def test_thin_neg_control_serializes_effect_as_none_not_nan():
    """Fix 1: a mapped NC that IS present but degenerate (unusable) -> the guard abstains with
    neg_control_effect=NaN internally, which must be EMITTED as None (strict JSON), while the rest
    of the abstain verdict is preserved."""
    rng = np.random.default_rng(83)
    n = 4000
    x = rng.standard_normal(n)
    t = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < _sig(0.5 * t + 0.3 * x)).astype(float)
    # NC whose non-null rows fall entirely in one arm -> not support-checkable -> guard abstains
    nc_bad = np.where(t == 1, (rng.random(n) < 0.5).astype(float), np.nan)
    store = pd.DataFrame({"treatment_flag": t, "conf": x, "retention": y, "pre_bad": nc_bad})

    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://thin-nc",
                                  neg_control_map={"retention": ["pre_bad"]})
    ce = {e.outcome: e for e in m.candidate_effects}["retention"]
    assert ce.supportable is True
    assert ce.neg_control_effect is None            # NaN sanitized to None at emission
    assert ce.neg_control_effects == {}             # no usable control survived
    assert ce.guard_trips is True                   # real abstain verdict preserved
    assert ce.guard_confidence == "low/abstain"
    assert ce.guard_reason
    assert ce.guard_floor == NC_GUARD_FLOOR
    assert ce.guard_ratio == NC_GUARD_RATIO


def test_manifest_json_has_no_nan_or_infinity_tokens():
    """Fix 1/3: the whole manifest serializes under strict JSON (allow_nan=False) — proving no
    NaN/Infinity leaks anywhere in the candidate grid, across a supportable+guarded candidate and
    a degenerate-NC (internally-NaN) candidate in the same build."""
    rng = np.random.default_rng(89)
    n = 4000
    conf = rng.standard_normal(n)
    flag = (rng.random(n) < _sig(1.3 * conf)).astype(int)
    x = rng.standard_normal(n)
    store = pd.DataFrame({
        "treatment_flag": flag,
        "conf": conf,
        "retention": (rng.random(n) < _sig(2.0 * conf - flag)).astype(float),
        "pre_signup": (rng.random(n) < _sig(1.8 * conf)).astype(float),          # valid NC
        "pre_bad": np.where(flag == 1, (rng.random(n) < 0.5).astype(float), np.nan),  # degenerate NC
        "conversion": (rng.random(n) < _sig(1.5 * conf - 0.5 * flag + 0.1 * x)).astype(float),
    })
    m = build_manifest_from_frame(
        store, "acme", "default", "v_1", "fixture://strict",
        neg_control_map={"retention": ["pre_signup"], "conversion": ["pre_bad"]})

    payload = dataclasses.asdict(m)
    # allow_nan=False raises ValueError if any NaN/Infinity is present -> must NOT raise
    text = json.dumps(payload, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text


def test_unsupportable_candidate_also_carries_the_guard_keys():
    """An unsupportable candidate (thin arms) still emits the full key set (None/empty), so the
    consumer schema never sees a missing key regardless of the support verdict."""
    # tiny frame -> arms below min_arm=200 -> unsupportable
    rng = np.random.default_rng(3)
    n = 120
    store = pd.DataFrame({
        "treatment_flag": (rng.random(n) < 0.5).astype(int),
        "conf": rng.standard_normal(n),
        "retention": (rng.random(n) < 0.5).astype(float),
    })
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://thin",
                                  neg_control_map={"retention": ["conf"]})
    ce = {e.outcome: e for e in m.candidate_effects}["retention"]
    assert ce.supportable is False
    d = dataclasses.asdict(ce)
    assert REQUIRED_GUARD_KEYS <= set(d)
    assert ce.guard_trips is None
    assert ce.neg_control_effects == {}
    assert ce.guard_floor is None
    assert ce.guard_ratio is None
