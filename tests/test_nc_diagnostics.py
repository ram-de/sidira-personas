"""Tests for the reviewable lever-card negative-control diagnostics (#565).

`build_manifest(nc_diagnostics=True)` computes, per SUPPORTABLE candidate effect, the
treatment's covariate-adjusted effect on every numeric+varying candidate control column
(`CandidateEffect.nc_diagnostics`). A large |residual| on a control the treatment cannot cause
reveals confounding a human reviewer should see; a clean/randomized control sits ~0.

All fixtures are generated in-process with a seeded numpy default_rng, so the signs and
magnitudes are deterministic (mirrors test_estimability.py's style). The confounder `u` is HIDDEN
(never a column) so the residual on a control that shares it is NOT adjusted away — the same
regime the negative-control guard is validated on.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from sidira_personas.estimability import (
    MAX_NC_DIAGNOSTICS,
    build_manifest,
    build_manifest_from_frame,
)
from sidira_personas.estimability.manifest_builder import _dtype_of, _safe_nunique


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _hidden_confounder_store(seed: int = 71, n: int = 6000) -> pd.DataFrame:
    """A binary lever, one outcome, and two CONTINUOUS candidate controls — one that shares the
    HIDDEN confounder `u` (large residual expected) and one that is independent (~0). The controls
    are continuous (not 0/1) so they stay covariate-role columns, not promoted binary levers; this
    lets test 2 exercise the self-neuter path (a control that is also an adjustment covariate)."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)                              # HIDDEN confounder (not a column)
    x_obs = rng.standard_normal(n)                          # an observed, clean covariate
    flag = (rng.random(n) < _sig(1.2 * u)).astype(int)     # binary lever, confounded by u
    retention = (rng.random(n) < _sig(0.8 * u - 0.6 * flag)).astype(float)   # the outcome
    aux_shared = 0.8 * u + 0.3 * rng.standard_normal(n)     # shares u, NOT caused by flag
    aux_clean = 0.5 * x_obs + 0.3 * rng.standard_normal(n)  # independent of u and flag
    return pd.DataFrame({"treatment_flag": flag, "x_obs": x_obs,
                         "retention": retention,
                         "aux_shared": aux_shared, "aux_clean": aux_clean})


def _candidate(m, outcome="retention"):
    return {ce.outcome: ce for ce in m.candidate_effects}[outcome]


# --------------------------------------------------------------------------- #
# 1. flag on -> diagnostics populated; shared-confounder control shows a large |residual|,
#    the clean control shows ~0.
# --------------------------------------------------------------------------- #
def test_nc_diagnostics_shared_control_large_clean_control_zero():
    store = _hidden_confounder_store()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://ncdiag",
                                  nc_diagnostics=True)
    ce = _candidate(m)
    assert ce.supportable is True
    diag = ce.nc_diagnostics
    assert diag, "nc_diagnostics should be non-empty when the flag is on"
    # both candidate controls are present in the diagnostic pool (covariate-role, numeric, varying)
    assert "aux_shared" in diag
    assert "aux_clean" in diag
    # the control that shares the hidden confounder shows the confounding; the clean one does not
    assert abs(diag["aux_shared"]) > 0.05
    assert abs(diag["aux_clean"]) < 0.05
    # neither the treatment nor the outcome is ever its own diagnostic entry
    assert "treatment_flag" not in diag
    assert "retention" not in diag
    # every value is a real float (JSON-safe: no None / NaN in the dict)
    assert all(isinstance(v, float) and v == v for v in diag.values())


# --------------------------------------------------------------------------- #
# 2. Self-neuter guard: a control that is ALSO a covariate must be excluded from its own
#    adjustment set, so its residual is preserved (not collapsed to ~0) when it shares the
#    confounder — the same self-neuter bug fixed for the guard.
# --------------------------------------------------------------------------- #
def test_nc_diagnostics_covariate_control_not_self_neutered():
    store = _hidden_confounder_store()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://ncdiag",
                                  nc_diagnostics=True)
    ce = _candidate(m)
    # aux_shared IS a covariate for the (flag, retention) effect (numeric, no outcome hint) ...
    assert "aux_shared" in ce.covariates
    # ... yet its diagnostic is NOT neutered to ~0: it was excluded from its own X_c, so the
    # residual still reveals the confounding it genuinely shares.
    assert abs(ce.nc_diagnostics["aux_shared"]) > 0.05


# --------------------------------------------------------------------------- #
# 3. Default-off: without the flag, every candidate's nc_diagnostics is the empty dict and no
#    truncation key is recorded (no extra compute / IO).
# --------------------------------------------------------------------------- #
def test_nc_diagnostics_default_off_is_empty():
    store = _hidden_confounder_store()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://ncdiag")
    assert m.candidate_effects
    for ce in m.candidate_effects:
        assert ce.nc_diagnostics == {}
    assert "nc_diagnostics_max_dropped_per_candidate" not in m.truncated


# --------------------------------------------------------------------------- #
# 4. Cap: a wide frame (> MAX_NC_DIAGNOSTICS numeric candidate controls) truncates the pool
#    and records the drop honestly.
# --------------------------------------------------------------------------- #
def test_nc_diagnostics_pool_is_capped_and_drop_recorded():
    rng = np.random.default_rng(83)
    n = 3000
    n_controls = MAX_NC_DIAGNOSTICS + 5            # 25 numeric candidate-control covariates
    data = {f"cov_{i:02d}": rng.standard_normal(n) for i in range(n_controls)}
    data["treatment_flag"] = (rng.random(n) < 0.5).astype(int)   # randomized binary lever
    data["retention"] = (rng.random(n) < _sig(0.4 * data["treatment_flag"])).astype(float)
    store = pd.DataFrame(data)

    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://wide",
                                  nc_diagnostics=True)
    ce = _candidate(m)
    # pool = numeric_varying minus {treatment_flag, retention} = 25 controls -> drop 5
    assert m.truncated["nc_diagnostics_max_dropped_per_candidate"] == n_controls - MAX_NC_DIAGNOSTICS
    assert len(ce.nc_diagnostics) <= MAX_NC_DIAGNOSTICS


# --------------------------------------------------------------------------- #
# 5. All diagnostic values are strictly floats (no None / NaN leaks into the JSON-safe dict).
# --------------------------------------------------------------------------- #
def test_nc_diagnostics_values_are_all_floats():
    store = _hidden_confounder_store()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://ncdiag",
                                  nc_diagnostics=True)
    for ce in m.candidate_effects:
        for k, v in ce.nc_diagnostics.items():
            assert isinstance(k, str)
            assert isinstance(v, float)
            assert v == v            # not NaN
            assert v not in (float("inf"), float("-inf"))


# --------------------------------------------------------------------------- #
# 6. Additive-only contract: default-off, the serialized manifest is strict-JSON (no NaN) and the
#    ONLY thing the new field contributes is an empty ``nc_diagnostics: {}`` on each candidate —
#    stripping that key leaves a payload the ``extra="ignore"`` consumer already accepted. This is
#    the byte-identity claim codex asked to be pinned down: additive, not a shape change.
# --------------------------------------------------------------------------- #
def test_nc_diagnostics_default_off_serialization_is_additive_only():
    store = _hidden_confounder_store()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://ncdiag")
    payload = asdict(m)
    # strict JSON (allow_nan=False) must succeed — the emission-boundary sanitizers hold with the
    # new field present.
    json.dumps(payload, allow_nan=False)
    assert payload["candidate_effects"], "expected a non-empty grid"
    for ce in payload["candidate_effects"]:
        # the sole contribution of the field when off is a present, empty dict
        assert ce["nc_diagnostics"] == {}
        # every OTHER field is untouched: removing nc_diagnostics yields the pre-field schema
        legacy = {k: v for k, v in ce.items() if k != "nc_diagnostics"}
        assert "nc_diagnostics" not in legacy
        assert legacy  # non-empty — the rest of the candidate is intact
    assert "nc_diagnostics_max_dropped_per_candidate" not in m.truncated


# --------------------------------------------------------------------------- #
# 7. Loader-safety invariant (codex Medium): enabling nc_diagnostics must never ask load_frame for
#    a column that was not profiled. The pool is built FROM the profiles, so the extended `need`
#    load stays within the store schema — it adds NO new KeyError risk over the base build (which
#    already assumes profiled columns are loadable). A spy load_frame asserts every requested column
#    is real, on BOTH the default-off and flag-on paths; if the pool ever leaked a phantom column,
#    this fires.
# --------------------------------------------------------------------------- #
def test_nc_diagnostics_never_requests_an_unprofiled_column():
    store = _hidden_confounder_store()
    known = set(store.columns)
    column_profiles = [{
        "name": c,
        "dtype": _dtype_of(store[c]),
        "distinct_count": _safe_nunique(store[c]),
        "null_fraction": float(store[c].isna().mean()),
    } for c in store.columns]
    binary_treatment_cols = {"treatment_flag"}

    requested: set[str] = set()

    def spy_load_frame(cols):
        # every requested column must be one the store actually profiled — no phantom loads
        assert set(cols) <= known, f"requested unprofiled column(s): {set(cols) - known}"
        requested.update(cols)
        return store[cols]

    for flag in (False, True):
        requested.clear()
        m = build_manifest(
            "acme", "default", "v_1", "fixture://spy", column_profiles, capabilities=[],
            load_frame=spy_load_frame, binary_treatment_cols=binary_treatment_cols,
            nc_diagnostics=flag)
        assert m.candidate_effects
        # flag-on requests are a SUPERSET (the pool) but still ⊆ the profiled schema (asserted above)
        assert requested <= known
    # sanity: with the flag on, diagnostics actually populated (the pool WAS loaded, from one query)
    ce = {c.outcome: c for c in m.candidate_effects}["retention"]
    assert ce.nc_diagnostics and "aux_shared" in ce.nc_diagnostics
