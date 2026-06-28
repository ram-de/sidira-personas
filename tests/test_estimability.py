"""Self-contained tests for the vendored estimability precompute (Phase 1).

Mirror the validated behaviors from research/empirical-uplift-spike without any external
CSVs: every fixture is generated in-process with a seeded numpy default_rng, so the signs
and guard verdicts are deterministic. These replace the source modules' __main__ self-tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sidira_personas.estimability import (
    build_manifest_from_frame,
    estimate,
    negative_control_guard,
)


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------- #
# estimate: recover the correct SIGN where naive diff-in-means is fooled
# --------------------------------------------------------------------------- #
def test_estimate_recovers_sign_under_confounding():
    rng = np.random.default_rng(7)
    n = 4000
    u = rng.standard_normal(n)                       # hidden-ish confounder, observed here as X
    t = (rng.random(n) < _sig(1.5 * u)).astype(int)  # treated more when u is high
    y = (rng.random(n) < _sig(2.0 * u - 1.0 * t)).astype(float)  # true effect of t is NEGATIVE
    df = pd.DataFrame({"T": t, "Y": y, "U": u})

    est = estimate(df, "T", "Y", ["U"])
    # naive comparison is positive (treated have high u -> high y): it gets the sign WRONG
    assert est.naive_ate > 0
    # the adjusted T-learner recovers the true negative sign
    assert est.t_learner_ate < 0
    assert est.sign_adjusted == "negative"
    # the diagnostic flags how much confounding was corrected
    assert est.naive_adjusted_gap > 0.1


# --------------------------------------------------------------------------- #
# negative_control_guard: clean RCT must NOT trip
# --------------------------------------------------------------------------- #
def test_guard_clean_randomized_does_not_trip():
    rng = np.random.default_rng(11)
    n = 6000
    x = rng.standard_normal(n)
    t = (rng.random(n) < 0.5).astype(int)             # randomized -> no confounding
    y = (rng.random(n) < _sig(0.9 * t + 0.5 * x)).astype(float)  # real positive effect
    nc = (rng.random(n) < _sig(0.5 * x)).astype(float)           # not caused by t, no shared bias
    df = pd.DataFrame({"T": t, "Y": y, "X": x, "NC": nc})

    g = negative_control_guard(df, "T", "Y", ["NC"], ["X"])
    assert abs(g.neg_control_effect) < 0.05          # nc residual ~ 0
    assert g.trips is False
    assert g.confidence == "high"


# --------------------------------------------------------------------------- #
# negative_control_guard: hidden confounder shared by the control => TRIP
# --------------------------------------------------------------------------- #
def test_guard_trips_on_hidden_confounder():
    rng = np.random.default_rng(13)
    n = 6000
    u = rng.standard_normal(n)                         # HIDDEN confounder (not in covariates)
    x_obs = rng.standard_normal(n)                     # the only observed covariate
    t = (rng.random(n) < _sig(1.2 * u)).astype(int)    # confounded by u
    y = (rng.random(n) < _sig(0.8 * u + 0.5 * t)).astype(float)
    nc = (rng.random(n) < _sig(0.8 * u)).astype(float)  # shares u, not caused by t
    df = pd.DataFrame({"T": t, "Y": y, "Xobs": x_obs, "NC": nc})

    g = negative_control_guard(df, "T", "Y", ["NC"], ["Xobs"])
    assert abs(g.neg_control_effect) > 0.05           # residual reveals the hidden confounding
    assert g.trips is True
    assert g.confidence == "low/abstain"


# --------------------------------------------------------------------------- #
# negative_control_guard: a SET of controls trips on CONSENSUS (max |residual|)
# --------------------------------------------------------------------------- #
def test_guard_set_trips_on_consensus_max():
    rng = np.random.default_rng(13)
    n = 6000
    u = rng.standard_normal(n)
    x_obs = rng.standard_normal(n)
    t = (rng.random(n) < _sig(1.2 * u)).astype(int)
    y = (rng.random(n) < _sig(0.8 * u + 0.5 * t)).astype(float)
    nc_strong = (rng.random(n) < _sig(0.8 * u)).astype(float)      # shares the confounder
    nc_weak = (rng.random(n) < _sig(0.4 * x_obs)).astype(float)    # clean -> ~0 residual
    df = pd.DataFrame({"T": t, "Y": y, "Xobs": x_obs,
                       "NC_strong": nc_strong, "NC_weak": nc_weak})

    g = negative_control_guard(df, "T", "Y", ["NC_weak", "NC_strong"], ["Xobs"])
    assert set(g.neg_control_effects) == {"NC_weak", "NC_strong"}
    # the decisive residual is the max-|effect| control (the strong one), not the weak one
    decisive = max(abs(v) for v in g.neg_control_effects.values())
    assert abs(g.neg_control_effect) == decisive
    assert abs(g.neg_control_effects["NC_strong"]) > abs(g.neg_control_effects["NC_weak"])
    assert g.trips is True
    assert g.confidence == "low/abstain"


# --------------------------------------------------------------------------- #
# build_manifest_from_frame: supportable effect + populated guard verdict
# --------------------------------------------------------------------------- #
def _confounded_store(seed: int = 21, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    conf = rng.standard_normal(n)                                  # continuous -> covariate
    flag = (rng.random(n) < _sig(1.3 * conf)).astype(int)         # binary lever -> treatment
    retention = (rng.random(n) < _sig(2.0 * conf - 1.0 * flag)).astype(float)
    conversion = (rng.random(n) < _sig(1.5 * conf - 0.5 * flag)).astype(float)
    return pd.DataFrame({"treatment_flag": flag, "conf": conf,
                         "retention": retention, "conversion": conversion})


def test_build_manifest_from_frame_supportable_with_guard():
    store = _confounded_store()
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://syn")

    roles = {c.name: c.role for c in m.columns}
    assert roles["treatment_flag"] == "treatment_candidate"
    assert roles["conf"] == "covariate"
    assert roles["retention"] == "outcome_candidate"

    by_outcome = {ce.outcome: ce for ce in m.candidate_effects}
    ce = by_outcome["retention"]
    assert ce.treatment == "treatment_flag"
    assert ce.supportable is True
    # true effect of the flag on retention is negative; naive comparison is fooled positive
    assert ce.naive_ate > 0
    assert ce.sign_adjusted == "negative"
    # the guard verdict is populated (not "unverified") because a sibling outcome exists as control
    assert ce.neg_controls == ["conversion"]
    assert ce.guard_confidence in ("high", "low/abstain")
    assert ce.guard_trips is not None
    assert ce.neg_control_effect is not None
    assert ce.guard_reason


# --------------------------------------------------------------------------- #
# build_manifest_from_frame: no low-cardinality lever => empty grid (route to LLM)
# --------------------------------------------------------------------------- #
def test_build_manifest_from_frame_no_treatment_yields_empty_grid():
    rng = np.random.default_rng(21)
    n = 2000
    store = pd.DataFrame({
        "retention": (rng.random(n) < 0.5).astype(float),   # outcome, never a treatment
        "conf": rng.standard_normal(n),                      # continuous covariate
        "spend": rng.standard_normal(n),                     # continuous outcome
    })
    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://noT")
    assert m.candidate_effects == []
    assert m.truncated == {}


# --------------------------------------------------------------------------- #
# build_manifest_from_frame: grid cap records truncation, does not crash
# --------------------------------------------------------------------------- #
def test_build_manifest_from_frame_grid_cap_records_truncation():
    rng = np.random.default_rng(31)
    n = 300
    data = {f"flag{i}": (rng.random(n) < 0.5).astype(int) for i in range(10)}  # 10 levers
    data["retention"] = (rng.random(n) < 0.5).astype(float)
    data["conf"] = rng.standard_normal(n)
    store = pd.DataFrame(data)

    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://cap")
    assert m.truncated.get("treatments_dropped") == 2          # 10 -> kept 8
    # the kept grid still built (8 treatments x 1 outcome), nothing crashed
    assert all(ce.outcome == "retention" for ce in m.candidate_effects)
