"""
causal_estimator.py — the route-agnostic EMPIRICAL engine core (router foundation).

Generalizes the validated logic from realdata_uplift.py (Cookie Cats) and
realdata_hillstrom.py (Hillstrom) into one reusable function pair the router calls:

  support_check(df, T, Y, X)  -> can we even estimate this empirically?  (no LLM)
  estimate(df, T, Y, X)       -> naive vs causal ATE + honesty diagnostics  (no LLM)

The router's contract: if support_check says supportable -> run estimate and ship the
adjusted ATE with its diagnostics; else hand the hypothesis to the LLM engine, flagged.

No LLM. Deps: numpy, pandas, scikit-learn.

Vendored verbatim from research/empirical-uplift-spike/causal_estimator.py (validated
across campaign iter1-iter5). The original module's __main__/_selftest block read external
Cookie Cats / Hillstrom CSVs; that behavior now lives in tests/test_estimability.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class Support:
    supportable: bool
    reasons: list[str]
    n_treat: int
    n_control: int
    overlap_fraction: float        # share of units with propensity in [0.05, 0.95]


@dataclass
class Estimate:
    n: int
    outcome_is_binary: bool
    naive_ate: float
    t_learner_ate: float           # standardization over the full sample (causal)
    ipw_ate: float                 # inverse-propensity weighted (causal)
    naive_adjusted_gap: float      # |naive - t_learner|: how much confounding was corrected
    overlap_fraction: float
    sign_adjusted: str             # 'positive' | 'negative'


@dataclass
class GuardedEstimate:
    estimate: Estimate             # the main treatment->outcome effect
    neg_control_effect: float      # the decisive (max-|effect|) negative-control residual
    neg_control_effects: dict      # {control_name: adjusted treatment->control effect}
    floor: float                   # max |effect| above this => residual confounding
    trips: bool                    # True => abstain (route to LLM), estimate not trustworthy
    confidence: str                # 'high' (trust empirical) | 'low/abstain' (route to LLM)
    reason: str


def _Xmat(df: pd.DataFrame, X: list[str]) -> np.ndarray:
    return df[X].to_numpy(float)

def _propensity(Xm: np.ndarray, T: np.ndarray) -> np.ndarray:
    return LogisticRegression(max_iter=3000).fit(Xm, T).predict_proba(Xm)[:, 1]


def _arm_proba(X_arm: np.ndarray, y_arm: np.ndarray, Xm_full: np.ndarray) -> np.ndarray:
    """Per-arm outcome probability for the binary T-learner, robust to a constant arm.

    `LogisticRegression.fit` needs both outcome classes present; if an arm's outcome is
    constant (only 0s or only 1s — which happens for a degenerate negative control the guard
    estimates WITHOUT going through support_check) the model is undefined. In that case the
    arm's predicted probability is exactly that constant for every row (a constant predictor),
    so the T-learner stays well-defined and never crashes."""
    # Defense in depth: an empty arm (zero rows after a null-bearing column is dropped) has no
    # outcome to read — `y_arm[0]` would IndexError. The negative_control_guard support-checks
    # each control so this should be unreachable, but never crash: predict a constant 0.
    if len(y_arm) == 0:
        return np.zeros(Xm_full.shape[0])
    if len(np.unique(y_arm)) < 2:
        return np.full(Xm_full.shape[0], float(y_arm[0]))
    return LogisticRegression(max_iter=3000).fit(X_arm, y_arm).predict_proba(Xm_full)[:, 1]


def support_check(df, treatment, outcome, covariates, min_arm=200, min_overlap=0.1) -> Support:
    """Pure data/stats gate: is this hypothesis empirically estimable from the data?"""
    # Drop rows with NaN in any needed column up front, so arm counts and the fitters
    # below see only usable rows (a null-bearing column must not crash sklearn).
    df = df.dropna(subset=[treatment, outcome, *covariates])
    reasons: list[str] = []
    T = df[treatment].to_numpy()
    Y = df[outcome].to_numpy(float)
    n_t, n_c = int((T == 1).sum()), int((T == 0).sum())
    # The estimator assumes a 0/1 treatment throughout (T==1 vs T==0). A multi-arm or
    # categorical lever (e.g. {30,40} or {0,1,2}) is OUT OF SCOPE for the empirical route and
    # must go to the LLM — flag it here so the non-frame build path / direct callers are
    # protected even when manifest discovery hasn't filtered it (a validated scope boundary).
    if set(pd.unique(df[treatment].dropna())) - {0, 1}:
        reasons.append("treatment not binary 0/1 (multi-arm/categorical out of scope)")
    if n_t < min_arm or n_c < min_arm:
        reasons.append(f"thin arm(s): treat={n_t}, control={n_c} (< {min_arm})")
    if df[outcome].nunique() < 2:
        reasons.append("outcome has no variation")
    # A binary outcome that is constant WITHIN an arm cannot fit the per-arm model in estimate()
    # (LogisticRegression needs both classes present in each arm). Mark it unsupportable here.
    binary = set(np.unique(Y)) <= {0.0, 1.0}
    if binary and ((np.unique(Y[T == 1]).size < 2) or (np.unique(Y[T == 0]).size < 2)):
        reasons.append("outcome constant within an arm (cannot fit per-arm model)")
    overlap = float("nan")
    if n_t >= 2 and n_c >= 2 and covariates:
        e = _propensity(_Xmat(df, covariates), T)
        overlap = float(np.mean((e > 0.05) & (e < 0.95)))
        if overlap < min_overlap:
            reasons.append(f"poor covariate overlap/positivity ({overlap:.2f} < {min_overlap})")
    return Support(supportable=not reasons, reasons=reasons,
                   n_treat=n_t, n_control=n_c, overlap_fraction=overlap)


def estimate(df, treatment, outcome, covariates) -> Estimate:
    """Naive vs causal ATE (treatment - control) with honesty diagnostics. No LLM."""
    # Drop rows with NaN in any needed column so the fitters never see NaN.
    df = df.dropna(subset=[treatment, outcome, *covariates])
    T = df[treatment].to_numpy()
    Y = df[outcome].to_numpy(float)
    binary = set(np.unique(Y)) <= {0.0, 1.0}
    naive = Y[T == 1].mean() - Y[T == 0].mean()

    # With no covariates there is no design matrix; the adjusted estimate degenerates to
    # the unadjusted diff-in-means (sklearn cannot fit an (n, 0) array).
    if not covariates:
        return Estimate(n=len(df), outcome_is_binary=binary, naive_ate=naive,
                        t_learner_ate=naive, ipw_ate=naive, naive_adjusted_gap=0.0,
                        overlap_fraction=1.0,
                        sign_adjusted="positive" if naive > 0 else "negative")

    Xm = _Xmat(df, covariates)

    # T-learner standardization: per-arm outcome model, predict both on the full sample.
    if binary:
        p1 = _arm_proba(Xm[T == 1], Y[T == 1], Xm)   # robust to a constant arm
        p0 = _arm_proba(Xm[T == 0], Y[T == 0], Xm)
        tl = (p1 - p0).mean()
    else:
        from sklearn.linear_model import LinearRegression
        m1 = LinearRegression().fit(Xm[T == 1], Y[T == 1])
        m0 = LinearRegression().fit(Xm[T == 0], Y[T == 0])
        tl = (m1.predict(Xm) - m0.predict(Xm)).mean()

    e = np.clip(_propensity(Xm, T), 0.02, 0.98)
    ipw = (T * Y / e - (1 - T) * Y / (1 - e)).mean()
    overlap = float(np.mean((e > 0.05) & (e < 0.95)))
    return Estimate(n=len(df), outcome_is_binary=binary, naive_ate=naive, t_learner_ate=tl,
                    ipw_ate=ipw, naive_adjusted_gap=abs(naive - tl), overlap_fraction=overlap,
                    sign_adjusted="positive" if tl > 0 else "negative")


def negative_control_guard(df, treatment, outcome, neg_controls, covariates,
                           floor=0.05, ratio=1.5) -> GuardedEstimate:
    """Trust the empirical estimate only if the negative-control outcome(s) show no residual.

    `neg_controls` is one metric name, or a SET of them, that the treatment should NOT cause but
    which share the confounding (e.g. pre-treatment behaviours). A non-zero adjusted
    treatment->control effect reveals confounding the overlap/positivity diagnostics cannot see.
    With a set, the guard trips on CONSENSUS — the max |residual| across controls — which recovers
    the bias a single weak control misses (validated: campaign iter1-iter4 — synthetic held-out +
    real Hillstrom RCT 100% specificity / 75% single-control sensitivity, lifted to full coverage
    by consensus; 0.05 floor transferred synthetic->real). On trip, the router ABSTAINS -> LLM path.

    Two ways to abstain (validated iter1-iter5): (1) HIDDEN confounding — any control's residual
    exceeds `floor`; (2) INDISTINGUISHABLE — the main effect is not larger than `ratio`x the bias
    the control reveals, so a true-null residual or a sub-bias real effect can't be told apart from
    confounding (closes the true-null gap; the cost is honestly deferring effects smaller than the
    bias to the LLM).

    NOTE (validated limits): power depends on at least one control STRONGLY coupled to the
    confounding (a weak set under-trips); nominating the controls is a domain/LLM judgment.
    """
    names = [neg_controls] if isinstance(neg_controls, str) else list(neg_controls)
    est = estimate(df, treatment, outcome, covariates)
    # Support-check each control before estimating it. The main outcome Y is gated by the caller,
    # but the controls are not: a null-bearing control whose non-null rows leave an arm EMPTY
    # (or constant/non-binary/thin) is not estimable, and estimating it would crash the per-arm
    # fitter (empty `y_arm` -> `y_arm[0]`). Only supportable controls contribute a residual; the
    # rest are skipped (this leaves valid-data behaviour identical — every usable control is kept).
    effects: dict = {}
    for nc in names:
        if support_check(df, treatment, nc, covariates).supportable:
            effects[nc] = float(estimate(df, treatment, nc, covariates).t_learner_ate)
    # No usable control survived support-checking (every mapped control had a thin/degenerate arm).
    # We cannot verify the estimate, so conservatively abstain -> route to LLM, rather than trust an
    # unverifiable effect or estimate a degenerate control.
    if not effects:
        return GuardedEstimate(
            estimate=est, neg_control_effect=float("nan"), neg_control_effects={},
            floor=floor, trips=True, confidence="low/abstain",
            reason="no usable negative control (thin/degenerate arms): cannot verify, route to LLM")
    decisive = max(effects.items(), key=lambda kv: abs(kv[1]))     # (name, effect) of max |effect|
    nc_mag = abs(decisive[1])
    main_mag = abs(est.t_learner_ate)
    hidden = nc_mag > floor                                        # strong residual = hidden confounding
    indistinct = main_mag <= ratio * nc_mag                       # effect not bigger than the bias (true-null/borderline)
    trips = bool(hidden or indistinct)
    reason = (f"negative control '{decisive[0]}' residual {decisive[1]:+.4f} > |{floor}|: hidden "
              "confounding, route to LLM" if hidden else
              f"main effect {est.t_learner_ate:+.4f} not > {ratio}x the bias the control reveals "
              f"({decisive[1]:+.4f}): indistinguishable from confounding, route to LLM" if indistinct else
              f"main effect {est.t_learner_ate:+.4f} clears the bias (control max {decisive[1]:+.4f}): "
              "estimate trusted")
    return GuardedEstimate(
        estimate=est, neg_control_effect=decisive[1], neg_control_effects=effects,
        floor=floor, trips=trips, confidence="low/abstain" if trips else "high", reason=reason)
