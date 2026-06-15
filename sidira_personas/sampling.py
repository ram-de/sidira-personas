"""Per-cluster joint sampling — the v1.1 producer side of the simulator §2 step-2 contract.

The GMM clusters live in PCA-embedding space, but everything the simulator renders (the
centroid and the per-agent individual row) lives in **adapted-feature space**. These helpers
therefore work directly on the ``adapted`` dict ({group: DataFrame}) — the same frames
``calibrate.top_distinctive_features`` reads — so the emitted fields are byte-comparable with
the centroid the consumer already has. No embedding inverse is needed.

Three pure functions:
- ``cluster_centroid_and_scale`` → full centroid mean + within-cluster std (the z-score denominator)
- ``population_bounds``          → per-feature population [min, max] for clamping draws in-distribution
- ``sample_synthetic_rows``      → N covariance-aware synthetic rows (joint structure preserved)

Synthetic-by-design: rows are drawn from a multivariate normal fit to the cluster (mean + ridged
covariance), never copied from real users, so the pack is privacy-clean by default. Real anonymized
rows stay reserved for the #222 benchmark path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Floor for a feature's within-cluster std so the consumer's z-score never divides by zero.
_SCALE_EPS = 1e-9
# Ridge added to the covariance diagonal before the Cholesky/eigh draw — keeps small or
# collinear clusters (singular empirical covariance) numerically sampleable.
_COV_RIDGE = 1e-6


def _concat_numeric(adapted: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all group frames column-wise into one numeric frame (adapted-feature space).

    The v1.1 fields are keyed by feature name (``{feature: value}``), matching the centroid the
    consumer already z-scores against, so a feature can only carry one value. If two groups happen
    to share a source feature name the concat would yield duplicate columns (and a bare ``lo[f]``
    would return a Series, not a scalar) — keep the first occurrence so every key stays unique.
    """
    frames = [f for f in adapted.values() if f.shape[1] > 0]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1)
    return out.loc[:, ~out.columns.duplicated()]


def cluster_centroid_and_scale(
    adapted: dict[str, pd.DataFrame], labels: np.ndarray, cluster_id: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Full per-feature cluster mean and within-cluster std (zero-floored)."""
    X = _concat_numeric(adapted)
    if X.empty:
        return {}, {}
    mask = labels == cluster_id
    sub = X.loc[mask]
    mean = sub.mean()
    std = sub.std(ddof=0).fillna(0.0)
    centroid_full = {f: float(v) for f, v in mean.items()}
    feature_scale = {f: float(max(v, _SCALE_EPS)) for f, v in std.items()}
    return centroid_full, feature_scale


def population_bounds(adapted: dict[str, pd.DataFrame]) -> dict[str, tuple[float, float]]:
    """Per-feature [min, max] over the whole population — clamp bounds for synthetic draws."""
    X = _concat_numeric(adapted)
    if X.empty:
        return {}
    lo = X.min()
    hi = X.max()
    return {f: (float(lo[f]), float(hi[f])) for f in X.columns}


def sample_synthetic_rows(
    adapted: dict[str, pd.DataFrame],
    labels: np.ndarray,
    cluster_id: int,
    n_rows: int,
    random_state: int,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, float]]:
    """Draw ``n_rows`` synthetic rows from the cluster's joint distribution.

    Covariance-aware (MVN over the concatenated adapted features with a ridged covariance) so the
    joint structure between features is preserved — the whole point of a "joint row" vs independent
    marginals. Each draw is clipped to ``bounds`` (default: population min/max) so synthetic rows
    stay physically plausible (flags/ratios in [0, 1], counts ≥ 0, etc.). Deterministic given
    ``random_state``.
    """
    X = _concat_numeric(adapted)
    if X.empty or n_rows <= 0:
        return []
    cols = list(X.columns)
    sub = X.loc[labels == cluster_id]

    if bounds is None:
        bounds = population_bounds(adapted)
    lo = np.array([bounds.get(c, (-np.inf, np.inf))[0] for c in cols], dtype=float)
    hi = np.array([bounds.get(c, (-np.inf, np.inf))[1] for c in cols], dtype=float)

    if len(sub) < 2:
        # Singleton/empty cluster: there is no within-cluster distribution to estimate, and
        # repeating the lone member's real feature vector would leak that user (the rows must stay
        # privacy-clean synthetic). Fall back to the POPULATION distribution so the draws are still
        # synthetic + plausible — just generic rather than persona-specific, which is the right
        # degradation for a persona with no real spread of its own.
        fit = X
    else:
        fit = sub
    mean = fit.mean().to_numpy(dtype=float)
    if len(fit) < 2:
        # Even the population fallback has <2 rows (the whole (sub)population is a single user):
        # no covariance is estimable. np.cov would return NaN, poisoning the pack — use zero
        # covariance (ridge only). Privacy is moot for a population of one; this only avoids NaN.
        cov = np.zeros((len(cols), len(cols)))
    else:
        cov = np.atleast_2d(np.cov(fit.to_numpy(dtype=float), rowvar=False))
    cov = cov + _COV_RIDGE * np.eye(len(cols))  # keep singular/collinear clusters sampleable

    rng = np.random.default_rng(random_state)
    draws = rng.multivariate_normal(mean, cov, size=n_rows, method="eigh")
    draws = np.clip(draws, lo, hi)

    return [{c: float(v) for c, v in zip(cols, row)} for row in draws]
