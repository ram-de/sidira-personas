"""Layer C — Persona Discovery. GMM with BIC-selected k."""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def fit_gmm_bic(X: np.ndarray, k_range: tuple[int, int], random_state: int = 42) -> tuple[GaussianMixture, int]:
    best_model, best_bic, best_k = None, np.inf, k_range[0]
    for k in range(k_range[0], k_range[1] + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=random_state, n_init=3, max_iter=200)
        gmm.fit(X)
        bic = gmm.bic(X)
        if bic < best_bic:
            best_model, best_bic, best_k = gmm, bic, k
    return best_model, best_k


def assign(model: GaussianMixture, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hard labels, max-posterior confidence, AND the full soft posterior matrix.

    The posterior vectors (rows sum to 1.0) are carried downstream so the pack can emit each
    persona's mean soft-membership and assignments.parquet can record per-user posteriors —
    the GMM already computes them; v1.0 discarded everything but the max (#7 soft posteriors).
    """
    probs = model.predict_proba(X)
    labels = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    return labels, confidence, probs
