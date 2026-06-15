"""Unit tests for the #7 stability + holdout diagnostics (sidira_personas/calibrate.py)."""
from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture

from sidira_personas.calibrate import (
    holdout_evaluation,
    jaccard_verdict,
    per_cluster_jaccard,
    posterior_entropy,
)


def _blobs(means, n=150, spread=1.0, seed=0):
    """Stack k isotropic Gaussian blobs in 4-D; return (X, base_labels)."""
    rng = np.random.default_rng(seed)
    dim = 4
    parts, labels = [], []
    for i, m in enumerate(means):
        parts.append(rng.normal(m, spread, size=(n, dim)))
        labels.extend([i] * n)
    X = np.vstack(parts)
    k = len(means)
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=42, n_init=3).fit(X)
    return X, gmm.predict(X), k


# --- jaccard_verdict bands (Hennig) ---

def test_jaccard_verdict_bands():
    assert jaccard_verdict(0.90) == "highly stable"
    assert jaccard_verdict(0.85) == "highly stable"
    assert jaccard_verdict(0.80) == "valid"
    assert jaccard_verdict(0.75) == "valid"
    assert jaccard_verdict(0.60) == "unstable"
    assert jaccard_verdict(0.50) == "dissolved"
    assert jaccard_verdict(0.10) == "dissolved"


# --- per_cluster_jaccard ---

def test_well_separated_clusters_are_highly_stable():
    X, labels, k = _blobs([0.0, 10.0, 20.0], seed=1)
    jac = per_cluster_jaccard(X, labels, k, iters=20, random_state=42)
    assert set(jac) == set(range(k))
    assert all(v >= 0.85 for v in jac.values()), jac
    assert all(jaccard_verdict(v) == "highly stable" for v in jac.values())


def test_overlapping_clusters_are_less_stable_than_separated():
    Xs, ls, ks = _blobs([0.0, 10.0], spread=1.0, seed=2)
    Xo, lo, ko = _blobs([0.0, 1.0], spread=1.0, seed=2)  # means nearly on top of each other
    sep = per_cluster_jaccard(Xs, ls, ks, iters=20, random_state=42)
    ovl = per_cluster_jaccard(Xo, lo, ko, iters=20, random_state=42)
    assert np.mean(list(ovl.values())) < np.mean(list(sep.values()))


def test_per_cluster_jaccard_is_deterministic():
    X, labels, k = _blobs([0.0, 10.0, 20.0], seed=3)
    a = per_cluster_jaccard(X, labels, k, iters=15, random_state=7)
    b = per_cluster_jaccard(X, labels, k, iters=15, random_state=7)
    assert a == b


# --- posterior_entropy ---

def test_posterior_entropy_crisp_is_zero():
    assert posterior_entropy([1.0, 0.0, 0.0]) == 0.0
    assert posterior_entropy({"P01": 1.0, "P02": 0.0}) == 0.0


def test_posterior_entropy_uniform_is_one():
    assert abs(posterior_entropy([0.25, 0.25, 0.25, 0.25]) - 1.0) < 1e-9
    assert abs(posterior_entropy({"P01": 0.5, "P02": 0.5}) - 1.0) < 1e-9


def test_posterior_entropy_partial_is_between():
    h = posterior_entropy([0.7, 0.2, 0.1])
    assert 0.0 < h < 1.0


def test_posterior_entropy_singleton_is_zero():
    assert posterior_entropy([1.0]) == 0.0
    assert posterior_entropy([]) == 0.0


# --- holdout_evaluation ---

def test_holdout_returns_expected_shape():
    X, _labels, k = _blobs([0.0, 10.0, 20.0], n=120, seed=4)
    h = holdout_evaluation(X, k, test_frac=0.25, random_state=42)
    assert h["n_train"] + h["n_test"] == len(X)
    assert h["n_test"] == round(len(X) * 0.25)
    assert np.isfinite(h["train_avg_loglik"])
    assert np.isfinite(h["test_avg_loglik"])
    assert h["loglik_gap"] == h["train_avg_loglik"] - h["test_avg_loglik"]
    assert 0.0 <= h["proportion_tv_distance"] <= 1.0


def test_holdout_proportion_drift_small_for_clean_clusters():
    X, _labels, k = _blobs([0.0, 10.0, 20.0], n=300, seed=5)
    h = holdout_evaluation(X, k, test_frac=0.2, random_state=42)
    # equal-sized, well-separated clusters → train/test proportions barely move
    assert h["proportion_tv_distance"] < 0.1


def test_holdout_is_deterministic():
    X, _labels, k = _blobs([0.0, 10.0], n=100, seed=6)
    assert holdout_evaluation(X, k, random_state=11) == holdout_evaluation(X, k, random_state=11)
