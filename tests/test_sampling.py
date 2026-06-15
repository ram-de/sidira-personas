"""Unit tests for the v1.1 joint-sampling math (sidira_personas/sampling.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sidira_personas.sampling import (
    cluster_centroid_and_scale,
    population_bounds,
    sample_synthetic_rows,
)


def _two_cluster_fixture(seed: int = 0):
    """Two well-separated Gaussian blobs over 3 correlated features, split into two groups."""
    rng = np.random.default_rng(seed)
    n = 200
    cov = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.1], [0.2, 0.1, 1.0]])
    a = rng.multivariate_normal([0.0, 0.0, 0.0], cov, size=n)
    b = rng.multivariate_normal([5.0, 5.0, 5.0], cov, size=n)
    X = np.vstack([a, b])
    labels = np.array([0] * n + [1] * n)
    adapted = {
        "g1": pd.DataFrame(X[:, :2], columns=["f0", "f1"]),
        "g2": pd.DataFrame(X[:, 2:], columns=["f2"]),
    }
    return adapted, labels


def test_centroid_and_scale_cover_all_features():
    adapted, labels = _two_cluster_fixture()
    centroid, scale = cluster_centroid_and_scale(adapted, labels, 0)
    assert set(centroid) == {"f0", "f1", "f2"}
    assert set(scale) == {"f0", "f1", "f2"}
    # cluster 0 centered near origin
    assert all(abs(centroid[f]) < 0.5 for f in centroid)
    # every scale strictly positive (zero-std floored)
    assert all(v > 0 for v in scale.values())


def test_feature_scale_floors_zero_std():
    adapted = {"g": pd.DataFrame({"const": [3.0, 3.0, 3.0], "var": [1.0, 2.0, 3.0]})}
    labels = np.array([0, 0, 0])
    _, scale = cluster_centroid_and_scale(adapted, labels, 0)
    assert scale["const"] > 0  # floored, never zero
    assert scale["var"] > scale["const"]


def test_sample_rows_count_and_keys():
    adapted, labels = _two_cluster_fixture()
    rows = sample_synthetic_rows(adapted, labels, 0, n_rows=64, random_state=42)
    assert len(rows) == 64
    assert all(set(r) == {"f0", "f1", "f2"} for r in rows)


def test_sample_rows_within_population_bounds():
    adapted, labels = _two_cluster_fixture()
    bounds = population_bounds(adapted)
    rows = sample_synthetic_rows(adapted, labels, 1, n_rows=128, random_state=7, bounds=bounds)
    for r in rows:
        for f, v in r.items():
            lo, hi = bounds[f]
            assert lo <= v <= hi


def test_sample_rows_preserve_joint_covariance():
    adapted, labels = _two_cluster_fixture()
    rows = sample_synthetic_rows(adapted, labels, 0, n_rows=2000, random_state=1)
    sampled = pd.DataFrame(rows)[["f0", "f1", "f2"]].to_numpy()
    src = pd.concat(adapted.values(), axis=1).loc[labels == 0][["f0", "f1", "f2"]].to_numpy()
    # f0/f1 correlation should be reproduced within a loose tolerance
    c_sampled = np.corrcoef(sampled[:, 0], sampled[:, 1])[0, 1]
    c_src = np.corrcoef(src[:, 0], src[:, 1])[0, 1]
    assert abs(c_sampled - c_src) < 0.15


def test_sample_rows_deterministic():
    adapted, labels = _two_cluster_fixture()
    a = sample_synthetic_rows(adapted, labels, 0, n_rows=32, random_state=99)
    b = sample_synthetic_rows(adapted, labels, 0, n_rows=32, random_state=99)
    assert a == b


def test_singleton_cluster_does_not_leak_real_row():
    # Single-member cluster: must NOT copy the lone member's real value into sample_rows (privacy).
    adapted = {"g": pd.DataFrame({"f0": [2.0, 4.0, 6.0, 8.0, 10.0]})}
    labels = np.array([0, 1, 1, 1, 1])  # cluster 0 has exactly one member (value 2.0)
    rows = sample_synthetic_rows(adapted, labels, 0, n_rows=50, random_state=0)
    assert len(rows) == 50
    vals = [r["f0"] for r in rows]
    # rows are synthetic population draws, not 50 copies of the real member's 2.0
    assert len({round(v, 6) for v in vals}) > 1
    assert not all(abs(v - 2.0) < 1e-9 for v in vals)
    # still clamped to population bounds
    assert all(2.0 <= v <= 10.0 for v in vals)


def test_single_row_population_emits_finite_rows():
    # Whole (sub)population is one user → no covariance estimable; must emit finite rows, never NaN.
    adapted = {"g": pd.DataFrame({"f0": [3.0], "f1": [5.0]})}
    labels = np.array([0])
    rows = sample_synthetic_rows(adapted, labels, 0, n_rows=4, random_state=0)
    assert len(rows) == 4
    assert all(np.isfinite(list(r.values())).all() for r in rows)


def test_duplicate_feature_names_across_groups_do_not_crash():
    # Two groups share feature name "shared" → concat would duplicate the column; keep first, no crash.
    adapted = {
        "g1": pd.DataFrame({"shared": [1.0, 2.0, 3.0], "a": [0.1, 0.2, 0.3]}),
        "g2": pd.DataFrame({"shared": [9.0, 9.0, 9.0], "b": [4.0, 5.0, 6.0]}),
    }
    labels = np.array([0, 0, 1])
    centroid, scale = cluster_centroid_and_scale(adapted, labels, 0)
    assert set(centroid) == {"shared", "a", "b"}  # de-duplicated, first occurrence wins
    assert centroid["shared"] == 1.5  # mean of g1's [1.0, 2.0], not g2's 9.0
    bounds = population_bounds(adapted)
    assert all(isinstance(v, tuple) and len(v) == 2 for v in bounds.values())
    rows = sample_synthetic_rows(adapted, labels, 0, n_rows=8, random_state=0)
    assert all(set(r) == {"shared", "a", "b"} for r in rows)


def test_empty_adapted_returns_empty():
    assert cluster_centroid_and_scale({}, np.array([0]), 0) == ({}, {})
    assert population_bounds({}) == {}
    assert sample_synthetic_rows({}, np.array([0]), 0, n_rows=4, random_state=0) == []
