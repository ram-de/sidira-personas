"""Layer D — Calibration. Weights (headcount/orders/revenue) + bootstrap stability + Persona Pack."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


def compute_weights(df: pd.DataFrame, labels: np.ndarray, schemes: list[str],
                    orders_col: str | None, revenue_col: str | None) -> dict[int, dict[str, float]]:
    weights: dict[int, dict[str, float]] = {}
    unique = sorted(set(labels.tolist()))
    for k in unique:
        mask = labels == k
        w: dict[str, float] = {}
        if "headcount" in schemes:
            w["by_headcount"] = float(mask.sum() / len(labels))
        if "orders" in schemes and orders_col and orders_col in df.columns:
            tot = float(df[orders_col].sum()) or 1.0
            w["by_orders"] = float(df.loc[mask, orders_col].sum() / tot)
        if "revenue" in schemes and revenue_col and revenue_col in df.columns:
            tot = float(df[revenue_col].sum()) or 1.0
            w["by_revenue"] = float(df.loc[mask, revenue_col].sum() / tot)
        weights[int(k)] = w
    return weights


def bootstrap_stability(X: np.ndarray, k: int, iters: int = 30,
                        random_state: int = 42) -> float:
    rng = np.random.default_rng(random_state)
    base = GaussianMixture(n_components=k, covariance_type="full",
                           random_state=random_state, n_init=1).fit(X).predict(X)
    scores = []
    for i in range(iters):
        idx = rng.choice(len(X), size=len(X), replace=True)
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=int(rng.integers(0, 1_000_000)), n_init=1).fit(X[idx])
        pred = gmm.predict(X)
        scores.append(adjusted_rand_score(base, pred))
    return float(np.mean(scores))


def top_distinctive_features(adapted: dict[str, pd.DataFrame], labels: np.ndarray,
                             cluster_id: int, top_n: int = 3, threshold_sigma: float = 0.5) -> list[dict]:
    mask = labels == cluster_id
    out: list[dict] = []
    for group, feats in adapted.items():
        if feats.shape[1] == 0:
            continue
        pop_mean = feats.mean()
        pop_std = feats.std().replace(0, 1.0)
        clu_mean = feats.loc[mask].mean()
        z = (clu_mean - pop_mean) / pop_std
        z_sorted = z.reindex(z.abs().sort_values(ascending=False).index)
        z_sorted = z_sorted[z_sorted.abs() >= threshold_sigma].head(top_n)
        for feat, zval in z_sorted.items():
            out.append({
                "feature": feat,
                "group": group,
                "centroid_value": float(clu_mean[feat]),
                "population_mean": float(pop_mean[feat]),
                "population_std": float(pop_std[feat]),
                "z_delta": float(zval),
            })
    return out
