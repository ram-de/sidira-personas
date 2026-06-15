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
    """Coarse pack-level stability: mean ARI between the base clustering and bootstrap refits.

    Retained for back-compat as ``PersonaPack.overall_stability``. It is a single global number
    that hides which clusters are solid and which are artifacts — use :func:`per_cluster_jaccard`
    for the per-persona signal that actually gates downstream simulation (#7).
    """
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


def per_cluster_jaccard(X: np.ndarray, base_labels: np.ndarray, k: int,
                        iters: int = 30, random_state: int = 42) -> dict[int, float]:
    """Hennig clusterboot-style per-cluster stability (Jaccard bootstrap).

    For each bootstrap resample we refit the GMM, classify the *full* dataset with the refit
    model, and for every base cluster record its maximum Jaccard overlap with any bootstrap
    cluster. The per-cluster score is the mean of those maxima across ``iters`` resamples.

    This surfaces what the global mean-ARI hides: a 0.67 overall can be 2–3 solid clusters
    (≥0.85) plus 2–3 artifacts (≤0.5). Hennig's interpretation bands are applied by
    :func:`jaccard_verdict`. Returns ``{cluster_id: mean_max_jaccard}`` over ``range(k)``.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    base_sets = {c: set(np.flatnonzero(base_labels == c).tolist()) for c in range(k)}
    acc: dict[int, list[float]] = {c: [] for c in range(k)}
    for _ in range(iters):
        idx = rng.choice(n, size=n, replace=True)
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                              random_state=int(rng.integers(0, 1_000_000)), n_init=1).fit(X[idx])
        boot_labels = gmm.predict(X)
        boot_sets = [set(np.flatnonzero(boot_labels == j).tolist()) for j in range(k)]
        for c, s in base_sets.items():
            if not s:
                acc[c].append(0.0)
                continue
            best = 0.0
            for t in boot_sets:
                if not t:
                    continue
                union = len(s | t)
                j = (len(s & t) / union) if union else 0.0
                if j > best:
                    best = j
            acc[c].append(best)
    return {c: (float(np.mean(v)) if v else 0.0) for c, v in acc.items()}


def jaccard_verdict(score: float) -> str:
    """Hennig clusterboot interpretation bands for a per-cluster Jaccard stability."""
    if score >= 0.85:
        return "highly stable"
    if score >= 0.75:
        return "valid"
    if score > 0.5:
        return "unstable"
    return "dissolved"


def posterior_entropy(posterior) -> float:
    """Normalized Shannon entropy of a soft-posterior vector, in [0, 1].

    0.0 = a crisp, single-component identity; 1.0 = uniform across all components (maximally
    mixed). Accepts a ``{persona_id: prob}`` dict or a sequence. Normalized by ``log(k)`` so it
    is comparable across packs with different ``k`` (#7 "posterior entropy" honesty field).
    """
    vals = list(posterior.values()) if isinstance(posterior, dict) else list(posterior)
    p = np.asarray(vals, dtype=float)
    k = p.size
    if k <= 1:
        return 0.0
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    h = float(-np.sum(p * np.log(p)))
    return h / float(np.log(k))


def holdout_evaluation(X: np.ndarray, k: int, test_frac: float = 0.2,
                       random_state: int = 42) -> dict:
    """Train/test generalization diagnostic for the GMM at the selected ``k``.

    A *diagnostic*, not the shipped fit: the production pack still fits on all rows, but with no
    holdout there was no evidence the clustering generalizes (#7 "add a holdout"). We fit on a
    train split and report the held-out average log-likelihood plus the train↔test cluster-
    proportion drift (total-variation distance). A large ``loglik_gap`` or ``proportion_tv``
    flags an overfit / non-generalizing k. Deterministic given ``random_state``.
    """
    n = len(X)
    rng = np.random.default_rng(random_state)
    perm = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    Xtr, Xte = X[train_idx], X[test_idx]
    gmm = GaussianMixture(n_components=k, covariance_type="full",
                          random_state=random_state, n_init=3, max_iter=200).fit(Xtr)
    train_ll = float(gmm.score(Xtr))
    test_ll = float(gmm.score(Xte))
    ktr = np.bincount(gmm.predict(Xtr), minlength=k) / len(Xtr)
    kte = np.bincount(gmm.predict(Xte), minlength=k) / len(Xte)
    tv = float(0.5 * np.abs(ktr - kte).sum())
    return {
        "test_frac": float(test_frac),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_avg_loglik": train_ll,
        "test_avg_loglik": test_ll,
        "loglik_gap": float(train_ll - test_ll),
        "proportion_tv_distance": tv,
    }


def _format_value(v, name: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, float):
        # ratios under 1 → percentage; counts → int-like; scores → 2dp
        if "rate" in name or "share" in name or "ratio" in name or "pct" in name:
            return f"{v*100:.1f}%"
        if abs(v) >= 100:
            return f"{v:,.0f}"
        if abs(v) >= 10:
            return f"{v:.1f}"
        return f"{v:.3f}"
    return str(v)


def profile_clusters(df_raw: pd.DataFrame, feature_groups: dict, labels: np.ndarray) -> dict:
    """Return per-cluster + population profile in ORIGINAL feature units.

    Returns:
        {
          "baseline": {feature: {value, value_fmt, kind}, ...},
          "clusters": {cluster_id: {feature: {value, value_fmt, kind, delta_vs_baseline, ...}}}
        }
    """
    feat_kinds: dict[str, tuple[str, str]] = {}     # name -> (group, type)
    for g, specs in feature_groups.items():
        for s in specs:
            feat_kinds[s["name"]] = (g, s["type"])

    baseline: dict[str, dict] = {}
    clusters: dict[int, dict[str, dict]] = {int(k): {} for k in sorted(set(labels.tolist()))}

    for name, (group, mtype) in feat_kinds.items():
        if name not in df_raw.columns:
            continue
        col = df_raw[name]

        if mtype in ("count", "score", "ratio", "tenure_days"):
            num = pd.to_numeric(col, errors="coerce")
            pop_mean = float(num.mean())
            baseline[name] = {
                "group": group, "kind": mtype, "value": pop_mean,
                "value_fmt": _format_value(pop_mean, name),
            }
            for cid in clusters:
                v = float(num[labels == cid].mean())
                clusters[cid][name] = {
                    "group": group, "kind": mtype, "value": v,
                    "value_fmt": _format_value(v, name),
                    "delta_abs": v - pop_mean,
                    "delta_pct": (v - pop_mean) / pop_mean if pop_mean else 0.0,
                }
        elif mtype == "flag":
            num = pd.to_numeric(col, errors="coerce").fillna(0).clip(0, 1)
            pop_rate = float(num.mean())
            baseline[name] = {
                "group": group, "kind": "flag", "value": pop_rate,
                "value_fmt": f"{pop_rate*100:.1f}% true",
            }
            for cid in clusters:
                v = float(num[labels == cid].mean())
                clusters[cid][name] = {
                    "group": group, "kind": "flag", "value": v,
                    "value_fmt": f"{v*100:.1f}% true",
                    "delta_abs": v - pop_rate,
                }
        elif mtype == "categorical":
            top_pop = col.fillna("UNK").value_counts(normalize=True)
            baseline[name] = {
                "group": group, "kind": "categorical",
                "value": top_pop.idxmax(), "value_fmt": f"mode={top_pop.idxmax()} ({top_pop.iloc[0]*100:.0f}%)",
                "distribution": top_pop.head(5).to_dict(),
            }
            for cid in clusters:
                sub = col[labels == cid].fillna("UNK").value_counts(normalize=True)
                if len(sub) == 0:
                    continue
                clusters[cid][name] = {
                    "group": group, "kind": "categorical",
                    "value": sub.idxmax(),
                    "value_fmt": f"{sub.idxmax()} ({sub.iloc[0]*100:.0f}%)",
                    "distribution": sub.head(5).to_dict(),
                }
        elif mtype == "json_ratios":
            keys = next((s["keys"] for s in feature_groups[group] if s["name"] == name), [])
            parsed = col.apply(lambda v: _safe_parse_json(v, keys))
            for k in keys:
                sub_col = parsed.apply(lambda d: d.get(k, 0.0))
                pop = float(sub_col.mean())
                qn = f"{name}__{k}"
                baseline[qn] = {
                    "group": group, "kind": "ratio", "value": pop,
                    "value_fmt": f"{pop*100:.1f}%",
                }
                for cid in clusters:
                    v = float(sub_col[labels == cid].mean())
                    clusters[cid][qn] = {
                        "group": group, "kind": "ratio", "value": v,
                        "value_fmt": f"{v*100:.1f}%",
                        "delta_abs": v - pop,
                        "delta_pct": (v - pop) / pop if pop else 0.0,
                    }
        elif mtype == "json_counts":
            keys = next((s["keys"] for s in feature_groups[group] if s["name"] == name), [])
            parsed = col.apply(lambda v: _safe_parse_json(v, keys))
            totals = parsed.apply(lambda d: sum(d.values()))
            for k in keys:
                share = parsed.apply(lambda d: d.get(k, 0.0)) / totals.replace(0, np.nan)
                share = share.fillna(0.0)
                pop = float(share.mean())
                qn = f"{name}__{k}"
                baseline[qn] = {
                    "group": group, "kind": "ratio", "value": pop,
                    "value_fmt": f"{pop*100:.1f}%",
                }
                for cid in clusters:
                    v = float(share[labels == cid].mean())
                    clusters[cid][qn] = {
                        "group": group, "kind": "ratio", "value": v,
                        "value_fmt": f"{v*100:.1f}%",
                        "delta_abs": v - pop,
                        "delta_pct": (v - pop) / pop if pop else 0.0,
                    }

    return {"baseline": baseline, "clusters": clusters}


def _safe_parse_json(v, keys: list[str]) -> dict:
    import json as _json
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return {k: 0.0 for k in keys}
    if isinstance(v, dict):
        return {k: float(v.get(k, 0.0) or 0.0) for k in keys}
    try:
        d = _json.loads(v)
        return {k: float(d.get(k, 0.0) or 0.0) for k in keys}
    except (TypeError, ValueError):
        return {k: 0.0 for k in keys}


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
