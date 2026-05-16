"""Layer A — Adapter. Company-specific. Normalizes raw metrics per feature group."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_count(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    s = np.log1p(s.clip(lower=0))
    mu, sigma = s.mean(), s.std() or 1.0
    return (s - mu) / sigma


def _normalize_score(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.fillna(s.median())
    mu, sigma = s.mean(), s.std() or 1.0
    return (s - mu) / sigma


def _normalize_ratio(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0, 1)
    return s


def _normalize_flag(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).clip(0, 1).astype(float)


def _onehot(s: pd.Series, name: str) -> pd.DataFrame:
    return pd.get_dummies(s.fillna("UNK"), prefix=name, dtype=float)


_FNS = {
    "count": _normalize_count,
    "score": _normalize_score,
    "ratio": _normalize_ratio,
    "flag": _normalize_flag,
}


def adapt_group(df: pd.DataFrame, metric_specs: list[dict]) -> pd.DataFrame:
    """Normalize the metrics for one feature group → returns a DataFrame of features."""
    cols: list[pd.DataFrame | pd.Series] = []
    for spec in metric_specs:
        name, mtype = spec["name"], spec["type"]
        if name not in df.columns:
            continue
        if mtype == "categorical":
            cols.append(_onehot(df[name], name))
        else:
            fn = _FNS.get(mtype)
            if fn is None:
                raise ValueError(f"unknown metric type {mtype} for {name}")
            cols.append(fn(df[name]).rename(name))
    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.concat(cols, axis=1)


def adapt_all(df: pd.DataFrame, feature_groups: dict[str, list[dict]]) -> dict[str, pd.DataFrame]:
    return {g: adapt_group(df, specs) for g, specs in feature_groups.items()}
