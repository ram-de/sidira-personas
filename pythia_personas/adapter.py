"""Layer A — Adapter. Company-specific. Normalizes raw metrics per feature group."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd


def _to_numeric_bool(s: pd.Series) -> pd.Series:
    """Coerce booleans, 0/1 floats, and 'true'/'false' strings into 0/1 floats."""
    if s.dtype == bool:
        return s.astype(float)
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().all():
        # try string mapping
        out = s.astype(str).str.lower().map({"true": 1.0, "false": 0.0, "t": 1.0, "f": 0.0})
    return out.fillna(0.0).clip(0, 1).astype(float)


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


def _onehot(s: pd.Series, name: str) -> pd.DataFrame:
    return pd.get_dummies(s.fillna("UNK").astype(str), prefix=name, dtype=float)


def _expand_json_ratios(s: pd.Series, name: str, keys: list[str]) -> pd.DataFrame:
    def parse(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return {k: 0.0 for k in keys}
        if isinstance(v, dict):
            d = v
        else:
            try:
                d = json.loads(v)
            except (TypeError, ValueError):
                return {k: 0.0 for k in keys}
        return {k: float(d.get(k, 0.0) or 0.0) for k in keys}

    parsed = s.apply(parse).tolist()
    df = pd.DataFrame(parsed, index=s.index)
    df = df.clip(0, 1).fillna(0.0)
    df.columns = [f"{name}__{k}" for k in keys]
    return df


def _tenure_days(s: pd.Series, snapshot: pd.Timestamp) -> pd.Series:
    """Days between date column and snapshot date → count-normalized."""
    dt = pd.to_datetime(s, errors="coerce")
    days = (snapshot - dt).dt.days.astype("float64")
    days = days.fillna(days.median() if days.notna().any() else 0.0)
    return _normalize_count(days.clip(lower=0))


def adapt_group(df: pd.DataFrame, metric_specs: list[dict],
                snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Normalize the metrics for one feature group → returns a DataFrame of features."""
    cols: list[pd.DataFrame | pd.Series] = []
    for spec in metric_specs:
        name = spec["name"]
        mtype = spec["type"]
        if name not in df.columns:
            continue

        if mtype == "count":
            cols.append(_normalize_count(df[name]).rename(name))
        elif mtype == "score":
            cols.append(_normalize_score(df[name]).rename(name))
        elif mtype == "ratio":
            cols.append(_normalize_ratio(df[name]).rename(name))
        elif mtype == "flag":
            cols.append(_to_numeric_bool(df[name]).rename(name))
        elif mtype == "categorical":
            cols.append(_onehot(df[name], name))
        elif mtype == "json_ratios":
            cols.append(_expand_json_ratios(df[name], name, spec["keys"]))
        elif mtype == "tenure_days":
            if snapshot is None:
                continue
            cols.append(_tenure_days(df[name], snapshot).rename(name))
        else:
            raise ValueError(f"unknown metric type {mtype} for {name}")

    if not cols:
        return pd.DataFrame(index=df.index)
    return pd.concat(cols, axis=1)


def adapt_all(df: pd.DataFrame, feature_groups: dict[str, list[dict]],
              snapshot: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    return {g: adapt_group(df, specs, snapshot=snapshot) for g, specs in feature_groups.items()}
