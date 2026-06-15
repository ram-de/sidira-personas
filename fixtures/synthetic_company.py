"""Synthetic company dataset generator — the end-to-end pipeline fixture (#12).

Replaces the purged DoorDash `config/` + `samples/` so the full pipeline
(Adapter → Encoder → Cluster → Calibrate → Compile) is runnable and CI-testable
with **no Snowflake, no API key, and no real data**.

The generated frame is paired with `config/synthetic.yaml`. It deliberately:

* spans all **7 fixed feature groups** and exercises **every adapter type**
  (count, score, ratio, flag, categorical, json_counts, json_ratios, tenure_days);
* embeds a small number of **separable latent segments** in the clustering groups
  (behavioral / motivations / pain / value / channel) so GMM+BIC recovers a real
  k>1 split — the demographic and loyalty groups are noisier and excluded from
  clustering, matching the real config's `clustering_exclude_groups`;
* carries a **prefilter column** (`cx_lifestage_detailed_status`) where the
  non-`Active` rows become the pipeline's FIXED_* hard-defined personas.

Everything is drawn from a seeded `numpy.random.Generator`, so `build()` is fully
deterministic — `build(seed=s)` always yields an identical frame.

Run as a script to (re)materialize the committed fixture parquet:

    python fixtures/synthetic_company.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --- Latent behavioral segments (centroids in the *clustering* feature space) -------------
# Each dict gives the per-feature segment mean; a shared spread is added as Gaussian noise and
# the draw is clipped to the feature's natural domain (ratios→[0,1], counts→≥0). The segments
# are spaced far enough apart that BIC recovers them, but they overlap enough to be realistic.
_SEGMENTS = [
    {  # S0 — power / high-frequency loyalist
        "name": "power",
        "total_orders_lifetime": 420, "order_frequency_28d": 14, "avg_days_between_orders": 2.0,
        "deal_seeker_flag": 0.15, "convenience_ratio": 0.80,
        "high_fee_rate": 0.10, "unsubscribe_rate": 0.02,
        "promo_redeemer_flag": 0.30, "out_of_pocket_per_order": 24.0, "discount_share": 0.15,
        "open_within_24h": 0.70, "click_within_24h": 0.45, "notifications_weekly": 9,
        "cat_mix": (0.15, 0.70, 0.15), "day_mix": (0.45, 0.55),
    },
    {  # S1 — deal-driven / promo-sensitive
        "name": "deal",
        "total_orders_lifetime": 130, "order_frequency_28d": 5, "avg_days_between_orders": 6.0,
        "deal_seeker_flag": 0.85, "convenience_ratio": 0.35,
        "high_fee_rate": 0.40, "unsubscribe_rate": 0.08,
        "promo_redeemer_flag": 0.90, "out_of_pocket_per_order": 11.0, "discount_share": 0.55,
        "open_within_24h": 0.55, "click_within_24h": 0.40, "notifications_weekly": 12,
        "cat_mix": (0.55, 0.25, 0.20), "day_mix": (0.50, 0.50),
    },
    {  # S2 — casual / low-engagement
        "name": "casual",
        "total_orders_lifetime": 22, "order_frequency_28d": 1, "avg_days_between_orders": 21.0,
        "deal_seeker_flag": 0.30, "convenience_ratio": 0.45,
        "high_fee_rate": 0.55, "unsubscribe_rate": 0.18,
        "promo_redeemer_flag": 0.25, "out_of_pocket_per_order": 17.0, "discount_share": 0.20,
        "open_within_24h": 0.20, "click_within_24h": 0.08, "notifications_weekly": 4,
        "cat_mix": (0.30, 0.45, 0.25), "day_mix": (0.60, 0.40),
    },
    {  # S3 — convenience / premium
        "name": "premium",
        "total_orders_lifetime": 240, "order_frequency_28d": 8, "avg_days_between_orders": 3.5,
        "deal_seeker_flag": 0.10, "convenience_ratio": 0.90,
        "high_fee_rate": 0.08, "unsubscribe_rate": 0.03,
        "promo_redeemer_flag": 0.15, "out_of_pocket_per_order": 31.0, "discount_share": 0.08,
        "open_within_24h": 0.60, "click_within_24h": 0.30, "notifications_weekly": 6,
        "cat_mix": (0.10, 0.55, 0.35), "day_mix": (0.35, 0.65),
    },
]

# Lifecycle split: the prefilter keeps `Active` for clustering; the rest become FIXED_* personas.
_LIFESTAGE = {"Active": 0.78, "Churned": 0.10, "Dormant": 0.08, "New": 0.04}
_REGIONS = ["West", "Midwest", "South", "Northeast"]
_NOTIF_CHANNELS = ["push", "email", "sms"]
_HV_SEGMENTS = ["high_value", "mid_value", "low_value"]
_SNAPSHOT = "2026-06-01"


def _clip_pos(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, None)


def _clip01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def build(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Build a deterministic synthetic userbase of `n` rows across 4 latent segments."""
    rng = np.random.default_rng(seed)

    # Assign each user a latent segment (roughly balanced, slightly uneven on purpose).
    seg_p = np.array([0.30, 0.25, 0.28, 0.17])
    seg_idx = rng.choice(len(_SEGMENTS), size=n, p=seg_p)

    def per_seg(field: str, spread: float, lo=None, hi=None) -> np.ndarray:
        mu = np.array([_SEGMENTS[s][field] for s in seg_idx], dtype=float)
        out = mu + rng.normal(0.0, 1.0, size=n) * spread
        if lo is not None or hi is not None:
            out = np.clip(out, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
        return out

    # --- numeric clustering features (segment-driven) ---
    total_orders = _clip_pos(per_seg("total_orders_lifetime", 35.0)).round()
    freq28 = _clip_pos(per_seg("order_frequency_28d", 2.0)).round()
    days_between = _clip_pos(per_seg("avg_days_between_orders", 2.5))
    convenience = _clip01(per_seg("convenience_ratio", 0.12))
    high_fee = _clip01(per_seg("high_fee_rate", 0.10))
    unsub = _clip01(per_seg("unsubscribe_rate", 0.04))
    out_of_pocket = _clip_pos(per_seg("out_of_pocket_per_order", 4.0))
    discount_share = _clip01(per_seg("discount_share", 0.10))
    open24 = _clip01(per_seg("open_within_24h", 0.12))
    click24 = _clip01(per_seg("click_within_24h", 0.10))
    notifs_weekly = _clip_pos(per_seg("notifications_weekly", 2.0)).round()

    # --- flags (Bernoulli on the segment's propensity) ---
    deal_seeker = (rng.random(n) < np.array([_SEGMENTS[s]["deal_seeker_flag"] for s in seg_idx])).astype(int)
    promo_redeemer = (rng.random(n) < np.array([_SEGMENTS[s]["promo_redeemer_flag"] for s in seg_idx])).astype(int)

    # --- json_ratios: category motivation (grocery, restaurant, retail), per-segment Dirichlet ---
    def cat_json(s_field: str) -> list[str]:
        out = []
        for s in seg_idx:
            base = np.array(_SEGMENTS[s][s_field], dtype=float)
            draw = rng.dirichlet(base * 12 + 0.5)
            out.append(json.dumps({"grocery": round(float(draw[0]), 4),
                                   "restaurant": round(float(draw[1]), 4),
                                   "retail": round(float(draw[2]), 4)}))
        return out

    category_motivation = cat_json("cat_mix")

    # --- json_counts: day-of-week order profile (weekday, weekend) as integer counts ---
    def day_json() -> list[str]:
        out = []
        for i, s in enumerate(seg_idx):
            wk, we = _SEGMENTS[s]["day_mix"]
            tot = max(1, int(total_orders[i] // 10) + 1)
            weekday = int(round(tot * wk))
            weekend = max(0, tot - weekday)
            out.append(json.dumps({"weekday": weekday, "weekend": weekend}))
        return out

    day_of_week_profile = day_json()

    # --- demographic / loyalty (excluded from clustering — noisier, weakly segment-linked) ---
    region = rng.choice(_REGIONS, size=n)
    device_count = _clip_pos(rng.normal(2.0, 1.0, size=n)).round() + 1
    # tenure: power/premium signed up longer ago on average
    tenure_bias = np.array([700, 400, 250, 600], dtype=float)[seg_idx]
    days_ago = _clip_pos(tenure_bias + rng.normal(0, 150, size=n)).astype(int)
    snapshot = pd.Timestamp(_SNAPSHOT)
    signup_date = (snapshot - pd.to_timedelta(days_ago, unit="D")).strftime("%Y-%m-%d")

    lifestage = rng.choice(list(_LIFESTAGE), size=n, p=list(_LIFESTAGE.values()))
    active_28d = ((lifestage == "Active") & (rng.random(n) < 0.9)).astype(int)

    notif_channel = rng.choice(_NOTIF_CHANNELS, size=n, p=[0.5, 0.35, 0.15])
    hv_segment = np.where(total_orders > 250, "high_value",
                          np.where(total_orders > 80, "mid_value", "low_value"))

    return pd.DataFrame({
        "consumer_id": [f"u{seed}_{i:06d}" for i in range(n)],
        "snapshot_date": _SNAPSHOT,
        # demographic_identity
        "signup_date": signup_date,
        "region": region,
        "device_count": device_count.astype(int),
        # behavioral_patterns
        "total_orders_lifetime": total_orders.astype(int),
        "order_frequency_28d": freq28.astype(int),
        "avg_days_between_orders": days_between.round(2),
        "day_of_week_profile": day_of_week_profile,
        # motivations_jtbd
        "deal_seeker_flag": deal_seeker,
        "convenience_ratio": convenience.round(4),
        "category_motivation": category_motivation,
        # pain_points_friction
        "high_fee_rate": high_fee.round(4),
        "unsubscribe_rate": unsub.round(4),
        # value_sensitivity
        "promo_redeemer_flag": promo_redeemer,
        "out_of_pocket_per_order": out_of_pocket.round(2),
        "discount_share": discount_share.round(4),
        # channel_tech_behavior
        "notification_channel": notif_channel,
        "open_within_24h": open24.round(4),
        "click_within_24h": click24.round(4),
        "notifications_weekly": notifs_weekly.astype(int),
        # loyalty_lifecycle
        "cx_lifestage_detailed_status": lifestage,
        "active_28d": active_28d,
        # baseline crosstab segment (not a feature group)
        "hv_segment": hv_segment,
    })


def main() -> None:
    out = Path(__file__).parent / "synthetic" / "users.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = build()
    df.to_parquet(out, index=False)
    print(f"wrote {len(df):,} rows × {df.shape[1]} cols → {out}")


if __name__ == "__main__":
    main()
