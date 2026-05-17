# Sample Persona Pack — Run #9 (enriched source, stability passes threshold)

Source: `PRODDB.RAMKUMARKARUNANIDHI.CHANNEL_TECH_FEATURES_10K` (10,000 users, snapshot 2026-05-14)
Schema has been enriched — ~20 previously-NULL columns now populated.

## Why this run matters

**Stability crossed the threshold for the first time: 0.670 ≥ 0.65.**

The previous runs had ~53% of users in one "average" mass with weak signals. With enriched features (DashPass status, app sessions, fee discounts, channel/category/time-of-day profiles, goto-merchant score, subscription health), that mass split into two large meaningful clusters: Healthy DashPass Subscribers (36%) and Discount-Driven Restaurant Regulars (40%). Stability rose from 0.439 → 0.670.

## The 10 personas

| ID | n | Headcount | Orders | Name |
|---|---|---|---|---|
| P01 | 408 | 4.6% | 3.9% | Australia Cohort |
| P02 | 154 | 1.7% | 1.2% | Caviar SMS Loyalist |
| P03 | 3,208 | 36.4% | 43.9% | **Healthy DashPass Subscriber** (NEW — was inside residual) |
| P04 | 238 | 2.7% | 2.6% | Notification-Fatigued Engaged User |
| P05 | 342 | 3.9% | 6.7% | **DashPass Grocery Power User** (sharper than before — high app sessions + NV freq) |
| P06 | 3,531 | 40.0% | 35.3% | **Discount-Driven Restaurant Regular** (NEW — was inside residual) |
| P07 | 942 | 10.7% | 6.4% | **Unreachable Disengaged User** (NEW — was hidden) |
| FIXED_DORMANT_RESURRECTED | 794 | 7.9% | 2.2% | Lifestage segment |
| FIXED_CHURNED_RESURRECTED | 258 | 2.6% | 0.5% | Lifestage segment |
| FIXED_NEW_CX | 125 | 1.3% | 0.0% | Lifestage segment |

## What feature enrichment unlocked

The newly populated columns that drove cluster separation:
- **DashPass status** — the 80/20 active/inactive split is now the single strongest divider
- **App session frequency 28d** — separates power users from passive engagement
- **Fee discount fields** (delivery / service / volume-with-discount-share) — distinguishes promo-funded users from full-price ones
- **Channel + category mix JSON profiles** — grocery vs restaurant vs catering became real signals
- **Notification reachability profile** — unreachable users were hidden before; now their own cluster
- **Health of subscription JTBD** — strongest motivational signal for DashPass quality
- **Goto-merchant loyalist score** — concentration of orders at favorite stores

## What's in this folder

- `persona_pack.json` — 10 personas with sizes, weights, top features, stability 0.670.
- `prompts/P0?.md` and `prompts/FIXED_*.md` — LLM-ready system prompts (~5KB each, with explicit baseline scope and full feature snapshot).
- `baseline_crosstab.csv` — discovered persona × `hv_mau_segment`.

## Still-NULL columns (next enrichment opportunity)

- PSM scores (`PRICE_SENSITIVITY_SCORE_PSM`, `PROMO_SENSITIVITY_MODEL_SCORE`, `IS_LOW_ROI_CX`)
- Notifications Hub (`HUB_*`)
- All loyalty fields (`HAS_REDEEMED_*`, `POINTS_*`, `ROAS`)
- UPSS lifestage & churn scores

Adding these would likely push stability past 0.75 and may further split the two large clusters into 3–4 sub-personas.
