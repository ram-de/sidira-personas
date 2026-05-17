# Sample Persona Pack — Run #5 (with baseline + named personas)

Source: `PRODDB.RAMKUMARKARUNANIDHI.CHANNEL_TECH_FEATURES_10K` (10,000 DoorDash users, snapshot 2026-05-14)

## What changed from the run #2 sample

1. **Every persona has a name** — auto-generated from z-delta predicates against the population baseline. Fallback to "Average Engaged User — <strongest delta>" so no two personas collide.
2. **Population baseline is in every prompt** — "Average DoorDash user" section now lists actual values for every populated feature, so the LLM has a reference frame.
3. **Persona's actual feature values are in every prompt** — full behavioral snapshot lists every populated feature with the persona's value alongside the baseline (e.g., `order_frequency_28d: 22.4 (baseline 16.6)`).

## The 10 personas

| ID | n | Name |
|---|---|---|
| P01 | 171 | Australia Cohort |
| P02 | 400 | Notification-Fatigued Engaged User |
| P03 | 473 | Average Engaged User — high Device Type Notif (data-quality artifact) |
| P04 | 4,648 | Average Engaged User — low NV Order Frequency 28D (the residual mass) |
| P05 | 1,543 | Average Engaged User — high Notification Message Type Reminders |
| P06 | 1,365 | Grocery Stockup Power User |
| P07 | 223 | Caviar SMS Loyalist |
| FIXED_CHURNED_RESURRECTED | 258 | Fixed — Churned Resurrected (lifestage segment) |
| FIXED_DORMANT_RESURRECTED | 794 | Fixed — Dormant Resurrected (lifestage segment) |
| FIXED_NEW_CX | 125 | Fixed — New Cx (lifestage segment) |

## What's in this folder

- `persona_pack.json` — the deliverable (10 personas with sizes, weights, top features, stability).
- `prompts/P0?.md` and `prompts/FIXED_*.md` — LLM-ready system prompts. Each is ~3KB and self-contained (baseline + persona snapshot + behavior guidance).
- `baseline_crosstab.csv` — discovered persona × `hv_mau_segment` baseline.

## Known limitations (carried from run #2)

- P04 still holds 53% of users with weak distinctive signals — populated features lack the resolution to separate Active users further. See repo `README.md` follow-ups (feature enrichment is the highest-leverage next step).
- Stability 0.439 below 0.65 threshold — niche clusters (Australia, Caviar, Notification-Fatigued, Grocery) are stable; the residual is fuzzy.
