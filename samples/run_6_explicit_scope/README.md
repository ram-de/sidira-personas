# Sample Persona Pack — Run #6 (explicit baseline scope + fixed-persona profiles)

Source: `PRODDB.RAMKUMARKARUNANIDHI.CHANNEL_TECH_FEATURES_10K` (10,000 DoorDash users, snapshot 2026-05-14)

## What changed from run #5

1. **Baseline scope is explicit in every prompt.** The "Baseline" section now states verbatim what reference set the numbers come from, e.g.:
   > **Scope:** Snapshot from `PRODDB.RAMKUMARKARUNANIDHI.CHANNEL_TECH_FEATURES_10K` | 10,000 users in source | Baseline = the 8,823 users where `cx_lifestage_detailed_status ∈ ['Active']` (prefiltered before clustering) | This is a sample, not the full DoorDash population.

   No more ambiguity about whether "average user" means "all DoorDash users" (it doesn't — it's a 8,823-user Active-only sample).

2. **Fixed personas (Dormant / Churned / New Cx) now have their own behavioral snapshot.** Previously they only had the baseline section but no per-persona values. Now each FIXED_* prompt shows the lifestage segment's actual feature values vs. the Active baseline — so the LLM understands, e.g., that New Cx users have ~7 lifetime orders vs. the Active baseline of 431.

## Why the scope distinction matters for Pythia

When a simulation agent is told "you are 2σ above the baseline on `nv_order_frequency_28d`", the agent's reaction is only valid if it understands what baseline it's compared against. If the baseline secretly excludes dormant users, then "2σ above" means something different than "2σ above all DoorDash". Stating the scope inline makes every prompt auditable.

## What's in this folder

- `persona_pack.json` — the deliverable (10 personas, sizes, weights, top features, stability).
- `prompts/P0?.md` and `prompts/FIXED_*.md` — LLM-ready system prompts with explicit baseline scope.
- `baseline_crosstab.csv` — discovered persona × `hv_mau_segment`.

## Carry-over limitations

- P04 still holds 53% of users with weak signals — feature enrichment is still the highest-leverage follow-up.
- Stability 0.439 below 0.65 threshold for the discovered clusters.
