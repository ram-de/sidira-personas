# pythia-personas

Per-company persona calibration for [Pythia](https://github.com/ram-de/pythia-scraper).

Takes a user-level dataset organized into a **fixed set of feature groups** (with variable metrics inside each group) and emits a **Persona Pack** — a versioned JSON artifact that defines the personas, their population weights, and an LLM system prompt for each persona.

The Persona Pack is consumed by the Pythia simulation engine to seed agent populations that faithfully represent a real company's userbase.

## The Fixed Schema

Seven feature groups. Constant across companies. Metrics inside each group are company-specific.

1. Demographic & Identity
2. Behavioral Patterns
3. Motivations & Jobs-to-be-Done
4. Pain Points & Friction
5. Value Sensitivity
6. Channel & Tech Behavior
7. Loyalty & Lifecycle Stage

## Pipeline

```
User dataset (Snowflake / Parquet / CSV)
   ↓ A — Adapter        normalize per group (company-specific)
   ↓ B — Encoder        PCA per group → fixed-width embedding
   ↓ C — Cluster        GMM (BIC-selected k) on concatenated embeddings
   ↓ D — Calibrate      weights (headcount / orders / revenue) + bootstrap stability
   ↓ E — Compile        persona centroid → LLM system prompt
   ↓
Persona Pack JSON  +  persona_prompts/*.md
```

## Quickstart

```bash
pip install -e .
pythia-personas run --config config/doordash.yaml
```

Outputs land in `outputs/<run_id>/`:
- `persona_pack.json`
- `prompts/P01.md … P0N.md`
- `assignments.parquet` (consumer_id → persona_id)
- `report.html` (cluster diagnostics, weights, stability)

## Status

v0.1 — pipeline runs end-to-end on real Snowflake data. First two runs documented below.

## Sample output

See [`samples/run_2_active_only/`](samples/run_2_active_only/) for a complete Persona Pack (10 personas, 7 prompts, baseline crosstab) from a 10K-user DoorDash run.

## Runs to date

| Run | Config | k | Stability | Headline finding |
|---|---|---|---|---|
| #1 | all users, no exclusions, k∈[5,12] | 12 | 0.797 | Demographics dominate; lifestage rediscovered as clusters |
| #2 | prefilter Active, exclude demo+lifestage from embedding, k∈[4,8] | 7 | 0.439 | Niche personas clean (grocery, app-uninstallers, Caviar); 53% mass remains |

## Follow-ups

1. **Feature enrichment (highest leverage).** Pull the columns currently NULL in the source snapshot — at minimum:
   - `PRICE_SENSITIVITY_SCORE_PSM`, `PROMO_SENSITIVITY_MODEL_SCORE`
   - `CONSUMER_OUT_OF_POCKET_PER_ORDER`, `AVG_DELIVERY_FEE_DISCOUNT`, `AVG_SERVICE_FEE_DISCOUNT`, `VOLUME_WITH_FEE_DISCOUNT_SHARE`
   - `GOTO_MERCHANT_LOYALIST_SCORE`
   - `APP_SESSION_FREQUENCY_28D`, `DAY_OF_WEEK_ORDER_PROFILE`, `TIME_OF_DAY_ORDER_PROFILE`
   - `CHANNEL_USAGE_MIX`, `CATEGORY_MIX_PROFILE` (richer than the JSON we have)
   - Loyalty fields if any users have them populated
2. **Stratified sample.** Current 10K is 88% Active. Pull a sample stratified across lifestage so dormant/churned have enough representation for the simulation engine to cover winback/resurrection hypotheses.
3. **Sub-cluster the residual mass** (Option 3 diagnostic). HDBSCAN on P04's 4,648 users to confirm whether the mass is genuinely amorphous or whether HDBSCAN finds substructure that GMM missed.
4. **LLM persona naming.** Wire the Anthropic call in `compile_prompts.py` — feed centroid feature deltas → get name + description + JTBD. Currently emits placeholder `"Persona P0X"`.
5. **Per-cluster stability.** Bootstrap stability is currently a single overall ARI. Switch to per-cluster Jaccard so unstable clusters can be flagged individually.
6. **HTML diagnostic report.** BIC curve, weight distributions, t-SNE scatter of the 40-dim embedding colored by cluster, baseline crosstab. Useful before any cluster ships to Pythia.
7. **Per-cluster orders/revenue concentration alerts.** When a small cluster carries an outsized share of orders (e.g., P06 at 15% headcount drives 22% orders), the Persona Pack should surface that as a "high leverage" flag for simulation cohort selection.
8. **Schema versioning.** Persona Pack v1.0 is current. Bump to v1.1 when feature enrichment lands (downstream consumers will need to know the schema changed).
