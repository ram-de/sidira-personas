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

v0.1 — scaffolding. Real pipeline lands incrementally.
