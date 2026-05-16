# Sample Persona Pack — Run #2

Source: `PRODDB.RAMKUMARKARUNANIDHI.CHANNEL_TECH_FEATURES_10K` (10,000 DoorDash users, snapshot 2026-05-14)

## Pipeline config

- **Prefilter:** `cx_lifestage_detailed_status ∈ {Active}` (8,823 clustered; 1,177 set aside as fixed lifestage personas)
- **Clustering exclude groups:** `demographic_identity`, `loyalty_lifecycle` (kept in profiling, dropped from embedding)
- **Encoder:** PCA, 8 dims per included group → 40-dim user embedding
- **Cluster:** GMM, BIC over k∈[4, 8] → **k=7 selected**
- **Stability:** 0.439 (below 0.65 threshold — see top-level notes)

## What's in this folder

- `persona_pack.json` — the deliverable. 10 personas (7 discovered + 3 fixed lifestage) with sizes, weights, top features, stability.
- `prompts/P0?.md` — LLM-ready system prompts for each persona (placeholder names/descriptions; LLM-naming hook not yet wired).
- `baseline_crosstab.csv` — crosstab of `hv_mau_segment` × discovered persona, useful for sanity checking against the pre-existing segmentation.

## Known limitations of this run

1. **P04 is 53% of users with one weak feature** — the populated features in this sample don't separate Active users finely enough.
2. **Stability is below threshold** for the discovered clusters; the niche clusters (grocery power, app-uninstallers, Caviar SMS, Australia) are stable; the mass cluster is not.
3. **Loyalty group is empty** in the source data, so loyalty-driven personas can't be discovered.

See repo root `README.md` for follow-up plan to enrich features.
