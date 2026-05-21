# pythia-personas — Milestones

Tracks progress from initial design through customer-ready persona pack delivery.

**Last updated:** 2026-05-20

## Phase tracker

| Phase | Status | Completion |
|---|---|---|
| **0. Architecture** | ✅ Done | 2026-04-15 |
| **1. Adapter + Encoder + Cluster + Calibrate** | ✅ Done | 2026-05-16 |
| **2. Prompt compilation + LLM naming** | ✅ Done | 2026-05-17 |
| **3. Round 1 feature enrichment** | ✅ Done | 2026-05-17 |
| **4. Round 2 enrichment (PSM, loyalty, UPSS, Hub)** | 🔲 Pending | ~1 week (gated on source data) |
| **5. R2 artifact integration** | 🔲 Pending | ~0.5 weeks |
| **6. Persona Pack schema v1.1 + validation** | 🔲 Pending | ~0.5 weeks |
| **7. First customer (Lumen MVP)** | 🔲 Pending | aligned with simulator Phase 7 |
| **8. Layer-3 calibration loop** | 🔲 Pending | aligned with simulator Phase 9 |

## Completed

### Phase 0 — Architecture (2026-04-15)
- 4-layer architecture finalized: Adapter (company-specific) / Encoder (generic) / Persona Model (generic) / Calibration (generic)
- Layer E added: prompt compiler — turns persona centroids into structured system prompts
- Persona Pack JSON identified as the deliverable contract
- Decision: unsupervised clustering, not classification — Pythia needs persona definitions, not labels
- Decision: skip pretraining/fine-tuning — use off-the-shelf LLMs with structured prompts

### Phase 1 — Adapter/Encoder/Cluster/Calibrate (2026-05-16)
- Repo created: `git@github.com-personal:ram-de/pythia-personas.git`
- Source: `PRODDB.RAMKUMARKARUNANIDHI.CHANNEL_TECH_FEATURES_10K` (146 cols, 10K users)
- 7 feature groups configured (constant across companies)
- Run #1 baseline: k=12, stability 0.797 (demographics dominated)
- Run #2: prefilter Active + exclude demo/lifestage from embedding, k=7, stability 0.439

### Phase 2 — Prompt compilation + LLM naming (2026-05-17)
- Run #5: population baseline + per-persona feature snapshots; auto-named via z-deltas; optional Claude polish
- Run #6: explicit baseline scope labels; per-persona behavioral snapshots for FIXED_* segments
- Architecture cleanup: removed all hard-coded DoorDash naming rules; LLM-first naming with `Persona##` fallback
- Run #10: end-to-end LLM naming via Groq verified (`llama-3.3-70b-versatile`); Anthropic as fallback
- Provider preference locked in: Groq → Anthropic → Persona## fallback

### Phase 3 — Round 1 feature enrichment (2026-05-17)
- 20+ previously-NULL columns repopulated: DashPass status, device footprint, notification reachability, app sessions, JSON profiles (dow/tod/channel/category), subscription health, goto-merchant score, fee discounts
- New `json_counts` adapter type added
- Stability: 0.439 → 0.670 (passes 0.65 threshold for first time)
- Residual mass split cleanly: P03 (36% DashPass Healthy) + P06 (40% Discount-Driven Restaurant) + P07 (10.7% Unreachable)
- Sample: `samples/run_10_llm_named_groq/`

---

## Future phases — detailed spec

### Phase 4 — Round 2 enrichment

Still-NULL high-leverage columns to land in source table:
- `PRICE_SENSITIVITY_SCORE_PSM`, `PROMO_SENSITIVITY_MODEL_SCORE`, `IS_LOW_ROI_CX`
- All loyalty fields: `HAS_REDEEMED_*`, `POINTS_*`, `ROAS`, `LOYALTY_MEMBER_AOV`, `TOTAL_REDEMPTIONS*`
- `UPSS_LIFESTAGE_ATTRIBUTE`, `UPSS_CHURN_ENGAGEMENT_SCORES`
- Notifications Hub: `HUB_NOTIFICATIONS_SENT`, `HUB_VIEW_TO_CLICK_RATE`, `HUB_REVISIT_RATE`
- `HIGH_PRICE_VARIANCE_ITEMS_EXPOSED`, `ORDER_DROP_AT_FEE_CHANGE`

Expected: stability → > 0.75, P03 (DashPass Healthy) and P06 (Discount-Driven Restaurant) split into 3–4 sub-personas each.

### Phase 5 — R2 artifact integration

**Goal:** Persona packs flow into the same R2 artifact store as feature packs, with the layout the simulator already expects.

**Tasks:**
- R2 layout per customer:
  ```
  customers/<customer_id>/personas/v_<N>/
  ├── manifest.json          # PersonaPack manifest
  ├── persona_pack.json      # full pack
  └── prompts/<persona_name>.md
  ```
- Pointer file `customers/<id>/personas/latest` → version string (parallel to crawler's pointer)
- `pythia_personas/storage.py` new module mirroring crawler's R2 + Local adapters
- CLI flag `--upload` or `--storage r2` to route output to R2 (default still local)
- Atomic pointer update only on successful run

**Acceptance:**
- `pythia-personas run --customer acme --storage r2` writes complete artifact set + updates `latest` pointer
- Simulator can read the pack via its `persona_loader.py` adapter
- A failed mid-run does not update the pointer

**Dependencies:** Phase 3 ✅ (sample structure already producing the right shape)

**Risks:**
- **Schema drift** with simulator's PersonaPack model — keep both in lockstep until `pythia-schemas` package factor-out happens
- **Pointer atomicity** — same risk as crawler; use write-then-read-back verification

**Estimated effort:** ~0.5 weeks

---

### Phase 6 — Persona Pack schema v1.1 + validation

**Goal:** Tighten the schema as the simulator's needs become clearer; add validation gates so a bad pack can't break a customer's simulations.

**Tasks:**
- Schema additions (informed by simulator integration in Phase 5):
  - `schema_version: "1.1"`
  - `customer_id` (so packs are self-identifying)
  - `behavioral_snapshot` made required per persona (currently optional)
  - `weights.revenue_weighted` populated (currently always None)
  - `stability_score` with version-bumped minimum (currently 0.439 in earlier runs — for v1.1 require ≥0.65)
- Validation gate in CI: every released pack must pass a `validate_pack.py` script before pointer update
- Migration: rename old `samples/run_*` to `samples/v_1_0/` and produce a new `v_1_1` sample
- Document schema-version policy in README: backwards-compat-one-minor-version

**Acceptance:**
- New packs land at schema_version 1.1 with all required fields
- Old packs (1.0) still readable by simulator (forwards-compat read)
- CI gate blocks a pack with stability < 0.65 from being promoted to `latest`

**Dependencies:** Phase 5

**Risks:**
- **Breaking changes** to simulator readers — mitigate by reading via shared Pydantic models with `Optional` fields and explicit defaults

**Estimated effort:** ~0.5 weeks

---

### Phase 7 — First customer (Lumen MVP)

Aligned with simulator Phase 7. Personas-specific requirements:
- Customer-supplied user table (Snowflake source, BigQuery source, or CSV)
- Customer-supplied feature-group mapping if their columns don't fit the 7-group default
- First persona pack reviewed by customer's data team before going to simulator
- Quarterly recalibration schedule agreed (the cadence for `run_crawl`-equivalent for personas)

**Acceptance:**
- First customer's persona pack lands; stability ≥0.65
- Customer's PMs run ≥3 simulations using the pack
- First quarterly recalibration completes unattended

**Dependencies:** Phases 4–6 + simulator Phase 6

---

### Phase 8 — Layer-3 calibration loop

Aligned with simulator Phase 9. Personas-specific responsibilities:
- Consume realistic-flag aggregations from simulator → adjust persona selection criteria
- Drop low-realism personas from next quarterly pack
- Boost weighting of high-realism personas
- Track per-persona accuracy contribution

**Goal:** A customer's persona pack gets *better* with use, not just refreshed. By month 12, personas selected for simulations are demonstrably more predictive than at month 1.

**Dependencies:** simulator Phase 9 — outcome capture + realism flagging must exist first

**Estimated effort:** ongoing; first measurable result aligned with simulator Phase 9 timeline

---

## Round 2 enrichment work (in flight — Phase 4 details)

## Other follow-ups (from README + session notes)

1. **Stratified sample** — current sample is 88% Active. Pull lifestage-stratified 10K (≥1K each for dormant, churned).
2. **Sub-cluster the residual P04** with HDBSCAN as a diagnostic (10-min spike).
3. **Per-cluster stability** (Jaccard) instead of single overall ARI.
4. **HTML diagnostic report** — BIC curve, weight distributions, t-SNE, baseline crosstab.
5. **High-leverage cluster alerts** — flag clusters where orders/revenue weight >> headcount weight.
6. **Persona Pack schema versioning** — bump to 1.1 when Round 2 enrichment lands.

## Decisions log

- ✅ Unsupervised clustering, not classification
- ✅ 5-layer architecture (Adapter / Encoder / Cluster / Calibrate / Compile Prompts)
- ✅ Persona Pack JSON as the deliverable contract
- ✅ No pretraining — use off-the-shelf LLMs with structured prompts
- ✅ Multiple weight schemes (head-count + orders + revenue) so simulations can choose
- ✅ Prefilter known lifestage segments as fixed personas; cluster only the residual
- ✅ LLM-first naming with generic `Persona##` fallback — keeps Layer E company-agnostic
- ✅ Groq default LLM provider; Anthropic fallback

## Related artifacts

- Feature-pack producer: [`pythia-crawler`](https://github.com/ram-de/pythia-crawler)
- Personal wiki entry: `~/vaults/personal/wiki/projects/pythia.md`
- System architecture: `~/vaults/personal/wiki/docs/pythia-system-architecture.md`
- Active working notes: `~/vaults/personal/sessions/persona-classifier/context.md`
