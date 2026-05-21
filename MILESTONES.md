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
| **4. Round 2 enrichment (PSM, loyalty, UPSS, Hub)** | 🔲 Pending | — |
| **5. R2 artifact integration** | 🔲 Pending | — |
| **6. First customer (Lumen MVP)** | 🔲 Pending | — |

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

## Next up (Phase 4 — Round 2 enrichment)

Still-NULL high-leverage columns to land in source table:
- `PRICE_SENSITIVITY_SCORE_PSM`, `PROMO_SENSITIVITY_MODEL_SCORE`, `IS_LOW_ROI_CX`
- All loyalty fields: `HAS_REDEEMED_*`, `POINTS_*`, `ROAS`, `LOYALTY_MEMBER_AOV`, `TOTAL_REDEMPTIONS*`
- `UPSS_LIFESTAGE_ATTRIBUTE`, `UPSS_CHURN_ENGAGEMENT_SCORES`
- Notifications Hub: `HUB_NOTIFICATIONS_SENT`, `HUB_VIEW_TO_CLICK_RATE`, `HUB_REVISIT_RATE`
- `HIGH_PRICE_VARIANCE_ITEMS_EXPOSED`, `ORDER_DROP_AT_FEE_CHANGE`

Expected: stability → > 0.75, P03 (DashPass Healthy) and P06 (Discount-Driven Restaurant) split into 3–4 sub-personas each.

## Phase 5 — R2 artifact integration (planned)

Wire `pythia-personas` outputs to the shared Cloudflare R2 artifact store using the same layout as `pythia-crawler`:

```
customers/<customer_id>/personas/v_<N>/
├── manifest.json
├── persona_pack.json
└── prompts/<persona_name>.md
```

Both producers writing to the same store is what makes them simultaneously consumable by `pythia-simulator`.

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
