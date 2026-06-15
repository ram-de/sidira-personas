# sidira-personas — Milestones

**Note:** This file is now a redirect. As of 2026-05-21, planning consolidated into the simulator repo (which is the integration point for all three repos).

| What | Where |
|---|---|
| Strategy, phases, critical path | [`sidira-simulator/ROADMAP.md`](https://github.com/ram-de/sidira-simulator/blob/main/ROADMAP.md) |
| Open tickets (this repo + simulator + crawler) | [Sidira GitHub Project](https://github.com/users/ram-de/projects/) |
| Current sprint narrative | [`sidira-simulator/CONTEXT.md`](https://github.com/ram-de/sidira-simulator/blob/main/CONTEXT.md) |
| Team conventions | [`sidira-simulator/AGENTS.md`](https://github.com/ram-de/sidira-simulator/blob/main/AGENTS.md) |

## This repo's current state

- ✅ LLM-first persona naming (Groq → Anthropic) with a generic numeric fallback
- ✅ Persona Pack schema v1.1 (additive: full centroids, within-cluster scale, sampleable rows, soft posteriors)
- 🔲 R2 artifact integration (writing to the same R2 layout the simulator reads from)
- 🔲 Per-cluster Jaccard stability + train/test holdout
- 🔲 Persona methodology rebuild (feature ontology + validation protocol)

Specs and ticket detail live in the GitHub Project under repo:sidira-personas.
