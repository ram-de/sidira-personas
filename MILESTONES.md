# sidira-personas — Milestones

**Note:** This file is now a redirect. As of 2026-05-21, planning consolidated into the simulator repo (which is the integration point for all three repos).

| What | Where |
|---|---|
| Strategy, phases, critical path | [`sidira-simulator/ROADMAP.md`](https://github.com/ram-de/sidira-simulator/blob/main/ROADMAP.md) |
| Open tickets (this repo + simulator + crawler) | [Sidira GitHub Project](https://github.com/users/ram-de/projects/) |
| Current sprint narrative | [`sidira-simulator/CONTEXT.md`](https://github.com/ram-de/sidira-simulator/blob/main/CONTEXT.md) |
| Team conventions | [`sidira-simulator/AGENTS.md`](https://github.com/ram-de/sidira-simulator/blob/main/AGENTS.md) |

## This repo's current state

- ✅ Round 1 personas (6 personas via Groq llama-3.3-70b naming, stability 0.670)
- 🔲 Round 2 enrichment — Pricing PSM, loyalty, Hub Notifications, UPSS lifestage. Should push stability >0.75 and split P03/P06.
- 🔲 R2 artifact integration (writing to the same R2 layout the simulator reads from)
- 🔲 Schema v1.1 with validation
- 🔲 Per-cluster stats emission (optional input for simulator Phase 2.5 variance modeling)

Specs and ticket detail live in the GitHub Project under repo:sidira-personas.
