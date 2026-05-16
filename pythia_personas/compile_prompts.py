"""Layer E — Compile a persona centroid into an LLM-ready system prompt.

For v0.1 the compiler is template-based and deterministic. Optionally an LLM
(Anthropic) can be called to produce richer name/description/JTBD text — that
hook is intentionally isolated here so the rest of the pipeline stays offline.
"""
from __future__ import annotations

from typing import Iterable

PROMPT_TEMPLATE = """# You are a {company_id} user. Reason and respond from this person's perspective.

## Persona: {name}
{description}

**Primary job-to-be-done:** {jtbd}

## Your profile vs. the average {company_id} user
{feature_lines}

## How to behave
- Stay grounded in the deltas above. Where a feature is much higher or lower than average, your behavior should reflect that.
- When asked about a hypothetical product change, react the way THIS profile would — not the way an average user would.
- Express hesitation, contradiction, and trade-offs when the feature evidence is mixed.
- Do not invent demographic details that aren't supported by the profile.
"""


def _format_delta(d: dict) -> str:
    sign = "+" if d["z_delta"] >= 0 else ""
    return f"- `{d['group']}.{d['feature']}` — {sign}{d['z_delta']:.2f}σ vs. population"


def compile_prompt(company_id: str, persona: dict, top_features: Iterable[dict]) -> str:
    feature_lines = "\n".join(_format_delta(f) for f in top_features) or "- (no strongly distinctive features)"
    return PROMPT_TEMPLATE.format(
        company_id=company_id,
        name=persona["name"],
        description=persona["description"],
        jtbd=persona["jtbd"],
        feature_lines=feature_lines,
    )
