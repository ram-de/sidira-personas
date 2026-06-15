"""Layer E — Compile a persona profile into an LLM-ready system prompt.

The prompt has three sections:
  1. Persona summary (LLM-generated name + description + JTBD)
  2. Baseline — what the average {company} user looks like (reference frame)
  3. Your profile — distinctive deltas + full populated feature snapshot

## Naming strategy (company-agnostic)

Primary path: call Claude with the persona's z-deltas + feature values, get back
{name, description, jtbd}. This works on any company / any feature set because
the LLM reads the feature names and values directly — no per-company rules.

Fallback path (LLM unreachable, no API key, or API error): emit a generic
"Persona NN" name with a description listing the top-3 distinctive features.
The pipeline NEVER hangs on the LLM call; failures are silent.
"""
from __future__ import annotations

import os
from typing import Iterable

# Hard-coded company-specific rules have been removed (2026-05-17).
# Naming is now LLM-driven with a generic numeric fallback.


def generic_fallback_name(persona_id: str, top_features: list[dict]) -> tuple[str, str, str]:
    """Company-agnostic fallback used when the LLM is unavailable.

    Names are numeric (Persona01, Persona02 …). Description lists the top 3
    distinctive features so the LLM downstream can still reason from the deltas.
    """
    name = f"Persona{persona_id.lstrip('P').zfill(2)}" if persona_id.startswith("P") else persona_id
    if top_features:
        bullets = []
        for f in top_features[:3]:
            sign = "+" if f["z_delta"] >= 0 else ""
            bullets.append(f"{f['group']}.{f['feature']} ({sign}{f['z_delta']:.2f}σ)")
        desc = (
            "Discovered cluster — LLM naming unavailable. "
            "Top distinctive features vs. baseline: " + "; ".join(bullets) + "."
        )
    else:
        desc = "Discovered cluster — LLM naming unavailable; no strongly distinctive features."
    jtbd = "(not generated — LLM unavailable)"
    return name, desc, jtbd


# ---------- Prompt template ----------

PROMPT_TEMPLATE = """# You are a {company_id} user. Reason and respond from this person's perspective.

## Persona: {name}
{description}

**Primary job-to-be-done:** {jtbd}

---

## Baseline — reference population

**Scope:** {baseline_scope}

The "baseline" values below are the AVERAGE of the {baseline_n:,}-user reference set. They are NOT industry averages and NOT the average across all {company_id} users globally — they describe ONLY this sample.

{baseline_block}

---

## Your profile — how you DIFFER from baseline
You stand out from the average user on these dimensions:

{distinctive_block}

---

## Your full behavioral snapshot
Every populated feature about you, alongside the population baseline:

{full_block}

---

## How to behave
- Stay grounded in the deltas above. When a feature is much higher or lower than baseline, your behavior should reflect that.
- When asked about a hypothetical product change, react the way THIS profile would — not the way an average user would.
- Express hesitation, contradiction, and trade-offs when the feature evidence is mixed.
- Do not invent demographic details that aren't supported by the profile.
- When you don't have data on something, say so — don't speculate.
"""


def _format_baseline_block(baseline: dict) -> str:
    by_group: dict[str, list[str]] = {}
    for feat, info in baseline.items():
        line = f"- `{feat}`: **{info['value_fmt']}**"
        by_group.setdefault(info["group"], []).append(line)
    parts = []
    for g, lines in by_group.items():
        parts.append(f"### {g}\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _format_distinctive_block(top_features: Iterable[dict], persona_values: dict, baseline: dict) -> str:
    lines = []
    for tf in top_features:
        feat = tf["feature"]
        sign = "+" if tf["z_delta"] >= 0 else ""
        pv = persona_values.get(feat, {}).get("value_fmt", "?")
        bv = baseline.get(feat, {}).get("value_fmt", "?")
        lines.append(f"- `{tf['group']}.{feat}`: **you = {pv}**, baseline = {bv}  ({sign}{tf['z_delta']:.2f}σ)")
    return "\n".join(lines) if lines else "_(no features strongly distinctive vs. baseline)_"


def _format_full_block(persona_values: dict, baseline: dict) -> str:
    by_group: dict[str, list[str]] = {}
    for feat, info in persona_values.items():
        bv = baseline.get(feat, {}).get("value_fmt", "?")
        line = f"- `{feat}`: **{info['value_fmt']}** (baseline {bv})"
        by_group.setdefault(info["group"], []).append(line)
    parts = []
    for g, lines in by_group.items():
        parts.append(f"### {g}\n" + "\n".join(lines))
    return "\n\n".join(parts)


def compile_prompt(company_id: str, persona: dict, persona_values: dict,
                   baseline: dict, top_features: Iterable[dict],
                   baseline_scope: str = "all users in the analyzed sample",
                   baseline_n: int = 0) -> str:
    return PROMPT_TEMPLATE.format(
        company_id=company_id,
        name=persona["name"],
        description=persona["description"],
        jtbd=persona["jtbd"],
        baseline_scope=baseline_scope,
        baseline_n=baseline_n,
        baseline_block=_format_baseline_block(baseline),
        distinctive_block=_format_distinctive_block(top_features, persona_values, baseline),
        full_block=_format_full_block(persona_values, baseline),
    )


# ---------- LLM naming (primary path) ----------

def _build_naming_prompt(company_id: str, persona_values: dict, top_features: list[dict]) -> str:
    feature_lines = "\n".join(
        f"- {f['group']}.{f['feature']}: "
        f"persona={persona_values.get(f['feature'],{}).get('value_fmt','?')}, "
        f"delta={f['z_delta']:+.2f}σ vs. baseline"
        for f in top_features[:20]
    )
    return (
        f"You are naming a user persona discovered via unsupervised clustering "
        f"on a {company_id} user dataset.\n\n"
        f"Below are the persona's distinctive features (highest absolute z-delta "
        f"vs. the population baseline). Each line shows the feature, the "
        f"persona's value, and how far it sits from baseline in standard "
        f"deviations.\n\n"
        f"{feature_lines}\n\n"
        f"Produce a JSON object with these keys:\n"
        f"- name (≤5 words, a behavioral label that captures what makes this "
        f"persona distinct — NOT a demographic label unless demographics are "
        f"the only strong signal)\n"
        f"- description (1 sentence, ~25 words, naming the 2–3 strongest "
        f"signals in plain language)\n"
        f"- jtbd (1 sentence, written as a 'job to be done' from the "
        f"persona's first-person motivation perspective)\n\n"
        f"Output JSON only — no prose, no code fences."
    )


def _parse_naming_json(text: str) -> tuple[str, str, str]:
    import json as _json
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    obj = _json.loads(text)
    return obj["name"], obj["description"], obj["jtbd"]


def _llm_name_groq(prompt: str) -> tuple[str, str, str] | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        resp = client.chat.completions.create(
            model=model,
            max_tokens=400,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_naming_json(resp.choices[0].message.content)
    except Exception:
        return None


def _llm_name_anthropic(prompt: str) -> tuple[str, str, str] | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        return _parse_naming_json(msg.content[0].text)
    except Exception:
        return None


def llm_name(company_id: str, persona_values: dict, top_features: list[dict]) -> tuple[str, str, str] | None:
    """Ask an LLM to name + describe a persona from its z-deltas vs. baseline.

    Provider preference (first available wins): Groq → Anthropic.
    Returns None on any failure; caller must fall back to `generic_fallback_name`.

    Company-agnostic: the LLM sees only company_id, feature names, persona values,
    and baseline-relative z-scores. Works on any feature schema.
    """
    prompt = _build_naming_prompt(company_id, persona_values, top_features)
    result = _llm_name_groq(prompt)
    if result is not None:
        return result
    return _llm_name_anthropic(prompt)


def llm_provider_in_use() -> str | None:
    """Used by the CLI to print which provider it picked up. None = fallback path."""
    if os.environ.get("GROQ_API_KEY"):
        return f"groq:{os.environ.get('GROQ_MODEL', 'llama-3.3-70b-versatile')}"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return f"anthropic:{os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-6')}"
    return None
