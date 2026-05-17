"""Layer E — Compile a persona profile into an LLM-ready system prompt.

The prompt has three sections:
  1. Persona summary (auto-generated name + description + JTBD)
  2. Baseline — what the average {company} user looks like (reference frame)
  3. Your profile — distinctive deltas + full populated feature snapshot

Deterministic naming: chooses a short tag from the persona's strongest
behavioral signals. Optional LLM polish via Anthropic if ANTHROPIC_API_KEY set.
"""
from __future__ import annotations

import os
from typing import Iterable

# ---------- Deterministic naming ----------

# Rules fire on z-deltas (vs population baseline), NOT absolute values.
# Order matters — first match wins. Put most specific niches at the top.
_NAMING_RULES = [
    # (predicate over deltas dict, name, description, jtbd)
    (
        lambda d: d.get("region_name_australia_oh", 0) > 1.0
                  or d.get("country_id_5_oh", 0) > 1.0,
        "Australia Cohort",
        "Australian DoorDash user. Category mix and notification reachability differ from the US baseline; grocery is a larger share of demand in this market.",
        "Get household and grocery essentials via DoorDash in a market with fewer restaurant alternatives.",
    ),
    (
        lambda d: d.get("consumer_experience_caviar_oh", 0) > 0.5
                  or d.get("notification_channel_sms_oh", 0) > 2.0,
        "Caviar SMS Loyalist",
        "Orders on the Caviar brand surface (not core DoorDash). SMS-reachable, not push-heavy. Expectations of restaurant quality and brand tone differ from mainstream DD users.",
        "Order from premium/curated restaurants on a brand they trust, with low promotional noise.",
    ),
    (
        lambda d: d.get("app_uninstall_after_notif", 0) > 0.5
                  or d.get("uninstall_within_24h", 0) > 0.5,
        "Notification-Fatigued Engaged User",
        "Orders regularly but marketing volume exceeds tolerance. Uninstalls and unsubscribes spike after notification campaigns. Sensitive to push/email frequency.",
        "Order when needed; aggressively reduce promotional noise from the app and brand.",
    ),
    (
        lambda d: d.get("weekly_grocery_stockup_flag", 0) > 0.7
                  and d.get("nv_order_frequency_28d", 0) > 0.5,
        "Grocery Stockup Power User",
        "Uses DoorDash as a weekly grocery delivery channel. NV-order frequency is materially above baseline; restaurant share is below average. Large basket sizes on a predictable cadence.",
        "Replace the weekly trip to the grocery store with predictable home delivery.",
    ),
    (
        lambda d: d.get("speed_seeker_flag", 0) > 0.5
                  and d.get("nv_order_frequency_28d", 0) > 0.3,
        "Fast Grocery Operator",
        "High-frequency grocery user who prioritizes delivery speed over price. Will pay a premium for same-day or express. App engagement is volatile.",
        "Get groceries fast for top-up missions and same-day household needs.",
    ),
    (
        lambda d: d.get("notification_message_type_reminders_oh", 0) > 0.5
                  and d.get("reminder_triggered_order_share", 0) > 0.3,
        "Reminder-Triggered Orderer",
        "Demand is largely triggered by cart-reminder notifications. Less proactive browsing; orders surface from re-engagement messaging.",
        "Re-engage with DoorDash when nudged about an abandoned cart or open session.",
    ),
    (
        lambda d: d.get("promo_triggered_order_share", 0) > 0.5
                  and d.get("promo_dependency_score", 0) > 0.5,
        "Promo-Triggered Bargain Hunter",
        "Heavy reliance on promotional discounts. Order frequency tracks promo eligibility; without a promo, conversion drops.",
        "Maximize savings — every order needs a deal to justify the fee structure.",
    ),
    (
        lambda d: d.get("order_frequency_28d", 0) > 1.0
                  and d.get("category_motivation_profile__restaurant", 0) > 0.3,
        "Daily Restaurant Power User",
        "Orders well above baseline frequency, restaurant-heavy. Habit is locked in and fee-tolerant within reason. Outsourcing meal prep as default.",
        "Outsource meal preparation as a default lifestyle pattern.",
    ),
    (
        lambda d: d.get("order_frequency_28d", 0) < -1.0
                  or d.get("total_orders_lifetime", 0) < -1.0,
        "Light Engaged User",
        "Active lifestage but at the low end of order frequency. Convenience-driven, not habit-driven. Increased friction risks churn.",
        "Use DoorDash for occasional convenience without a routine commitment.",
    ),
]

_FALLBACK = (
    "Average Engaged User",
    "Steady, moderate-frequency DoorDash user. Restaurant-leaning category mix, average notification engagement, mainstream behavior with no strongly distinctive signals.",
    "Get reliably delivered meals as a regular convenience tool.",
)


def auto_name(persona_values: dict, top_features: list[dict]) -> tuple[str, str, str]:
    """Pick a name from the persona's z-deltas (top_features) vs. baseline.

    Builds a delta dict keyed by feature name (raw + one-hot keys with `_oh` suffix
    to distinguish category-membership signals).
    """
    deltas: dict = {}
    for tf in top_features:
        feat = tf["feature"]
        deltas[feat] = tf["z_delta"]
        # tag one-hot expansions for cleaner predicates
        if "_" in feat and feat.split("_")[0] in {"region", "country", "consumer", "notification", "device"}:
            deltas[feat.lower() + "_oh"] = tf["z_delta"]
    # also expose categorical OH names lowercased
    for feat in list(deltas.keys()):
        if feat.endswith("_oh"):
            continue
        deltas[feat.lower() + "_oh"] = deltas[feat]

    for predicate, name, desc, jtbd in _NAMING_RULES:
        try:
            if predicate(deltas):
                return name, desc, jtbd
        except Exception:
            continue

    # Fallback: differentiate by strongest signed delta so the same fallback
    # name doesn't collide across multiple residual clusters.
    if top_features:
        strongest = max(top_features, key=lambda f: abs(f["z_delta"]))
        sign = "high" if strongest["z_delta"] > 0 else "low"
        feat_short = strongest["feature"].split("__")[-1].replace("_", " ").title()[:30]
        suffix = f" — {sign} {feat_short}"
        name = _FALLBACK[0] + suffix
        desc = (_FALLBACK[1] + f" Most distinctive: {strongest['group']}.{strongest['feature']} "
                f"({'+' if strongest['z_delta'] >= 0 else ''}{strongest['z_delta']:.2f}σ).")
        return name, desc, _FALLBACK[2]
    return _FALLBACK


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


# ---------- Optional LLM polish ----------

def llm_polish_name(persona_values: dict, top_features: list[dict]) -> tuple[str, str, str] | None:
    """If ANTHROPIC_API_KEY is set, ask Claude for a name/desc/JTBD. Returns None on any failure."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        feature_lines = "\n".join(
            f"- {f['group']}.{f['feature']}: persona={persona_values.get(f['feature'],{}).get('value_fmt','?')}, "
            f"delta={f['z_delta']:+.2f}σ"
            for f in top_features[:15]
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": (
                    "Given this DoorDash user persona's distinctive features (vs population), "
                    "produce a JSON object with keys: name (≤5 words), description (1 sentence), "
                    "jtbd (one short sentence, 'Job to be done' phrasing). No prose, just JSON.\n\n"
                    f"{feature_lines}"
                ),
            }],
        )
        import json as _json
        text = msg.content[0].text.strip()
        # tolerate fenced code
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        obj = _json.loads(text)
        return obj["name"], obj["description"], obj["jtbd"]
    except Exception:
        return None
