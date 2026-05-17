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
_REMOVED_NAMING_RULES: list = [
    # (predicate over deltas dict, name, description, jtbd)
    (
        lambda d: d.get("notification_reachability_status_unk_oh", 0) > 0.5
                  or d.get("notification_reachability_profile_unk_oh", 0) > 0.5,
        "Unreachable Disengaged User",
        "Active orderer but unreachable via notifications — profile flagged UNK on reach status. Engagement happens through direct app use, not marketing. Lower order frequency and fewer NV orders than baseline.",
        "Order on DoorDash without marketing pull; surfaces only when the user comes to the app directly.",
    ),
    (
        lambda d: d.get("region_name_australia_oh", 0) > 0.5
                  or d.get("country_name_australia_oh", 0) > 0.5,
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
        lambda d: d.get("app_session_frequency_28d", 0) > 1.0
                  and d.get("nv_order_frequency_28d", 0) > 0.5
                  and d.get("dashpass_status_active_oh", 0) > 0.2,
        "DashPass Grocery Power User",
        "DashPass subscriber with high app engagement and grocery-leaning NV orders. Treats DoorDash as an everyday grocery delivery channel; subscription justifies frequent use.",
        "Maximize DashPass value through frequent grocery + restaurant orders.",
    ),
    (
        lambda d: d.get("dashpass_status_active_oh", 0) > 0.3
                  and d.get("health_of_subscription_jtbd", 0) > 0.3
                  and d.get("high_fee_order_rate", 0) < 0,
        "Healthy DashPass Subscriber",
        "Active DashPass member where the subscription is paying off — health-of-subscription is well above baseline, fee burden is below baseline. Stable, profitable user for the marketplace.",
        "Use DashPass as a cost-saving wrapper around regular delivery orders.",
    ),
    (
        lambda d: d.get("dashpass_status_inactive_oh", 0) > 0.3
                  and d.get("category_mix_profile__restaurant", 0) > 0.2
                  and d.get("device_type_notif_ios_oh", 0) > 0.3,
        "iOS Restaurant Mainstream (no DashPass)",
        "Restaurant-heavy iOS user without DashPass. Pays full fees per order; somewhat fee-tolerant but receives more fee discounts than DashPass members. Mainstream engaged but unsubscribed.",
        "Order restaurant delivery occasionally without committing to a subscription.",
    ),
    (
        lambda d: d.get("health_of_subscription_jtbd", 0) < -0.2
                  and d.get("avg_delivery_fee_discount", 0) > 0.2
                  and d.get("device_type_notif_ios_oh", 0) > 0.3,
        "Discount-Driven Restaurant Regular",
        "Lukewarm on DashPass (low subscription health) but receives larger-than-average fee discounts; orders concentrate on restaurants via iOS. Engagement is fee-promotion sensitive — pricing changes will move this segment.",
        "Order restaurant delivery when the fee math works out — promotions and discounts matter.",
    ),
    (
        lambda d: d.get("device_footprint", 0) > 0.7
                  and (d.get("channel_usage_mix__catering", 0) > 0.3
                       or d.get("notification_reachability_profile_email,sm_oh", 0) > 0.3),
        "Multi-Device Catering / Power-Channel User",
        "Uses multiple devices and orders across non-mainstream channels (catering, multi-channel reachability). Power user pattern with broader engagement surface area than typical.",
        "Manage household / team / business orders across devices and channels.",
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

def llm_name(company_id: str, persona_values: dict, top_features: list[dict]) -> tuple[str, str, str] | None:
    """Ask Claude to name + describe a persona from its z-deltas vs. baseline.

    Returns None on any failure (no API key, import error, network error, parse
    error). The caller must fall back to `generic_fallback_name`.

    Company-agnostic: the LLM sees only the company_id, feature names, and
    persona values + baseline-relative z-scores. Works on any feature schema.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        feature_lines = "\n".join(
            f"- {f['group']}.{f['feature']}: "
            f"persona={persona_values.get(f['feature'],{}).get('value_fmt','?')}, "
            f"delta={f['z_delta']:+.2f}σ vs. baseline"
            for f in top_features[:20]
        )
        prompt = (
            f"You are naming a user persona discovered via unsupervised clustering "
            f"on a {company_id} user dataset.\n\n"
            f"Below is the persona's distinctive features (highest absolute z-delta "
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
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        import json as _json
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        obj = _json.loads(text)
        return obj["name"], obj["description"], obj["jtbd"]
    except Exception:
        return None
