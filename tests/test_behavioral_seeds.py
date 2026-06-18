"""Per-persona ``behavioral_seeds`` generation (issue #342).

The producer's LLM naming step now emits, in the SAME call that names/describes a
persona, a ``behavioral_seeds`` object — the five identity-draw pools the
simulator's ``identity_seed`` service draws from. These tests cover:

  * the JSON parse/coerce helper (well-formed, partial, garbage, absent);
  * the pipeline wiring — every persona (discovered + FIXED) carries
    ``behavioral_seeds`` when the LLM returns them, and ``None`` under
    ``--no-llm`` (so the simulator falls back, exactly as a v1.0 pack does).

No network: the LLM is stubbed at ``pipeline.llm_name``.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import yaml

import sidira_personas.compile_prompts as cp
import sidira_personas.pipeline as pipeline
from sidira_personas.schema import BehavioralSeeds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
import synthetic_company  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "synthetic.yaml"

_SEED_KEYS = (
    "age_bands",
    "locale_bands",
    "tenure_bands",
    "current_state_pool",
    "recent_gripe_pool",
)


def _cfg_and_df():
    return yaml.safe_load(CONFIG.read_text()), synthetic_company.build()


def _well_formed_seeds_dict() -> dict:
    return {
        "age_bands": ["25-34", "35-44"],
        "locale_bands": ["urban_us", "suburban_us", "rural_us"],
        "tenure_bands": ["established (1-3y)", "veteran (3y+)"],
        "current_state_pool": [
            "weekday evening, in a hurry",
            "weekend, browsing options",
            "post-work, decision-fatigued",
            "lunch break, time-boxed",
            "late night, comfort mood",
        ],
        "recent_gripe_pool": [
            "fees added up at checkout",
            "delivery later than estimated",
            "promo code didn't apply",
            "order arrived cold",
        ],
    }


# --- helper: JSON parse / coerce --------------------------------------------


def test_parse_naming_json_extracts_behavioral_seeds():
    import json

    payload = json.dumps(
        {
            "name": "Power Orderer",
            "description": "Orders often, app-native.",
            "jtbd": "Dinner handled fast.",
            "behavioral_seeds": _well_formed_seeds_dict(),
        }
    )
    name, desc, jtbd, seeds = cp._parse_naming_json(payload)
    assert name == "Power Orderer"
    assert isinstance(seeds, dict)
    assert seeds["age_bands"] == ["25-34", "35-44"]
    # every list survives, most-common-first order preserved
    for k in _SEED_KEYS:
        assert isinstance(seeds[k], list) and seeds[k]


def test_parse_naming_json_seeds_absent_is_none():
    import json

    payload = json.dumps({"name": "N", "description": "d", "jtbd": "j"})
    *_head, seeds = cp._parse_naming_json(payload)
    assert seeds is None


def test_coerce_behavioral_seeds_filters_garbage():
    raw = {
        "age_bands": ["25-34", "", "  ", 7, None, "35-44"],
        "locale_bands": "not a list",
        "tenure_bands": [],
        "current_state_pool": ["in a hurry"],
        "recent_gripe_pool": ["cold food"],
    }
    out = cp._coerce_behavioral_seeds(raw)
    assert out["age_bands"] == ["25-34", "35-44"]  # blanks + non-strings dropped
    assert out["locale_bands"] == []  # non-list → empty
    assert out["current_state_pool"] == ["in a hurry"]


def test_coerce_behavioral_seeds_all_empty_collapses_to_none():
    """A hollow object (every list empty) must collapse to None so it never
    masks the simulator's fallback with an empty-but-present schema."""
    assert cp._coerce_behavioral_seeds({k: [] for k in _SEED_KEYS}) is None
    assert cp._coerce_behavioral_seeds(None) is None
    assert cp._coerce_behavioral_seeds("garbage") is None


# --- pipeline wiring: LLM on -------------------------------------------------


def test_pipeline_emits_behavioral_seeds_when_llm_on(monkeypatch):
    """With a (stubbed) LLM returning seeds, EVERY persona — discovered AND the
    prefilter FIXED_* one — carries a well-formed BehavioralSeeds."""
    cfg, df = _cfg_and_df()
    monkeypatch.setattr(pipeline, "llm_provider_in_use", lambda provider=None: "groq:stub")
    monkeypatch.setattr(
        pipeline,
        "llm_name",
        lambda *a, **k: ("Seg", "A described audience.", "Get it done", _well_formed_seeds_dict()),
    )
    res = pipeline.run_pipeline(df, cfg, run_id="seeds-on", no_llm=False)

    assert any(p.id.startswith("FIXED") for p in res.pack.personas), "fixture should yield a FIXED persona"
    for p in res.pack.personas:
        assert isinstance(p.behavioral_seeds, BehavioralSeeds), p.id
        bs = p.behavioral_seeds
        assert bs.age_bands == ["25-34", "35-44"]
        assert 4 <= len(bs.current_state_pool) <= 8
        assert 3 <= len(bs.recent_gripe_pool) <= 6
        # serializes cleanly for the wire (dataclasses.asdict → JSON → R2)
        assert asdict(p)["behavioral_seeds"]["tenure_bands"]


def test_pipeline_no_llm_leaves_behavioral_seeds_none():
    """--no-llm → no per-persona pools; every persona's behavioral_seeds is None
    so the simulator falls back to its generic default pool (unchanged behavior)."""
    cfg, df = _cfg_and_df()
    res = pipeline.run_pipeline(df, cfg, run_id="seeds-off", no_llm=True)
    assert res.pack.personas
    for p in res.pack.personas:
        assert p.behavioral_seeds is None, p.id


def test_pipeline_tolerates_legacy_3tuple_naming(monkeypatch):
    """A naming path that returns the legacy 3-tuple (no seeds) must not break the
    build — behavioral_seeds is simply None. Guards the back-compat unpack."""
    cfg, df = _cfg_and_df()
    monkeypatch.setattr(pipeline, "llm_provider_in_use", lambda provider=None: "groq:stub")
    monkeypatch.setattr(pipeline, "llm_name", lambda *a, **k: ("Seg", "desc", "jtbd"))
    res = pipeline.run_pipeline(df, cfg, run_id="legacy", no_llm=False)
    for p in res.pack.personas:
        assert p.behavioral_seeds is None, p.id


def test_pipeline_seeds_none_when_llm_returns_no_seeds(monkeypatch):
    """LLM names the persona but omits/garbles behavioral_seeds (4-tuple with
    None tail) → behavioral_seeds None, simulator falls back."""
    cfg, df = _cfg_and_df()
    monkeypatch.setattr(pipeline, "llm_provider_in_use", lambda provider=None: "groq:stub")
    monkeypatch.setattr(pipeline, "llm_name", lambda *a, **k: ("Seg", "desc", "jtbd", None))
    res = pipeline.run_pipeline(df, cfg, run_id="noseeds", no_llm=False)
    for p in res.pack.personas:
        assert p.behavioral_seeds is None, p.id
