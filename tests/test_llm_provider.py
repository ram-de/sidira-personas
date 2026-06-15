"""Provider-forcing for LLM naming (#268).

The simulator binding passes the wizard's chosen naming model through
``run_pipeline(provider=...)`` → ``llm_name(provider=...)``. A forced provider
must route only to that backend and never silently fall through to the other.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

import sidira_personas.compile_prompts as cp
import sidira_personas.pipeline as pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
import synthetic_company  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "synthetic.yaml"


def _stub_providers(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cp, "_llm_name_groq", lambda p: (calls.append("groq") or ("G", "g", "g")))
    monkeypatch.setattr(
        cp, "_llm_name_anthropic", lambda p: (calls.append("anthropic") or ("A", "a", "a"))
    )
    return calls


def test_forced_groq_routes_only_to_groq(monkeypatch):
    calls = _stub_providers(monkeypatch)
    out = cp.llm_name("co", {}, [], provider="groq")
    assert out[0] == "G"
    assert calls == ["groq"]


def test_forced_anthropic_routes_only_to_anthropic(monkeypatch):
    calls = _stub_providers(monkeypatch)
    out = cp.llm_name("co", {}, [], provider="anthropic")
    assert out[0] == "A"
    assert calls == ["anthropic"]


def test_forced_groq_failure_does_not_fall_through(monkeypatch):
    """Forced groq returning None must NOT silently use anthropic."""
    calls: list[str] = []
    monkeypatch.setattr(cp, "_llm_name_groq", lambda p: (calls.append("groq") or None))
    monkeypatch.setattr(
        cp, "_llm_name_anthropic", lambda p: (calls.append("anthropic") or ("A", "a", "a"))
    )
    out = cp.llm_name("co", {}, [], provider="groq")
    assert out is None
    assert calls == ["groq"]  # anthropic never consulted


def test_forced_anthropic_failure_does_not_fall_through(monkeypatch):
    """Forced anthropic returning None must NOT silently use groq."""
    calls: list[str] = []
    monkeypatch.setattr(
        cp, "_llm_name_groq", lambda p: (calls.append("groq") or ("G", "g", "g"))
    )
    monkeypatch.setattr(cp, "_llm_name_anthropic", lambda p: (calls.append("anthropic") or None))
    out = cp.llm_name("co", {}, [], provider="anthropic")
    assert out is None
    assert calls == ["anthropic"]  # groq never consulted


def test_unknown_provider_raises_not_auto_selects(monkeypatch):
    """An unrecognized forced provider is a misconfiguration, not auto-select."""
    calls = _stub_providers(monkeypatch)
    with pytest.raises(ValueError, match="unknown LLM provider"):
        cp.llm_name("co", {}, [], provider="Anthropic")  # wrong casing
    with pytest.raises(ValueError, match="unknown LLM provider"):
        cp.llm_name("co", {}, [], provider="claude-sonnet-4-6")  # model id, not provider
    assert calls == []  # never reached a backend
    with pytest.raises(ValueError, match="unknown LLM provider"):
        cp.llm_provider_in_use(provider="gpt")


def test_default_prefers_groq_then_falls_back(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(cp, "_llm_name_groq", lambda p: (calls.append("groq") or None))
    monkeypatch.setattr(
        cp, "_llm_name_anthropic", lambda p: (calls.append("anthropic") or ("A", "a", "a"))
    )
    out = cp.llm_name("co", {}, [], provider=None)
    assert out[0] == "A"
    assert calls == ["groq", "anthropic"]  # tried groq first, fell through


def test_provider_in_use_honors_forced_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Forced anthropic with no anthropic key → None (no silent fallback to groq).
    assert cp.llm_provider_in_use(provider="anthropic") is None
    # Forced groq with a groq key present → groq label.
    assert cp.llm_provider_in_use(provider="groq").startswith("groq:")
    # Unforced → first available (groq).
    assert cp.llm_provider_in_use().startswith("groq:")


def test_run_pipeline_threads_provider_to_llm_name(monkeypatch):
    """End-to-end: run_pipeline(provider=...) reaches llm_name in every naming path."""
    seen: list[str | None] = []

    def _spy(company_id, persona_values, top_features, provider=None):
        seen.append(provider)
        return ("Persona X", "desc", "jtbd")

    # Patch the name bound inside pipeline's module namespace (where it's called).
    monkeypatch.setattr(pipeline, "llm_name", _spy)
    cfg = yaml.safe_load(CONFIG.read_text())
    df = synthetic_company.build()
    pipeline.run_pipeline(df, cfg, run_id="prov-test", no_llm=False, provider="anthropic")
    assert seen, "llm_name was never called"
    assert all(p == "anthropic" for p in seen)
