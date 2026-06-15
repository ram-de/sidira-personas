"""Provider-forcing for LLM naming (#268).

The simulator binding passes the wizard's chosen naming model through
``run_pipeline(provider=...)`` → ``llm_name(provider=...)``. A forced provider
must route only to that backend and never silently fall through to the other.
"""
from __future__ import annotations

import sidira_personas.compile_prompts as cp


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
