"""Strict LLM-naming mode for production (real-customer) pack builds (#342).

The build silently falls back to generic ``Persona##`` names (raw-stat jargon
descriptions) when the LLM is unavailable. That is fine for research iteration
but a correctness landmine for a real-customer pack: downstream behavioral
simulation only recovers who-responds-more heterogeneity when the audience is
described in plain language (validated: enriched descriptions rank-rho +0.457
vs. ~0 for jargon fallbacks). ``require_llm`` makes naming mandatory — the build
fails loudly instead of silently shipping a degraded pack.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

import sidira_personas.pipeline as pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
import synthetic_company  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "synthetic.yaml"


def _cfg_and_df():
    return yaml.safe_load(CONFIG.read_text()), synthetic_company.build()


def test_require_llm_and_no_llm_are_mutually_exclusive():
    cfg, df = _cfg_and_df()
    with pytest.raises(ValueError, match="mutually exclusive"):
        pipeline.run_pipeline(df, cfg, run_id="x", no_llm=True, require_llm=True)


def test_require_llm_fails_fast_when_no_provider(monkeypatch):
    """No usable provider must abort at preflight, BEFORE the clustering run."""
    cfg, df = _cfg_and_df()
    monkeypatch.setattr(pipeline, "llm_provider_in_use", lambda provider=None: None)
    # If this raised only after clustering, the spy below would have been called.
    called = []
    monkeypatch.setattr(pipeline, "llm_name", lambda *a, **k: called.append(1))
    with pytest.raises(RuntimeError, match="no usable LLM naming provider"):
        pipeline.run_pipeline(df, cfg, run_id="x", require_llm=True)
    assert not called, "naming (hence clustering) ran despite preflight failure"


def test_require_llm_aborts_on_midbuild_naming_failure(monkeypatch):
    """Provider reachable at preflight but returns None mid-build → loud abort."""
    cfg, df = _cfg_and_df()
    monkeypatch.setattr(pipeline, "llm_provider_in_use", lambda provider=None: "groq:stub")
    monkeypatch.setattr(pipeline, "llm_name", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="LLM naming failed for"):
        pipeline.run_pipeline(df, cfg, run_id="x", require_llm=True)


def test_require_llm_succeeds_and_carries_llm_descriptions(monkeypatch):
    """When naming works, the pack builds and no persona carries a fallback marker."""
    cfg, df = _cfg_and_df()
    monkeypatch.setattr(pipeline, "llm_provider_in_use", lambda provider=None: "groq:stub")
    monkeypatch.setattr(
        pipeline, "llm_name",
        lambda *a, **k: ("Value Seekers", "Lower-spending shoppers who buy on discount.", "Save money"),
    )
    res = pipeline.run_pipeline(df, cfg, run_id="x", require_llm=True)
    for p in res.pack.personas:
        assert "LLM naming unavailable" not in p.description
        assert "not generated" not in p.jtbd


def test_default_still_falls_back_silently(monkeypatch):
    """Without require_llm, a None naming result silently uses the generic fallback
    (unchanged research-iteration behavior)."""
    cfg, df = _cfg_and_df()
    monkeypatch.setattr(pipeline, "llm_provider_in_use", lambda provider=None: "groq:stub")
    monkeypatch.setattr(pipeline, "llm_name", lambda *a, **k: None)
    res = pipeline.run_pipeline(df, cfg, run_id="x")  # require_llm defaults False
    discovered = [p for p in res.pack.personas if not p.id.startswith("FIXED")]
    assert any("LLM naming unavailable" in p.description for p in discovered)
