"""Library-level test for the extracted ``run_pipeline`` entry point (#268).

The CLI smoke test (``test_pipeline_smoke``) covers the thin wrapper; this drives
the pure function directly — an in-memory DataFrame in, a ``PipelineResult`` out,
no source read and no file writes — which is exactly how the simulator backend's
persona-build worker will call it.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
import synthetic_company  # noqa: E402

from sidira_personas import PipelineResult, run_pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "synthetic.yaml"


def _run() -> PipelineResult:
    cfg = yaml.safe_load(CONFIG.read_text())
    df = synthetic_company.build()
    return run_pipeline(df, cfg, run_id="test-run", no_llm=True)


def test_returns_in_memory_result():
    res = _run()
    assert isinstance(res, PipelineResult)
    assert isinstance(res.assignments, pd.DataFrame)
    assert res.pack.run_id == "test-run"
    assert res.pack.schema_version == "1.1"


def test_pack_is_asdict_serializable():
    """The simulator serializes the pack via dataclasses.asdict → JSON to R2."""
    res = _run()
    pack = asdict(res.pack)
    assert pack["run_id"] == "test-run"
    assert 3 <= pack["k_selected"] <= 6
    discovered = [p for p in pack["personas"] if not p["id"].startswith("FIXED")]
    assert len(discovered) == pack["k_selected"]


def test_one_prompt_per_persona():
    res = _run()
    assert set(res.prompts) == {p.id for p in res.pack.personas}
    assert all(text.strip() for text in res.prompts.values())


def test_assignments_carry_soft_posteriors():
    res = _run()
    asg = res.assignments
    assert "consumer_id" in asg.columns
    assert "confidence" in asg.columns
    post_cols = [c for c in asg.columns if c.startswith("post_P")]
    assert len(post_cols) == res.pack.k_selected
    assert (asg[post_cols].sum(axis=1) - 1.0).abs().max() < 1e-6


def test_log_callback_receives_progress():
    cfg = yaml.safe_load(CONFIG.read_text())
    df = synthetic_company.build()
    lines: list[str] = []
    run_pipeline(df, cfg, run_id="test-run", no_llm=True, log=lines.append)
    assert any("Layer C" in ln for ln in lines)


def test_baseline_crosstab_present_when_configured():
    """config sets baseline_segment_column: hv_segment → crosstab returned."""
    res = _run()
    assert res.baseline_crosstab is not None
    assert "TOTAL" in res.baseline_crosstab.index
