"""Pipeline wiring for the estimability manifest (Phase 2).

The manifest is an ADDITIVE, OFF-by-default artifact emitted beside the persona pack. These
tests assert the four wiring invariants: (1) off by default → no manifest, pack unchanged;
(2) enabled → a well-formed manifest AND a byte-identical pack (money path unaffected);
(3) a failing manifest build NEVER breaks the pack; (4) unhashable (dict/list) columns are
profiled defensively and never crash the builder.

Runs on the committed synthetic fixture with ``no_llm=True`` (no Snowflake, no API key).
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sidira_personas.estimability import build_manifest_from_frame
from sidira_personas.io import read_local
from sidira_personas.pipeline import run_pipeline

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "synthetic.yaml"
DATA = REPO / "fixtures" / "synthetic" / "users.parquet"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return read_local(str(DATA))


@pytest.fixture(scope="module")
def runs(df, cfg):
    """Run the pipeline once with estimability OFF and once ON (same run_id), for reuse."""
    off_cfg = copy.deepcopy(cfg)
    off_cfg.pop("estimability", None)
    on_cfg = copy.deepcopy(cfg)
    on_cfg.setdefault("estimability", {})["enabled"] = True
    off = run_pipeline(df, off_cfg, run_id="fixed-run", no_llm=True)
    on = run_pipeline(df, on_cfg, run_id="fixed-run", no_llm=True)
    return off, on


# --------------------------------------------------------------------------- #
# 1. off by default
# --------------------------------------------------------------------------- #
def test_off_when_no_estimability_block(df, cfg):
    base = copy.deepcopy(cfg)
    base.pop("estimability", None)
    res = run_pipeline(df, base, run_id="t-off", no_llm=True)
    assert res.estimability_manifest is None
    assert res.pack.personas  # the pack still builds normally


def test_off_when_enabled_false(df, cfg):
    # The committed synthetic.yaml ships an estimability block with enabled: false.
    base = copy.deepcopy(cfg)
    assert base.get("estimability", {}).get("enabled") is False
    res = run_pipeline(df, base, run_id="t-disabled", no_llm=True)
    assert res.estimability_manifest is None


# --------------------------------------------------------------------------- #
# 2. enabled → well-formed manifest, pack unchanged
# --------------------------------------------------------------------------- #
def test_enabled_produces_well_formed_manifest(runs):
    _off, on = runs
    m = on.estimability_manifest
    assert isinstance(m, dict)
    assert m.get("status") != "error", m
    for key in ("manifest_version", "columns", "candidate_effects", "truncated"):
        assert key in m, f"missing manifest key: {key}"
    # synthetic store has 0/1 *_flag levers and order/active-hinted numeric outcomes
    assert len(m["candidate_effects"]) >= 1, (
        f"expected >=1 candidate effect; got {len(m['candidate_effects'])}"
    )


def test_pack_identical_between_off_and_on(runs):
    """The manifest is additive: the pack (money path) must be byte-identical with it off vs on."""
    off, on = runs
    p_off, p_on = asdict(off.pack), asdict(on.pack)
    # created_at is utcnow() and differs between the two calls; run_id is pinned identical.
    p_off.pop("created_at")
    p_on.pop("created_at")
    assert p_off == p_on
    pd.testing.assert_frame_equal(off.assignments, on.assignments)


# --------------------------------------------------------------------------- #
# 3. never breaks the build
# --------------------------------------------------------------------------- #
def test_manifest_failure_never_breaks_pack(df, cfg, monkeypatch):
    import sidira_personas.estimability as est_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("kaboom")

    # The pipeline does `from .estimability import build_manifest_from_frame` at call time,
    # so patching the package attribute is picked up.
    monkeypatch.setattr(est_mod, "build_manifest_from_frame", _boom)

    on_cfg = copy.deepcopy(cfg)
    on_cfg.setdefault("estimability", {})["enabled"] = True
    res = run_pipeline(df, on_cfg, run_id="t-boom", no_llm=True)

    assert res.estimability_manifest["status"] == "error"
    assert "RuntimeError" in res.estimability_manifest["reason"]
    assert "kaboom" in res.estimability_manifest["reason"]
    # the pack still built normally despite the manifest failure
    assert res.pack.personas
    assert len(res.assignments) == len(df[df["cx_lifestage_detailed_status"] == "Active"])


# --------------------------------------------------------------------------- #
# 4. unhashable column safe (dict cells)
# --------------------------------------------------------------------------- #
def test_unhashable_column_does_not_crash_builder():
    rng = np.random.default_rng(5)
    n = 800
    store = pd.DataFrame({
        # dict-valued cells -> unhashable -> would crash a naive nunique()/unique()
        "json_counts": [{"weekday": int(i % 4), "weekend": int(i % 3)} for i in range(n)],
        "deal_flag": (rng.random(n) < 0.5).astype(int),        # 0/1 lever -> treatment
        "conf": rng.standard_normal(n),                        # continuous covariate
        "retention": (rng.random(n) < 0.5).astype(float),      # order/retention-hinted outcome
    })

    m = build_manifest_from_frame(store, "acme", "default", "v_1", "fixture://unhash")

    roles = {c.name: c.role for c in m.columns}
    dcount = {c.name: c.distinct_count for c in m.columns}
    # the dict column is a covariate, never a treatment/outcome, with an unresolved distinct_count
    assert roles["json_counts"] == "covariate"
    assert dcount["json_counts"] is None
    assert roles["deal_flag"] == "treatment_candidate"
    assert roles["retention"] == "outcome_candidate"
    # well-formed: the (deal_flag -> retention) effect exists (supportable or not)
    assert len(m.candidate_effects) >= 1
