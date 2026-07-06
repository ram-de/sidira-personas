"""Config-gated estimability emit + build-time auto negative-control (M1 iter-2, #572).

Drives the pure `run_pipeline` entry point over the synthetic fixture (no Snowflake, no API
key) and asserts:

  1. estimability OFF (the default) -> `result.estimability_manifest is None`, and the money
     path (pack / prompts / assignments) is unchanged — the emit is inert when disabled.
  2. estimability ON -> a strict-JSON manifest dict whose candidate effects carry a REAL
     `guard_trips` bool (not None), proving the auto negative-control map made the guard run.
  3. `default_neg_control_map` maps each outcome candidate to OTHER outcome candidates only,
     never itself, capped at `max_controls`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
import synthetic_company  # noqa: E402

from sidira_personas.estimability import (  # noqa: E402
    default_neg_control_map,
)
from sidira_personas.estimability.manifest_builder import _classify  # noqa: E402
from sidira_personas.pipeline import run_pipeline  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "synthetic.yaml"


def _base_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text())


@pytest.fixture(scope="module")
def df():
    return synthetic_company.build()


@pytest.fixture(scope="module")
def result_off(df):
    cfg = _base_cfg()  # synthetic.yaml ships estimability.enabled: false
    return run_pipeline(df, cfg, run_id="synth-off", no_llm=True)


@pytest.fixture(scope="module")
def result_on(df):
    cfg = _base_cfg()
    cfg["estimability"] = {"enabled": True, "max_treatments": 8, "max_outcomes": 8}
    return run_pipeline(df, cfg, run_id="synth-on", no_llm=True)


# --- 1. default OFF is inert -------------------------------------------------

def test_manifest_is_none_when_disabled(result_off):
    assert result_off.estimability_manifest is None


def test_money_path_unchanged_when_disabled(result_off):
    """The pack / prompts / assignments the pipeline has always produced are intact — the
    estimability knob does not perturb the money path when it is off."""
    pack = result_off.pack
    assert pack.schema_version == "1.1"
    assert 3 <= pack.k_selected <= 6
    # one compiled prompt per persona, and assignments carry the confidence column
    assert set(result_off.prompts) == {p.id for p in pack.personas}
    assert "confidence" in result_off.assignments.columns


# --- 2. enabled -> guard actually runs ---------------------------------------

def test_manifest_emitted_with_candidate_effects(result_on):
    m = result_on.estimability_manifest
    assert isinstance(m, dict)
    assert m.get("status") != "error", f"estimability build errored: {m}"
    assert m["candidate_effects"], "expected at least one candidate effect"


def test_at_least_one_guard_trip_is_a_real_bool(result_on):
    """The auto negative-control map must make the guard RUN — so at least one candidate
    carries guard_trips as a True/False bool, not None (which would be CANT_VALIDATE)."""
    effects = result_on.estimability_manifest["candidate_effects"]
    real_bools = [e for e in effects if isinstance(e["guard_trips"], bool)]
    assert real_bools, "no candidate had guard_trips as a real bool — guard never ran"


def test_manifest_is_strict_json_safe(result_on):
    # No NaN/inf leaks — the downstream Pydantic consumer + R2 sidecar need strict JSON.
    json.dumps(result_on.estimability_manifest, allow_nan=False)


def test_enabled_does_not_disturb_the_pack(result_on):
    """Enabling estimability only ADDS the manifest; the pack itself is still a normal v1.1 pack."""
    assert result_on.estimability_manifest is not None
    assert result_on.pack.schema_version == "1.1"
    assert set(result_on.prompts) == {p.id for p in result_on.pack.personas}


# --- 3. the auto negative-control heuristic ----------------------------------

def test_default_neg_control_map_maps_outcomes_to_other_outcomes(df):
    # Build column profiles the same way build_manifest_from_frame does (via _classify roles).
    prof = [{
        "name": c,
        "dtype": ("bool" if str(df[c].dtype) == "bool"
                  else ("number" if df[c].dtype.kind in "ifu" else "string")),
        "distinct_count": int(df[c].nunique()) if df[c].dtype.kind in "ifub" else None,
        "null_fraction": float(df[c].isna().mean()),
    } for c in df.columns]

    outcome_names = {c.name for c in _classify(prof)
                     if c.role == "outcome_candidate" and c.has_variation
                     and c.dtype in ("number", "bool")}
    assert outcome_names, "fixture should expose numeric outcome candidates"

    nc_map = default_neg_control_map(prof, max_controls=5)
    assert nc_map, "expected a non-empty auto negative-control map"
    for y, controls in nc_map.items():
        assert y in outcome_names                       # keys are outcome candidates
        assert len(controls) <= 5                        # capped at max_controls
        assert y not in controls                         # never itself
        assert set(controls) <= outcome_names - {y}      # only OTHER outcome candidates


def test_default_neg_control_map_respects_cap(df):
    prof = [{
        "name": c,
        "dtype": ("bool" if str(df[c].dtype) == "bool"
                  else ("number" if df[c].dtype.kind in "ifu" else "string")),
        "distinct_count": None,
        "null_fraction": 0.0,
    } for c in df.columns]
    nc_map = default_neg_control_map(prof, max_controls=1)
    assert all(len(v) <= 1 for v in nc_map.values())
