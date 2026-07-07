"""Config-driven outcome/lever discovery (M1 iter-4, #572).

The manifest builder discovers outcomes via a hardcoded commerce keyword list
(`OUTCOME_HINTS`). On a non-commerce vertical (streaming: `completion_rate`,
`autoplay_enabled`, ...) almost nothing matches and the grid starves. These tests prove the
ADDITIVE, backward-compatible config path: a customer declares real KPIs (`outcomes`), extra
hint tokens (`extra_outcome_hints`), and levers (`treatments`), and non-conforming
declarations are RECORDED on `truncated`, never silently dropped. Defaults (all None) stay
byte-identical to the hint-only behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sidira_personas.estimability as est_pkg
from sidira_personas.estimability import build_manifest_from_frame
from sidira_personas.pipeline import run_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
import synthetic_company  # noqa: E402


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _pairs(m) -> set[tuple[str, str]]:
    """The (treatment, outcome) candidate grid the manifest actually fit."""
    return {(c.treatment, c.outcome) for c in m.candidate_effects}


def _streaming_frame(n: int = 2000) -> pd.DataFrame:
    """Non-commerce (streaming) fixture: NO column name matches a commerce hint.

    Two binary levers (`autoplay_enabled`, `hd_enabled`), a float KPI `completion_rate`, and
    two numeric covariates. Balanced-ish propensity keeps both arms well over the estimator's
    min-per-arm and overlap floors, so a declared outcome yields a supportable candidate.
    """
    rng = np.random.default_rng(4)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    autoplay = (rng.random(n) < _sig(0.4 * x1)).astype(int)
    hd = (rng.random(n) < 0.5).astype(int)
    completion = _sig(0.3 * x1 - 0.2 * x2 + 0.4 * autoplay) + 0.01 * rng.standard_normal(n)
    return pd.DataFrame({
        "weekly_viewing_hours": x1,          # covariate (non-hint numeric)
        "device_age_months": x2,             # covariate
        "autoplay_enabled": autoplay,        # binary lever
        "hd_enabled": hd,                    # binary lever
        "completion_rate": completion,       # float KPI, matches NO hint
    })


# --- 1. declared outcome recognized; not an outcome by default ---------------------------

def test_non_hint_outcome_absent_by_default_but_declared_recognized():
    df = _streaming_frame()
    m_default = build_manifest_from_frame(df, "c", "s", "v", "fs")
    # completion_rate matches no commerce hint -> no candidate names it as an outcome.
    assert all(c.outcome != "completion_rate" for c in m_default.candidate_effects)

    m_cfg = build_manifest_from_frame(df, "c", "s", "v", "fs", outcomes=["completion_rate"])
    got = {c.outcome for c in m_cfg.candidate_effects}
    assert "completion_rate" in got
    # the declared outcome yields at least one SUPPORTABLE empirical candidate.
    assert any(c.outcome == "completion_rate" and c.supportable for c in m_cfg.candidate_effects)


# --- 2. extra_outcome_hints token also recognizes it -------------------------------------

def test_extra_outcome_hint_token_recognizes_column():
    df = _streaming_frame()
    m = build_manifest_from_frame(df, "c", "s", "v", "fs", extra_outcome_hints=["completion"])
    assert any(c.outcome == "completion_rate" for c in m.candidate_effects)


# --- 3. declared treatments restrict the lever set; non-binary/absent -> truncated -------

def test_declared_treatments_restrict_and_record_non_binary():
    df = _streaming_frame()
    # Two binary levers exist; declaring only one restricts the promoted levers to it.
    m = build_manifest_from_frame(
        df, "c", "s", "v", "fs",
        outcomes=["completion_rate"],
        treatments=["autoplay_enabled", "device_age_months"],  # 2nd is a non-binary covariate
    )
    levers = {c.treatment for c in m.candidate_effects}
    assert levers == {"autoplay_enabled"}          # hd_enabled excluded (not declared)
    assert "hd_enabled" not in levers
    # the declared-but-non-binary lever is recorded honestly, not silently dropped.
    assert m.truncated.get("treatments_declared_not_binary") == ["device_age_months"]


def test_declared_absent_treatment_recorded():
    df = _streaming_frame()
    m = build_manifest_from_frame(
        df, "c", "s", "v", "fs",
        outcomes=["completion_rate"],
        treatments=["autoplay_enabled", "nonexistent_lever"],
    )
    # The FIT grid still contains only the real lever; the declared-but-absent lever now ALSO
    # gets an explicit ungroundable stub row (#564) — LLM route with a reason, never silent.
    fitted = {c.treatment for c in m.candidate_effects if c.n_treat or c.n_control}
    assert fitted == {"autoplay_enabled"}
    stub = {c.treatment: c for c in m.candidate_effects}["nonexistent_lever"]
    assert stub.supportable is False
    assert stub.grounding_complete is False
    assert "no_treatment_column" in stub.grounding_gaps
    assert m.truncated.get("treatments_declared_not_binary") == ["nonexistent_lever"]


# --- declared outcome present-but-non-numeric / absent are recorded ----------------------

def test_declared_outcome_non_numeric_and_absent_recorded():
    df = _streaming_frame()
    df["plan_tier"] = "gold"        # present but non-numeric -> not empirically estimable
    m = build_manifest_from_frame(
        df, "c", "s", "v", "fs",
        outcomes=["plan_tier", "ghost_metric"],   # 2nd absent from the frame entirely
    )
    assert m.truncated.get("outcomes_declared_not_numeric") == ["plan_tier"]
    assert m.truncated.get("outcomes_declared_not_found") == ["ghost_metric"]


def test_declared_outcome_non_varying_recorded_separately():
    """A present, numeric, but CONSTANT declared outcome is recorded under its own reason key
    (`non_varying`), NOT mislabeled as non-numeric — accurate operator-facing accounting."""
    df = _streaming_frame()
    df["flat_metric"] = 0.5          # present, numeric, but constant -> cannot be an outcome
    m = build_manifest_from_frame(df, "c", "s", "v", "fs", outcomes=["flat_metric"])
    assert m.truncated.get("outcomes_declared_non_varying") == ["flat_metric"]
    # and it is NOT misfiled under the non-numeric key.
    assert "flat_metric" not in m.truncated.get("outcomes_declared_not_numeric", [])


# --- YAML-scalar robustness: a bare `outcomes: completion_rate` must not become a char set ---

def test_declared_outcome_bare_string_scalar_coerced():
    df = _streaming_frame()
    # A single string (as a bare YAML scalar would arrive) is treated as ONE column name,
    # not set("completion_rate") (a set of characters that would silently match nothing).
    m = build_manifest_from_frame(df, "c", "s", "v", "fs", outcomes="completion_rate")
    assert any(c.outcome == "completion_rate" and c.supportable for c in m.candidate_effects)
    assert m.truncated == {}          # no spurious not_found from stray single characters


def test_declared_treatment_bare_string_scalar_coerced():
    df = _streaming_frame()
    m = build_manifest_from_frame(
        df, "c", "s", "v", "fs", outcomes=["completion_rate"], treatments="autoplay_enabled")
    assert {c.treatment for c in m.candidate_effects} == {"autoplay_enabled"}


# --- 4. backward-compat: new params default-None -> identical candidate set --------------

def _commerce_frame(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    x1 = rng.standard_normal(n)
    promo = (rng.random(n) < _sig(0.4 * x1)).astype(int)   # binary lever
    revenue = _sig(0.3 * x1 + 0.5 * promo) + 0.01 * rng.standard_normal(n)  # hint-named KPI
    return pd.DataFrame({
        "tenure_months": x1,        # covariate
        "promo_shown": promo,       # binary lever
        "revenue": revenue,         # matches the "revenue" commerce hint
    })


def test_backward_compat_defaults_unchanged_on_hint_fixture():
    df = _commerce_frame()
    m_bare = build_manifest_from_frame(df, "c", "s", "v", "fs")
    # hint-based discovery finds revenue as the outcome with promo_shown as the lever.
    assert _pairs(m_bare) == {("promo_shown", "revenue")}
    # explicitly passing the new params as None must be byte-identical to omitting them.
    m_none = build_manifest_from_frame(
        df, "c", "s", "v", "fs",
        outcomes=None, extra_outcome_hints=None, treatments=None,
    )
    assert _pairs(m_none) == _pairs(m_bare)
    assert m_none.truncated == m_bare.truncated == {}


# --- 5. pipeline-level: run_pipeline threads estimability.* into the builder --------------

def test_run_pipeline_threads_config_keys_to_builder(monkeypatch):
    captured: dict = {}

    def _spy(df, **kwargs):  # noqa: ARG001
        captured.update(kwargs)
        # Return a tiny valid manifest so run_pipeline's asdict/logging path is exercised.
        return build_manifest_from_frame(df, kwargs["customer_id"], kwargs["setup_id"],
                                         kwargs["pack_version"], kwargs["feature_store"])

    # run_pipeline does `from .estimability import build_manifest_from_frame` at call time, so
    # patching the attribute on the estimability package is what the pipeline picks up.
    monkeypatch.setattr(est_pkg, "build_manifest_from_frame", _spy)

    df = synthetic_company.build()
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "synthetic.yaml").read_text())
    cfg["estimability"] = {
        "enabled": True,
        "outcomes": ["convenience_ratio"],
        "extra_outcome_hints": ["convenience"],
        "treatments": ["deal_seeker_flag"],
    }
    run_pipeline(df, cfg, run_id="cfg-thread", no_llm=True)
    assert captured.get("outcomes") == ["convenience_ratio"]
    assert captured.get("extra_outcome_hints") == ["convenience"]
    assert captured.get("treatments") == ["deal_seeker_flag"]
