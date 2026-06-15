"""End-to-end pipeline smoke test on the synthetic fixture (#12).

Runs the FULL pipeline (Adapter → Encoder → Cluster → Calibrate → Compile) via the
real CLI against a freshly generated synthetic userbase — no Snowflake, no API key,
no real data — and asserts the v1.1 + #7 contract on the emitted Persona Pack.

This is the integration layer the unit tests (test_sampling / test_stability /
test_schema) don't cover: config parsing, per-group adapters, PCA encode, GMM
k-selection, prefilter → FIXED_* personas, pack compile, and parquet emission.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))
import synthetic_company  # noqa: E402

from sidira_personas.cli import cli  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "config" / "synthetic.yaml"


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Generate synthetic data, run the CLI end-to-end with --no-llm, return the output dir."""
    tmp = tmp_path_factory.mktemp("synth_pipeline")
    data_path = tmp / "users.parquet"
    synthetic_company.build().to_parquet(data_path, index=False)

    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["source"]["path"] = str(data_path)
    cfg_path = tmp / "synthetic.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    out_dir = tmp / "out"
    result = CliRunner().invoke(
        cli, ["run", "--config", str(cfg_path), "--no-llm", "--out", str(out_dir)]
    )
    assert result.exit_code == 0, f"pipeline failed:\n{result.output}\n{result.exception}"

    runs = list(out_dir.glob("synth-*"))
    assert len(runs) == 1, f"expected one run dir, got {runs}"
    return runs[0]


@pytest.fixture(scope="module")
def pack(pipeline_run):
    return json.loads((pipeline_run / "persona_pack.json").read_text())


# --- the committed fixture stays in sync with the generator ---

def test_committed_fixture_matches_generator():
    """The checked-in parquet must be exactly what `build()` produces (guards drift)."""
    committed = REPO / "fixtures" / "synthetic" / "users.parquet"
    assert committed.exists(), "run `python fixtures/synthetic_company.py` to materialize it"
    pd.testing.assert_frame_equal(synthetic_company.build(), pd.read_parquet(committed))


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(synthetic_company.build(seed=3), synthetic_company.build(seed=3))


# --- pack-level v1.1 + #7 contract ---

def test_schema_version_is_1_1(pack):
    assert pack["schema_version"] == "1.1"


def test_clustering_found_a_real_split(pack):
    # 4 latent segments → BIC may merge/over-split, but it must be a genuine multi-cluster result.
    assert 3 <= pack["k_selected"] <= 6
    discovered = [p for p in pack["personas"] if not p["id"].startswith("FIXED")]
    assert len(discovered) == pack["k_selected"]


def test_notes_carry_stability_and_holdout(pack):
    notes = pack["notes"]
    assert "stability_method" in notes
    assert set(notes["per_cluster_jaccard"]) == {p["id"] for p in pack["personas"]
                                                 if not p["id"].startswith("FIXED")}
    holdout = notes["holdout"]
    for key in ("train_avg_loglik", "test_avg_loglik", "loglik_gap",
                "proportion_tv_distance", "n_train", "n_test"):
        assert key in holdout
    assert 0.0 <= holdout["proportion_tv_distance"] <= 1.0


def test_at_least_one_cluster_is_stable(pack):
    """A well-separated segment should clear Hennig's "valid" band somewhere in the pack."""
    jac = pack["notes"]["per_cluster_jaccard"].values()
    assert max(jac) >= 0.75


# --- discovered-persona fields (the simulator step-2 contract) ---

def test_discovered_personas_carry_v11_fields(pack):
    discovered = [p for p in pack["personas"] if not p["id"].startswith("FIXED")]
    assert discovered
    for p in discovered:
        assert p["centroid_full"], f"{p['id']} has empty centroid_full"
        assert p["feature_scale"], f"{p['id']} has empty feature_scale"
        assert all(v > 0 for v in p["feature_scale"].values()), f"{p['id']} has non-positive scale"
        assert set(p["centroid_full"]) == set(p["feature_scale"])
        assert len(p["sample_rows"]) == 32  # config: sampling.n_synthetic_rows
        assert set(p["sample_rows"][0]) == set(p["centroid_full"])
        assert abs(sum(p["posterior_mean"].values()) - 1.0) < 1e-6
        assert p["stability"] is not None
        assert p["stability_verdict"] in {"highly stable", "valid", "unstable", "dissolved"}
        assert 0.0 <= p["posterior_entropy"] <= 1.0


def test_sample_rows_stay_within_population_bounds(pack, pipeline_run):
    """Synthetic draws are clipped in-distribution — ratios/flags never escape [0,1]."""
    ratio_like = ("convenience_ratio", "high_fee_rate", "unsubscribe_rate",
                  "discount_share", "open_within_24h", "click_within_24h")
    for p in pack["personas"]:
        for row in p["sample_rows"]:
            for feat in ratio_like:
                if feat in row:
                    assert -0.01 <= row[feat] <= 1.01, f"{p['id']} {feat}={row[feat]}"


# --- FIXED_* personas (prefilter → hard-defined segments) ---

def test_fixed_personas_exist_and_are_null_on_gmm_fields(pack):
    fixed = [p for p in pack["personas"] if p["id"].startswith("FIXED")]
    assert fixed, "prefilter should have produced FIXED_* personas from non-Active rows"
    for p in fixed:
        assert p["stability"] is None
        assert p["stability_verdict"] == "n/a (hard-defined)"
        assert p["posterior_mean"] is None
        assert p["posterior_entropy"] is None
        # but they still carry the sampleable v1.1 structure, computed within their own subset
        assert p["centroid_full"]
        assert len(p["sample_rows"]) == 32


# --- on-disk artifacts ---

def test_assignments_carry_soft_posteriors(pipeline_run, pack):
    asg = pd.read_parquet(pipeline_run / "assignments.parquet")
    assert "consumer_id" in asg.columns
    assert "confidence" in asg.columns
    post_cols = [c for c in asg.columns if c.startswith("post_P")]
    assert len(post_cols) == pack["k_selected"]
    # soft posteriors over the GMM components sum to ~1 per row
    assert (asg[post_cols].sum(axis=1) - 1.0).abs().max() < 1e-6


def test_a_prompt_is_written_per_persona(pipeline_run, pack):
    prompts = list((pipeline_run / "prompts").glob("*.md"))
    assert len(prompts) == len(pack["personas"])
