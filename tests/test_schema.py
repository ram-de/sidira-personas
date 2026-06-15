"""Round-trip + back-compat tests for the v1.1 Persona Pack schema."""
from __future__ import annotations

import json

from sidira_personas.schema import (
    PERSONA_PACK_SCHEMA_VERSION,
    FeatureDelta,
    Persona,
    PersonaPack,
    PersonaWeights,
)


def _persona(**overrides) -> Persona:
    base = dict(
        id="P01",
        name="Test",
        description="d",
        jtbd="j",
        size=10,
        weights=PersonaWeights(by_headcount=0.5),
        stability=0.8,
        top_features=[FeatureDelta("f0", "g", 1.0, 0.0, 1.0, 1.0)],
        centroid={"f0": 1.0},
    )
    base.update(overrides)
    return Persona(**base)


def test_schema_version_is_1_1():
    assert PERSONA_PACK_SCHEMA_VERSION == "1.1"


def test_v1_0_persona_still_constructs_via_defaults():
    """A v1.0-style Persona (new fields omitted) constructs and defaults are empty/None."""
    p = _persona()
    assert p.centroid_full == {}
    assert p.feature_scale == {}
    assert p.sample_rows == []
    assert p.posterior_mean is None


def test_pack_round_trips_through_json(tmp_path):
    p = _persona(
        centroid_full={"f0": 1.0, "f1": 2.0},
        feature_scale={"f0": 0.5, "f1": 0.25},
        sample_rows=[{"f0": 1.1, "f1": 2.2}, {"f0": 0.9, "f1": 1.8}],
        posterior_mean={"P01": 0.7, "P02": 0.3},
    )
    pack = PersonaPack(
        schema_version=PERSONA_PACK_SCHEMA_VERSION,
        company_id="c",
        run_id="r",
        created_at="2026-01-01",
        n_users=10,
        feature_groups=["g"],
        personas=[p],
        cluster_method="gmm",
        k_selected=1,
        encoder_method="pca",
        embedding_dim_per_group=8,
        overall_stability=0.8,
    )
    path = tmp_path / "pack.json"
    pack.to_json(str(path))
    loaded = json.loads(path.read_text())

    assert loaded["schema_version"] == "1.1"
    lp = loaded["personas"][0]
    assert lp["centroid_full"] == {"f0": 1.0, "f1": 2.0}
    assert lp["feature_scale"] == {"f0": 0.5, "f1": 0.25}
    assert len(lp["sample_rows"]) == 2
    assert abs(sum(lp["posterior_mean"].values()) - 1.0) < 1e-9
