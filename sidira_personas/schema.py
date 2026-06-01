"""Persona Pack JSON schema — versioned contract consumed by the Sidira simulation engine."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json

PERSONA_PACK_SCHEMA_VERSION = "1.0"


@dataclass
class FeatureDelta:
    feature: str
    group: str
    centroid_value: float
    population_mean: float
    population_std: float
    z_delta: float


@dataclass
class PersonaWeights:
    by_headcount: float
    by_orders: float | None = None
    by_revenue: float | None = None


@dataclass
class Persona:
    id: str
    name: str
    description: str
    jtbd: str
    size: int
    weights: PersonaWeights
    stability: float
    top_features: list[FeatureDelta]
    centroid: dict[str, float]


@dataclass
class PersonaPack:
    schema_version: str
    company_id: str
    run_id: str
    created_at: str
    n_users: int
    feature_groups: list[str]
    personas: list[Persona]
    cluster_method: str
    k_selected: int
    encoder_method: str
    embedding_dim_per_group: int
    overall_stability: float
    notes: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
