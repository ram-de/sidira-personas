"""Persona Pack JSON schema — versioned contract consumed by the Sidira simulation engine.

v1.1 (additive over v1.0) adds the four fields the simulator's §2 step-2 individual-row
layer consumes (see ram-de/sidira-personas#7). All live in **adapted-feature space** — the
same space the centroid values come from — so the consumer z-scores a sampled row directly:
``z = (row[f] - centroid_full[f]) / feature_scale[f]``. New ``Persona`` fields:

- ``centroid_full``  — cluster mean over ALL adapted features (not just the top-distinctive subset).
- ``feature_scale``  — within-cluster std per feature (zero-floored), the z-score denominator.
- ``sample_rows``    — N privacy-clean synthetic rows (covariance-aware draws), keyed by list index.
- ``posterior_mean`` — mean GMM posterior over the cluster's members ({persona_id: prob}); the
  soft-membership hedge. ``None`` for FIXED_* personas (hard-defined, no GMM posterior).

Every new field is defaulted so v1.0 packs still parse unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import json

PERSONA_PACK_SCHEMA_VERSION = "1.1"


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
class BehavioralSeeds:
    """Per-persona identity-draw pools consumed by the simulator's identity_seed
    service (``draw_identity_seed``). Each list is ordered most-common-first so a
    uniform draw yields a realistic-ish distribution for THIS segment.

    LLM-generated alongside the persona name/description (see compile_prompts),
    grounded in the cluster's distinctive z-deltas + feature values. When the LLM
    is unavailable (``--no-llm`` / no key), the producer emits ``None`` and the
    simulator falls back to its generic ``_DEFAULT_BEHAVIORAL_SEEDS`` pool — same
    degraded-but-functional behavior as a v1.0 pack that omits the field.

    Mirrors the consumer's ``BehavioralSeedSchema`` field-for-field (5 lists).
    """

    age_bands: list[str] = field(default_factory=list)
    locale_bands: list[str] = field(default_factory=list)
    tenure_bands: list[str] = field(default_factory=list)
    current_state_pool: list[str] = field(default_factory=list)
    recent_gripe_pool: list[str] = field(default_factory=list)


@dataclass
class Persona:
    id: str
    name: str
    description: str
    jtbd: str
    size: int
    weights: PersonaWeights
    stability: float | None
    top_features: list[FeatureDelta]
    centroid: dict[str, float]
    # --- v1.1 (additive; see module docstring) ---
    centroid_full: dict[str, float] = field(default_factory=dict)
    feature_scale: dict[str, float] = field(default_factory=dict)
    sample_rows: list[dict[str, float]] = field(default_factory=list)
    posterior_mean: dict[str, float] | None = None
    # --- #342 (additive) — per-persona identity-draw pools for the simulator's
    # identity_seed service. None when LLM naming is off/unavailable (the consumer
    # then falls back to its generic default pool, exactly as a v1.0 pack does). ---
    behavioral_seeds: BehavioralSeeds | None = None
    # --- #7 honesty fields (additive) ---
    # ``stability`` now carries the per-cluster Jaccard (Hennig clusterboot) for discovered
    # personas, and ``None`` for FIXED_* hard-defined segments (no GMM bootstrap applies — the
    # old hard-coded 1.0 was a fiction). ``stability_verdict`` is its interpretation band.
    stability_verdict: str | None = None
    # Normalized entropy of ``posterior_mean`` (0 = crisp identity, 1 = maximally mixed); ``None``
    # for FIXED_* personas. Measures assignment ambiguity / cluster overlap — a high value means a
    # falsely-crisp identity, which is what collapses downstream simulated-agent variance.
    posterior_entropy: float | None = None


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
