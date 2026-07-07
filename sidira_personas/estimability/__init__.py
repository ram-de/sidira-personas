"""Estimability precompute — the router's empirical (no-LLM) foundation.

Phase 1 (PURE): the validated causal-estimator core plus a build-time manifest builder.
No Snowflake / R2 / pipeline wiring — those land in Phase 2.

Public API:
  - support_check / estimate / negative_control_guard  (causal_estimator)
  - Support / Estimate / GuardedEstimate dataclasses
  - build_manifest / build_manifest_from_frame / default_neg_control_map  (manifest_builder)
  - Manifest / CandidateEffect / ColumnEntry / CapabilityLink / CoverageEntry dataclasses
"""
from __future__ import annotations

from .causal_estimator import (
    Estimate,
    GuardedEstimate,
    Support,
    estimate,
    negative_control_guard,
    support_check,
)
from .manifest_builder import (
    BEHAVIOURAL_NULL_THRESHOLD,
    GROUNDING_GAP_TOKENS,
    MAX_NC_DIAGNOSTICS,
    MIN_GROUNDING_COVARIATES,
    CandidateEffect,
    CapabilityLink,
    ColumnEntry,
    CoverageEntry,
    Manifest,
    build_manifest,
    build_manifest_from_frame,
    default_neg_control_map,
)

__all__ = [
    "Support",
    "Estimate",
    "GuardedEstimate",
    "support_check",
    "estimate",
    "negative_control_guard",
    "ColumnEntry",
    "CapabilityLink",
    "CoverageEntry",
    "CandidateEffect",
    "Manifest",
    "build_manifest",
    "build_manifest_from_frame",
    "default_neg_control_map",
    "MAX_NC_DIAGNOSTICS",
    "BEHAVIOURAL_NULL_THRESHOLD",
    "MIN_GROUNDING_COVARIATES",
    "GROUNDING_GAP_TOKENS",
]
