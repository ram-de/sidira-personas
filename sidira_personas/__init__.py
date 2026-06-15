"""sidira-personas — persona discovery pipeline (Adapter → Encoder → Cluster → Calibrate → Compile).

The public library entry point is :func:`run_pipeline`, which runs the full
pipeline over an in-memory DataFrame and returns the artifacts (used by the CLI
and by the simulator backend's persona-build worker).
"""
from __future__ import annotations

from .pipeline import PipelineResult, run_pipeline

__all__ = ["run_pipeline", "PipelineResult"]
