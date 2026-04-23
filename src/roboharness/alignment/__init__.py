"""Numeric alignment metrics for retargeted robot motion.

Public API for comparing a robot's current pose against an authored T-pose
contract. See ``docs/gmr-alignment-sop.md`` for the geometric definition of
"aligned", the workflow for authoring a spec, and common quaternion pitfalls.
"""

from __future__ import annotations

from roboharness.alignment.metrics import (
    TposeSpec,
    compute_deviations,
    load_tpose_spec,
    total_deviation,
    worst_k,
)

__all__ = [
    "TposeSpec",
    "compute_deviations",
    "load_tpose_spec",
    "total_deviation",
    "worst_k",
]
