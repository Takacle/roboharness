"""Numeric alignment metrics for retargeted robot motion.

Public API for comparing a robot's current pose against an authored T-pose
contract. See ``docs/gmr-alignment-sop.md`` for the geometric definition of
"aligned", the workflow for authoring a spec, and common quaternion pitfalls.

New-robot setup API (skeleton maps → body matching → config generation):
    ``skeleton_maps``, ``body_matcher``, ``config_gen``, ``gmr_register``.
"""

from __future__ import annotations

from roboharness.alignment.metrics import (
    TposeSpec,
    compute_deviations,
    compute_direct_patch,
    compute_position_deviations,
    load_tpose_spec,
    total_deviation,
    total_position_deviation,
    worst_k,
    worst_k_position,
)
from roboharness.alignment.optimize import optimize_scales
from roboharness.alignment.patch import apply_patch

__all__ = [
    "TposeSpec",
    "apply_patch",
    "compute_deviations",
    "compute_direct_patch",
    "compute_position_deviations",
    "load_tpose_spec",
    "optimize_scales",
    "total_deviation",
    "total_position_deviation",
    "worst_k",
    "worst_k_position",
]
