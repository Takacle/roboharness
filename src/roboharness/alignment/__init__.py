"""Compatibility shim — re-exports from ``gmr_harness.alignment``.

.. deprecated::
    Use ``gmr_harness.alignment`` directly. This module will be removed
    in a future release.

All sub-modules (``body_matcher``, ``config_gen``, etc.) are re-export
shims from ``gmr_harness.alignment.*``. New code should import directly
from ``gmr_harness.alignment``.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment is deprecated; use gmr_harness.alignment instead.",
    DeprecationWarning,
    stacklevel=2,
)

from roboharness.alignment.metrics import (  # noqa: E402
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
from roboharness.alignment.optimize import optimize_scales  # noqa: E402
from roboharness.alignment.patch import apply_patch  # noqa: E402

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
