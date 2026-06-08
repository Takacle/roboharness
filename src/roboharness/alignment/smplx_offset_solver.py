"""Compatibility shim — re-exports from gmr_harness.alignment.smplx_offset_solver.

.. deprecated::
    Use gmr_harness.alignment.smplx_offset_solver instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.smplx_offset_solver is deprecated; "
    "use gmr_harness.alignment.smplx_offset_solver instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.smplx_offset_solver import *  # noqa: E402,F403
