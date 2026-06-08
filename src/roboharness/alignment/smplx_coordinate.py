"""Compatibility shim — re-exports from gmr_harness.alignment.smplx_coordinate.

.. deprecated::
    Use gmr_harness.alignment.smplx_coordinate instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.smplx_coordinate is deprecated; "
    "use gmr_harness.alignment.smplx_coordinate instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.smplx_coordinate import *  # noqa: E402,F403
