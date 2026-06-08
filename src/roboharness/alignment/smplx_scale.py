"""Compatibility shim — re-exports from gmr_harness.alignment.smplx_scale.

.. deprecated::
    Use gmr_harness.alignment.smplx_scale instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.smplx_scale is deprecated; "
    "use gmr_harness.alignment.smplx_scale instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.smplx_scale import *  # noqa: E402,F403
