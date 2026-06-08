"""Compatibility shim — re-exports from gmr_harness.alignment.patch.

.. deprecated::
    Use gmr_harness.alignment.patch instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.patch is deprecated; use gmr_harness.alignment.patch instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.patch import *  # noqa: E402,F403
from gmr_harness.alignment.patch import (  # noqa: E402,F401
    SCALE_BOUNDS,
    _quats_close,
    _resolve_quat_spec,
    _resolve_scale_spec,
)
