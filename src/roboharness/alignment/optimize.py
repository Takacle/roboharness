"""Compatibility shim — re-exports from gmr_harness.alignment.optimize.

.. deprecated::
    Use gmr_harness.alignment.optimize instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.optimize is deprecated; use gmr_harness.alignment.optimize instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.optimize import *  # noqa: E402,F403
