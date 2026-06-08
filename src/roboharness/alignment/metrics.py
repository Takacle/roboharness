"""Compatibility shim — re-exports from gmr_harness.alignment.metrics.

.. deprecated::
    Use gmr_harness.alignment.metrics instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.metrics is deprecated; use gmr_harness.alignment.metrics instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.metrics import *  # noqa: E402,F403
