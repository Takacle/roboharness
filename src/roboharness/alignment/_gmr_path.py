"""Compatibility shim — re-exports from gmr_harness.alignment._gmr_path.

.. deprecated::
    Use gmr_harness.alignment._gmr_path instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment._gmr_path is deprecated; use gmr_harness.alignment._gmr_path instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment._gmr_path import *  # noqa: E402,F403
