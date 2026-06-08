"""Compatibility shim — re-exports from gmr_harness.alignment._gmr_params.

.. deprecated::
    Use gmr_harness.alignment._gmr_params instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment._gmr_params is deprecated; "
    "use gmr_harness.alignment._gmr_params instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment._gmr_params import *  # noqa: E402,F403
