"""Compatibility shim — re-exports from gmr_harness.alignment.config_gen.

.. deprecated::
    Use gmr_harness.alignment.config_gen instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.config_gen is deprecated; use gmr_harness.alignment.config_gen instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.config_gen import *  # noqa: E402,F403
