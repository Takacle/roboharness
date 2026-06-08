"""Compatibility shim — re-exports from gmr_harness.alignment.gmr_register.

.. deprecated::
    Use gmr_harness.alignment.gmr_register instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.gmr_register is deprecated; "
    "use gmr_harness.alignment.gmr_register instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.gmr_register import *  # noqa: E402,F403
