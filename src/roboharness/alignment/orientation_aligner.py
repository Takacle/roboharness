"""Compatibility shim — re-exports from gmr_harness.alignment.orientation_aligner.

.. deprecated::
    Use gmr_harness.alignment.orientation_aligner instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.orientation_aligner is deprecated; "
    "use gmr_harness.alignment.orientation_aligner instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.orientation_aligner import *  # noqa: E402,F403
from gmr_harness.alignment.orientation_aligner import _resolve_includes  # noqa: E402,F401
