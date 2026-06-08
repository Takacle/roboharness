"""Compatibility shim — re-exports from gmr_harness.alignment.body_matcher.

.. deprecated::
    Use gmr_harness.alignment.body_matcher instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.body_matcher is deprecated; "
    "use gmr_harness.alignment.body_matcher instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.body_matcher import *  # noqa: E402,F403
