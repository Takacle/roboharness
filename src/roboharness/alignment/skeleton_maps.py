"""Compatibility shim — re-exports from gmr_harness.alignment.skeleton_maps.

.. deprecated::
    Use gmr_harness.alignment.skeleton_maps instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.skeleton_maps is deprecated; "
    "use gmr_harness.alignment.skeleton_maps instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.skeleton_maps import *  # noqa: E402,F403
