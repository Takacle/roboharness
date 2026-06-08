"""Compatibility shim — re-exports from ``gmr_harness.alignment.smplx_template``.

.. deprecated::
    Use ``gmr_harness.alignment.smplx_template`` instead.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "roboharness.alignment.smplx_template is deprecated; "
    "use gmr_harness.alignment.smplx_template instead.",
    DeprecationWarning,
    stacklevel=2,
)

from gmr_harness.alignment.smplx_template import *  # noqa: E402,F403
from gmr_harness.alignment.smplx_template import _REQUIRED_JOINTS  # noqa: E402,F401
