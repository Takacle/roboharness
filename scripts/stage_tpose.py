"""Compatibility wrapper — delegates to ``gmr_harness.cli.stage_tpose``.

.. deprecated::
    Use ``gmr-harness stage`` directly. This script is kept for backward
    compatibility and will be removed in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "scripts/stage_tpose.py is deprecated; use 'gmr-harness stage' instead.",
    DeprecationWarning,
    stacklevel=2,
)


def main(argv: list[str] | None = None) -> None:
    from gmr_harness.cli.stage_tpose import main as _gmr_main

    _gmr_main(argv)


if __name__ == "__main__":
    main()
