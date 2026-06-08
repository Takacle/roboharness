"""Compatibility wrapper — delegates to ``gmr_harness.cli.validate``.

.. deprecated::
    Use ``gmr-harness validate`` directly. This script is kept for backward
    compatibility and will be removed in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "examples/gmr_tpose_validate.py is deprecated; use 'gmr-harness validate' instead.",
    DeprecationWarning,
    stacklevel=2,
)


def main(argv: list[str] | None = None) -> None:
    from gmr_harness.cli.validate import main as _gmr_main

    _gmr_main(argv)


if __name__ == "__main__":
    main()
