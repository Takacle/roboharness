"""Compatibility wrapper — delegates to ``gmr-harness agent --tune_mode scale``.

.. deprecated::
    Use ``gmr-harness agent`` directly. This script is kept for backward
    compatibility and will be removed in a future release.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "examples/gmr_alignment_inspector.py is deprecated; use 'gmr-harness agent' instead.",
    DeprecationWarning,
    stacklevel=2,
)


def main(argv: list[str] | None = None) -> None:
    from gmr_harness.cli.agent import main as _gmr_main

    _gmr_main(argv)


if __name__ == "__main__":
    main()
