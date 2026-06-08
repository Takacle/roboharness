"""Unified CLI entry point: ``gmr-harness <subcommand>``.

Uses manual subcommand dispatch so each sub-module owns its own argparse
parser.  This avoids the empty-parser problem and ensures ``gmr-harness
setup --help`` shows the real flags.

Supports ``python -m gmr_harness.cli.main`` as well.
"""

from __future__ import annotations

import sys

_SUBCOMMANDS = {
    "setup": ("gmr_harness.cli.setup_robot", "main"),
    "stage": ("gmr_harness.cli.stage_tpose", "main"),
    "validate": ("gmr_harness.cli.validate", "main"),
    "agent": ("gmr_harness.cli.agent", "main"),
}


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return

    subcmd = args[0]
    if subcmd not in _SUBCOMMANDS:
        print(f"Unknown subcommand: {subcmd!r}", file=sys.stderr)
        _print_help()
        sys.exit(1)

    module_name, func_name = _SUBCOMMANDS[subcmd]
    import importlib

    mod = importlib.import_module(module_name)
    func = getattr(mod, func_name)
    func(args[1:])


def _print_help() -> None:
    print(
        "usage: gmr-harness <subcommand> [options]\n"
        "\n"
        "GMR-Harness: Alignment toolchain for General Motion Retargeting\n"
        "\n"
        "Subcommands:\n"
        "  setup      Setup a new robot in the GMR alignment pipeline\n"
        "  stage      Stage a robot at T-pose and dump alignment spec\n"
        "  validate   Validate retargeted T-pose against committed spec\n"
        "  agent      AI-driven automatic IK config optimization\n"
        "\n"
        "Use 'gmr-harness <subcommand> --help' for subcommand details."
    )


if __name__ == "__main__":
    main()
