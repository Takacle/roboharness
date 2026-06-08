"""Tests for gmr_harness.cli.main — dispatch and help."""

from __future__ import annotations

import subprocess
import sys

from gmr_harness.cli.main import _SUBCOMMANDS, main


def test_help_runs_without_gmr():
    result = subprocess.run(
        [sys.executable, "-m", "gmr_harness.cli.main", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "gmr-harness" in result.stdout
    assert "setup" in result.stdout


def test_help_via_main_argv():
    main(["--help"])


def test_unknown_subcommand_exits():
    try:
        main(["nonexistent_cmd"])
        raise AssertionError("should have exited")
    except SystemExit as exc:
        assert exc.code == 1


def test_all_subcommands_registered():
    assert "setup" in _SUBCOMMANDS
    assert "stage" in _SUBCOMMANDS
    assert "validate" in _SUBCOMMANDS
    assert "agent" in _SUBCOMMANDS


def test_subcommand_modules_importable():
    import importlib

    for _name, (mod_name, _func) in _SUBCOMMANDS.items():
        mod = importlib.import_module(mod_name)
        assert hasattr(mod, "main")
