"""Smoke tests for legacy wrapper scripts — verify --help works without GMR."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]


def _wrap_help(rel_path: str, warn: bool = False) -> tuple[int, str, str]:
    full_path = _REPO / rel_path
    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(_REPO / "packages" / "gmr-harness" / "src"),
    }
    cmd = [sys.executable]
    if warn:
        cmd.append("-Wd")
    cmd.extend([str(full_path), "--help"])
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.returncode, result.stdout, result.stderr


def test_setup_robot_help():
    rc, out, err = _wrap_help("scripts/setup_robot.py")
    assert rc == 0, f"exit {rc}: stderr={err}"
    assert "Setup a new robot" in out


def test_setup_robot_deprecation_warning():
    _, _, err = _wrap_help("scripts/setup_robot.py", warn=True)
    assert "deprecated" in err or "DeprecationWarning" in err


def test_stage_tpose_help():
    rc, out, err = _wrap_help("scripts/stage_tpose.py")
    assert rc == 0, f"exit {rc}: stderr={err}"
    assert "Stage a robot" in out or "T-pose" in out


def test_gmr_alignment_agent_help():
    rc, out, err = _wrap_help("examples/gmr_alignment_agent.py")
    assert rc == 0, f"exit {rc}: stderr={err}"
    assert "GMR Alignment Agent" in out


def test_gmr_tpose_validate_help():
    rc, out, err = _wrap_help("examples/gmr_tpose_validate.py")
    assert rc == 0, f"exit {rc}: stderr={err}"
    assert "Validate" in out or "T-pose" in out or "deviation" in out
