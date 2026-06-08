"""Tests for --from_step artifact validation in setup_robot."""

from __future__ import annotations

import contextlib
import json
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_gmr_env(tmp_path):
    gmr_root = tmp_path / "GMR"
    gmr_root.mkdir()
    ik_dir = gmr_root / "general_motion_retargeting" / "ik_configs"
    ik_dir.mkdir(parents=True, exist_ok=True)
    params_dir = gmr_root / "general_motion_retargeting"
    params_dir.mkdir(parents=True, exist_ok=True)

    params_py = params_dir / "params.py"
    params_py.write_text(
        "import pathlib\n"
        "HERE = pathlib.Path(__file__).parent\n"
        "IK_CONFIG_ROOT = HERE / 'ik_configs'\n"
        "ASSET_ROOT = HERE / '..' / 'assets'\n"
        "ROBOT_XML_DICT = {'test_robot': ASSET_ROOT / 'test_robot' / 'robot.xml'}\n"
        "ROBOT_BASE_DICT = {'test_robot': 'pelvis'}\n"
        "VIEWER_CAM_DISTANCE_DICT = {'test_robot': 2.5}\n"
        "IK_CONFIG_DICT = {\n"
        "    'bvh': {'test_robot': IK_CONFIG_ROOT / 'bvh_to_test_robot.json'},\n"
        "}\n"
    )

    xml_dir = gmr_root / "assets" / "test_robot"
    xml_dir.mkdir(parents=True)
    xml = xml_dir / "robot.xml"
    xml.write_text(
        "<mujoco><worldbody>"
        '<body name="pelvis"><body name="left_leg"><joint name="lj"/><geom/></body>'
        '<body name="right_leg"><joint name="rj"/><geom/></body></body>'
        "</worldbody></mujoco>"
    )

    return gmr_root


def test_from_step_skip_match_warns_missing_config(mock_gmr_env, capsys):
    with patch("gmr_harness.cli.setup_robot._get_gmr_root", return_value=mock_gmr_env):
        from gmr_harness.cli.setup_robot import main

        with contextlib.suppress(SystemExit):
            main(
                [
                    "--robot",
                    "test_robot",
                    "--from_step",
                    "2",
                    "--dry_run",
                    "--no-interactive",
                ]
            )

    captured = capsys.readouterr()
    assert "WARNING: expected config not found" in captured.out


def test_from_step_skip_register_warns_missing(mock_gmr_env, capsys):
    with patch("gmr_harness.cli.setup_robot._get_gmr_root", return_value=mock_gmr_env):
        from gmr_harness.cli.setup_robot import main

        with contextlib.suppress(SystemExit):
            main(
                [
                    "--robot",
                    "test_robot",
                    "--from_step",
                    "3",
                    "--dry_run",
                    "--no-interactive",
                ]
            )

    captured = capsys.readouterr()
    assert "Verified" in captured.out and "registered in params.py" in captured.out


def test_from_step_skip_stage_warns_missing_spec(mock_gmr_env, tmp_path, capsys):
    with patch("gmr_harness.cli.setup_robot._get_gmr_root", return_value=mock_gmr_env):
        from gmr_harness.cli.setup_robot import main

        with contextlib.suppress(SystemExit):
            main(
                [
                    "--robot",
                    "test_robot",
                    "--from_step",
                    "5",
                    "--dry_run",
                    "--no-interactive",
                    "--output_dir",
                    str(tmp_path / "out"),
                ]
            )

    captured = capsys.readouterr()
    assert "expected spec not found" in captured.out


def test_from_step_skip_stage_finds_existing_spec(mock_gmr_env, tmp_path, capsys):
    spec_dir = tmp_path / "out"
    spec_dir.mkdir()
    spec_file = spec_dir / "test_robot.json"
    spec_file.write_text(json.dumps({"links": {}, "qpos": []}))

    with patch("gmr_harness.cli.setup_robot._get_gmr_root", return_value=mock_gmr_env):
        from gmr_harness.cli.setup_robot import main

        with contextlib.suppress(SystemExit):
            main(
                [
                    "--robot",
                    "test_robot",
                    "--from_step",
                    "5",
                    "--dry_run",
                    "--no-interactive",
                    "--output_dir",
                    str(spec_dir),
                ]
            )

    captured = capsys.readouterr()
    assert "Verified spec exists" in captured.out


def test_from_step_resume_at_validate_skips_earlier_steps(mock_gmr_env, capsys, tmp_path):
    spec_dir = tmp_path / "out"
    spec_dir.mkdir()
    spec_file = spec_dir / "test_robot.json"
    spec_file.write_text(json.dumps({"links": {}, "qpos": []}))

    with patch("gmr_harness.cli.setup_robot._get_gmr_root", return_value=mock_gmr_env):
        from gmr_harness.cli.setup_robot import main

        with contextlib.suppress(SystemExit):
            main(
                [
                    "--robot",
                    "test_robot",
                    "--from_step",
                    "6",
                    "--dry_run",
                    "--no-interactive",
                    "--output_dir",
                    str(spec_dir),
                ]
            )

    captured = capsys.readouterr()
    assert "Skipping steps 0-1" in captured.out
    assert "Skipping step 2" in captured.out
    assert "Skipping step 4" in captured.out
    assert "Verified spec exists" in captured.out
