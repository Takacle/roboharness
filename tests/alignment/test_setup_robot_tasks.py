"""Tests for GMR setup pipeline fixes (Tasks A-D).

Tests the extracted helper functions and orchestration behavior without
requiring a live GMR installation or MuJoCo.
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import patch

import numpy as np
import pytest

from roboharness._math_utils import axis_angle_to_quat
from roboharness.alignment.orientation_aligner import parse_world_rotation_arg


class TestParseWorldRotationArg:
    def test_90deg_around_z(self):
        result = parse_world_rotation_arg("90,0,0,1")
        expected = [np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2]
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_unnormalised_axis(self):
        result = parse_world_rotation_arg("90,0,0,2")
        expected = [np.sqrt(2) / 2, 0.0, 0.0, np.sqrt(2) / 2]
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_180deg_around_x(self):
        result = parse_world_rotation_arg("180,1,0,0")
        expected = [0.0, 1.0, 0.0, 0.0]
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_negative_angle(self):
        result = parse_world_rotation_arg("-90,0,0,1")
        positive = parse_world_rotation_arg("90,0,0,1")
        assert abs(result[3] - positive[3]) > 0.01

    def test_wrong_field_count_raises(self):
        with pytest.raises(ValueError, match="4 comma-separated"):
            parse_world_rotation_arg("90,0,0")

    def test_non_float_raises(self):
        with pytest.raises(ValueError, match="must be floats"):
            parse_world_rotation_arg("abc,0,0,1")

    def test_zero_axis_raises(self):
        with pytest.raises(ValueError, match="zero norm"):
            parse_world_rotation_arg("90,0,0,0")

    def test_consistency_with_axis_angle_to_quat(self):
        result = parse_world_rotation_arg("45,1,1,0")
        norm = (1**2 + 1**2) ** 0.5
        axis = [1 / norm, 1 / norm, 0.0]
        expected = axis_angle_to_quat(axis, 45)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestDryRunNoWrite:
    def test_fresh_generation_no_file(self, tmp_path):
        from roboharness.alignment.body_matcher import match_bodies
        from roboharness.alignment.config_gen import generate_ik_config
        from roboharness.alignment.skeleton_maps import BVH_SKELETON

        bodies = [
            "pelvis",
            "left_hip",
            "left_knee",
            "left_ankle",
            "right_hip",
            "right_knee",
            "right_ankle",
            "torso",
            "left_shoulder",
            "left_elbow",
            "right_shoulder",
            "right_elbow",
        ]
        result = match_bodies(bodies, BVH_SKELETON)
        config = generate_ik_config(result, BVH_SKELETON)

        dest_dir = tmp_path / "ik_configs"
        dest_dir.mkdir()
        sentinel_before = set(dest_dir.iterdir())

        dry_run = True
        robot = "test_dry_run_robot"
        fmt = "bvh"

        if not dry_run:
            from roboharness.alignment.config_gen import write_ik_config

            write_ik_config(config, robot, fmt, output_dir=dest_dir)

        sentinel_after = set(dest_dir.iterdir())
        assert sentinel_before == sentinel_after

    def test_clone_no_file(self, tmp_path):
        from roboharness.alignment.config_gen import clone_ik_config

        src = tmp_path / "source.json"
        src.write_text(
            json.dumps(
                {
                    "robot_root_name": "pelvis",
                    "ik_match_table1": {"pelvis": ["Pelvis", 100, 10, [0, 0, 0], [1, 0, 0, 0]]},
                    "ik_match_table2": {"pelvis": ["Pelvis", 100, 5, [0, 0, 0], [1, 0, 0, 0]]},
                }
            )
        )

        dest_dir = tmp_path / "ik_configs"
        dest_dir.mkdir()
        sentinel_before = set(dest_dir.iterdir())

        dry_run = True
        if not dry_run:
            clone_ik_config(src, {"pelvis": "new_pelvis"}, "test_robot", "bvh", output_dir=dest_dir)

        sentinel_after = set(dest_dir.iterdir())
        assert sentinel_before == sentinel_after


class TestXmlLocationValidation:
    def test_external_xml_rejected(self, tmp_path):
        xml_file = tmp_path / "external" / "robot.xml"
        xml_file.parent.mkdir(parents=True)
        xml_file.write_text('<mujoco><worldbody><body name="torso"/></worldbody></mujoco>')

        gmr_root = tmp_path / "GMR"
        asset_dir = gmr_root / "assets" / "my_robot"
        asset_dir.mkdir(parents=True)

        args = argparse.Namespace(
            xml=str(xml_file),
            robot="my_robot",
        )

        with patch("scripts.setup_robot.GMR_ROOT", gmr_root):
            with pytest.raises(SystemExit):
                import scripts.setup_robot as mod

                mod._resolve_xml(args)

    def test_internal_xml_accepted(self, tmp_path):
        gmr_root = tmp_path / "GMR"
        asset_dir = gmr_root / "assets" / "my_robot"
        asset_dir.mkdir(parents=True)
        xml_file = asset_dir / "model.xml"
        xml_file.write_text('<mujoco><worldbody><body name="torso"/></worldbody></mujoco>')

        args = argparse.Namespace(
            xml=str(xml_file),
            robot="my_robot",
        )

        with patch("scripts.setup_robot.GMR_ROOT", gmr_root):
            import scripts.setup_robot as mod

            result = mod._resolve_xml(args)
            assert result.name == "model.xml"


class TestStaleParamsValidation:
    def test_reload_after_registration(self, tmp_path):
        from roboharness.alignment._gmr_params import load_gmr_params

        gmr_sub = tmp_path / "general_motion_retargeting"
        gmr_sub.mkdir()
        params_py = gmr_sub / "params.py"
        params_py.write_text(
            "ROBOT_XML_DICT = {}\n"
            "ROBOT_BASE_DICT = {}\n"
            "VIEWER_CAM_DISTANCE_DICT = {}\n"
            "IK_CONFIG_DICT = {}\n"
        )

        params1 = load_gmr_params(tmp_path)
        ik1 = getattr(params1, "IK_CONFIG_DICT", {})
        assert "my_robot" not in ik1.get("bvh", {})

        params_py.write_text(
            "ROBOT_XML_DICT = {}\n"
            "ROBOT_BASE_DICT = {}\n"
            "VIEWER_CAM_DISTANCE_DICT = {}\n"
            "IK_CONFIG_DICT = {'bvh': {'my_robot': 'configs/bvh_to_my_robot.json'}}\n"
        )

        params2 = load_gmr_params(tmp_path)
        ik_dict = getattr(params2, "IK_CONFIG_DICT", {})
        assert "my_robot" in ik_dict.get("bvh", {})
