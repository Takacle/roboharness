"""Tests for GMR setup pipeline fixes (Tasks A-D).

Tests the extracted helper functions and orchestration behavior without
requiring a live GMR installation or MuJoCo.
"""

from __future__ import annotations

import argparse
import json
from io import StringIO
from unittest.mock import MagicMock, patch

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


class TestDryRunOrchestration:
    """Test that dry_run=true in setup_robot.py never calls write helpers."""

    def _make_gmr_tree(self, tmp_path):
        gmr_root = tmp_path / "GMR"
        asset_dir = gmr_root / "assets" / "my_robot"
        asset_dir.mkdir(parents=True)
        xml_file = asset_dir / "model.xml"
        xml_file.write_text(
            "<mujoco><worldbody>"
            '<body name="pelvis">'
            '  <body name="left_hip"/>'
            '  <body name="left_knee"/>'
            '  <body name="right_hip"/>'
            '  <body name="right_knee"/>'
            '  <body name="torso">'
            '    <body name="left_shoulder"/>'
            '    <body name="left_elbow"/>'
            '    <body name="right_shoulder"/>'
            '    <body name="right_elbow"/>'
            "  </body>"
            "</body>"
            "</worldbody></mujoco>"
        )
        ik_dir = gmr_root / "general_motion_retargeting" / "ik_configs"
        ik_dir.mkdir(parents=True)
        params_py = gmr_root / "general_motion_retargeting" / "params.py"
        params_py.write_text(
            "ROBOT_XML_DICT = {}\n"
            "ROBOT_BASE_DICT = {}\n"
            "VIEWER_CAM_DISTANCE_DICT = {}\n"
            "IK_CONFIG_DICT = {}\n"
        )
        return gmr_root, xml_file

    def test_dry_run_fresh_generation_skips_write(self, tmp_path):
        gmr_root, xml_file = self._make_gmr_tree(tmp_path)

        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.write_ik_config") as mock_write,
            patch("scripts.setup_robot.clone_ik_config"),
            patch("scripts.setup_robot.extract_xml_body_names") as mock_extract,
            patch("scripts.setup_robot.register_in_params") as mock_reg,
            patch("scripts.setup_robot.update_script_choices"),
        ):
            mock_extract.return_value = [
                "pelvis",
                "left_hip",
                "left_knee",
                "right_hip",
                "right_knee",
                "torso",
                "left_shoulder",
                "left_elbow",
                "right_shoulder",
                "right_elbow",
            ]
            mock_reg.return_value = []

            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "my_robot",
                "--xml",
                str(xml_file),
                "--dry_run",
            ]
            with patch("sys.argv", test_args), patch("sys.stdout", new_callable=StringIO):
                mod.main()

            mock_write.assert_not_called()

    def test_dry_run_clone_skips_write(self, tmp_path):
        gmr_root, xml_file = self._make_gmr_tree(tmp_path)
        ik_configs = gmr_root / "general_motion_retargeting" / "ik_configs"
        src_name = "bvh_to_src_robot.json"
        src_config = ik_configs / src_name
        src_config.write_text(
            json.dumps(
                {
                    "robot_root_name": "pelvis",
                    "ik_match_table1": {"pelvis": ["Pelvis", 100, 10, [0, 0, 0], [1, 0, 0, 0]]},
                    "ik_match_table2": {"pelvis": ["Pelvis", 100, 5, [0, 0, 0], [1, 0, 0, 0]]},
                }
            )
        )

        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.clone_ik_config") as mock_clone,
            patch("scripts.setup_robot.write_ik_config"),
            patch("scripts.setup_robot.extract_xml_body_names") as mock_extract,
            patch("scripts.setup_robot._find_clone_source", return_value=src_config),
            patch("scripts.setup_robot.register_in_params", return_value=[]),
            patch("scripts.setup_robot.update_script_choices", return_value=[]),
        ):
            mock_extract.return_value = [
                "pelvis",
                "left_hip",
                "left_knee",
                "right_hip",
                "right_knee",
                "torso",
                "left_shoulder",
                "left_elbow",
                "right_shoulder",
                "right_elbow",
            ]

            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "my_robot",
                "--xml",
                str(xml_file),
                "--clone_from",
                "src_robot",
                "--formats",
                "bvh",
                "--dry_run",
            ]
            with patch("sys.argv", test_args), patch("sys.stdout", new_callable=StringIO):
                mod.main()

            mock_clone.assert_not_called()

    def test_dry_run_prints_preview(self, tmp_path):
        gmr_root, xml_file = self._make_gmr_tree(tmp_path)

        output = StringIO()
        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.write_ik_config"),
            patch("scripts.setup_robot.extract_xml_body_names") as mock_extract,
            patch("scripts.setup_robot.register_in_params", return_value=[]),
            patch("scripts.setup_robot.update_script_choices", return_value=[]),
            patch("sys.stdout", output),
        ):
            mock_extract.return_value = [
                "pelvis",
                "left_hip",
                "left_knee",
                "right_hip",
                "right_knee",
                "torso",
                "left_shoulder",
                "left_elbow",
                "right_shoulder",
                "right_elbow",
            ]

            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "my_robot",
                "--xml",
                str(xml_file),
                "--dry_run",
            ]
            with patch("sys.argv", test_args):
                mod.main()

        text = output.getvalue()
        assert "dry_run" in text.lower() or "Dry run" in text


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

        with patch("scripts.setup_robot.GMR_ROOT", gmr_root), pytest.raises(SystemExit):
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

    def test_nested_xml_rejected(self, tmp_path):
        gmr_root = tmp_path / "GMR"
        asset_dir = gmr_root / "assets" / "my_robot"
        subdir = asset_dir / "variants"
        subdir.mkdir(parents=True)
        xml_file = subdir / "model.xml"
        xml_file.write_text('<mujoco><worldbody><body name="torso"/></worldbody></mujoco>')

        args = argparse.Namespace(
            xml=str(xml_file),
            robot="my_robot",
        )

        with patch("scripts.setup_robot.GMR_ROOT", gmr_root), pytest.raises(SystemExit):
            import scripts.setup_robot as mod

            mod._resolve_xml(args)


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


class TestSmplxValidationNotSkipped:
    def _make_args(self, **overrides):
        defaults = dict(
            tpose_motion="/path/to/tpose.bvh",
            skip_solve=False,
            skip_validate=False,
            tpose_src="smplx",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_smplx_validates_by_default(self):
        args = self._make_args()
        should_validate = args.tpose_motion and not args.skip_solve and not args.skip_validate
        assert should_validate is True

    def test_smplx_skip_validate_flag(self):
        args = self._make_args(skip_validate=True)
        should_validate = args.tpose_motion and not args.skip_solve and not args.skip_validate
        assert should_validate is False

    def test_smplx_skip_solve_still_skips_validation(self):
        args = self._make_args(skip_solve=True)
        should_validate = args.tpose_motion and not args.skip_solve and not args.skip_validate
        assert should_validate is False

    def test_bvh_validates_by_default(self):
        args = self._make_args(tpose_src="bvh")
        should_validate = args.tpose_motion and not args.skip_solve and not args.skip_validate
        assert should_validate is True

    def test_no_tpose_motion_skips_validation(self):
        args = self._make_args(tpose_motion=None)
        should_validate = args.tpose_motion and not args.skip_solve and not args.skip_validate
        assert not should_validate


class TestValidationCommandConstructed:
    """Verify that Step 6 actually constructs the gmr_tpose_validate.py command."""

    def _make_gmr_tree(self, tmp_path):
        gmr_root = tmp_path / "GMR"
        asset_dir = gmr_root / "assets" / "smplx_robot"
        asset_dir.mkdir(parents=True)
        xml_file = asset_dir / "model.xml"
        xml_file.write_text(
            "<mujoco><worldbody>"
            '<body name="pelvis">'
            '  <body name="torso"/>'
            "</body>"
            "</worldbody></mujoco>"
        )
        ik_dir = gmr_root / "general_motion_retargeting" / "ik_configs"
        ik_dir.mkdir(parents=True)
        params_py = gmr_root / "general_motion_retargeting" / "params.py"
        params_py.write_text(
            "ROBOT_XML_DICT = {}\n"
            "ROBOT_BASE_DICT = {}\n"
            "VIEWER_CAM_DISTANCE_DICT = {}\n"
            f"IK_CONFIG_DICT = {{'smplx': {{'smplx_robot': "
            f"'{ik_dir / 'smplx_to_smplx_robot.json'}'}}}}\n"
        )
        (ik_dir / "smplx_to_smplx_robot.json").write_text("{}")
        return gmr_root

    def test_smplx_validation_command_invoked(self, tmp_path):
        gmr_root = self._make_gmr_tree(tmp_path)

        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="PASSED", stderr="")

        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.extract_xml_body_names", return_value=["pelvis", "torso"]),
            patch("scripts.setup_robot.register_in_params", return_value=[]),
            patch("scripts.setup_robot.update_script_choices", return_value=[]),
            patch("scripts.setup_robot._solve_smplx_offsets", return_value=True),
            patch("scripts.setup_robot.subprocess.run", mock_run),
            patch("sys.stdout", new_callable=StringIO),
        ):
            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "smplx_robot",
                "--tpose_motion",
                "/path/to/tpose.npz",
                "--tpose_src",
                "smplx",
                "--formats",
                "smplx",
            ]
            with patch("sys.argv", test_args):
                mod.main()

        validate_calls = [
            c for c in mock_run.call_args_list if any("gmr_tpose_validate" in str(a) for a in c[0])
        ]
        assert len(validate_calls) >= 1, (
            f"Expected gmr_tpose_validate.py to be invoked, "
            f"but subprocess.run calls were: {mock_run.call_args_list}"
        )

    def test_smplx_validation_command_contains_robot_and_src(self, tmp_path):
        gmr_root = self._make_gmr_tree(tmp_path)

        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="PASSED", stderr="")

        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.extract_xml_body_names", return_value=["pelvis", "torso"]),
            patch("scripts.setup_robot.register_in_params", return_value=[]),
            patch("scripts.setup_robot.update_script_choices", return_value=[]),
            patch("scripts.setup_robot._solve_smplx_offsets", return_value=True),
            patch("scripts.setup_robot.subprocess.run", mock_run),
            patch("sys.stdout", new_callable=StringIO),
        ):
            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "smplx_robot",
                "--tpose_motion",
                "/path/to/tpose.npz",
                "--tpose_src",
                "smplx",
                "--formats",
                "smplx",
            ]
            with patch("sys.argv", test_args):
                mod.main()

        validate_calls = [
            c for c in mock_run.call_args_list if any("gmr_tpose_validate" in str(a) for a in c[0])
        ]
        cmd = validate_calls[0][0][0]
        assert "--robot" in cmd
        assert "smplx_robot" in cmd
        assert "--src" in cmd
        assert "smplx" in cmd

    def test_smplx_skip_validate_no_command(self, tmp_path):
        gmr_root = self._make_gmr_tree(tmp_path)

        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.extract_xml_body_names", return_value=["pelvis", "torso"]),
            patch("scripts.setup_robot.register_in_params", return_value=[]),
            patch("scripts.setup_robot.update_script_choices", return_value=[]),
            patch("scripts.setup_robot._solve_smplx_offsets", return_value=True),
            patch("scripts.setup_robot.subprocess.run", mock_run),
            patch("sys.stdout", new_callable=StringIO),
        ):
            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "smplx_robot",
                "--tpose_motion",
                "/path/to/tpose.npz",
                "--tpose_src",
                "smplx",
                "--formats",
                "smplx",
                "--skip_validate",
            ]
            with patch("sys.argv", test_args):
                mod.main()

        validate_calls = [
            c for c in mock_run.call_args_list if any("gmr_tpose_validate" in str(a) for a in c[0])
        ]
        assert len(validate_calls) == 0


class TestStageCommandConstructed:
    """Verify Step 4 passes the selected T-pose source into stage_tpose.py."""

    def _make_gmr_tree(self, tmp_path):
        gmr_root = tmp_path / "GMR"
        asset_dir = gmr_root / "assets" / "smplx_robot"
        asset_dir.mkdir(parents=True)
        xml_file = asset_dir / "model.xml"
        xml_file.write_text(
            "<mujoco><worldbody>"
            '<body name="pelvis">'
            '  <body name="torso"/>'
            "</body>"
            "</worldbody></mujoco>"
        )
        ik_dir = gmr_root / "general_motion_retargeting" / "ik_configs"
        ik_dir.mkdir(parents=True)
        params_py = gmr_root / "general_motion_retargeting" / "params.py"
        ik_config_path = ik_dir / "smplx_to_smplx_robot.json"
        params_py.write_text(
            "ROBOT_XML_DICT = {}\n"
            "ROBOT_BASE_DICT = {}\n"
            "VIEWER_CAM_DISTANCE_DICT = {}\n"
            f"IK_CONFIG_DICT = {{'smplx': {{'smplx_robot': {str(ik_config_path)!r}}}}}\n"
        )
        ik_config_path.write_text("{}")
        return gmr_root

    def test_stage_command_uses_tpose_src(self, tmp_path):
        gmr_root = self._make_gmr_tree(tmp_path)

        mock_run = MagicMock()
        mock_run.return_value = MagicMock(returncode=0, stdout="line1\nline2\nline3", stderr="")

        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.extract_xml_body_names", return_value=["pelvis", "torso"]),
            patch("scripts.setup_robot.register_in_params", return_value=[]),
            patch("scripts.setup_robot.update_script_choices", return_value=[]),
            patch("scripts.setup_robot._solve_smplx_offsets", return_value=True),
            patch("scripts.setup_robot.subprocess.run", mock_run),
            patch("sys.stdout", new_callable=StringIO),
        ):
            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "smplx_robot",
                "--tpose_motion",
                "/path/to/tpose.npz",
                "--tpose_src",
                "smplx",
                "--formats",
                "smplx",
                "--skip_validate",
            ]
            with patch("sys.argv", test_args):
                mod.main()

        stage_calls = [
            c for c in mock_run.call_args_list if any("stage_tpose" in str(a) for a in c[0])
        ]
        assert len(stage_calls) == 1
        cmd = stage_calls[0][0][0]
        assert "--src" in cmd
        assert cmd[cmd.index("--src") + 1] == "smplx"


class TestConfigWritesToGmrRoot:
    """Prove that write_ik_config and clone_ik_config use GMR_ROOT-based dir."""

    def _make_gmr_tree(self, tmp_path):
        gmr_root = tmp_path / "custom_gmr_root"
        asset_dir = gmr_root / "assets" / "my_robot"
        asset_dir.mkdir(parents=True)
        xml_file = asset_dir / "model.xml"
        xml_file.write_text(
            "<mujoco><worldbody>"
            '<body name="pelvis">'
            '  <body name="left_hip"/>'
            '  <body name="left_knee"/>'
            '  <body name="right_hip"/>'
            '  <body name="right_knee"/>'
            '  <body name="torso">'
            '    <body name="left_shoulder"/>'
            '    <body name="left_elbow"/>'
            '    <body name="right_shoulder"/>'
            '    <body name="right_elbow"/>'
            "  </body>"
            "</body>"
            "</worldbody></mujoco>"
        )
        ik_dir = gmr_root / "general_motion_retargeting" / "ik_configs"
        ik_dir.mkdir(parents=True)
        params_py = gmr_root / "general_motion_retargeting" / "params.py"
        params_py.write_text(
            "ROBOT_XML_DICT = {}\n"
            "ROBOT_BASE_DICT = {}\n"
            "VIEWER_CAM_DISTANCE_DICT = {}\n"
            "IK_CONFIG_DICT = {}\n"
        )
        return gmr_root, xml_file, ik_dir

    def test_write_goes_to_patched_gmr_root(self, tmp_path):
        gmr_root, xml_file, ik_dir = self._make_gmr_tree(tmp_path)

        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.extract_xml_body_names") as mock_extract,
            patch("scripts.setup_robot.register_in_params", return_value=[]),
            patch("scripts.setup_robot.update_script_choices", return_value=[]),
            patch("sys.stdout", new_callable=StringIO),
        ):
            mock_extract.return_value = [
                "pelvis",
                "left_hip",
                "left_knee",
                "right_hip",
                "right_knee",
                "torso",
                "left_shoulder",
                "left_elbow",
                "right_shoulder",
                "right_elbow",
            ]

            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "my_robot",
                "--xml",
                str(xml_file),
                "--formats",
                "bvh",
            ]
            with patch("sys.argv", test_args):
                mod.main()

        expected = ik_dir / "bvh_to_my_robot.json"
        assert expected.exists(), (
            f"Config not written to {expected}; ik_dir contents: {list(ik_dir.iterdir())}"
        )

    def test_clone_goes_to_patched_gmr_root(self, tmp_path):
        gmr_root, xml_file, ik_dir = self._make_gmr_tree(tmp_path)

        src_config = ik_dir / "bvh_to_src_robot.json"
        src_config.write_text(
            json.dumps(
                {
                    "robot_root_name": "pelvis",
                    "ik_match_table1": {"pelvis": ["Pelvis", 100, 10, [0, 0, 0], [1, 0, 0, 0]]},
                    "ik_match_table2": {"pelvis": ["Pelvis", 100, 5, [0, 0, 0], [1, 0, 0, 0]]},
                }
            )
        )

        with (
            patch("scripts.setup_robot.GMR_ROOT", gmr_root),
            patch("scripts.setup_robot.extract_xml_body_names") as mock_extract,
            patch("scripts.setup_robot._find_clone_source", return_value=src_config),
            patch("scripts.setup_robot.register_in_params", return_value=[]),
            patch("scripts.setup_robot.update_script_choices", return_value=[]),
            patch("sys.stdout", new_callable=StringIO),
        ):
            mock_extract.return_value = [
                "pelvis",
                "left_hip",
                "left_knee",
                "right_hip",
                "right_knee",
                "torso",
                "left_shoulder",
                "left_elbow",
                "right_shoulder",
                "right_elbow",
            ]

            import scripts.setup_robot as mod

            test_args = [
                "setup_robot.py",
                "--robot",
                "my_robot",
                "--xml",
                str(xml_file),
                "--clone_from",
                "src_robot",
                "--formats",
                "bvh",
            ]
            with patch("sys.argv", test_args):
                mod.main()

        expected = ik_dir / "bvh_to_my_robot.json"
        assert expected.exists(), (
            f"Cloned config not written to {expected}; ik_dir contents: {list(ik_dir.iterdir())}"
        )
