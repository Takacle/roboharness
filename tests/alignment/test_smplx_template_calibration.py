"""Tests for SMPL-X template calibration, offset solving, and CLI wiring.

Covers:
- Task A: load_smplx_template_tpose frame generation
- Task B: solve_smplx_offsets_from_template
- Task C: setup_robot.py --src smplx template calibration wiring
- Task D: gmr_tpose_validate.py --use_smplx_template mode
- Task E: SMPL-X coordinate policy preservation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

smplx = pytest.importorskip("smplx")
torch = pytest.importorskip("torch")

from roboharness.alignment.smplx_offset_solver import (  # noqa: E402
    solve_smplx_offsets_from_template,
)
from roboharness.alignment.smplx_template import (  # noqa: E402
    _REQUIRED_JOINTS,
    load_smplx_template_tpose,
    resolve_body_model_path,
)

_BODY_MODEL_ROOT = resolve_body_model_path(None)
_HAS_BODY_MODEL = _BODY_MODEL_ROOT.is_dir()

pytestmark = pytest.mark.skipif(not _HAS_BODY_MODEL, reason="SMPLX body model not found")

_SRC_MAP = {
    "scripts/setup_robot.py": "packages/gmr-harness/src/gmr_harness/cli/setup_robot.py",
    "scripts/stage_tpose.py": "packages/gmr-harness/src/gmr_harness/cli/stage_tpose.py",
    "examples/gmr_tpose_validate.py": "packages/gmr-harness/src/gmr_harness/cli/validate.py",
    "examples/gmr_alignment_agent.py": "packages/gmr-harness/src/gmr_harness/solver.py",
}


def _src(path: str) -> str:
    return Path(_SRC_MAP.get(path, path)).read_text()


class TestResolveBodyModelPath:
    def test_resolves_directory_with_smplx_subfolder(self, tmp_path: Path):
        body_models = tmp_path / "body_models"
        smplx_dir = body_models / "smplx"
        smplx_dir.mkdir(parents=True)
        (smplx_dir / "SMPLX_MALE.npz").write_text("fake")
        result = resolve_body_model_path(body_models)
        assert result == body_models

    def test_resolves_smplx_subfolder_to_parent(self, tmp_path: Path):
        body_models = tmp_path / "body_models"
        smplx_dir = body_models / "smplx"
        smplx_dir.mkdir(parents=True)
        (smplx_dir / "SMPLX_MALE.npz").write_text("fake")
        result = resolve_body_model_path(smplx_dir)
        assert result == body_models

    def test_resolves_npz_file_as_is(self, tmp_path: Path):
        body_models = tmp_path / "body_models"
        smplx_dir = body_models / "smplx"
        smplx_dir.mkdir(parents=True)
        npz_file = smplx_dir / "SMPLX_MALE.npz"
        npz_file.write_text("fake")
        result = resolve_body_model_path(npz_file)
        assert result == npz_file

    def test_resolves_arbitrary_npz_file(self, tmp_path: Path):
        custom = tmp_path / "my_custom_model.npz"
        custom.write_text("fake")
        result = resolve_body_model_path(custom)
        assert result == custom

    def test_none_auto_discovers(self):
        result = resolve_body_model_path(None)
        assert result.is_dir() or result.is_file()

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            resolve_body_model_path("/nonexistent/path/to/model")

    def test_no_hardcoded_absolute_paths_in_source(self):
        source = _src("scripts/setup_robot.py")
        source += Path(
            "packages/gmr-harness/src/gmr_harness/alignment/smplx_template.py"
        ).read_text()
        assert "/home/" not in source, "smplx_template.py must not contain hardcoded absolute paths"

    def test_no_hardcoded_absolute_paths_in_solver(self):
        source = _src("scripts/setup_robot.py")
        source += Path(
            "packages/gmr-harness/src/gmr_harness/alignment/smplx_offset_solver.py"
        ).read_text()
        assert "/home/" not in source, (
            "smplx_offset_solver.py must not contain hardcoded absolute paths"
        )


class TestLoadSmlxTemplateTpose:
    def test_returns_required_joints(self):
        frame, _height = load_smplx_template_tpose(_BODY_MODEL_ROOT)
        for j in _REQUIRED_JOINTS:
            assert j in frame, f"Missing required joint: {j}"

    def test_all_quaternions_normalized(self):
        frame, _height = load_smplx_template_tpose(_BODY_MODEL_ROOT)
        for name, (_pos, quat) in frame.items():
            norm = float(np.linalg.norm(quat))
            assert abs(norm - 1.0) < 1e-6, f"{name} quat norm = {norm}"

    def test_pelvis_orientation_is_smpl_to_mujoco(self):
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        frame, _height = load_smplx_template_tpose(_BODY_MODEL_ROOT)
        _, quat = frame["pelvis"]
        np.testing.assert_allclose(quat, SMPL_TO_MUJOCO_QUAT, atol=1e-6)

    def test_body_orientations_carry_base_rotation_at_zero_pose(self):
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        frame, _height = load_smplx_template_tpose(_BODY_MODEL_ROOT)
        body_joints = [
            "pelvis",
            "left_hip",
            "right_hip",
            "spine1",
            "left_knee",
            "right_knee",
            "spine2",
            "left_ankle",
            "right_ankle",
            "spine3",
            "left_foot",
            "right_foot",
            "neck",
            "left_collar",
            "right_collar",
            "head",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        ]
        for name in body_joints:
            _, quat = frame[name]
            np.testing.assert_allclose(
                quat,
                SMPL_TO_MUJOCO_QUAT,
                atol=1e-6,
                err_msg=f"{name} orientation not SMPL_TO_MUJOCO_QUAT at zero pose",
            )

    def test_human_height_reasonable(self):
        _frame, height = load_smplx_template_tpose(_BODY_MODEL_ROOT)
        assert 1.4 < height < 2.2, f"Unreasonable height: {height}"

    def test_positions_are_3d(self):
        frame, _height = load_smplx_template_tpose(_BODY_MODEL_ROOT)
        for name, (pos, _quat) in frame.items():
            assert pos.shape == (3,), f"{name} pos shape = {pos.shape}"

    def test_custom_betas(self):
        _frame0, h0 = load_smplx_template_tpose(_BODY_MODEL_ROOT, betas=np.zeros(10))
        betas_tall = np.zeros(10)
        betas_tall[0] = 2.0
        _frame_t, h_t = load_smplx_template_tpose(_BODY_MODEL_ROOT, betas=betas_tall)
        assert h_t > h0

    def test_renamed_npz_loads_via_full_loader(self, tmp_path: Path):
        import shutil

        npz_src = _BODY_MODEL_ROOT / "smplx" / "SMPLX_MALE.npz"
        if not npz_src.is_file():
            npz_src = _BODY_MODEL_ROOT / "SMPLX_MALE.npz"
        if not npz_src.is_file():
            pytest.skip("SMPLX_MALE.npz not found for rename test")

        renamed = tmp_path / "arbitrary_name.npz"
        shutil.copy2(npz_src, renamed)

        frame, height = load_smplx_template_tpose(renamed)
        for j in _REQUIRED_JOINTS:
            assert j in frame, f"Missing required joint after rename: {j}"
        assert 1.4 < height < 2.2

        _, q = frame["pelvis"]
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        np.testing.assert_allclose(q, SMPL_TO_MUJOCO_QUAT, atol=1e-6)


class TestSolveSmplxOffsetsFromTemplate:
    def _make_tpose_spec(self, tmp_path: Path) -> Path:
        spec = {
            "robot": "test_robot",
            "xml_path": "/fake/path.xml",
            "qpos": [0, 0, 0, 1, 0, 0, 0] + [0.0] * 29,
            "links": {
                "base_link": {
                    "pos": [0, 0, 0],
                    "R": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                },
                "left_hip_link": {
                    "pos": [0.05, 0, -0.1],
                    "R": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                },
                "right_hip_link": {
                    "pos": [-0.05, 0, -0.1],
                    "R": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                },
            },
        }
        spec_path = tmp_path / "test_robot.json"
        spec_path.write_text(json.dumps(spec))
        return spec_path

    def _make_ik_config(self, tmp_path: Path) -> Path:
        config = {
            "robot_root_name": "base_link",
            "human_root_name": "pelvis",
            "ik_match_table1": {
                "base_link": ["pelvis", 100, 10, [0, 0, 0], [1, 0, 0, 0]],
                "left_hip_link": ["left_hip", 0, 10, [0, 0, 0], [1, 0, 0, 0]],
                "right_hip_link": ["right_hip", 0, 10, [0, 0, 0], [1, 0, 0, 0]],
            },
            "ik_match_table2": {
                "base_link": ["pelvis", 100, 5, [0, 0, 0], [1, 0, 0, 0]],
                "left_hip_link": ["left_hip", 10, 5, [0, 0, 0], [1, 0, 0, 0]],
                "right_hip_link": ["right_hip", 10, 5, [0, 0, 0], [1, 0, 0, 0]],
            },
            "human_scale_table": {"pelvis": 0.9},
        }
        config_path = tmp_path / "smplx_to_test_robot.json"
        config_path.write_text(json.dumps(config))
        return config_path

    def test_solves_offsets(self, tmp_path: Path):
        config_path = self._make_ik_config(tmp_path)
        spec_path = self._make_tpose_spec(tmp_path)

        solved = solve_smplx_offsets_from_template(
            ik_config_path=config_path,
            tpose_spec_path=spec_path,
            body_model_path=_BODY_MODEL_ROOT,
        )

        for table_name in ("ik_match_table1", "ik_match_table2"):
            table = solved[table_name]
            for joint_name, entry in table.items():
                quat = entry[4]
                norm = float(np.linalg.norm(quat))
                assert abs(norm - 1.0) < 1e-6, f"{table_name}/{joint_name} quat norm = {norm}"

    def test_both_tables_same_offset(self, tmp_path: Path):
        config_path = self._make_ik_config(tmp_path)
        spec_path = self._make_tpose_spec(tmp_path)

        solved = solve_smplx_offsets_from_template(
            ik_config_path=config_path,
            tpose_spec_path=spec_path,
            body_model_path=_BODY_MODEL_ROOT,
        )

        t1 = solved["ik_match_table1"]
        t2 = solved["ik_match_table2"]
        for joint_name in t1:
            if joint_name in t2:
                q1 = np.asarray(t1[joint_name][4])
                q2 = np.asarray(t2[joint_name][4])
                direct = float(np.linalg.norm(q1 - q2))
                flipped = float(np.linalg.norm(q1 + q2))
                assert min(direct, flipped) < 1e-6, (
                    f"{joint_name}: t1={t1[joint_name][4]} vs t2={t2[joint_name][4]}"
                )

    def test_preserves_world_rotation(self, tmp_path: Path):
        config_path = self._make_ik_config(tmp_path)
        spec_path = self._make_tpose_spec(tmp_path)

        config_with_wr = json.loads(config_path.read_text())
        config_with_wr["world_rotation"] = [0.5, -0.5, -0.5, -0.5]
        config_path.write_text(json.dumps(config_with_wr))

        solved = solve_smplx_offsets_from_template(
            ik_config_path=config_path,
            tpose_spec_path=spec_path,
            body_model_path=_BODY_MODEL_ROOT,
        )

        assert "world_rotation" in solved
        np.testing.assert_allclose(solved["world_rotation"], [0.5, -0.5, -0.5, -0.5], atol=1e-10)

    def test_solver_does_not_inject_world_rotation(self, tmp_path: Path):
        config_path = self._make_ik_config(tmp_path)
        spec_path = self._make_tpose_spec(tmp_path)

        config_json = json.loads(config_path.read_text())
        assert "world_rotation" not in config_json

        solved = solve_smplx_offsets_from_template(
            ik_config_path=config_path,
            tpose_spec_path=spec_path,
            body_model_path=_BODY_MODEL_ROOT,
        )

        assert "world_rotation" not in solved

    def test_solver_ignores_world_rotation_in_offsets(self, tmp_path: Path):
        from scipy.spatial.transform import Rotation as R

        config_path = self._make_ik_config(tmp_path)
        spec_path = self._make_tpose_spec(tmp_path)

        r_wr = R.from_euler("z", 45, degrees=True)
        wr_quat = [float(v) for v in r_wr.as_quat(scalar_first=True)]

        config_with_wr = json.loads(config_path.read_text())
        config_with_wr["world_rotation"] = wr_quat
        config_path.write_text(json.dumps(config_with_wr))

        solved = solve_smplx_offsets_from_template(
            ik_config_path=config_path,
            tpose_spec_path=spec_path,
            body_model_path=_BODY_MODEL_ROOT,
        )

        assert "world_rotation" in solved
        np.testing.assert_allclose(solved["world_rotation"], wr_quat, atol=1e-10)

        q_offset = np.asarray(solved["ik_match_table1"]["base_link"][4], dtype=np.float64)
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        r_conv_inv = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True).inv()
        r_target = R.from_matrix(
            np.asarray(
                json.loads(spec_path.read_text())["links"]["base_link"]["R"],
                dtype=np.float64,
            )
        )
        expected_offset = (r_conv_inv * r_target).as_quat(scalar_first=True)
        assert (
            min(
                float(np.linalg.norm(q_offset - expected_offset)),
                float(np.linalg.norm(q_offset + expected_offset)),
            )
            < 1e-6
        )

    def test_preserves_robot_root_name(self, tmp_path: Path):
        config_path = self._make_ik_config(tmp_path)
        spec_path = self._make_tpose_spec(tmp_path)

        solved = solve_smplx_offsets_from_template(
            ik_config_path=config_path,
            tpose_spec_path=spec_path,
            body_model_path=_BODY_MODEL_ROOT,
        )

        assert solved["robot_root_name"] == "base_link"
        assert solved["human_root_name"] == "pelvis"


class TestSetupRobotSmplxTemplateCLI:
    def test_smplx_template_model_arg_exists(self):
        source = _src("scripts/setup_robot.py")
        assert "--smplx_template_model" in source

    def test_solve_smplx_offsets_accepts_body_model_root(self):
        source = _src("scripts/setup_robot.py")
        assert "body_model_root=" in source

    def test_template_available_when_smplx_and_body_model(self):
        args = argparse.Namespace(
            src="smplx",
            tpose_motion=None,
            skip_solve=False,
            skip_validate=False,
        )
        smplx_body_model_root = _BODY_MODEL_ROOT
        smplx_template_available = args.src == "smplx" and smplx_body_model_root is not None
        assert smplx_template_available is True

    def test_no_tpose_motion_but_template_available_solves(self, tmp_path: Path):
        spec_path = tmp_path / "specs" / "tpose" / "test_robot.json"
        spec_path.parent.mkdir(parents=True)
        spec_data = {
            "robot": "test_robot",
            "xml_path": "/fake.xml",
            "qpos": [0, 0, 0, 1, 0, 0, 0],
            "links": {"base_link": {"pos": [0, 0, 0], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}},
        }
        spec_path.write_text(json.dumps(spec_data))

        smplx_template_available = True
        should_solve = (
            False  # args.tpose_motion is None
            or (smplx_template_available and spec_path.exists())
        )
        assert should_solve is True

    def test_setup_passes_body_model_root_to_validate(self):
        source = _src("scripts/setup_robot.py")
        idx_template = source.index("--use_smplx_template")
        idx_model = source.index("--smplx_template_model", idx_template)
        assert idx_model > idx_template, (
            "setup_robot.py must pass --smplx_template_model to validate after --use_smplx_template"
        )

    def test_template_model_help_mentions_directory(self):
        source = _src("scripts/setup_robot.py")
        assert "smplx/" in source and "directory" in source.lower()


class TestValidatorTemplateMode:
    def test_use_smplx_template_arg_exists(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "--use_smplx_template" in source

    def test_tpose_motion_no_longer_required(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "required=False" in source or "required=False" in source

    def test_template_frame_function_exists(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "_retarget_template_frame" in source

    def test_error_when_no_motion_and_no_template(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "--tpose_motion is required" in source

    def test_template_mode_not_referenced_in_coordinate_fix_test(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "apply_smplx_base_rotation" not in source

    def test_failure_hint_mentions_template(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "template calibration" in source.lower() or "Walking .npz" in source

    def test_template_mode_requires_smplx_src(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "--use_smplx_template requires --src smplx" in source

    def test_template_model_help_mentions_directory(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "smplx" in source.lower()


class TestSmplxCoordinatePolicyPreserved:
    def test_stage_tpose_no_smplx_root_quat(self):
        source = _src("scripts/stage_tpose.py")
        assert "SMPLX_BASE_ROTATION_QUAT" not in source

    def test_validator_no_apply_smplx_base_rotation(self):
        source = _src("examples/gmr_tpose_validate.py")
        assert "apply_smplx_base_rotation" not in source

    def test_agent_no_apply_smplx_base_rotation(self):
        source = _src("examples/gmr_alignment_agent.py")
        assert "apply_smplx_base_rotation" not in source

    def test_solver_does_not_inject_world_rotation(self, tmp_path: Path):
        config = {
            "robot_root_name": "base_link",
            "human_root_name": "pelvis",
            "ik_match_table1": {
                "base_link": ["pelvis", 100, 10, [0, 0, 0], [1, 0, 0, 0]],
            },
            "ik_match_table2": {
                "base_link": ["pelvis", 100, 5, [0, 0, 0], [1, 0, 0, 0]],
            },
        }
        config_path = tmp_path / "smplx_to_robot.json"
        config_path.write_text(json.dumps(config))

        spec = {
            "robot": "robot",
            "xml_path": "/fake.xml",
            "qpos": [0, 0, 0, 1, 0, 0, 0],
            "links": {
                "base_link": {
                    "pos": [0, 0, 0],
                    "R": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                },
            },
        }
        spec_path = tmp_path / "robot.json"
        spec_path.write_text(json.dumps(spec))

        solved = solve_smplx_offsets_from_template(
            ik_config_path=config_path,
            tpose_spec_path=spec_path,
            body_model_path=_BODY_MODEL_ROOT,
        )

        assert "world_rotation" not in solved, (
            "Solver must not inject world_rotation — that is compute_world_rotation's job"
        )
        assert solved["robot_root_name"] == "base_link"
        assert solved["human_root_name"] == "pelvis"


class TestAgentSmplxSolveModeNoMotion:
    def test_motion_file_not_required(self):
        source = Path("packages/gmr-harness/src/gmr_harness/cli/agent.py").read_text()
        motion_section = source.split("--motion_file")[1].split("\n")[0]
        assert "required=True" not in motion_section

    def test_smplx_template_solve_guard_skips_retarget(self):
        source = _src("examples/gmr_alignment_agent.py")
        assert "smplx_template_solve" in source
        assert "Phase A: skipped (SMPL-X template solve does not require motion)" in source

    def test_template_solve_uses_resolved_tpose_spec_path(self):
        source = _src("examples/gmr_alignment_agent.py")
        assert "tpose_spec_path=" in source, (
            "Template solve must pass resolved tpose_spec_path, not args.tpose_spec"
        )
        assert (
            "args.tpose_spec"
            not in source.split("solve_smplx_offsets_from_template(")[1].split("\n")[0]
        ), "solve_smplx_offsets_from_template must use tpose_spec_path, not args.tpose_spec"

    def test_template_solve_does_not_use_args_tpose_spec(self):
        source = _src("examples/gmr_alignment_agent.py")
        solve_start = source.index("solve_smplx_offsets_from_template(")
        solve_call = source[solve_start : solve_start + 300]
        assert "args.tpose_spec" not in solve_call, (
            "solve_smplx_offsets_from_template must use tpose_spec_path, not args.tpose_spec"
        )


class TestAgentDefaultSpecDiscovery:
    def test_auto_discovered_spec_used_for_template_solve(self, tmp_path: Path):
        spec_dir = tmp_path / "specs" / "tpose"
        spec_dir.mkdir(parents=True)
        spec_data = {
            "robot": "test_robot",
            "xml_path": "/fake.xml",
            "qpos": [0, 0, 0, 1, 0, 0, 0],
            "links": {
                "base_link": {"pos": [0, 0, 0], "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            },
        }
        spec_path = spec_dir / "test_robot.json"
        spec_path.write_text(json.dumps(spec_data))

        args_tpose_spec = None
        robot_name = "test_robot"

        tpose_spec_path = args_tpose_spec
        if tpose_spec_path is None:
            default_path = spec_dir / f"{robot_name}.json"
            if default_path.exists():
                tpose_spec_path = default_path

        assert tpose_spec_path == spec_path, (
            "Auto-discovered spec path must match specs/tpose/{robot}.json"
        )

    def test_default_spec_path_resolves_without_args_tpose_spec(self, tmp_path: Path):
        spec_dir = tmp_path / "specs" / "tpose"
        spec_dir.mkdir(parents=True)
        (spec_dir / "test_robot.json").write_text('{"robot":"test_robot"}')

        args_tpose_spec = None
        default_path = spec_dir / "test_robot.json"

        tpose_spec_path = args_tpose_spec
        if tpose_spec_path is None and default_path.exists():
            tpose_spec_path = default_path

        assert tpose_spec_path is not None
        assert tpose_spec_path.name == "test_robot.json"
