"""Tests for SMPL-X coordinate conversion module.

Validates:
- SMPL_TO_MUJOCO_QUAT normalization and axis mapping
- smpl_to_mujoco_frame() transforms positions and orientations
- smpl_to_mujoco_world_rotation() returns correct quaternion
- Consistency with legacy SMPLX_BASE_ROTATION_QUAT
- classify_smplx_frame_convention() detects Y-up vs Z-up data
- Heading preservation through skip-conversion for AMASS data
- Offset contract: template-computed offsets work with AMASS frames
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from roboharness.alignment.smplx_coordinate import (
    SMPL_TO_MUJOCO_QUAT,
    classify_smplx_frame_convention,
    normalize_to_pelvis_z,
    smpl_to_mujoco_frame,
    smpl_to_mujoco_world_rotation,
    validate_smplx_runtime_config,
)


def _yup_frame(pelvis_quat=None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if pelvis_quat is None:
        pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0])
    return {
        "pelvis": (np.array([0.0, 0.9, 0.1]), pelvis_quat),
        "head": (np.array([0.0, 1.7, 0.05]), np.array([1.0, 0.0, 0.0, 0.0])),
    }


def _zup_frame(heading_deg: float = 0.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    t = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
    heading = R.from_rotvec([0.0, 0.0, np.radians(heading_deg)])
    pelvis_q = (heading * t).as_quat(scalar_first=True)
    pelvis_pos = np.array([0.1, 0.05, 0.95])
    head_q = pelvis_q.copy()
    head_pos = np.array([0.1, 0.05, 1.75])
    return {
        "pelvis": (pelvis_pos, pelvis_q),
        "head": (head_pos, head_q),
    }


class TestSmplToMujocoQuat:
    def test_is_normalized(self):
        norm = float(np.linalg.norm(SMPL_TO_MUJOCO_QUAT))
        assert abs(norm - 1.0) < 1e-10

    def test_maps_up_y_to_z(self):
        r = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        np.testing.assert_allclose(r.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-8)

    def test_maps_left_x_to_y(self):
        r = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        np.testing.assert_allclose(r.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-8)

    def test_maps_forward_z_to_x(self):
        r = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        np.testing.assert_allclose(r.apply([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-8)

    def test_legacy_inverse_consistency(self):
        from roboharness._math_utils import SMPLX_BASE_ROTATION_QUAT

        r_legacy = R.from_quat(SMPLX_BASE_ROTATION_QUAT, scalar_first=True).inv()
        r_new = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        np.testing.assert_allclose(
            r_legacy.as_quat(scalar_first=True),
            r_new.as_quat(scalar_first=True),
            atol=1e-10,
        )

    def test_identity_quat_stays_identity(self):
        r = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = (r * R.from_quat(q_identity, scalar_first=True)).as_quat(scalar_first=True)
        np.testing.assert_allclose(result, SMPL_TO_MUJOCO_QUAT, atol=1e-8)


class TestSmplToMujocoFrame:
    def test_transforms_positions(self):
        frame = {
            "pelvis": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            "head": (np.array([0.0, 1.8, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        result = smpl_to_mujoco_frame(frame)
        np.testing.assert_allclose(result["pelvis"][0], [0.0, 0.0, 1.0], atol=1e-8)
        np.testing.assert_allclose(result["head"][0], [0.0, 0.0, 1.8], atol=1e-8)

    def test_transforms_orientations(self):
        frame = {
            "pelvis": (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        result = smpl_to_mujoco_frame(frame)
        _, q = result["pelvis"]
        np.testing.assert_allclose(q, SMPL_TO_MUJOCO_QUAT, atol=1e-8)

    def test_preserves_number_of_joints(self):
        frame = {f"joint_{i}": (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])) for i in range(10)}
        result = smpl_to_mujoco_frame(frame)
        assert len(result) == 10

    def test_does_not_modify_input(self):
        pos_orig = np.array([1.0, 2.0, 3.0])
        quat_orig = np.array([1.0, 0.0, 0.0, 0.0])
        frame = {"joint": (pos_orig.copy(), quat_orig.copy())}
        smpl_to_mujoco_frame(frame)
        np.testing.assert_array_equal(frame["joint"][0], pos_orig)
        np.testing.assert_array_equal(frame["joint"][1], quat_orig)

    def test_all_quaternions_normalized(self):
        frame = {
            "a": (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
            "b": (np.zeros(3), np.array([0.707, 0.707, 0.0, 0.0])),
        }
        result = smpl_to_mujoco_frame(frame)
        for name, (_, q) in result.items():
            norm = float(np.linalg.norm(q))
            assert abs(norm - 1.0) < 1e-6, f"{name} quat norm = {norm}"


class TestSmplToMujocoWorldRotation:
    def test_returns_runtime_quat(self):
        wr = smpl_to_mujoco_world_rotation()
        assert wr == SMPL_TO_MUJOCO_QUAT

    def test_is_list_of_floats(self):
        wr = smpl_to_mujoco_world_rotation()
        assert isinstance(wr, list)
        assert len(wr) == 4
        assert all(isinstance(v, float) for v in wr)


class TestClassifySmplxFrameConvention:
    def test_classifies_yup_native_data(self):
        frames = [_yup_frame() for _ in range(5)]
        assert classify_smplx_frame_convention(frames) == "y"

    def test_classifies_zup_amass_data(self):
        frames = [_zup_frame(heading_deg=45.0) for _ in range(5)]
        assert classify_smplx_frame_convention(frames) == "z"

    def test_classifies_zup_with_heading_180(self):
        frames = [_zup_frame(heading_deg=180.0) for _ in range(5)]
        assert classify_smplx_frame_convention(frames) == "z"

    def test_classifies_zup_with_negative_heading(self):
        frames = [_zup_frame(heading_deg=-90.0) for _ in range(5)]
        assert classify_smplx_frame_convention(frames) == "z"

    def test_raises_on_empty_frames(self):
        with pytest.raises(RuntimeError, match="No SMPLX frames"):
            classify_smplx_frame_convention([])

    def test_raises_on_missing_pelvis(self):
        frames = [{"head": (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))}]
        with pytest.raises(KeyError, match="pelvis"):
            classify_smplx_frame_convention(frames)

    def test_raises_on_invalid_quaternion_norm(self):
        bad_frame = {
            "pelvis": (np.zeros(3), np.array([0.5, 0.0, 0.0, 0.0])),
        }
        with pytest.raises(ValueError, match="invalid norm"):
            classify_smplx_frame_convention([bad_frame])

    def test_raises_on_ambiguous_convention(self):
        pelvis_q = R.from_rotvec([np.pi / 4, 0.0, 0.0]).as_quat(scalar_first=True)
        frames = [
            {"pelvis": (np.zeros(3), pelvis_q)},
        ]
        with pytest.raises(ValueError, match="Ambiguous"):
            classify_smplx_frame_convention(frames, max_samples=1)

    def test_mixed_frames_raises_ambiguous(self):
        mixed = [_yup_frame() for _ in range(3)] + [_zup_frame() for _ in range(3)]
        with pytest.raises(ValueError, match="Ambiguous"):
            classify_smplx_frame_convention(mixed, max_samples=6)

    def test_respects_max_samples(self):
        frames = [_yup_frame() for _ in range(2)] + [_zup_frame() for _ in range(50)]
        result = classify_smplx_frame_convention(frames, max_samples=2)
        assert result == "y"


class TestHeadingPreservation:
    def test_amass_heading_preserved_without_conversion(self):
        heading_deg = 123.0
        frame = _zup_frame(heading_deg=heading_deg)
        q_before = frame["pelvis"][1].copy()

        heading_r = R.from_rotvec([0.0, 0.0, np.radians(heading_deg)])
        t = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        expected_q = (heading_r * t).as_quat(scalar_first=True)
        np.testing.assert_allclose(q_before, expected_q, atol=1e-10)

    def test_heading_survives_offset_application(self):
        heading_deg = 45.0
        frame = _zup_frame(heading_deg=heading_deg)
        q_amass = frame["pelvis"][1]

        t = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        r_target = R.identity()
        r_offset = t.inv() * r_target

        q_robot = (R.from_quat(q_amass, scalar_first=True) * r_offset).as_quat(scalar_first=True)
        heading_r = R.from_rotvec([0.0, 0.0, np.radians(heading_deg)])
        expected = (heading_r * r_target).as_quat(scalar_first=True)
        np.testing.assert_allclose(q_robot, expected, atol=1e-6)

    def test_different_headings_produce_different_poses(self):
        f0 = _zup_frame(heading_deg=0.0)
        f90 = _zup_frame(heading_deg=90.0)
        q0 = f0["pelvis"][1]
        q90 = f90["pelvis"][1]
        assert not np.allclose(q0, q90)

    def test_yup_conversion_preserves_identity_heading(self):
        frame = _yup_frame()
        result = smpl_to_mujoco_frame(frame)
        q_result = result["pelvis"][1]
        r_result = R.from_quat(q_result, scalar_first=True)
        local_y = r_result.apply([0.0, 1.0, 0.0])
        np.testing.assert_allclose(local_y, [0.0, 0.0, 1.0], atol=1e-6)


class TestOffsetContractWithAmassData:
    def test_amass_offsets_produce_correct_robot_pose(self):
        t = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        r_target = R.from_rotvec([0.1, 0.2, 0.3])

        template_pelvis_q = t.as_quat(scalar_first=True)
        r_offset = R.from_quat(template_pelvis_q, scalar_first=True).inv() * r_target

        heading_deg = 90.0
        amass_frame = _zup_frame(heading_deg=heading_deg)
        amass_pelvis_q = amass_frame["pelvis"][1]

        q_robot = (R.from_quat(amass_pelvis_q, scalar_first=True) * r_offset).as_quat(
            scalar_first=True
        )

        heading_r = R.from_rotvec([0.0, 0.0, np.radians(heading_deg)])
        expected = (heading_r * r_target).as_quat(scalar_first=True)
        np.testing.assert_allclose(q_robot, expected, atol=1e-6)

    def test_zero_heading_amass_gives_target_rest(self):
        t = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        r_target = R.from_rotvec([0.1, 0.2, 0.3])
        r_offset = t.inv() * r_target

        amass_frame = _zup_frame(heading_deg=0.0)
        amass_pelvis_q = amass_frame["pelvis"][1]

        q_robot = (R.from_quat(amass_pelvis_q, scalar_first=True) * r_offset).as_quat(
            scalar_first=True
        )
        expected = r_target.as_quat(scalar_first=True)
        np.testing.assert_allclose(q_robot, expected, atol=1e-6)

    def test_amass_position_offset_is_world_consistent(self):
        pos_template = np.array([0.0, 0.0, 0.95])
        pos_amass = np.array([0.1, 0.05, 0.95])

        r_target = R.identity()
        pos_target = np.array([0.0, 0.0, 0.9])

        pos_offset_template = r_target.inv().apply(pos_target - pos_template)
        pos_offset_amass = r_target.inv().apply(pos_target - pos_amass)

        np.testing.assert_allclose(pos_offset_template, [0.0, 0.0, -0.05], atol=1e-6)
        assert pos_offset_template[2] < 0
        assert abs(pos_offset_amass[2] - (-0.05)) < 0.15


class TestOrientationAlignerUsesConverter:
    def test_compute_world_rotation_does_not_use_smplx_world_rotation(self, tmp_path: Path):
        source = Path("src/roboharness/alignment/orientation_aligner.py").read_text()
        assert "smpl_to_mujoco_world_rotation" not in source, (
            "orientation_aligner.py must not call smpl_to_mujoco_world_rotation"
        )

    def test_no_direct_smplx_base_import_in_compute_world_rotation(self):
        source = Path("src/roboharness/alignment/orientation_aligner.py").read_text()
        assert "from roboharness._math_utils import SMPLX_BASE_ROTATION_QUAT" not in source, (
            "orientation_aligner.py must not import SMPLX_BASE_ROTATION_QUAT directly"
        )


class TestSolverUsesPipeline:
    def test_solver_does_not_import_smpl_to_mujoco_frame(self):
        source = Path("src/roboharness/alignment/smplx_offset_solver.py").read_text()
        assert "smpl_to_mujoco_frame" not in source, (
            "smplx_offset_solver.py must not import smpl_to_mujoco_frame — "
            "the template loader returns Z-up directly"
        )

    def test_solver_imports_scale_module(self):
        source = Path("src/roboharness/alignment/smplx_offset_solver.py").read_text()
        assert "apply_human_scale" in source, (
            "smplx_offset_solver.py must use apply_human_scale from pipeline"
        )

    def test_solver_does_not_import_legacy_constant(self):
        source = Path("src/roboharness/alignment/smplx_offset_solver.py").read_text()
        assert "SMPLX_BASE_ROTATION_QUAT" not in source, (
            "smplx_offset_solver.py must not import SMPLX_BASE_ROTATION_QUAT"
        )

    def test_solver_does_not_inject_world_rotation(self):
        source = Path("src/roboharness/alignment/smplx_offset_solver.py").read_text()
        assert "smpl_to_mujoco_world_rotation" not in source, (
            "smplx_offset_solver.py must not import smpl_to_mujoco_world_rotation"
        )


class TestValidateSmplxRuntimeConfig:
    def test_raises_on_legacy_base_world_rotation(self):
        config = {"world_rotation": [0.5, 0.5, 0.5, 0.5]}
        with pytest.raises(ValueError, match="legacy base"):
            validate_smplx_runtime_config(config, "smplx_to_test.json")

    def test_passes_on_none_world_rotation(self):
        config = {"world_rotation": None}
        validate_smplx_runtime_config(config, "smplx_to_test.json")

    def test_passes_on_geometry_based_world_rotation(self):
        config = {"world_rotation": [0.707, 0.0, 0.0, 0.707]}
        validate_smplx_runtime_config(config, "smplx_to_test.json")

    def test_passes_when_no_world_rotation_key(self):
        config = {}
        validate_smplx_runtime_config(config, "smplx_to_test.json")

    def test_passes_when_converted_at_loader_false(self):
        config = {"world_rotation": [0.5, 0.5, 0.5, 0.5]}
        validate_smplx_runtime_config(config, "smplx_to_test.json", converted_at_loader=False)


class TestNormalizeToPelvisZ:
    def test_pelvis_becomes_zero(self):
        frame = {
            "pelvis": (np.array([0.5, -0.2, 0.95]), np.array([1.0, 0.0, 0.0, 0.0])),
            "head": (np.array([0.5, -0.2, 1.72]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        normalize_to_pelvis_z(frame)
        np.testing.assert_allclose(frame["pelvis"][0], [0.5, -0.2, 0.0], atol=1e-8)

    def test_child_joints_shifted(self):
        frame = {
            "pelvis": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            "knee": (np.array([0.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.0, 0.0])),
            "foot": (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        normalize_to_pelvis_z(frame)
        np.testing.assert_allclose(frame["pelvis"][0][2], 0.0, atol=1e-8)
        np.testing.assert_allclose(frame["knee"][0][2], -0.5, atol=1e-8)
        np.testing.assert_allclose(frame["foot"][0][2], -1.0, atol=1e-8)

    def test_explicit_pelvis_z_param(self):
        frame = {
            "pelvis": (np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0])),
            "head": (np.array([0.1, 0.2, 1.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        normalize_to_pelvis_z(frame, pelvis_z=0.3)
        np.testing.assert_allclose(frame["pelvis"][0][2], 0.0, atol=1e-8)
        np.testing.assert_allclose(frame["head"][0][2], 0.7, atol=1e-8)

    def test_quaternions_unchanged(self):
        q = np.array([0.707, 0.707, 0.0, 0.0])
        frame = {
            "pelvis": (np.array([0.0, 0.0, 0.95]), q),
        }
        normalize_to_pelvis_z(frame)
        np.testing.assert_array_equal(frame["pelvis"][1], q)

    def test_x_y_unchanged(self):
        frame = {
            "pelvis": (np.array([1.5, -2.5, 0.95]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        normalize_to_pelvis_z(frame)
        np.testing.assert_allclose(frame["pelvis"][0][0], 1.5, atol=1e-8)
        np.testing.assert_allclose(frame["pelvis"][0][1], -2.5, atol=1e-8)

    def test_no_op_on_zero_pelvis(self):
        frame = {
            "pelvis": (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            "knee": (np.array([0.0, 0.0, -0.5]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        knee_z_before = frame["knee"][0][2].copy()
        normalize_to_pelvis_z(frame)
        np.testing.assert_allclose(frame["knee"][0][2], knee_z_before, atol=1e-8)

    def test_no_op_when_missing_pelvis(self):
        frame = {
            "head": (np.array([0.0, 0.0, 1.7]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        head_z_before = frame["head"][0][2].copy()
        normalize_to_pelvis_z(frame)
        np.testing.assert_allclose(frame["head"][0][2], head_z_before, atol=1e-8)
