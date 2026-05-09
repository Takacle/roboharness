"""Tests for SMPL-X coordinate conversion module.

Validates:
- SMPL_TO_MUJOCO_QUAT normalization and axis mapping
- smpl_to_mujoco_frame() transforms positions and orientations
- smpl_to_mujoco_world_rotation() returns correct quaternion
- Consistency with legacy SMPLX_BASE_ROTATION_QUAT
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


class TestSmplToMujocoQuat:
    def test_is_normalized(self):
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        norm = float(np.linalg.norm(SMPL_TO_MUJOCO_QUAT))
        assert abs(norm - 1.0) < 1e-10

    def test_maps_up_y_to_z(self):
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        r = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        np.testing.assert_allclose(r.apply([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-8)

    def test_maps_left_x_to_y(self):
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        r = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        np.testing.assert_allclose(r.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-8)

    def test_maps_forward_z_to_x(self):
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        r = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        np.testing.assert_allclose(r.apply([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-8)

    def test_legacy_inverse_consistency(self):
        from roboharness._math_utils import SMPLX_BASE_ROTATION_QUAT
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        r_legacy = R.from_quat(SMPLX_BASE_ROTATION_QUAT, scalar_first=True).inv()
        r_new = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        np.testing.assert_allclose(
            r_legacy.as_quat(scalar_first=True),
            r_new.as_quat(scalar_first=True),
            atol=1e-10,
        )

    def test_identity_quat_stays_identity(self):
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        r = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
        q_identity = np.array([1.0, 0.0, 0.0, 0.0])
        result = (r * R.from_quat(q_identity, scalar_first=True)).as_quat(scalar_first=True)
        np.testing.assert_allclose(result, SMPL_TO_MUJOCO_QUAT, atol=1e-8)


class TestSmplToMujocoFrame:
    def test_transforms_positions(self):
        from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame

        frame = {
            "pelvis": (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
            "head": (np.array([0.0, 1.8, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        result = smpl_to_mujoco_frame(frame)
        np.testing.assert_allclose(result["pelvis"][0], [0.0, 0.0, 1.0], atol=1e-8)
        np.testing.assert_allclose(result["head"][0], [0.0, 0.0, 1.8], atol=1e-8)

    def test_transforms_orientations(self):
        from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame

        frame = {
            "pelvis": (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])),
        }
        result = smpl_to_mujoco_frame(frame)
        _, q = result["pelvis"]
        from roboharness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

        np.testing.assert_allclose(q, SMPL_TO_MUJOCO_QUAT, atol=1e-8)

    def test_preserves_number_of_joints(self):
        from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame

        frame = {f"joint_{i}": (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])) for i in range(10)}
        result = smpl_to_mujoco_frame(frame)
        assert len(result) == 10

    def test_does_not_modify_input(self):
        from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame

        pos_orig = np.array([1.0, 2.0, 3.0])
        quat_orig = np.array([1.0, 0.0, 0.0, 0.0])
        frame = {"joint": (pos_orig.copy(), quat_orig.copy())}
        smpl_to_mujoco_frame(frame)
        np.testing.assert_array_equal(frame["joint"][0], pos_orig)
        np.testing.assert_array_equal(frame["joint"][1], quat_orig)

    def test_all_quaternions_normalized(self):
        from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame

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
        from roboharness.alignment.smplx_coordinate import (
            SMPL_TO_MUJOCO_QUAT,
            smpl_to_mujoco_world_rotation,
        )

        wr = smpl_to_mujoco_world_rotation()
        assert wr == SMPL_TO_MUJOCO_QUAT

    def test_is_list_of_floats(self):
        from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_world_rotation

        wr = smpl_to_mujoco_world_rotation()
        assert isinstance(wr, list)
        assert len(wr) == 4
        assert all(isinstance(v, float) for v in wr)


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
        import pytest

        from roboharness.alignment.smplx_coordinate import validate_smplx_runtime_config

        config = {"world_rotation": [0.5, 0.5, 0.5, 0.5]}
        with pytest.raises(ValueError, match="legacy base"):
            validate_smplx_runtime_config(config, "smplx_to_test.json")

    def test_passes_on_none_world_rotation(self):
        from roboharness.alignment.smplx_coordinate import validate_smplx_runtime_config

        config = {"world_rotation": None}
        validate_smplx_runtime_config(config, "smplx_to_test.json")

    def test_passes_on_geometry_based_world_rotation(self):
        from roboharness.alignment.smplx_coordinate import validate_smplx_runtime_config

        config = {"world_rotation": [0.707, 0.0, 0.0, 0.707]}
        validate_smplx_runtime_config(config, "smplx_to_test.json")

    def test_passes_when_no_world_rotation_key(self):
        from roboharness.alignment.smplx_coordinate import validate_smplx_runtime_config

        config = {}
        validate_smplx_runtime_config(config, "smplx_to_test.json")

    def test_passes_when_converted_at_loader_false(self):
        from roboharness.alignment.smplx_coordinate import validate_smplx_runtime_config

        config = {"world_rotation": [0.5, 0.5, 0.5, 0.5]}
        validate_smplx_runtime_config(config, "smplx_to_test.json", converted_at_loader=False)
