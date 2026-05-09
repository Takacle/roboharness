"""Tests for SMPL-X T-pose coordinate consistency after loader-boundary refactor.

Validates:
1. stage_tpose.py does NOT apply SMPL-X base root quaternion (robot stays upright)
2. gmr_tpose_validate.py checks for identity root quaternion for SMPL-X
3. gmr_alignment_agent.py solve_mode uses raw spec without base rotation
4. compute_world_rotation uses geometry-based R_mat for SMPLX (not hardcoded base)
5. SMPLX post-conversion frame convention produces proper det>0 rotations
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R


def _read(rel_path: str) -> str:
    return Path(rel_path).read_text()


class TestStageTposeSmplxRootQuat:
    def test_smplx_tpose_does_not_modify_root(self):
        source = _read("scripts/stage_tpose.py")
        assert "SMPLX_BASE_ROTATION_QUAT" not in source, (
            "stage_tpose.py must not reference SMPLX_BASE_ROTATION_QUAT — "
            "frame conversion is handled via world_rotation in the IK config."
        )

    def test_smplx_root_quat_import_removed(self):
        source = _read("scripts/stage_tpose.py")
        assert "from roboharness._math_utils import SMPLX_BASE_ROTATION_QUAT" not in source


class TestValidatorSmplxIdentityCheck:
    def test_source_no_apply_smplx_base_rotation(self):
        source = _read("examples/gmr_tpose_validate.py")
        assert "apply_smplx_base_rotation" not in source, (
            "gmr_tpose_validate.py must not reference apply_smplx_base_rotation"
        )

    def test_source_checks_identity_root(self):
        source = _read("examples/gmr_tpose_validate.py")
        assert "identity" in source
        assert "stage_tpose.py --src smplx" in source

    def test_source_no_smplx_base_rotation_import(self):
        source = _read("examples/gmr_tpose_validate.py")
        assert "SMPLX_BASE_ROTATION_QUAT" not in source

    def test_source_prints_deviation_info(self):
        source = _read("examples/gmr_tpose_validate.py")
        assert "deviation from identity" in source


class TestAgentNoSmplxBaseRotation:
    def test_source_no_apply_smplx_base_rotation(self):
        source = _read("examples/gmr_alignment_agent.py")
        assert "apply_smplx_base_rotation" not in source, (
            "gmr_alignment_agent.py must not reference apply_smplx_base_rotation"
        )


@dataclass
class _FakeMatch:
    mapping: dict[str, str]


_MINIMAL_XML = textwrap.dedent("""\
    <mujoco>
      <worldbody>
        <body name="base_link" pos="0 0 0.66">
          <body name="spine" pos="0 0 0.3"/>
          <body name="left_hip" pos="0 0.1 -0.3"/>
          <body name="right_hip" pos="0 -0.1 -0.3"/>
          <body name="left_shoulder" pos="0.15 0 0.35"/>
          <body name="right_shoulder" pos="-0.15 0 0.35"/>
        </body>
      </worldbody>
    </mujoco>
""")

_MINIMAL_XML_SMPLX_ALIGNED = textwrap.dedent("""\
    <mujoco>
      <worldbody>
        <body name="base_link" pos="0 0 0.66">
          <body name="spine" pos="0 0 0.3"/>
          <body name="left_hip" pos="0 0.1 -0.3"/>
          <body name="right_hip" pos="0 -0.1 -0.3"/>
          <body name="left_shoulder" pos="0 0.15 0.35"/>
          <body name="right_shoulder" pos="0 -0.15 0.35"/>
        </body>
      </worldbody>
    </mujoco>
""")


class TestComputeWorldRotationSmplx:
    """Tests for compute_world_rotation with src_format='smplx' after refactor."""

    @pytest.fixture()
    def robot_xml(self, tmp_path: Path) -> Path:
        xml_path = tmp_path / "test_robot.xml"
        xml_path.write_text(_MINIMAL_XML)
        return xml_path

    @pytest.fixture()
    def robot_xml_smplx_aligned(self, tmp_path: Path) -> Path:
        xml_path = tmp_path / "test_robot_smplx.xml"
        xml_path.write_text(_MINIMAL_XML_SMPLX_ALIGNED)
        return xml_path

    def _make_match(self):
        return _FakeMatch(
            mapping={
                "root": "base_link",
                "left_hip": "left_hip",
                "right_hip": "right_hip",
                "left_shoulder": "left_shoulder",
                "right_shoulder": "right_shoulder",
                "spine": "spine",
            }
        )

    def test_smplx_returns_non_none_for_bvh_aligned_robot(self, robot_xml: Path):
        from roboharness.alignment.orientation_aligner import compute_world_rotation

        result = compute_world_rotation(robot_xml, self._make_match(), src_format="smplx")
        assert result is not None, (
            "For a robot with left=+X (BVH convention), SMPLX should need a rotation"
        )

    def test_smplx_returns_none_for_smplx_aligned_robot(self, robot_xml_smplx_aligned: Path):
        from roboharness.alignment.orientation_aligner import compute_world_rotation

        result = compute_world_rotation(
            robot_xml_smplx_aligned, self._make_match(), src_format="smplx"
        )
        assert result is None, (
            "SMPLX compute_world_rotation should return None for a robot whose "
            "geometry matches the SMPLX post-conversion frame (X=forward, Y=left, Z=up)"
        )

    def test_bvh_returns_none_for_identity_robot(self, robot_xml: Path):
        from roboharness.alignment.orientation_aligner import compute_world_rotation

        result = compute_world_rotation(robot_xml, self._make_match(), src_format="bvh")
        assert result is None, (
            "BVH compute_world_rotation should return None for an identity-geometry robot"
        )

    def test_smplx_frame_matrix_has_positive_det(self, robot_xml: Path):
        from roboharness.alignment.orientation_aligner import compute_world_rotation

        result = compute_world_rotation(robot_xml, self._make_match(), src_format="smplx")
        if result is None:
            return

        r = R.from_quat(np.asarray(result, dtype=np.float64), scalar_first=True)
        mat = r.as_matrix()
        assert np.linalg.det(mat) > 0, "SMPLX world_rotation must be a proper rotation (det > 0)"

    def test_smplx_axis_mapping_for_bvh_aligned_robot(self, robot_xml: Path):
        """R * e_forward == robot_forward, R * e_left == robot_left, R * e_up == robot_up."""
        from roboharness.alignment.orientation_aligner import compute_world_rotation

        result = compute_world_rotation(robot_xml, self._make_match(), src_format="smplx")
        if result is None:
            pytest.skip("SMPLX returned None for this robot geometry")

        r = R.from_quat(np.asarray(result, dtype=np.float64), scalar_first=True)
        mat = r.as_matrix()

        up = mat @ np.array([0.0, 0.0, 1.0])
        assert np.dot(up, np.array([0.0, 0.0, 1.0])) > 0.9, (
            f"SMPLX R*[0,0,1] should map close to robot up (+Z), got {up}"
        )

        lft = mat @ np.array([0.0, 1.0, 0.0])
        assert np.dot(lft, np.array([1.0, 0.0, 0.0])) > 0.9, (
            f"SMPLX R*[0,1,0] should map close to robot left (+X), got {lft}"
        )

        fwd = mat @ np.array([1.0, 0.0, 0.0])
        assert np.dot(fwd, np.array([0.0, -1.0, 0.0])) > 0.9, (
            f"SMPLX R*[1,0,0] should map close to robot forward (-Y), got {fwd}"
        )

    def test_smplx_does_not_use_hardcoded_base_rotation(self, robot_xml: Path):
        source = Path("src/roboharness/alignment/orientation_aligner.py").read_text()
        assert "smpl_to_mujoco_world_rotation" not in source, (
            "orientation_aligner.py must not import smpl_to_mujoco_world_rotation"
        )

    def test_no_direct_smplx_base_import_in_compute_world_rotation(self):
        source = Path("src/roboharness/alignment/orientation_aligner.py").read_text()
        assert "from roboharness._math_utils import SMPLX_BASE_ROTATION_QUAT" not in source, (
            "orientation_aligner.py must not import SMPLX_BASE_ROTATION_QUAT directly"
        )
