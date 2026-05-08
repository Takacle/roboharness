"""Tests for SMPL-X T-pose coordinate consistency fix.

Validates:
1. stage_tpose.py applies SMPL-X base root quaternion when --src smplx
2. gmr_tpose_validate.py does NOT call apply_smplx_base_rotation(spec)
3. gmr_alignment_agent.py solve_mode uses raw spec without base rotation
"""

from __future__ import annotations

from pathlib import Path


def _read(rel_path: str) -> str:
    return Path(rel_path).read_text()


class TestStageTposeSmplxRootQuat:
    def test_smplx_tpose_source_has_root_quat_logic(self):
        source = _read("scripts/stage_tpose.py")
        assert 'args.src == "smplx"' in source
        assert "SMPLX_BASE_ROTATION_QUAT" in source
        assert "qpos[3:7]" in source
        assert "args.qpos is None" in source
        assert "args.qpos_file is None" in source
        assert "model.jnt_type[0]" in source

    def test_smplx_root_quat_not_applied_for_explicit_qpos(self):
        source = _read("scripts/stage_tpose.py")
        idx_smplx = source.index('args.src == "smplx"')
        idx_qpos_none = source.index("args.qpos is None", idx_smplx)
        idx_qpos_file_none = source.index("args.qpos_file is None", idx_smplx)
        assert idx_qpos_none < idx_qpos_file_none
        idx_assignment = source.index("qpos[3:7]", idx_smplx)
        assert idx_qpos_none < idx_assignment
        assert idx_qpos_file_none < idx_assignment

    def test_smplx_root_quat_imports_constant(self):
        source = _read("scripts/stage_tpose.py")
        assert "from roboharness._math_utils import SMPLX_BASE_ROTATION_QUAT" in source

    def test_smplx_root_quat_logs_message(self):
        source = _read("scripts/stage_tpose.py")
        assert "Applied SMPL-X base root quaternion" in source


class TestValidatorNoSmplxBaseRotation:
    def test_source_no_apply_smplx_base_rotation(self):
        source = _read("examples/gmr_tpose_validate.py")
        assert "apply_smplx_base_rotation" not in source, (
            "gmr_tpose_validate.py must not reference apply_smplx_base_rotation"
        )

    def test_source_has_smplx_diagnostics(self):
        source = _read("examples/gmr_tpose_validate.py")
        assert "SMPL-X root quaternion angle" in source
        assert "stage_tpose.py --src smplx" in source

    def test_source_has_smplx_failure_hint(self):
        source = _read("examples/gmr_tpose_validate.py")
        assert "SMPL-X large-angle failure hint" in source
        assert "stage_tpose.py --src smplx" in source


class TestAgentNoSmplxBaseRotation:
    def test_source_no_apply_smplx_base_rotation(self):
        source = _read("examples/gmr_alignment_agent.py")
        assert "apply_smplx_base_rotation" not in source, (
            "gmr_alignment_agent.py must not reference apply_smplx_base_rotation"
        )
