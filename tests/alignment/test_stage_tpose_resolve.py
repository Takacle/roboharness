"""Tests for stage_tpose robot/config resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_resolve_robot_falls_back_to_generated_robot_config(tmp_path, monkeypatch):
    import scripts.stage_tpose as stage_tpose

    gmr_root = tmp_path / "GMR"
    asset_dir = gmr_root / "assets" / "engineai_pm01"
    asset_dir.mkdir(parents=True)
    xml_path = asset_dir / "pm_v2.xml"
    xml_path.write_text(
        "<mujoco><worldbody>"
        '<body name="LINK_BASE">'
        '<body name="LINK_SHOULDER_ROLL_L"/>'
        '<body name="LINK_ELBOW_PITCH_L"/>'
        "</body>"
        "</worldbody></mujoco>"
    )

    ik_dir = gmr_root / "general_motion_retargeting" / "ik_configs"
    ik_dir.mkdir(parents=True)
    generated_config = ik_dir / "bvh_to_engineai_pm01.json"
    generated_config.write_text(
        '{"ik_match_table1": {"LINK_SHOULDER_ROLL_L": "LeftArm"}, '
        '"ik_match_table2": {"LINK_ELBOW_PITCH_L": "LeftForeArm"}}'
    )

    stale_config = ik_dir / "bvh_to_pm01.json"
    params = SimpleNamespace(
        ROBOT_XML_DICT={"engineai_pm01": xml_path},
        IK_CONFIG_DICT={"bvh": {"engineai_pm01": stale_config}},
        VIEWER_CAM_DISTANCE_DICT={"engineai_pm01": 3.0},
    )

    monkeypatch.setattr(stage_tpose, "GMR_ROOT", gmr_root)
    monkeypatch.setattr(stage_tpose, "load_gmr_params", lambda _root: params)

    resolved_xml, link_names, cam_distance = stage_tpose._resolve_robot("engineai_pm01", "bvh")

    assert resolved_xml == xml_path
    assert link_names == ["LINK_ELBOW_PITCH_L", "LINK_SHOULDER_ROLL_L"]
    assert cam_distance == 3.0


def test_qpos_input_cannot_be_same_as_output_spec(tmp_path):
    import scripts.stage_tpose as stage_tpose

    spec_path = tmp_path / "specs" / "tpose" / "engineai_pm01.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_text('{"qpos": [0.0]}')

    with pytest.raises(ValueError, match="must not be the same path"):
        stage_tpose._guard_qpos_input_not_output(spec_path, spec_path)


def test_qpos_input_can_be_separate_from_output_spec(tmp_path):
    import scripts.stage_tpose as stage_tpose

    qpos_source = tmp_path / "inputs" / "engineai_pm01_qpos.json"
    spec_path = tmp_path / "specs" / "tpose" / "engineai_pm01.json"

    stage_tpose._guard_qpos_input_not_output(qpos_source, spec_path)
