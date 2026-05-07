"""Tests for roboharness.alignment.config_gen."""

from __future__ import annotations

import json

import pytest

from roboharness.alignment.body_matcher import MatchResult, match_bodies
from roboharness.alignment.config_gen import generate_ik_config, write_ik_config
from roboharness.alignment.skeleton_maps import BVH_SKELETON, SMPLX_SKELETON


class TestGenerateIkConfig:
    @pytest.fixture()
    def h1_match(self):
        bodies = [
            "pelvis", "left_hip_yaw_link", "left_hip_roll_link",
            "left_hip_pitch_link", "left_knee_link", "left_ankle_link",
            "right_hip_yaw_link", "right_hip_roll_link", "right_hip_pitch_link",
            "right_knee_link", "right_ankle_link", "torso_link",
            "left_shoulder_pitch_link", "left_shoulder_roll_link",
            "left_shoulder_yaw_link", "left_elbow_link",
            "right_shoulder_pitch_link", "right_shoulder_roll_link",
            "right_shoulder_yaw_link", "right_elbow_link",
        ]
        return match_bodies(bodies, SMPLX_SKELETON)

    def test_config_is_valid_json(self, h1_match):
        config = generate_ik_config(h1_match, SMPLX_SKELETON)
        text = json.dumps(config)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_root_name(self, h1_match):
        config = generate_ik_config(h1_match, SMPLX_SKELETON)
        assert config["robot_root_name"] == "pelvis"
        assert config["human_root_name"] == "pelvis"

    def test_both_tables_present(self, h1_match):
        config = generate_ik_config(h1_match, SMPLX_SKELETON)
        assert "ik_match_table1" in config
        assert "ik_match_table2" in config
        assert config["use_ik_match_table1"] is True
        assert config["use_ik_match_table2"] is True

    def test_table_entry_structure(self, h1_match):
        config = generate_ik_config(h1_match, SMPLX_SKELETON)
        for table_key in ("ik_match_table1", "ik_match_table2"):
            for body_name, entry in config[table_key].items():
                assert len(entry) == 5, f"{table_key}[{body_name}] has {len(entry)} items"
                human_bone, pos_w, rot_w, pos_off, quat = entry
                assert isinstance(human_bone, str)
                assert isinstance(pos_w, int)
                assert isinstance(rot_w, int)
                assert len(pos_off) == 3
                assert len(quat) == 4

    def test_scale_table_present(self, h1_match):
        config = generate_ik_config(h1_match, SMPLX_SKELETON)
        assert "human_scale_table" in config
        assert len(config["human_scale_table"]) > 0

    def test_identity_quaternions(self, h1_match):
        config = generate_ik_config(h1_match, SMPLX_SKELETON)
        for table_key in ("ik_match_table1", "ik_match_table2"):
            for _body, entry in config[table_key].items():
                assert entry[4] == [1.0, 0.0, 0.0, 0.0]

    def test_root_has_high_pos_weight(self, h1_match):
        config = generate_ik_config(h1_match, SMPLX_SKELETON)
        root_entry = config["ik_match_table1"].get("pelvis")
        assert root_entry is not None
        assert root_entry[1] > 0  # pos_weight > 0

    def test_foot_has_high_pos_weight(self, h1_match):
        config = generate_ik_config(h1_match, SMPLX_SKELETON)
        foot_entry = config["ik_match_table1"].get("left_ankle_link")
        assert foot_entry is not None
        assert foot_entry[1] > 0  # pos_weight > 0

    def test_bvh_skeleton_produces_bvh_names(self):
        bodies = [
            "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
            "torso_link", "left_shoulder_roll_link", "left_elbow_link",
            "right_shoulder_roll_link", "right_elbow_link",
        ]
        match = match_bodies(bodies, BVH_SKELETON)
        config = generate_ik_config(match, BVH_SKELETON)
        assert config["human_root_name"] == "Hips"
        for table_key in ("ik_match_table1", "ik_match_table2"):
            for _body, entry in config[table_key].items():
                assert entry[0] in BVH_SKELETON.scale_defaults


class TestWriteIkConfig:
    def test_writes_to_tmpdir(self, tmp_path):
        match = MatchResult(mapping={"root": "pelvis", "spine": "torso_link"})
        config = generate_ik_config(match, SMPLX_SKELETON)
        path = write_ik_config(config, "test_robot", "smplx", output_dir=tmp_path)
        assert path.exists()
        assert path.name == "smplx_to_test_robot.json"
        loaded = json.loads(path.read_text())
        assert loaded["robot_root_name"] == "pelvis"
