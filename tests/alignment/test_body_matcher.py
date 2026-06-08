"""Tests for roboharness.alignment.body_matcher."""

from __future__ import annotations

from typing import ClassVar

from roboharness.alignment.body_matcher import match_bodies
from roboharness.alignment.skeleton_maps import BVH_SKELETON, SMPLX_SKELETON


class TestUnitreeH1:
    BODIES: ClassVar[list[str]] = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "left_knee_link",
        "left_ankle_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "right_knee_link",
        "right_ankle_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
    ]

    def test_smplx_root(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("root") == "pelvis"

    def test_smplx_spine(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("spine") == "torso_link"

    def test_smplx_left_hip(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("left_hip") in self.BODIES

    def test_smplx_left_knee(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("left_knee") == "left_knee_link"

    def test_smplx_left_foot(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("left_foot") in self.BODIES

    def test_smplx_left_shoulder(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("left_shoulder") in self.BODIES

    def test_smplx_left_elbow(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("left_elbow") == "left_elbow_link"

    def test_smplx_right_symmetry(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        for role in ("hip", "knee", "foot", "shoulder", "elbow"):
            assert f"left_{role}" in r.mapping, f"missing left_{role}"
            assert f"right_{role}" in r.mapping, f"missing right_{role}"
        # H1 has no wrist links — this is expected
        assert "left_wrist" in r.unmatched_roles


class TestUnitreeG1:
    BODIES: ClassVar[list[str]] = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "left_toe_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "right_toe_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_pitch_link",
        "right_wrist_yaw_link",
    ]

    def test_smplx_all_roles_matched(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert not r.unmatched_roles, f"Unmatched: {r.unmatched_roles}"

    def test_smplx_root(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping["root"] == "pelvis"


class TestBoosterT1:
    BODIES: ClassVar[list[str]] = [
        "Trunk",
        "H1",
        "H2",
        "AL1",
        "AL2",
        "AL3",
        "left_hand_link",
        "AR1",
        "AR2",
        "AR3",
        "right_hand_link",
        "Waist",
        "Hip_Pitch_Left",
        "Hip_Roll_Left",
        "Hip_Yaw_Left",
        "Shank_Left",
        "Ankle_Cross_Left",
        "left_foot_link",
        "Hip_Pitch_Right",
        "Hip_Roll_Right",
        "Hip_Yaw_Right",
        "Shank_Right",
        "Ankle_Cross_Right",
        "right_foot_link",
    ]

    def test_smplx_root(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON, root_body_hint="Waist")
        assert r.mapping["root"] == "Waist"

    def test_smplx_spine(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON, root_body_hint="Waist")
        assert r.mapping.get("spine") == "Trunk"

    def test_smplx_left_shoulder(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON, root_body_hint="Waist")
        assert r.mapping.get("left_shoulder") == "AL2"

    def test_smplx_left_elbow(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON, root_body_hint="Waist")
        assert r.mapping.get("left_elbow") == "AL3"

    def test_smplx_left_knee(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON, root_body_hint="Waist")
        assert r.mapping.get("left_knee") == "Shank_Left"


class TestFourierN1:
    BODIES: ClassVar[list[str]] = [
        "base_link",
        "left_thigh_pitch_link",
        "left_thigh_roll_link",
        "left_thigh_yaw_link",
        "left_shank_pitch_link",
        "left_foot_roll_link",
        "left_foot_pitch_link",
        "right_thigh_pitch_link",
        "right_thigh_roll_link",
        "right_thigh_yaw_link",
        "right_shank_pitch_link",
        "right_foot_roll_link",
        "right_foot_pitch_link",
        "imu_link",
        "waist_yaw_link",
        "torso_link",
        "left_upper_arm_pitch_link",
        "left_upper_arm_roll_link",
        "left_upper_arm_yaw_link",
        "left_lower_arm_pitch_link",
        "left_hand_yaw_link",
        "left_end_effector_link",
        "right_upper_arm_pitch_link",
        "right_upper_arm_roll_link",
        "right_upper_arm_yaw_link",
        "right_lower_arm_pitch_link",
        "right_hand_yaw_link",
        "right_end_effector_link",
    ]

    def test_smplx_root(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping["root"] == "base_link"

    def test_smplx_left_hip(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("left_hip") in self.BODIES

    def test_smplx_left_shoulder(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("left_shoulder") in self.BODIES


class TestKuavoS45:
    BODIES: ClassVar[list[str]] = [
        "base_link",
        "leg_l1_link",
        "leg_l2_link",
        "leg_l3_link",
        "leg_l4_link",
        "leg_l5_link",
        "leg_l6_link",
        "leg_r1_link",
        "leg_r2_link",
        "leg_r3_link",
        "leg_r4_link",
        "leg_r5_link",
        "leg_r6_link",
        "zarm_l1_link",
        "zarm_l2_link",
        "zarm_l3_link",
        "zarm_l4_link",
        "zarm_l5_link",
        "zarm_l6_link",
        "zarm_l7_link",
        "zarm_r1_link",
        "zarm_r2_link",
        "zarm_r3_link",
        "zarm_r4_link",
        "zarm_r5_link",
        "zarm_r6_link",
        "zarm_r7_link",
        "zhead_1_link",
        "zhead_2_link",
    ]

    def test_smplx_root(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping["root"] == "base_link"

    def test_smplx_left_shoulder(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON)
        assert r.mapping.get("left_shoulder") == "zarm_l2_link"


class TestOverrides:
    BODIES: ClassVar[list[str]] = ["base_link", "leg_a", "leg_b", "arm_x", "arm_y"]

    def test_override_takes_priority(self):
        r = match_bodies(
            self.BODIES,
            SMPLX_SKELETON,
            overrides={"left_hip": "leg_a"},
        )
        assert r.mapping.get("left_hip") == "leg_a"

    def test_root_hint(self):
        r = match_bodies(self.BODIES, SMPLX_SKELETON, root_body_hint="base_link")
        assert r.mapping["root"] == "base_link"


class TestEngineAiNaming:
    BODIES: ClassVar[list[str]] = [
        "LINK_BASE",
        "LINK_HIP_PITCH_L",
        "LINK_HIP_ROLL_L",
        "LINK_HIP_YAW_L",
        "LINK_KNEE_PITCH_L",
        "LINK_ANKLE_PITCH_L",
        "LINK_ANKLE_ROLL_L",
        "LINK_FOOT_L",
        "LINK_HIP_PITCH_R",
        "LINK_HIP_ROLL_R",
        "LINK_HIP_YAW_R",
        "LINK_KNEE_PITCH_R",
        "LINK_ANKLE_PITCH_R",
        "LINK_ANKLE_ROLL_R",
        "LINK_FOOT_R",
        "LINK_TORSO_YAW",
        "LINK_SHOULDER_PITCH_L",
        "LINK_SHOULDER_ROLL_L",
        "LINK_SHOULDER_YAW_L",
        "LINK_ELBOW_PITCH_L",
        "LINK_ELBOW_YAW_L",
        "LINK_ELBOW_END_L",
        "LINK_SHOULDER_PITCH_R",
        "LINK_SHOULDER_ROLL_R",
        "LINK_SHOULDER_YAW_R",
        "LINK_ELBOW_PITCH_R",
        "LINK_ELBOW_YAW_R",
        "LINK_ELBOW_END_R",
    ]

    def test_bvh_engineai_pm01_names_match_all_roles(self):
        r = match_bodies(self.BODIES, BVH_SKELETON)
        assert r.mapping == {
            "root": "LINK_BASE",
            "spine": "LINK_TORSO_YAW",
            "left_hip": "LINK_HIP_ROLL_L",
            "right_hip": "LINK_HIP_ROLL_R",
            "left_knee": "LINK_KNEE_PITCH_L",
            "right_knee": "LINK_KNEE_PITCH_R",
            "left_foot": "LINK_ANKLE_ROLL_L",
            "right_foot": "LINK_ANKLE_ROLL_R",
            "left_shoulder": "LINK_SHOULDER_ROLL_L",
            "right_shoulder": "LINK_SHOULDER_ROLL_R",
            "left_elbow": "LINK_ELBOW_PITCH_L",
            "right_elbow": "LINK_ELBOW_PITCH_R",
            "left_wrist": "LINK_ELBOW_END_L",
            "right_wrist": "LINK_ELBOW_END_R",
        }
        assert r.unmatched_roles == []


class TestMatchResult:
    def test_empty_bodies(self):
        r = match_bodies([], SMPLX_SKELETON)
        assert r.mapping.get("root") is None
        assert len(r.unmatched_roles) > 0

    def test_unmatched_bodies_tracked(self):
        bodies = ["pelvis", "left_knee_link", "torso_link", "mystery_body"]
        r = match_bodies(bodies, SMPLX_SKELETON)
        assert "mystery_body" in r.unmatched_bodies
