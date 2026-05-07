"""Match robot body names to human skeleton roles via heuristic rules.

Pure logic module — takes a list of body names and a ``HumanSkeleton``,
returns a ``MatchResult`` with the mapping and any unmatched items.
No IO, no external dependencies beyond ``skeleton_maps``.

The matching is based on regex patterns extracted from 18+ existing robots'
MuJoCo XML body naming conventions. When heuristics fail, the caller handles
interactive resolution using ``unmatched_roles`` from the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from roboharness.alignment.skeleton_maps import HumanSkeleton


@dataclass
class MatchResult:
    """Output of body-name matching."""

    mapping: dict[str, str] = field(default_factory=dict)
    unmatched_roles: list[str] = field(default_factory=list)
    unmatched_bodies: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Rule:
    """One candidate pattern for a given role."""

    pattern: str
    flags: int = re.IGNORECASE

    def matches(self, body_name: str) -> bool:
        return bool(re.search(self.pattern, body_name, self.flags))


_RULES: dict[str, list[_Rule]] = {
    "left_hip": [
        _Rule(r"^left_hip_roll_link$"),
        _Rule(r"^left_hip_yaw_link$"),
        _Rule(r"^left_hip_pitch_link$"),
        _Rule(r"^left_thigh_roll_link$"),
        _Rule(r"^left_thigh_pitch_link$"),
        _Rule(r"^left_thigh_yaw_link$"),
        _Rule(r"^Hip_Yaw_Left$"),
        _Rule(r"^Hip_Roll_Left$"),
        _Rule(r"^Hip_Pitch_Left$"),
        _Rule(r"^Left_Hip_Roll$"),
        _Rule(r"^Left_Hip_Yaw$"),
        _Rule(r"^Left_Hip_Pitch$"),
        _Rule(r"^hip_yaw_l_link$"),
        _Rule(r"^hip_roll_l_link$"),
        _Rule(r"^hip_pitch_l_link$"),
        _Rule(r"^leg_l1_link$"),
        _Rule(r"^l_hip_roll_link$"),
        _Rule(r"^l_hip_pitch_link$"),
        _Rule(r"left.*hip.*roll"),
        _Rule(r"left.*hip.*yaw"),
    ],
    "right_hip": [
        _Rule(r"^right_hip_roll_link$"),
        _Rule(r"^right_hip_yaw_link$"),
        _Rule(r"^right_hip_pitch_link$"),
        _Rule(r"^right_thigh_roll_link$"),
        _Rule(r"^right_thigh_pitch_link$"),
        _Rule(r"^right_thigh_yaw_link$"),
        _Rule(r"^Hip_Yaw_Right$"),
        _Rule(r"^Hip_Roll_Right$"),
        _Rule(r"^Hip_Pitch_Right$"),
        _Rule(r"^Right_Hip_Roll$"),
        _Rule(r"^Right_Hip_Yaw$"),
        _Rule(r"^Right_Hip_Pitch$"),
        _Rule(r"^hip_yaw_r_link$"),
        _Rule(r"^hip_roll_r_link$"),
        _Rule(r"^hip_pitch_r_link$"),
        _Rule(r"^leg_r1_link$"),
        _Rule(r"^r_hip_roll_link$"),
        _Rule(r"^r_hip_pitch_link$"),
        _Rule(r"right.*hip.*roll"),
        _Rule(r"right.*hip.*yaw"),
    ],
    "left_knee": [
        _Rule(r"^left_knee_link$"),
        _Rule(r"^left_shank(?:_pitch)?_link$"),
        _Rule(r"^left_calf_link$"),
        _Rule(r"^Shank_Left$"),
        _Rule(r"^Left_Shank$"),
        _Rule(r"^knee_pitch_l_link$"),
        _Rule(r"^leg_l3_link$"),
        _Rule(r"^l_thigh_link$"),
        _Rule(r"^left_lower_arm_pitch_link$"),
        _Rule(r"left.*knee"),
        _Rule(r"left.*shank"),
    ],
    "right_knee": [
        _Rule(r"^right_knee_link$"),
        _Rule(r"^right_shank(?:_pitch)?_link$"),
        _Rule(r"^right_calf_link$"),
        _Rule(r"^Shank_Right$"),
        _Rule(r"^Right_Shank$"),
        _Rule(r"^knee_pitch_r_link$"),
        _Rule(r"^leg_r3_link$"),
        _Rule(r"^r_thigh_link$"),
        _Rule(r"^right_lower_arm_pitch_link$"),
        _Rule(r"right.*knee"),
        _Rule(r"right.*shank"),
    ],
    "left_foot": [
        _Rule(r"^left_ankle_roll_link$"),
        _Rule(r"^left_ankle_pitch_link$"),
        _Rule(r"^left_ankle_link$"),
        _Rule(r"^left_toe_link$"),
        _Rule(r"^left_foot(?:_roll)?_link$"),
        _Rule(r"^Ankle_Cross_Left$"),
        _Rule(r"^left_foot_link$"),
        _Rule(r"^ank_roll_link$"),
        _Rule(r"^ankle_pitch_l_link$"),
        _Rule(r"^ankle_roll_l_link$"),
        _Rule(r"^ank_l"),
        _Rule(r"^leg_l5_link$"),
        _Rule(r"^l_ankle_roll_link$"),
        _Rule(r"^LL_FOOT$"),
        _Rule(r"^left_foot_pitch_link$"),
        _Rule(r"left.*ankle.*roll"),
        _Rule(r"left.*foot"),
    ],
    "right_foot": [
        _Rule(r"^right_ankle_roll_link$"),
        _Rule(r"^right_ankle_pitch_link$"),
        _Rule(r"^right_ankle_link$"),
        _Rule(r"^right_toe_link$"),
        _Rule(r"^right_foot(?:_roll)?_link$"),
        _Rule(r"^Ankle_Cross_Right$"),
        _Rule(r"^right_foot_link$"),
        _Rule(r"^ank_roll_link_2$"),
        _Rule(r"^ankle_pitch_r_link$"),
        _Rule(r"^ankle_roll_r_link$"),
        _Rule(r"^ank_r"),
        _Rule(r"^leg_r5_link$"),
        _Rule(r"^r_ankle_roll_link$"),
        _Rule(r"^LR_FOOT$"),
        _Rule(r"^right_foot_pitch_link$"),
        _Rule(r"right.*ankle.*roll"),
        _Rule(r"right.*foot"),
    ],
    "left_shoulder": [
        _Rule(r"^left_shoulder_roll_link$"),
        _Rule(r"^left_shoulder_yaw_link$"),
        _Rule(r"^left_shoulder_pitch_link$"),
        _Rule(r"^shoulder_roll_l_link$"),
        _Rule(r"^shoulder_pitch_l_link$"),
        _Rule(r"^shoulder_yaw_l_link$"),
        _Rule(r"^left_upper_arm_roll_link$"),
        _Rule(r"^l_shoulder_roll_link$"),
        _Rule(r"^l_shoulder_pitch_link$"),
        _Rule(r"^AL2$"),
        _Rule(r"^Left_Arm_2$"),
        _Rule(r"^zarm_l2_link$"),
        _Rule(r"^left_arm_link2$"),
        _Rule(r"^sho_pitch_link$"),
        _Rule(r"^2xl430_gears_3$"),
        _Rule(r"left.*shoulder.*roll"),
        _Rule(r"left.*shoulder"),
        _Rule(r"shoulder.*_l_"),
    ],
    "right_shoulder": [
        _Rule(r"^right_shoulder_roll_link$"),
        _Rule(r"^right_shoulder_yaw_link$"),
        _Rule(r"^right_shoulder_pitch_link$"),
        _Rule(r"^shoulder_roll_r_link$"),
        _Rule(r"^shoulder_pitch_r_link$"),
        _Rule(r"^shoulder_yaw_r_link$"),
        _Rule(r"^right_upper_arm_roll_link$"),
        _Rule(r"^r_shoulder_roll_link$"),
        _Rule(r"^r_shoulder_pitch_link$"),
        _Rule(r"^AR2$"),
        _Rule(r"^Right_Arm_2$"),
        _Rule(r"^zarm_r2_link$"),
        _Rule(r"^right_arm_link2$"),
        _Rule(r"^sho_pitch_link_2$"),
        _Rule(r"^2xl430_gears_6$"),
        _Rule(r"right.*shoulder.*roll"),
        _Rule(r"right.*shoulder"),
        _Rule(r"shoulder.*_r_"),
    ],
    "left_elbow": [
        _Rule(r"^left_elbow_link$"),
        _Rule(r"^left_lower_arm_pitch_link$"),
        _Rule(r"^elbow_pitch_l_link$"),
        _Rule(r"^elbow_roll_l_link$"),
        _Rule(r"^l_elbow_link$"),
        _Rule(r"^AL3$"),
        _Rule(r"^Left_Arm_3$"),
        _Rule(r"^zarm_l3_link$"),
        _Rule(r"^left_arm_link3$"),
        _Rule(r"^left_hand_yaw_link$"),
        _Rule(r"left.*elbow"),
        _Rule(r"elbow.*_l_"),
    ],
    "right_elbow": [
        _Rule(r"^right_elbow_link$"),
        _Rule(r"^right_lower_arm_pitch_link$"),
        _Rule(r"^elbow_pitch_r_link$"),
        _Rule(r"^elbow_roll_r_link$"),
        _Rule(r"^r_elbow_link$"),
        _Rule(r"^AR3$"),
        _Rule(r"^Right_Arm_3$"),
        _Rule(r"^zarm_r3_link$"),
        _Rule(r"^right_arm_link3$"),
        _Rule(r"^right_hand_yaw_link$"),
        _Rule(r"right.*elbow"),
        _Rule(r"elbow.*_r_"),
    ],
    "left_wrist": [
        _Rule(r"^left_wrist_yaw_link$"),
        _Rule(r"^left_wrist_roll_link$"),
        _Rule(r"^left_wrist_pitch_link$"),
        _Rule(r"^wrist_pitch_l_link$"),
        _Rule(r"^wrist_roll_l_link$"),
        _Rule(r"^wrist_yaw_l_link$"),
        _Rule(r"^left_end_effector_link$"),
        _Rule(r"^left_hand_link$"),
        _Rule(r"^zarm_l7_link$"),
        _Rule(r"^left_arm_link7$"),
        _Rule(r"^l_wrist_link$"),
        _Rule(r"^left_hand$"),
        _Rule(r"left.*wrist"),
        _Rule(r"wrist.*_l_"),
    ],
    "right_wrist": [
        _Rule(r"^right_wrist_yaw_link$"),
        _Rule(r"^right_wrist_roll_link$"),
        _Rule(r"^right_wrist_pitch_link$"),
        _Rule(r"^wrist_pitch_r_link$"),
        _Rule(r"^wrist_roll_r_link$"),
        _Rule(r"^wrist_yaw_r_link$"),
        _Rule(r"^right_end_effector_link$"),
        _Rule(r"^right_hand_link$"),
        _Rule(r"^zarm_r7_link$"),
        _Rule(r"^right_arm_link7$"),
        _Rule(r"^r_wrist_link$"),
        _Rule(r"^right_hand$"),
        _Rule(r"right.*wrist"),
        _Rule(r"wrist.*_r_"),
    ],
    "spine": [
        _Rule(r"^torso_link$"),
        _Rule(r"^Trunk$"),
        _Rule(r"^waist_link$"),
        _Rule(r"^torso_link2$"),
        _Rule(r"^H1$"),
        _Rule(r"^waist_yaw_link$"),
        _Rule(r"^torso_link4$"),
        _Rule(r"^head$"),
        _Rule(r"torso"),
        _Rule(r"trunk"),
    ],
}

_ROOT_PATTERNS: list[_Rule] = [
    _Rule(r"^pelvis$"),
    _Rule(r"^base_link$"),
    _Rule(r"^Waist$"),
    _Rule(r"^Trunk$"),
    _Rule(r"^torso$"),
    _Rule(r"^imu_2$"),
    _Rule(r"^torso_link4$"),
    _Rule(r"^Base_link$"),
]


def _match_root(
    body_names: list[str],
    root_hint: str | None,
) -> str | None:
    if root_hint and root_hint in body_names:
        return root_hint
    for pat in _ROOT_PATTERNS:
        for name in body_names:
            if pat.matches(name):
                return name
    return None


def match_bodies(
    robot_body_names: list[str],
    skeleton: HumanSkeleton,
    *,
    root_body_hint: str | None = None,
    overrides: dict[str, str] | None = None,
) -> MatchResult:
    """Map robot body names to skeleton roles.

    Parameters
    ----------
    robot_body_names:
        All body names extracted from the MuJoCo XML.
    skeleton:
        The target human skeleton (SMPL-X or BVH).
    root_body_hint:
        Optional explicit root body name.
    overrides:
        ``{role: robot_body_name}`` manual overrides. Highest priority.

    Returns
    -------
    ``MatchResult`` with mapping, unmatched roles, and unmatched bodies.
    """
    overrides = overrides or {}
    used_bodies: set[str] = set()
    mapping: dict[str, str] = {}

    root = _match_root(robot_body_names, root_body_hint)
    if root is not None:
        mapping["root"] = root
        used_bodies.add(root)

    for role in skeleton.role_to_joint:
        if role == "root":
            continue
        if role in overrides:
            body = overrides[role]
            if body in robot_body_names:
                mapping[role] = body
                used_bodies.add(body)
            continue
        rules = _RULES.get(role, [])
        for rule in rules:
            for body in robot_body_names:
                if body in used_bodies:
                    continue
                if rule.matches(body):
                    mapping[role] = body
                    used_bodies.add(body)
                    break
            if role in mapping:
                break

    all_roles = set(skeleton.role_to_joint)
    matched_roles = set(mapping)
    unmatched_roles = sorted(all_roles - matched_roles)
    unmatched_bodies = sorted(b for b in robot_body_names if b not in used_bodies)

    return MatchResult(
        mapping=mapping,
        unmatched_roles=unmatched_roles,
        unmatched_bodies=unmatched_bodies,
    )
