"""Human skeleton definitions for different motion-capture formats.

Pure data module — zero IO, zero external dependencies. Each skeleton declares
the joint names, default scale factors, and fallback relationships used by
``body_matcher`` and ``config_gen`` to produce IK configs.

Two built-in skeletons are provided:

* ``SMPLX_SKELETON`` — SMPL-X body model joint names (lowercase, underscore).
* ``BVH_SKELETON`` — BVH / LAFAN1 / SOMA joint names (PascalCase).

Consumers select a skeleton by calling ``get_skeleton(src_format)`` where
*src_format* is one of ``"smplx"``, ``"bvh"``, ``"fbx"``, ``"fbx_offline"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ROLES: list[str] = [
    "root",
    "spine",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


@dataclass(frozen=True)
class HumanSkeleton:
    """Declaration of a human skeleton's joint naming conventions."""

    name: str
    root_name: str
    role_to_joint: dict[str, str]
    scale_defaults: dict[str, float]
    fallback_map: dict[str, str] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)


_SMPLX_ROLE_MAP: dict[str, str] = {
    "root": "pelvis",
    "spine": "spine3",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_foot": "left_foot",
    "right_foot": "right_foot",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
}

_SMPLX_SCALES: dict[str, float] = {
    "pelvis": 0.9,
    "spine3": 0.9,
    "left_hip": 0.9,
    "right_hip": 0.9,
    "left_knee": 0.9,
    "right_knee": 0.9,
    "left_foot": 0.9,
    "right_foot": 0.9,
    "left_shoulder": 0.8,
    "right_shoulder": 0.8,
    "left_elbow": 0.8,
    "right_elbow": 0.8,
    "left_wrist": 0.8,
    "right_wrist": 0.8,
}

_SMPLX_FALLBACK: dict[str, str] = {
    "left_collar": "left_shoulder",
    "right_collar": "right_shoulder",
    "left_ankle": "left_foot",
    "right_ankle": "right_foot",
    "spine1": "spine3",
    "spine2": "spine3",
    "neck": "spine3",
    "head": "spine3",
}

_SMPLX_EDGES: list[tuple[str, str]] = [
    ("pelvis", "spine1"),
    ("spine1", "spine2"),
    ("spine2", "spine3"),
    ("spine3", "neck"),
    ("neck", "head"),
    ("spine3", "left_collar"),
    ("left_collar", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("spine3", "right_collar"),
    ("right_collar", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_foot"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_foot"),
]

SMPLX_SKELETON = HumanSkeleton(
    name="smplx",
    root_name="pelvis",
    role_to_joint=dict(_SMPLX_ROLE_MAP),
    scale_defaults=dict(_SMPLX_SCALES),
    fallback_map=dict(_SMPLX_FALLBACK),
    edges=[*_SMPLX_EDGES],
)


_BVH_ROLE_MAP: dict[str, str] = {
    "root": "Hips",
    "spine": "Spine2",
    "left_hip": "LeftUpLeg",
    "right_hip": "RightUpLeg",
    "left_knee": "LeftLeg",
    "right_knee": "RightLeg",
    "left_foot": "LeftFootMod",
    "right_foot": "RightFootMod",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}

_BVH_SCALES: dict[str, float] = {
    "Hips": 0.9,
    "Spine2": 0.9,
    "LeftUpLeg": 0.9,
    "RightUpLeg": 0.9,
    "LeftLeg": 0.9,
    "RightLeg": 0.9,
    "LeftFootMod": 0.9,
    "RightFootMod": 0.9,
    "LeftArm": 0.75,
    "RightArm": 0.75,
    "LeftForeArm": 0.75,
    "RightForeArm": 0.75,
    "LeftHand": 0.75,
    "RightHand": 0.75,
}

_BVH_FALLBACK: dict[str, str] = {
    "LeftToe": "LeftFootMod",
    "RightToe": "RightFootMod",
    "LeftFoot": "LeftFootMod",
    "RightFoot": "RightFootMod",
    "LeftShoulder": "LeftArm",
    "RightShoulder": "RightArm",
    "Spine": "Spine2",
    "Spine1": "Spine2",
    "Neck": "Spine2",
    "Head": "Spine2",
}

_BVH_EDGES: list[tuple[str, str]] = [
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Head"),
    ("Spine2", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("LeftFoot", "LeftToe"),
    ("LeftFoot", "LeftFootMod"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
    ("RightFoot", "RightToe"),
    ("RightFoot", "RightFootMod"),
    ("Hips", "Spine2"),
    ("Spine2", "LeftArm"),
    ("Spine2", "RightArm"),
    ("LeftLeg", "LeftFootMod"),
    ("RightLeg", "RightFootMod"),
]

BVH_SKELETON = HumanSkeleton(
    name="bvh",
    root_name="Hips",
    role_to_joint=dict(_BVH_ROLE_MAP),
    scale_defaults=dict(_BVH_SCALES),
    fallback_map=dict(_BVH_FALLBACK),
    edges=[*_BVH_EDGES],
)


_SKELETON_REGISTRY: dict[str, HumanSkeleton] = {
    "smplx": SMPLX_SKELETON,
    "bvh": BVH_SKELETON,
    "fbx": BVH_SKELETON,
    "fbx_offline": BVH_SKELETON,
}


def get_skeleton(src_format: str) -> HumanSkeleton:
    """Return the skeleton definition for a motion source format."""
    skel = _SKELETON_REGISTRY.get(src_format)
    if skel is None:
        raise ValueError(
            f"Unknown src_format {src_format!r}; choose from {sorted(_SKELETON_REGISTRY)}"
        )
    return skel
