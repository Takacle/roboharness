"""Generate GMR IK config JSON from a body-name matching result.

Takes a ``MatchResult`` (from ``body_matcher``) and a ``HumanSkeleton``,
produces a complete IK config dict suitable for writing to
``general_motion_retargeting/ik_configs/<src>_to_<robot>.json``.

This module handles structure and defaults — quaternion offsets start as
identity and are refined later by ``solve_mode`` or the VLM alignment agent.

When an ``xml_path`` is supplied, ``world_rotation`` is auto-computed from
the robot's default-pose body geometry.  Pass ``world_rotation_override``
to override the automatic value (or set to ``"none"`` to force no rotation).
"""

from __future__ import annotations

import json
from pathlib import Path

from roboharness._math_utils import IDENTITY_QUAT
from roboharness.alignment.body_matcher import MatchResult
from roboharness.alignment.skeleton_maps import HumanSkeleton

_IDENTITY_QUAT = IDENTITY_QUAT  # keep existing local alias for backward compat
_ZERO_OFFSET: list[float] = [0.0, 0.0, 0.0]


def _weight_for_role(role: str) -> tuple[int, int, int, int]:
    """Return (t1_pos, t1_rot, t2_pos, t2_rot) for a role."""
    if role == "root":
        return (100, 10, 100, 5)
    if role in ("left_foot", "right_foot"):
        return (100, 10, 50, 10)
    if role == "spine":
        return (0, 100, 0, 10)
    if role in ("left_hip", "right_hip"):
        return (0, 10, 10, 5)
    if role in ("left_knee", "right_knee"):
        return (0, 10, 10, 5)
    if role in ("left_shoulder", "right_shoulder"):
        return (0, 10, 10, 5)
    if role in ("left_elbow", "right_elbow"):
        return (0, 10, 10, 5)
    if role in ("left_wrist", "right_wrist"):
        return (0, 10, 10, 5)
    return (0, 10, 10, 5)


def generate_ik_config(
    match: MatchResult,
    skeleton: HumanSkeleton,
    *,
    ground_height: float = 0.0,
    human_height_assumption: float = 1.8,
    xml_path: Path | None = None,
    src_format: str | None = None,
    world_rotation_override: list[float] | None = None,
) -> dict:
    """Build a full IK config dict from match results.

    The config uses identity quaternion offsets — call
    ``gmr_alignment_agent.py --solve_mode`` or ``apply_patch`` to refine.

    When *xml_path* is provided, ``world_rotation`` is auto-detected from
    the robot's default-pose body geometry via ``orientation_aligner``.
    Pass *world_rotation_override* to replace the automatic value
    (a falsy value like ``None`` or ``[]`` leaves auto-detection active;
    use ``--world_rot none`` to explicitly skip).
    """
    from roboharness.alignment.orientation_aligner import compute_world_rotation

    root_body = match.mapping.get("root", "pelvis")

    table1: dict[str, list] = {}
    table2: dict[str, list] = {}

    for role, robot_body in match.mapping.items():
        joint_name = skeleton.role_to_joint[role]
        t1p, t1r, t2p, t2r = _weight_for_role(role)
        entry_t1 = [joint_name, t1p, t1r, list(_ZERO_OFFSET), list(_IDENTITY_QUAT)]
        entry_t2 = [joint_name, t2p, t2r, list(_ZERO_OFFSET), list(_IDENTITY_QUAT)]
        table1[robot_body] = entry_t1
        table2[robot_body] = entry_t2

    scale_table: dict[str, float] = {}
    for role, _robot_body in match.mapping.items():
        joint_name = skeleton.role_to_joint[role]
        default = skeleton.scale_defaults.get(joint_name, 0.9)
        scale_table[joint_name] = default

    fallback_offsets: dict[str, list[float]] = {}
    for child_joint, parent_joint in skeleton.fallback_map.items():
        if child_joint not in scale_table and parent_joint in scale_table:
            fallback_offsets[child_joint] = list(_IDENTITY_QUAT)
            if parent_joint not in scale_table:
                parent_role = None
                for r, j in skeleton.role_to_joint.items():
                    if j == parent_joint:
                        parent_role = r
                        break
                if parent_role and parent_role in match.mapping:
                    scale_table[parent_joint] = skeleton.scale_defaults.get(parent_joint, 0.9)

    config: dict = {
        "robot_root_name": root_body,
        "human_root_name": skeleton.root_name,
        "ground_height": ground_height,
        "human_height_assumption": human_height_assumption,
        "use_ik_match_table1": True,
        "use_ik_match_table2": True,
        "human_scale_table": dict(sorted(scale_table.items())),
        "ik_match_table1": dict(sorted(table1.items())),
        "ik_match_table2": dict(sorted(table2.items())),
    }
    if fallback_offsets:
        config["_fallback_offsets"] = dict(sorted(fallback_offsets.items()))

    # ── world_rotation ──
    wr: list[float] | None = None
    if xml_path is not None and xml_path.exists():
        fmt = src_format or skeleton.name
        wr = compute_world_rotation(xml_path, match, src_format=fmt)
    if world_rotation_override is not None:
        wr = world_rotation_override or None
    if wr is not None and wr:
        config["world_rotation"] = [round(x, 10) for x in wr]

    return config


def write_ik_config(
    config: dict,
    robot_name: str,
    src_format: str,
    output_dir: Path | None = None,
) -> Path:
    """Write an IK config dict to JSON. Returns the written path.

    If *output_dir* is ``None``, writes to
    ``GMR/general_motion_retargeting/ik_configs/`` (auto-detected).
    """
    if output_dir is None:
        output_dir = _default_ik_config_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{src_format}_to_{robot_name}.json"
    with path.open("w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    return path


def _default_ik_config_dir() -> Path:
    """Auto-locate ``GMR/general_motion_retargeting/ik_configs/``."""
    roboharness_root = Path(__file__).resolve().parents[3]
    gmr_root = roboharness_root.parent / "GMR"
    if not gmr_root.exists():
        gmr_root = roboharness_root / "GMR"
    return gmr_root / "general_motion_retargeting" / "ik_configs"


def clone_ik_config(
    source_config_path: Path,
    new_body_mapping: dict[str, str],
    robot_name: str,
    src_format: str,
    output_dir: Path | None = None,
) -> Path:
    """Clone an existing IK config with body name substitutions.

    *new_body_mapping* maps ``old_robot_body → new_robot_body``. Only bodies
    present in the mapping are renamed; others are preserved as-is.

    Same-format only (smplx→smplx, bvh→bvh).
    """
    with source_config_path.open() as f:
        config = json.load(f)

    for table_key in ("ik_match_table1", "ik_match_table2"):
        table = config.get(table_key, {})
        new_table: dict[str, list] = {}
        for body_name, entry in table.items():
            new_name = new_body_mapping.get(body_name, body_name)
            new_table[new_name] = entry
        config[table_key] = new_table

    root = config.get("robot_root_name", "")
    config["robot_root_name"] = new_body_mapping.get(root, root)

    return write_ik_config(config, robot_name, src_format, output_dir)
