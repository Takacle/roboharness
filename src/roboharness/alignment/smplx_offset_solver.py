"""Solve SMPL-X IK config offsets from the canonical template frame.

Replaces the previous approach of assuming identity quaternions for all human
joints. Instead, uses the SMPL-X body model zero-pose to generate a synthetic
calibration frame, then computes:

    offset = inverse(human_joint_world_quat) * robot_link_expected_world_quat

for every robot link in the IK config whose mapped SMPL-X joint is available
in the template frame and whose robot link exists in the T-pose spec.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roboharness.alignment.metrics import load_tpose_spec
from roboharness.alignment.smplx_template import (
    load_smplx_template_tpose,
    resolve_body_model_path,
)


def solve_smplx_offsets_from_template(
    ik_config_path: Path,
    tpose_spec_path: Path,
    body_model_path: Path | str | None = None,
    gender: str = "male",
) -> dict:
    """Solve SMPL-X offsets using the canonical template T-pose.

    Parameters
    ----------
    ik_config_path:
        Path to the existing ``smplx_to_<robot>.json`` IK config.
    tpose_spec_path:
        Path to ``specs/tpose/<robot>.json``.
    body_model_path:
        Path to the SMPL-X body model (directory, ``smplx/`` subfolder, or
        ``.npz`` file). ``None`` auto-discovers via ``GMR_ROOT``.
    gender:
        Body model gender.

    Returns
    -------
    Updated IK config dict with solved quaternion offsets in both
    ``ik_match_table1`` and ``ik_match_table2``.
    """
    from scipy.spatial.transform import Rotation as R

    body_model_resolved = resolve_body_model_path(body_model_path)

    frame, _human_height = load_smplx_template_tpose(body_model_resolved, gender=gender)

    with ik_config_path.open() as f:
        config: dict = json.load(f)
    spec = load_tpose_spec(tpose_spec_path)

    offsets_solved: dict[str, list[float]] = {}

    for table_name in ("ik_match_table1", "ik_match_table2"):
        table: dict = config.get(table_name, {})
        for robot_joint_name, entry in table.items():
            human_bone_name: str = entry[0]

            if human_bone_name not in frame:
                continue
            if robot_joint_name not in spec.get("links", {}):
                continue

            if robot_joint_name in offsets_solved:
                q_offset = offsets_solved[robot_joint_name]
                entry[4] = q_offset
                continue

            _, q_human = frame[human_bone_name]
            q_human_arr = np.asarray(q_human, dtype=np.float64)
            q_human_arr = q_human_arr / (np.linalg.norm(q_human_arr) + 1e-12)

            R_expected = np.asarray(spec["links"][robot_joint_name]["R"], dtype=np.float64)
            r_human = R.from_quat(q_human_arr, scalar_first=True)
            r_target = R.from_matrix(R_expected)
            r_offset = r_human.inv() * r_target
            q_offset = [float(v) for v in r_offset.as_quat(scalar_first=True)]

            offsets_solved[robot_joint_name] = q_offset
            entry[4] = q_offset

    if "world_rotation" in config and config["world_rotation"] is not None:
        del config["world_rotation"]

    return config


def write_solved_config(
    config: dict,
    output_path: Path,
) -> Path:
    """Write a solved config dict to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")
    return output_path
