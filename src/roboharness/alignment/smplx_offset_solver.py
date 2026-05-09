"""Solve SMPL-X IK config offsets from the canonical template frame.

Pipeline architecture (BVH-style, conversion at loader boundary):

    1. ``load_smplx_template_tpose`` — generate Z-up template frame
    2. ``normalize_to_pelvis_z`` — shift pelvis to Z=0 (dataset-agnostic)
    3. ``apply_human_scale`` — scale positions per bone scale factors
    4. ``apply_world_rotation_to_frame`` — apply config world_rotation (matches runtime)
    5. ``compute_joint_offsets`` — pure offset computation per joint

Public API: ``solve_smplx_offsets_from_template()`` — unchanged signature.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from roboharness.alignment.metrics import TposeSpec, load_tpose_spec
from roboharness.alignment.smplx_scale import apply_human_scale
from roboharness.alignment.smplx_template import (
    load_smplx_template_tpose,
    resolve_body_model_path,
)


def _check_stale_smplx_config(config: dict, config_path: Path) -> None:
    """Fail-fast if the config contains a legacy SMPL-X base world_rotation.

    After the loader-boundary refactor, SMPL-X data arrives in Z-up at the
    GMR runtime.  A stale config with ``world_rotation = [0.5, 0.5, 0.5, 0.5]``
    would apply the Y→Z conversion a second time, producing incorrect results.

    Raises ``ValueError`` so the caller must regenerate or migrate the config.
    """
    from roboharness.alignment.smplx_coordinate import validate_smplx_runtime_config

    validate_smplx_runtime_config(config, config_path, converted_at_loader=True)


def _apply_rotation_to_frame(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
    quat_wxyz: list[float],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Apply a rotation to every joint position and orientation in a frame."""
    from scipy.spatial.transform import Rotation as R

    r_wr = R.from_quat(np.asarray(quat_wxyz, dtype=np.float64), scalar_first=True)
    transformed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (pos, quat) in frame.items():
        new_pos = r_wr.apply(np.asarray(pos, dtype=np.float64))
        new_quat = (
            r_wr * R.from_quat(np.asarray(quat, dtype=np.float64), scalar_first=True)
        ).as_quat(scalar_first=True)
        transformed[name] = (new_pos, new_quat)
    return transformed


def compute_joint_offsets(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
    spec: TposeSpec,
    config: dict,
    *,
    ground_height: float = 0.0,
) -> dict[str, tuple[list[float], list[float]]]:
    """Compute per-joint quaternion and position offsets (pure computation).

    Parameters
    ----------
    frame:
        Frame after world_rotation has been applied (if any), matching the
        state the GMR runtime will be in when it applies offsets.
        ``{joint: (pos, quat_wxyz)}``.
    spec:
        T-pose spec dict with ``spec["links"][joint_name]["R"]`` and ``["pos"]``.
    config:
        IK config dict (read-only; ``ik_match_table1/2`` used for mapping).
    ground_height:
        Config ``ground_height`` value (added to Z component of pos offset).

    Returns
    -------
    ``{robot_joint_name: (quat_offset, pos_offset)}`` where both are
    scalar-first ``[w, x, y, z]`` / ``[x, y, z]`` lists.
    """
    from scipy.spatial.transform import Rotation as R

    offsets: dict[str, tuple[list[float], list[float]]] = {}

    for table_name in ("ik_match_table1", "ik_match_table2"):
        table: dict = config.get(table_name, {})
        for robot_joint_name, entry in table.items():
            human_bone_name: str = entry[0]

            if human_bone_name not in frame:
                continue
            if robot_joint_name not in spec.get("links", {}):
                continue

            if robot_joint_name in offsets:
                q_offset, p_offset = offsets[robot_joint_name]
                entry[3] = p_offset
                entry[4] = q_offset
                continue

            _, q_human = frame[human_bone_name]
            q_human_arr = np.asarray(q_human, dtype=np.float64)
            q_human_arr = q_human_arr / (np.linalg.norm(q_human_arr) + 1e-12)

            R_expected = np.asarray(spec["links"][robot_joint_name]["R"], dtype=np.float64)
            pos_human = np.asarray(frame[human_bone_name][0], dtype=np.float64)
            pos_target = np.asarray(spec["links"][robot_joint_name]["pos"], dtype=np.float64)

            r_human = R.from_quat(q_human_arr, scalar_first=True)
            r_target = R.from_matrix(R_expected)
            r_offset = r_human.inv() * r_target
            q_offset = [float(v) for v in r_offset.as_quat(scalar_first=True)]

            ground = ground_height * np.array([0.0, 0.0, 1.0], dtype=np.float64)
            pos_offset = r_target.inv().apply(pos_target - pos_human) + ground
            p_offset = [float(v) for v in pos_offset]

            offsets[robot_joint_name] = (q_offset, p_offset)
            entry[3] = p_offset
            entry[4] = q_offset

    return offsets


def solve_smplx_offsets_from_template(
    ik_config_path: Path,
    tpose_spec_path: Path,
    body_model_path: Path | str | None = None,
    gender: str = "male",
) -> dict:
    """Solve SMPL-X offsets using the canonical template T-pose.

    Pipeline: load (Z-up) → normalise pelvis Z → scale → world_rotation → solve.

    The pelvis-Z normalisation shifts all positions so the pelvis sits at
    Z=0 before offsets are computed.  This keeps the position offsets
    independent of any per-dataset ground reference (AMASS ``trans`` vs
    body-model ``transl``) and makes one config work across datasets.

    The world_rotation application matches the GMR runtime order
    (scale → world_rotation → offset), ensuring solved offsets are
    consistent with what the runtime will apply.

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
    body_model_resolved = resolve_body_model_path(body_model_path)

    with ik_config_path.open() as f:
        config: dict = json.load(f)

    _check_stale_smplx_config(config, ik_config_path)

    frame, human_height = load_smplx_template_tpose(body_model_resolved, gender=gender)

    from roboharness.alignment.smplx_coordinate import normalize_to_pelvis_z

    normalize_to_pelvis_z(frame)

    human_root_name = str(config.get("human_root_name", "pelvis"))
    scale_table_raw = config.get("human_scale_table", {})
    height_assumption = float(config.get("human_height_assumption", human_height))
    frame = apply_human_scale(
        frame,
        scale_table_raw,
        human_root_name=human_root_name,
        height_assumption=height_assumption,
        human_height=human_height,
    )

    wr = config.get("world_rotation")
    if wr:
        frame = _apply_rotation_to_frame(frame, wr)

    spec = load_tpose_spec(tpose_spec_path)
    ground_height = float(config.get("ground_height", 0.0))
    compute_joint_offsets(frame, spec, config, ground_height=ground_height)

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
