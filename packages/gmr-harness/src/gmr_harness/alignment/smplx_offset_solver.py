"""Solve SMPL-X IK config offsets from the canonical template frame.

Pipeline architecture (Y-up native, compatible with GMR runtime):

    1. ``load_smplx_template_tpose`` — generate Y-up template frame
    2. ``apply_human_scale`` — scale positions per bone scale factors
    3. ``compute_joint_offsets`` — compute offsets using the proven formula
       ``Q_offset = SMPL_TO_MUJOCO.inv() * R_target_Zup`` with zero position
       offsets, matching the convention used by all existing working configs.

The offsets are compatible with GMR's native ``get_smplx_data()`` /
``get_smplx_data_offline_fast()`` which return Y-up SMPL-X data.

Public API: ``solve_smplx_offsets_from_template()`` — unchanged signature.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gmr_harness.alignment.metrics import TposeSpec, load_tpose_spec
from gmr_harness.alignment.smplx_scale import apply_human_scale
from gmr_harness.alignment.smplx_template import (
    load_smplx_template_tpose,
    resolve_body_model_path,
)

_ZERO_OFFSET: list[float] = [0.0, 0.0, 0.0]


def _check_stale_smplx_config(config: dict, config_path: Path) -> None:
    """Fail-fast if the config contains a legacy SMPL-X base world_rotation.

    A stale config with ``world_rotation = [0.5, 0.5, 0.5, 0.5]`` would
    double-apply the Y→Z conversion.

    Raises ``ValueError`` so the caller must regenerate or migrate the config.
    """
    from gmr_harness.alignment.smplx_coordinate import validate_smplx_runtime_config

    validate_smplx_runtime_config(config, config_path, converted_at_loader=False)


def compute_joint_offsets(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
    spec: TposeSpec,
    config: dict,
    *,
    ground_height: float = 0.0,
) -> dict[str, tuple[list[float], list[float]]]:
    """Compute per-joint quaternion and position offsets for Y-up SMPL-X data.

    Uses the proven convention: ``Q_offset = SMPL_TO_MUJOCO.inv() * R_target``
    where ``R_target`` is the robot body rotation in the T-pose spec (Z-up).
    Position offsets are set to zero — the rotational offset handles coordinate
    alignment, matching all existing working GMR configs.

    Parameters
    ----------
    frame:
        Y-up human template frame (unused for rotation but kept for API compat).
    spec:
        T-pose spec dict with ``spec["links"][joint_name]["R"]`` and ``["pos"]``.
    config:
        IK config dict (read-only; ``ik_match_table1/2`` used for mapping).
    ground_height:
        Config ``ground_height`` value (unused with zero position offsets).

    Returns
    -------
    ``{robot_joint_name: (quat_offset, pos_offset)}`` where both are
    scalar-first ``[w, x, y, z]`` / ``[x, y, z]`` lists.
    """
    from scipy.spatial.transform import Rotation as R

    from gmr_harness.alignment.smplx_coordinate import SMPL_TO_MUJOCO_QUAT

    r_conv_inv = R.from_quat(
        np.asarray(SMPL_TO_MUJOCO_QUAT, dtype=np.float64), scalar_first=True
    ).inv()

    offsets: dict[str, tuple[list[float], list[float]]] = {}

    for table_name in ("ik_match_table1", "ik_match_table2"):
        table: dict = config.get(table_name, {})
        for robot_joint_name, entry in table.items():
            if robot_joint_name not in spec.get("links", {}):
                continue

            if robot_joint_name in offsets:
                q_offset, p_offset = offsets[robot_joint_name]
                entry[3] = p_offset
                entry[4] = q_offset
                continue

            R_expected = np.asarray(spec["links"][robot_joint_name]["R"], dtype=np.float64)
            r_target = R.from_matrix(R_expected)
            r_offset = r_conv_inv * r_target
            q_offset = [float(v) for v in r_offset.as_quat(scalar_first=True)]

            offsets[robot_joint_name] = (q_offset, list(_ZERO_OFFSET))
            entry[3] = list(_ZERO_OFFSET)
            entry[4] = q_offset

    return offsets


def solve_smplx_offsets_from_template(
    ik_config_path: Path,
    tpose_spec_path: Path,
    body_model_path: Path | str | None = None,
    gender: str = "male",
) -> dict:
    """Solve SMPL-X offsets using the canonical template T-pose.

    Pipeline: load template → compute offsets from T-pose spec.

    The offsets use the convention ``Q_offset = SMPL_TO_MUJOCO.inv() * R_target``
    with zero position offsets, which is compatible with GMR's native Y-up
    SMPL-X data (``get_smplx_data_offline_fast()``).

    Parameters
    ----------
    ik_config_path:
        Path to the existing ``smplx_to_<robot>.json`` IK config.
    tpose_spec_path:
        Path to ``specs/tpose/<robot>.json``.
    body_model_path:
        Path to the SMPL-X body model. ``None`` auto-discovers via ``GMR_ROOT``.
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

    frame, human_height = load_smplx_template_tpose(body_model_resolved, gender=gender, yup=True)

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
