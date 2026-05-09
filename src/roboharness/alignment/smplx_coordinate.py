"""Unified SMPL-X to MuJoCo coordinate conversion.

Provides a single source of truth for the Y-up (SMPL-X) → Z-up (MuJoCo)
coordinate frame conversion used by the offset solver, world-rotation
computation, and validation pipeline.

Coordinate mapping:

=========  ========  ===========
Axis       SMPL-X    MuJoCo
=========  ========  ===========
Up         +Y        +Z
Left       +X        +Y
Forward    +Z        +X
=========  ========  ===========

The conversion is a 120-degree rotation about the (1,1,1)/sqrt(3) axis,
represented by the runtime quaternion ``SMPL_TO_MUJOCO_QUAT``.

Legacy note
-----------
The old constant ``SMPLX_BASE_ROTATION_QUAT`` (in ``_math_utils``) is stored
in row-vector convention `[0.5, -0.5, -0.5, -0.5]` and required ``.inv()``
at every call site.  ``SMPL_TO_MUJOCO_QUAT`` is the runtime form (the inverse)
and requires no inversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SMPL_TO_MUJOCO_QUAT: list[float] = [0.5, 0.5, 0.5, 0.5]


def smpl_to_mujoco_frame(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Convert a Y-up SMPL-X template frame to Z-up MuJoCo coordinates.

    Parameters
    ----------
    frame:
        ``{joint_name: (position_3d, quat_wxyz)}`` in SMPL-X Y-up frame.

    Returns
    -------
    New frame dict with positions and orientations transformed to Z-up.
    Quaternions are scalar-first ``[w, x, y, z]``.
    """
    from scipy.spatial.transform import Rotation as R

    r_conv = R.from_quat(
        np.asarray(SMPL_TO_MUJOCO_QUAT, dtype=np.float64),
        scalar_first=True,
    )
    transformed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, (pos, quat) in frame.items():
        p = np.asarray(pos, dtype=np.float64)
        q = np.asarray(quat, dtype=np.float64)
        new_pos = r_conv.apply(p)
        new_quat = (r_conv * R.from_quat(q, scalar_first=True)).as_quat(scalar_first=True)
        transformed[name] = (new_pos, new_quat)
    return transformed


def _is_legacy_base_world_rotation(wr: list[float] | None) -> bool:
    """Return True if *wr* is the legacy SMPL-X base world_rotation."""
    if wr is None:
        return False
    base = SMPL_TO_MUJOCO_QUAT
    return len(wr) == 4 and all(abs(a - b) < 1e-6 for a, b in zip(wr, base, strict=True))


def validate_smplx_runtime_config(
    config: dict,
    config_path: str | Path,
    *,
    converted_at_loader: bool = True,
) -> None:
    """Validate a SMPL-X IK config for compatibility with loader-boundary conversion.

    After the loader-boundary refactor, SMPL-X data arrives in Z-up at GMR
    runtime.  A stale config with ``world_rotation = [0.5, 0.5, 0.5, 0.5]``
    would apply the Y→Z conversion a second time.

    Raises ``ValueError`` when a stale config is detected.
    """
    if not converted_at_loader:
        return
    wr = config.get("world_rotation")
    if _is_legacy_base_world_rotation(wr):
        raise ValueError(
            f"SMPL-X config {config_path} contains the legacy base "
            "world_rotation [0.5, 0.5, 0.5, 0.5].  After the loader-boundary "
            "refactor, SMPL-X data is already Z-up when it reaches GMR runtime.  "
            "This world_rotation will double-apply the Y→Z conversion.  "
            "Regenerate the config via:\n"
            "  python scripts/setup_robot.py --robot <robot> --src smplx "
            "--auto_register --update_scripts"
        )


def smpl_to_mujoco_world_rotation() -> list[float]:
    """Return the ``world_rotation`` quaternion for SMPL-X IK configs.

    .. deprecated::
        This function is no longer used for SMPL-X IK config world_rotation.
        The coordinate conversion is now applied at the loading boundary
        (in ``load_smplx()`` and ``load_smplx_template_tpose()``), and
        ``compute_world_rotation()`` computes the fine-tuning alignment from
        robot geometry.  Kept for backward compatibility.
    """
    return list(SMPL_TO_MUJOCO_QUAT)
