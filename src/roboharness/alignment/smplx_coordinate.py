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


def classify_smplx_frame_convention(
    frames: list[dict[str, tuple[np.ndarray, np.ndarray]]],
    max_samples: int = 30,
) -> str:
    """Classify whether GMR SMPLX frames are native Y-up or AMASS Z-up.

    Examines pelvis orientation across the first *max_samples* frames.  For
    native SMPLX data the body-local up axis (Y) aligns with world Y; for
    AMASS data (where ``global_orient`` was applied in the wrong convention)
    body-local Y aligns with world Z.

    Parameters
    ----------
    frames:
        GMR SMPLX loader output frames (before any conversion).
    max_samples:
        Number of early frames to sample.

    Returns
    -------
    ``"y"`` for native Y-up data (needs ``smpl_to_mujoco_frame()``),
    ``"z"`` for AMASS Z-up data (skip conversion — already Z-up).

    Raises
    ------
    RuntimeError
        If *frames* is empty.
    KeyError
        If ``"pelvis"`` is missing from the first frame.
    ValueError
        If a pelvis quaternion has non-unit norm, or if the convention is
        ambiguous (median Y-score and Z-score within 0.25).
    """
    from scipy.spatial.transform import Rotation as R

    if not frames:
        raise RuntimeError("No SMPLX frames to classify")
    if "pelvis" not in frames[0]:
        raise KeyError("Frame missing 'pelvis' joint for convention detection")

    n = min(max_samples, len(frames))
    y_scores: list[float] = []
    z_scores: list[float] = []

    for i in range(n):
        q = np.asarray(frames[i]["pelvis"][1], dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 0.9 or norm > 1.1:
            raise ValueError(f"Frame {i} pelvis quaternion has invalid norm {norm:.4f}")
        rq = R.from_quat(q, scalar_first=True)
        ly = rq.apply(np.array([0.0, 1.0, 0.0]))
        y_scores.append(float(ly[1]))
        z_scores.append(float(ly[2]))

    y_median = float(np.median(y_scores))
    z_median = float(np.median(z_scores))
    margin = abs(y_median - z_median)

    if margin < 0.25:
        raise ValueError(
            f"Ambiguous SMPLX convention (Y={y_median:.3f}, Z={z_median:.3f}, "
            f"margin={margin:.3f}). Cannot auto-detect coordinate system."
        )

    return "y" if y_median > z_median else "z"


def normalize_to_pelvis_z(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    pelvis_z: float | None = None,
) -> None:
    """Shift all positions in *frame* so that the pelvis sits at Z=0.

    Normalising both the template and runtime frames to a common pelvis
    Z reference makes computed position offsets independent of any
    per-dataset ground reference (i.e.  the solution works for AMASS /
    ACCAD, native SMPL-X, and any future data source without tuning).

    Parameters
    ----------
    frame:
        Single SMPL-X frame dict (positions in Z-up MuJoCo convention).
    pelvis_z:
        Reference pelvis Z to subtract.  When ``None`` the current
        ``frame["pelvis"]`` Z is used.
    """
    if pelvis_z is None:
        if "pelvis" not in frame:
            return
        pelvis_z = float(frame["pelvis"][0][2])
    offset = np.array([0.0, 0.0, -pelvis_z], dtype=np.float64)
    for name in frame:
        frame[name] = (frame[name][0] + offset, frame[name][1])
