"""Shared math utilities — quaternion/vector operations and rotation conversions.

Consolidates duplicated quaternion normalization, vector normalization, and
rotation matrix converters that previously lived in ``alignment/patch.py``,
``robots/unitree_g1/locomotion.py``, ``alignment/metrics.py``, and
``alignment/orientation_aligner.py``.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np

IDENTITY_QUAT: list[float] = [1.0, 0.0, 0.0, 0.0]


def normalize_quat(q: list[float]) -> list[float]:
    """Return L2-normalized quaternion; identity on near-zero input."""
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-9:
        return list(IDENTITY_QUAT)
    return [v / norm for v in q]


def normalize_vector(
    v: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Return L2-normalized vector; *fallback* on near-zero input.

    When *fallback* is ``None`` the input vector is returned unchanged.
    """
    n = float(np.linalg.norm(v))
    if n < 1e-10:
        source = v if fallback is None else fallback
        return cast("np.ndarray", np.asarray(source, dtype=float).copy())
    return cast("np.ndarray", np.asarray(v / n, dtype=float))


def quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    """Hamilton product ``q1 * q2``, scalar-first ``[w, x, y, z]``."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a 3x3 rotation matrix to ``(unit_axis, angle_radians)``.

    Returns ``(z_axis, 0.0)`` for (near-)identity matrices. For angles near
    π, uses the symmetric extraction to stay numerically stable.
    """
    trace = float(np.clip(R[0, 0] + R[1, 1] + R[2, 2], -1.0, 3.0))
    cos_angle = (trace - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))

    if angle < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0

    if np.pi - angle < 1e-6:
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]]) + 1.0
        diag = np.clip(diag, 0.0, None)
        axis = np.sqrt(diag * 0.5)
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            return np.array([0.0, 0.0, 1.0]), angle
        return axis / norm, angle

    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
    )
    axis = axis / (2.0 * np.sin(angle))
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0]), angle
    return axis / norm, angle


def rotation_matrix_to_quat(M: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to scalar-first ``[w, x, y, z]`` quaternion."""
    # Lazy import to avoid hard dependency on scipy at module level.
    from scipy.spatial.transform import Rotation as R

    q = R.from_matrix(M).as_quat(scalar_first=True)
    return [float(x) for x in q]


def axis_angle_to_quat(axis: list[float], angle_deg: float) -> list[float]:
    """Convert axis-angle to scalar-first ``[w, x, y, z]`` quaternion."""
    half = math.radians(angle_deg) / 2.0
    s = math.sin(half)
    return [math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s]


# .. deprecated::
#   Use ``SMPL_TO_MUJOCO_QUAT`` from ``roboharness.alignment.smplx_coordinate``
#   instead. This constant uses the historical row-vector convention; the new
#   constant is in runtime form and requires no ``.inv()`` call.
SMPLX_BASE_ROTATION_QUAT: list[float] = [0.5, -0.5, -0.5, -0.5]
