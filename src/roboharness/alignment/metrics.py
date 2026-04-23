"""Pose-deviation metrics against an authored T-pose spec.

The T-pose spec is a JSON file produced by ``scripts/stage_tpose.py`` that
freezes each controlled link's expected world-frame orientation (as a 3x3
rotation matrix) when the robot is physically staged at T-pose. At runtime
we forward-simulate a candidate qpos, read ``data.xmat``, and compute the
axis-angle of ``R_actual @ R_expected.T`` per link — a single scalar
"rotation error" with an accompanying axis that directly suggests the fix.

This module is a pure sensor: no optimization, no policy, no IO side effects
beyond reading an XML and a JSON. Agents and tests call it to get a ground
truth signal that vision cannot provide.

Requires ``pip install -e ".[demo]"`` (for ``mujoco``). The ``mujoco`` import
is lazy so this module can be imported in environments without it; only
``compute_deviations`` actually needs it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

import numpy as np


class LinkFrame(TypedDict):
    """Expected world-frame pose of one link at T-pose."""

    pos: list[float]  # [x, y, z]
    R: list[list[float]]  # 3x3 rotation, row-major


class TposeSpec(TypedDict):
    """Full T-pose contract for a robot.

    ``xml_path`` is recorded for provenance; callers pass the xml path
    explicitly to ``compute_deviations`` so the spec is not bound to an
    absolute path at load time.
    """

    robot: str
    xml_path: str
    qpos: list[float]  # the staged qpos that produced this spec
    links: dict[str, LinkFrame]


class Deviation(TypedDict):
    axis: list[float]  # unit rotation axis (world frame)
    angle_deg: float  # non-negative


def load_tpose_spec(path: str | Path) -> TposeSpec:
    """Load a T-pose spec JSON.

    The returned dict is validated structurally (required keys present,
    each link has a 3x3 R and 3-vector pos). Numeric content is trusted —
    the authoring tool is responsible for producing sane values.
    """
    p = Path(path)
    with p.open() as f:
        data = json.load(f)

    for required in ("robot", "xml_path", "qpos", "links"):
        if required not in data:
            raise ValueError(f"T-pose spec at {p} missing required key: {required!r}")

    links = data["links"]
    if not isinstance(links, dict) or not links:
        raise ValueError(f"T-pose spec at {p} has empty or malformed 'links'")

    for name, entry in links.items():
        if "pos" not in entry or "R" not in entry:
            raise ValueError(f"Link {name!r} in {p} missing 'pos' or 'R'")
        if len(entry["pos"]) != 3:
            raise ValueError(f"Link {name!r} pos must be length-3, got {entry['pos']!r}")
        R = entry["R"]
        if len(R) != 3 or any(len(row) != 3 for row in R):
            raise ValueError(f"Link {name!r} R must be 3x3, got shape {len(R)}x?")

    return cast("TposeSpec", data)


def _rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a 3x3 rotation matrix to (unit_axis, angle_radians).

    Returns ``(z_axis, 0.0)`` when the matrix is (near-)identity — the axis
    is undefined there but we pick a stable convention so callers don't
    have to special-case zero rotations. For angles near pi we use the
    symmetric extraction that stays numerically stable.
    """
    # Clamp trace to valid arccos domain — floating-point error can push
    # trace slightly outside [-1, 3] even for valid rotations.
    trace = float(np.clip(R[0, 0] + R[1, 1] + R[2, 2], -1.0, 3.0))
    cos_angle = (trace - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))

    if angle < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0

    if np.pi - angle < 1e-6:
        # Near-180° rotation: standard axis-angle extraction is unstable
        # (sin(angle) ~ 0). Use the diagonal of R + I instead.
        diag = np.array([R[0, 0], R[1, 1], R[2, 2]]) + 1.0
        diag = np.clip(diag, 0.0, None)
        axis = np.sqrt(diag * 0.5)
        # Fix signs from off-diagonals
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


def compute_deviations(
    qpos: np.ndarray,
    xml_path: str | Path,
    spec: TposeSpec,
) -> dict[str, Deviation]:
    """Compute per-link axis-angle deviation from the spec.

    For each link named in ``spec['links']``, we read the body's world-frame
    rotation ``R_actual`` from ``data.xmat`` after ``mj_forward``, and compute
    the residual rotation ``R_err = R_actual @ R_expected.T``. Its axis-angle
    is the per-link deviation: angle is a non-negative scalar "how wrong",
    axis is the world-frame direction that would rotate R_actual onto R_expected.

    Links present in the spec but not found in the XML are skipped with
    ``angle_deg = nan``; callers can filter these out or treat them as hard
    errors depending on context.
    """
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised only without mujoco
        raise ImportError(
            "compute_deviations requires mujoco; install with: pip install -e '.[demo]'"
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != (model.nq,):
        raise ValueError(f"qpos shape mismatch: got {qpos.shape}, model expects ({model.nq},)")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    report: dict[str, Deviation] = {}
    for link_name, frame in spec["links"].items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if body_id < 0:
            report[link_name] = {"axis": [0.0, 0.0, 1.0], "angle_deg": float("nan")}
            continue

        R_actual = np.asarray(data.xmat[body_id]).reshape(3, 3).copy()
        R_expected = np.asarray(frame["R"], dtype=np.float64)
        R_err = R_actual @ R_expected.T
        axis, angle = _rotation_matrix_to_axis_angle(R_err)
        report[link_name] = {
            "axis": [float(axis[0]), float(axis[1]), float(axis[2])],
            "angle_deg": float(np.degrees(angle)),
        }

    return report


def total_deviation(report: dict[str, Deviation]) -> float:
    """Sum of |angle_deg| across all links, ignoring NaN entries.

    The monotone scalar Phase 2's regression gate uses: a patch that increases
    this value made things worse and should be reverted.
    """
    return float(sum(d["angle_deg"] for d in report.values() if not np.isnan(d["angle_deg"])))


def worst_k(report: dict[str, Deviation], k: int = 5) -> list[tuple[str, float]]:
    """Top-k links by |angle_deg|, descending. NaN entries are excluded."""
    finite = [(name, d["angle_deg"]) for name, d in report.items() if not np.isnan(d["angle_deg"])]
    finite.sort(key=lambda kv: kv[1], reverse=True)
    return finite[:k]
