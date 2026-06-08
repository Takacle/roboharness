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

from gmr_harness._math_utils import rotation_matrix_to_axis_angle


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


class PosDeviation(TypedDict):
    pos_err_m: float  # Euclidean distance in metres
    direction: list[float]  # unit vector from actual to expected (world frame)


def load_tpose_spec(path: str | Path) -> TposeSpec:
    """Load a T-pose spec JSON.

    The returned dict is validated structurally (required keys present,
    each link has a 3x3 R and 3-vector pos). Numeric content is trusted —
    the authoring tool is responsible for producing sane values.

    If ``xml_path`` is relative, it is resolved against GMR_ROOT at load time
    and stored in ``_resolved_xml_path`` for downstream consumers.
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

    xml = Path(data["xml_path"])
    if not xml.is_absolute():
        try:
            from gmr_harness.alignment._gmr_path import find_gmr_root

            resolved = find_gmr_root() / xml
            if resolved.exists():
                data["_resolved_xml_path"] = str(resolved)
        except FileNotFoundError:
            pass
    else:
        data["_resolved_xml_path"] = str(xml)

    return cast("TposeSpec", data)


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
        axis, angle = rotation_matrix_to_axis_angle(R_err)
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


def compute_position_deviations(
    qpos: np.ndarray,
    xml_path: str | Path,
    spec: TposeSpec,
) -> dict[str, PosDeviation]:
    """Compute per-link Euclidean position error against the T-pose spec.

    For each link named in ``spec['links']``, read the body's world-frame
    position from ``data.xpos`` after ``mj_forward``, and return the
    Euclidean distance from the expected position in the spec.

    Links present in the spec but not found in the XML are skipped with
    ``pos_err_m = nan``.

    Parameters
    ----------
    qpos:
        Candidate joint configuration (length ``model.nq``).
    xml_path:
        Path to the robot MuJoCo XML.
    spec:
        T-pose spec loaded via ``load_tpose_spec``.

    Returns
    -------
    ``{link_name: {"pos_err_m": float, "direction": [dx, dy, dz]}}`` where
    *pos_err_m* is the Euclidean distance (metres) and *direction* is a
    unit vector pointing from the actual body position to the expected one.
    """
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "compute_position_deviations requires mujoco; install with: pip install -e '.[demo]'"
        ) from exc

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.shape != (model.nq,):
        raise ValueError(f"qpos shape mismatch: got {qpos.shape}, model expects ({model.nq},)")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    report: dict[str, PosDeviation] = {}
    for link_name, frame in spec["links"].items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        if body_id < 0:
            report[link_name] = {"pos_err_m": float("nan"), "direction": [0.0, 0.0, 1.0]}
            continue

        pos_actual = np.asarray(data.xpos[body_id], dtype=np.float64)
        pos_expected = np.asarray(frame["pos"], dtype=np.float64)
        diff = pos_actual - pos_expected
        err_m = float(np.linalg.norm(diff))
        direction = (
            list(-diff / err_m) if err_m > 1e-9 else [0.0, 0.0, 1.0]
        )  # from actual to expected

        report[link_name] = {"pos_err_m": err_m, "direction": direction}

    return report


def total_position_deviation(report: dict[str, PosDeviation]) -> float:
    """Sum of ``pos_err_m`` across all links, ignoring NaN entries."""
    return float(sum(d["pos_err_m"] for d in report.values() if not np.isnan(d["pos_err_m"])))


def worst_k_position(report: dict[str, PosDeviation], k: int = 5) -> list[tuple[str, float]]:
    """Top-k links by ``pos_err_m``, descending. NaN entries are excluded."""
    finite = [(name, d["pos_err_m"]) for name, d in report.items() if not np.isnan(d["pos_err_m"])]
    finite.sort(key=lambda kv: kv[1], reverse=True)
    return finite[:k]


def worst_k(report: dict[str, Deviation], k: int = 5) -> list[tuple[str, float]]:
    """Top-k links by |angle_deg|, descending. NaN entries are excluded."""
    finite = [(name, d["angle_deg"]) for name, d in report.items() if not np.isnan(d["angle_deg"])]
    finite.sort(key=lambda kv: kv[1], reverse=True)
    return finite[:k]


def compute_direct_patch(
    human_data: dict,
    config: dict,
    tpose_spec: dict,
    preserve: set[str] | None = None,
) -> dict:
    """Compute correct IK config quaternions directly from human orientations.

    At T-pose, the GMR IK solver targets each robot link with:
        ``Q_target = Q_human_world * Q_ik_offset``   (scipy right-multiply)
        ``P_target = P_human_world + Q_target.apply(P_ik_offset)``

    where *Q_human_world* is the human bone quaternion after scaling and
    world_rotation.  Given the expected robot orientation *R_expected* from
    the T-pose spec, the correct offset is:
        ``Q_ik_offset = Q_human_world^{-1} * R_expected``
    The matching position offset is:
        ``P_ik_offset = R_expected^{-1} * (P_expected - P_human_world)``

    This function computes that offset for every joint in *config* whose
    human bone name appears in *human_data* and whose robot joint name
    appears in *tpose_spec*.

    Parameters
    ----------
    human_data:
        Output of ``scaled_human_reference(retargeter, tpose_frame)`` —
        a dict ``{bone_name: (pos, quat)}`` where *quat* is scalar-first
        ``[w, x, y, z]``.
    config:
        The IK config dict (with ``ik_match_table1`` / ``ik_match_table2``).
    tpose_spec:
        The T-pose spec dict (with ``links`` containing expected ``R`` matrices).
    preserve:
        Joint names to leave unchanged (e.g. ``{"pelvis", "torso_link"}``
        for joints that were already manually tuned for correct facing
        direction).

    Returns
    -------
    A patch dict with ``mode: "set"`` for every computed quaternion and
    ``pos_offset`` for every link that also has an expected position.
    """
    from scipy.spatial.transform import Rotation as R

    patch_t1: dict[str, dict] = {}
    patch_t2: dict[str, dict] = {}
    preserve_set = preserve or set()

    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = config.get(table_name, {})
        for robot_joint_name, entry in table.items():
            if robot_joint_name in preserve_set:
                continue
            human_bone_name = entry[0]
            if human_bone_name not in human_data:
                continue
            if robot_joint_name not in tpose_spec.get("links", {}):
                continue

            p_human, q_human = human_data[human_bone_name]
            p_human_arr = np.asarray(p_human, dtype=np.float64)
            q_human_arr = np.asarray(q_human, dtype=np.float64)
            norm = float(np.linalg.norm(q_human_arr))
            if norm < 1e-9:
                continue
            q_human_arr = q_human_arr / norm

            R_expected = np.asarray(tpose_spec["links"][robot_joint_name]["R"], dtype=np.float64)
            p_expected = np.asarray(
                tpose_spec["links"][robot_joint_name].get("pos", p_human_arr), dtype=np.float64
            )
            r_human = R.from_quat(q_human_arr, scalar_first=True)
            r_target = R.from_matrix(R_expected)
            r_offset = r_human.inv() * r_target
            q_offset = [float(v) for v in r_offset.as_quat(scalar_first=True)]
            pos_offset = [float(v) for v in r_target.inv().apply(p_expected - p_human_arr)]

            spec = {"mode": "set", "quat": q_offset, "pos_offset": pos_offset}
            if table_name == "ik_match_table1":
                patch_t1[robot_joint_name] = spec
            else:
                patch_t2[robot_joint_name] = spec

    return {"ik_match_table1": patch_t1, "ik_match_table2": patch_t2}
