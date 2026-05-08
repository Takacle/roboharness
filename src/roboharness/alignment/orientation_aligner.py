"""Auto-detect world_rotation from robot body geometry.

Compares the robot's default-pose body positions (parsed from MuJoCo XML)
against the expected human coordinate convention and returns the quaternion
that aligns the two frames.

The human convention used depends on ``src_format``:

* ``"bvh"`` — after the BVH/LAFAN1 loader, the coordinate axes are
  X=right, Y=forward, Z=up.
* ``"smplx"`` — the SMPL-X convention (X=right, Y=up, Z=forward) is
  incompatible with a pure rotation for the typical robot (X=forward,
  Z=up).  The IK solver handles the alignment via the root's free
  joint, so we return ``None``.

Supports robots with missing arms, head, or legs via fallback logic.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from roboharness._math_utils import (
    normalize_vector,
)
from roboharness._math_utils import (
    rotation_matrix_to_quat as _rotation_matrix_to_quat_scalar_first,
)
from roboharness.alignment.body_matcher import MatchResult


def _parse_pos(s: str) -> np.ndarray:
    return np.fromstring(s, dtype=np.float64, sep=" ")


def _parse_quat(s: str) -> np.ndarray:
    """Parse MuJoCo quat (w x y z) → xyzw for scipy."""
    vals = _parse_pos(s)
    return np.array([vals[1], vals[2], vals[3], vals[0]], dtype=np.float64)


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector *v* by quaternion *q* (xyzw)."""
    return R.from_quat(q).apply(v)  # type: ignore[no-any-return]


def _resolve_includes(xml_elem: ET.Element, base_dir: Path) -> None:
    """Recursively resolve ``<include file=\"...\"/>`` in-place.

    Included files are parsed and their children are grafted into *xml_elem*
    at the include position.  Nested includes are handled as well.
    """
    includes = list(xml_elem.iter("include"))
    for inc in includes:
        rel = inc.attrib.get("file", "")
        if not rel:
            continue
        inc_path = base_dir / rel
        if not inc_path.exists():
            continue
        inc_tree = ET.parse(str(inc_path))
        inc_root = inc_tree.getroot()
        _resolve_includes(inc_root, inc_path.parent)
        parent = xml_elem  # fallback — find parent below
        for ancestor in xml_elem.iter():
            for c in list(ancestor):
                if c is inc:
                    parent = ancestor
                    break
        insert_idx = list(parent).index(inc)
        grafted = list(inc_root)
        parent[insert_idx : insert_idx + 1] = grafted


def _collect_body_positions(xml_path: Path) -> dict[str, np.ndarray]:
    """Return {body_name: world_position} for the default (zero-DoF) pose.

    Walks the XML body tree with identity joint rotations.  The root body
    is placed at its XML ``pos`` (the *worldbody* offset) so that all
    positions share a common coordinate frame.

    Handles ``<include>`` elements by recursively merging included XMLs.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    _resolve_includes(root, xml_path.parent)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"<worldbody> not found in {xml_path}")
    root_elem = worldbody.find("body")
    if root_elem is None:
        raise ValueError(f"no <body> under <worldbody> in {xml_path}")

    positions: dict[str, np.ndarray] = {}

    def _walk(node: ET.Element, parent_pos: np.ndarray, parent_quat: np.ndarray) -> None:
        local_pos = _parse_pos(node.attrib.get("pos", "0 0 0"))
        local_quat = _parse_quat(node.attrib.get("quat", "1 0 0 0"))
        world_pos = parent_pos + _quat_rotate(parent_quat, local_pos)
        world_quat = (R.from_quat(parent_quat) * R.from_quat(local_quat)).as_quat()
        name = node.attrib.get("name")
        if name:
            positions[name] = world_pos
        for child in node.findall("body"):
            _walk(child, world_pos, world_quat)

    root_local_pos = _parse_pos(root_elem.attrib.get("pos", "0 0 0"))
    root_quat = _parse_quat(root_elem.attrib.get("quat", "1 0 0 0"))
    root_name = root_elem.attrib.get("name")
    if root_name:
        positions[root_name] = root_local_pos.copy()
    for child in root_elem.findall("body"):
        _walk(child, root_local_pos, root_quat)

    return positions


def compute_world_rotation(
    xml_path: Path,
    match: MatchResult,
    *,
    src_format: str = "bvh",
) -> list[float] | None:
    """Compute ``world_rotation`` from robot default-pose body geometry.

    Parameters
    ----------
    xml_path:
        Path to the robot's MuJoCo XML.
    match:
        Body-name matching result (role → robot body name).
    src_format:
        Human motion format; one of ``"bvh"``, ``"smplx"``, ``"fbx"``,
        ``"fbx_offline"``.

    Returns
    -------
    ``[w, x, y, z]`` quaternion list (scalar-first), or ``None`` when no
    rotation is needed / cannot be determined.
    """
    if src_format in ("smplx",):
        return None

    positions = _collect_body_positions(xml_path)

    # ── locate key landmarks ──
    root_pos = positions.get(match.mapping.get("root", ""))
    if root_pos is None:
        return None

    lh = positions.get(match.mapping.get("left_hip", ""))
    rh = positions.get(match.mapping.get("right_hip", ""))
    ls = positions.get(match.mapping.get("left_shoulder", ""))
    rs = positions.get(match.mapping.get("right_shoulder", ""))
    spine_rb = match.mapping.get("spine", "")
    spine_pos = positions.get(spine_rb)

    hip_pos = root_pos
    if lh is not None and rh is not None:
        hip_pos = (lh + rh) / 2.0
    elif lh is not None:
        hip_pos = lh
    elif rh is not None:
        hip_pos = rh

    # ── robot up ──
    robot_up: np.ndarray | None = None
    if spine_pos is not None:
        robot_up = normalize_vector(spine_pos - hip_pos)
    if robot_up is None or np.linalg.norm(robot_up) < 1e-8:
        robot_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # ── robot lateral / right ──
    lat_raw: np.ndarray | None = None
    if ls is not None and rs is not None:
        lat_raw = ls - rs
    elif lh is not None and rh is not None:
        lat_raw = lh - rh
    else:
        lat_raw = np.cross(robot_up, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(lat_raw) < 1e-8:
            lat_raw = np.cross(robot_up, np.array([0.0, 1.0, 0.0]))
    robot_left = normalize_vector(lat_raw)

    # ── robot forward ──
    # Post-loader convention for BVH/FBX: X=left, Y=forward, Z=up
    # forward = cross(up, left)  →  robot's -X direction (backward)
    robot_forward = normalize_vector(np.cross(robot_up, robot_left))
    robot_left = normalize_vector(np.cross(robot_forward, robot_up))
    robot_frame = np.column_stack([robot_left, robot_forward, robot_up])

    # Ensure a proper rotation (det = +1) via SVD projection.
    U, _, Vt = np.linalg.svd(robot_frame)
    R_mat = U @ Vt
    if np.linalg.det(R_mat) < 0.0:
        Vt[-1, :] *= -1
        R_mat = U @ Vt

    # Skip if the rotation is effectively identity.
    if np.allclose(R_mat, np.eye(3), atol=1e-4):
        return None

    return _rotation_matrix_to_quat_scalar_first(R_mat)


def parse_world_rotation_arg(value: str) -> list[float]:
    """Parse a ``--world_rot`` CLI value into a scalar-first quaternion.

    Input format: ``"angle,axis_x,axis_y,axis_z"`` where *angle* is in
    degrees and the axis does not need to be pre-normalised.

    Returns
    -------
    ``[w, x, y, z]`` normalised quaternion list (scalar-first).

    Raises
    ------
    ``ValueError`` on wrong number of fields, non-float values, or
    zero-length axis.
    """
    from roboharness._math_utils import axis_angle_to_quat

    parts = value.split(",")
    if len(parts) != 4:
        raise ValueError(
            f"--world_rot requires 4 comma-separated values (angle,ax,ay,az), got {len(parts)}"
        )
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"--world_rot values must be floats, got {value!r}") from None
    angle_deg, ax, ay, az = nums
    norm = (ax * ax + ay * ay + az * az) ** 0.5
    if norm < 1e-9:
        raise ValueError("--world_rot axis has zero norm")
    axis = [ax / norm, ay / norm, az / norm]
    return axis_angle_to_quat(axis, angle_deg)


def extract_xml_body_names(xml_path: Path) -> list[str]:
    """Extract all body names from a MuJoCo XML file via regex.

    This is a lightweight alternative to ``_collect_body_positions`` for
    cases where only body names are needed, not world-space positions.
    """
    import re

    return sorted(set(re.findall(r'<body\s+name="([^"]+)"', xml_path.read_text())))


def apply_smplx_base_rotation(spec: dict) -> dict:
    """Pre-multiply all ``R`` matrices in a T-pose spec by the SMPL-X base rotation.

    The SMPL-X coordinate frame (Y-up, X=left) differs from MuJoCo (Z-up, X=forward).
    This helper applies the conversion so that downstream ``compute_direct_patch``
    and ``compute_deviations`` operate in a consistent frame.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    from roboharness._math_utils import SMPLX_BASE_ROTATION_QUAT

    r_base = R.from_quat(SMPLX_BASE_ROTATION_QUAT, scalar_first=True)
    mat_base = r_base.as_matrix()
    mod_links: dict[str, dict] = {}
    for name, info in spec.get("links", {}).items():
        R_body = np.asarray(info["R"], dtype=np.float64)
        mod_links[name] = {**info, "R": (mat_base @ R_body).tolist()}
    return {**spec, "links": mod_links}
