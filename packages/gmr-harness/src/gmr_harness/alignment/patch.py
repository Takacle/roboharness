"""IK config patching for quaternion offsets and scale values.

Extracted from ``examples/gmr_alignment_agent.py`` so the logic is

  1. unit-testable in isolation from GMR, MuJoCo, and the VLM, and
  2. reusable by any future tuner (not only the VLM loop).

A GMR IK config has two dictionaries — ``ik_match_table1`` (strict) and
``ik_match_table2`` (relaxed, same joints) — that MUST share the same
quaternion offset per joint; see ``docs/gmr-alignment-sop.md`` §7. The
historical ``apply_patch`` in the VLM example did not enforce this, so a
VLM that only patched table1 silently drifted the tables apart, and the
IK solver pulled the joint back to the stale table2 value. This module's
``apply_patch`` auto-mirrors any single-table patch and, optionally,
raises when explicit per-table entries disagree.

In addition to quaternion/position offset patches (``ik_match_table1/2`` and
``world_rotation``), ``apply_patch`` accepts scale-related keys:

  - ``human_scale_table``: ``{bone_name: {"mode": "set"|"mul", "value": float}}``
  - ``human_height_assumption``: ``{"mode": "set"|"mul", "value": float}``

All quaternions are scalar-first ``[w, x, y, z]`` — the same convention
used by ``mj_ref_quat`` and the IK config JSON on disk.
"""

from __future__ import annotations

import copy
from typing import Literal

from gmr_harness._math_utils import normalize_quat, quat_multiply

quat_normalize = normalize_quat  # backward-compat alias

Quat = list[float]  # length-4, scalar-first [w, x, y, z]
Vec3 = list[float]  # length-3 position offset [x, y, z]

SCALE_BOUNDS: tuple[float, float] = (0.2, 2.5)
"""Clamp range for individual bone scale values in ``human_scale_table``."""

HEIGHT_BOUNDS: tuple[float, float] = (0.5, 3.0)
"""Clamp range for ``human_height_assumption`` (metres)."""

WEIGHT_BOUNDS: tuple[float, float] = (0.0, 1000.0)
"""Clamp range for ``pos_weight`` / ``rot_weight`` in ik_match_table entries."""


def _resolve_quat_spec(spec: object, current: Quat) -> Quat:
    """Resolve one patch entry (dict or raw list) into a normalized quaternion.

    Accepts either ``{"mode": "set"|"mul", "quat": [w,x,y,z]}`` or a bare
    ``[w,x,y,z]`` (treated as ``set``). When the outer dict contains other
    keys (e.g. ``pos_weight``), ``spec["quat"]`` itself may be the nested
    quaternion spec dict — this is resolved transparently.

    ``mul`` applies as ``q_new = patch * current`` per the SOP.
    """
    if isinstance(spec, dict):
        inner = spec["quat"]
        spec = inner if isinstance(inner, dict) else spec
    # From here, spec is either a bare [w,x,y,z] or {"mode":"...","quat":[...]}
    if isinstance(spec, dict):
        mode = spec.get("mode", "set")
        quat = spec["quat"]
    else:
        mode = "set"
        quat = spec

    quat = normalize_quat(list(quat))
    if mode == "mul":
        quat = normalize_quat(quat_multiply(quat, current))
    elif mode != "set":
        raise ValueError(f"Unknown patch mode {mode!r}; use 'set' or 'mul'.")
    return quat


def _is_pos_offset_patch(spec: object) -> bool:
    """Return True if *spec* contains a position-offset patch."""
    return isinstance(spec, dict) and "pos_offset" in spec


def _resolve_pos_offset_spec(spec: object) -> Vec3:
    """Resolve a position-offset patch into a length-3 float vector."""
    if not isinstance(spec, dict):
        raise ValueError("Position offset patch must be a dict with 'pos_offset'.")
    pos = spec["pos_offset"]
    if len(pos) != 3:
        raise ValueError(f"pos_offset must be length 3, got {pos!r}")
    return [float(v) for v in pos]


def _resolve_scale_spec(
    spec: object,
    current: float,
    bounds: tuple[float, float] = SCALE_BOUNDS,
) -> float:
    """Resolve one scale patch entry into a float, enforcing bounds.

    Accepts either ``{"mode": "set"|"mul", "value": 0.75}`` or a bare
    ``0.75`` (treated as ``set``). ``mul`` applies as ``new = current * value``.
    """
    if isinstance(spec, dict):
        mode = spec.get("mode", "set")
        value = spec["value"]
    elif isinstance(spec, (int, float)):
        mode = "set"
        value = float(spec)
    else:
        raise ValueError(
            f"Invalid scale spec {spec!r}; expected float or "
            "{'mode': 'set'|'mul', 'value': float}"
        )

    value = float(value)
    if mode == "mul":
        value = current * value
    elif mode != "set":
        raise ValueError(f"Unknown scale patch mode {mode!r}; use 'set' or 'mul'.")

    lo, hi = bounds
    if not (lo <= value <= hi):
        raise ValueError(f"Scale value {value} out of bounds [{lo}, {hi}]")

    return value


MirrorPolicy = Literal["auto", "strict", "off"]


def _is_quat_patch(spec: object) -> bool:
    """True if *spec* contains quaternion patching instructions.

    A bare list is always a quaternion patch. A dict is a quaternion patch
    only if it has a ``"quat"`` key (weight-only patches have other keys).
    """
    if not isinstance(spec, dict):
        return True
    return "quat" in spec


def _apply_weight_changes(
    entry: list,
    new_pw: float | None,
    new_rw: float | None,
) -> None:
    """Apply resolved weight changes to one ik_match_table entry."""
    if new_pw is not None:
        entry[1] = new_pw  # pos_weight at index 1
    if new_rw is not None:
        entry[2] = new_rw  # rot_weight at index 2


def _mirror_single_weight(
    v1: float | None,
    v2: float | None,
    fname: str,
    joint: str,
    mirror: MirrorPolicy,
) -> tuple[float | None, float | None]:
    """Mirror a single weight field per table1↔table2 policy."""
    if v1 is not None and v2 is not None:
        if mirror == "strict" and abs(v1 - v2) > 1e-9:
            raise ValueError(
                f"table1/table2 {fname} for {joint!r} disagree: {v1} vs {v2}. See SOP §7."
            )
        return v1, v2
    if v1 is not None:
        if mirror == "strict":
            raise ValueError(
                f"{fname} for {joint!r} patched only in "
                "ik_match_table1. Per SOP §7, both tables "
                "must be patched together."
            )
        return v1, v1
    if v2 is not None:
        if mirror == "strict":
            raise ValueError(
                f"{fname} for {joint!r} patched only in "
                "ik_match_table2. Per SOP §7, both tables "
                "must be patched together."
            )
        return v2, v2
    return None, None


def apply_patch(
    config: dict,
    patch: dict,
    mirror: MirrorPolicy = "auto",
) -> dict:
    """Apply a VLM / agent patch to a GMR IK config, enforcing table1↔table2.

    Parameters
    ----------
    config:
        The IK config loaded from ``<src>_to_<robot>.json`` — mutated
        inside a deep copy, never in place.
    patch:
        Optional keys:

        - ``ik_match_table1`` / ``ik_match_table2``: ``{joint: quat_spec}``
          where ``quat_spec`` is ``[w,x,y,z]`` or
          ``{"mode": "set"|"mul", "quat": [w,x,y,z]}``.
        - ``world_rotation``: top-level ``[w,x,y,z]`` (always ``set``).
        - ``human_scale_table``: ``{bone: value_spec}`` where ``value_spec``
          is a bare ``float`` (set) or ``{"mode": "set"|"mul", "value": f}``.
          Values are clamped to ``[0.2, 2.5]``.
        - ``human_height_assumption``: bare ``float`` (set) or
          ``{"mode": "set"|"mul", "value": f}``. Clamped to ``[0.5, 3.0]``.
    mirror:
        How to reconcile table1/table2 for the same joint:

        - ``"auto"`` (default): if a joint is patched in only one table,
          the same resolved quaternion is auto-applied to the other
          table. If both tables are patched with disagreeing quats, the
          disagreement is tolerated (caller knows what they're doing).
        - ``"strict"``: a joint patched in only one table is an error;
          both-table patches must resolve to the same quaternion.
        - ``"off"``: legacy behavior — each table patched independently.

    Returns
    -------
    A new config dict. The input is never mutated.
    """
    result = copy.deepcopy(config)

    p1: dict = patch.get("ik_match_table1") or {}
    p2: dict = patch.get("ik_match_table2") or {}
    t1: dict = result.get("ik_match_table1") or {}
    t2: dict = result.get("ik_match_table2") or {}

    joints = set(p1) | set(p2)
    for joint in joints:
        in1 = joint in p1
        in2 = joint in p2
        # Guard: SOP §7 says table1 and table2 hold the same joint set. If
        # the config violates that independently of the patch, do not try
        # to invent a missing entry — leave the anomaly alone so a real
        # config bug is visible rather than masked.
        has1 = joint in t1
        has2 = joint in t2
        if not (has1 or has2):
            continue

        pe1 = p1[joint] if in1 else None
        pe2 = p2[joint] if in2 else None

        # --- Quaternion ---
        qp1 = _is_quat_patch(pe1) if in1 else False
        qp2 = _is_quat_patch(pe2) if in2 else False

        current1: Quat = t1[joint][4] if has1 else [1.0, 0.0, 0.0, 0.0]
        current2: Quat = t2[joint][4] if has2 else [1.0, 0.0, 0.0, 0.0]

        new1: Quat | None = _resolve_quat_spec(pe1, current1) if qp1 else None
        new2: Quat | None = _resolve_quat_spec(pe2, current2) if qp2 else None

        # --- Position offset ---
        posp1 = _is_pos_offset_patch(pe1) if in1 else False
        posp2 = _is_pos_offset_patch(pe2) if in2 else False
        new_pos1: Vec3 | None = _resolve_pos_offset_spec(pe1) if posp1 else None
        new_pos2: Vec3 | None = _resolve_pos_offset_spec(pe2) if posp2 else None

        # --- Extract weight patches ---
        pw1: float | None = None
        rw1: float | None = None
        pw2: float | None = None
        rw2: float | None = None

        if isinstance(pe1, dict):
            if "pos_weight" in pe1:
                pw1 = _resolve_scale_spec(
                    pe1["pos_weight"], t1[joint][1] if has1 else 0, WEIGHT_BOUNDS
                )
            if "rot_weight" in pe1:
                rw1 = _resolve_scale_spec(
                    pe1["rot_weight"], t1[joint][2] if has1 else 0, WEIGHT_BOUNDS
                )
        if isinstance(pe2, dict):
            if "pos_weight" in pe2:
                pw2 = _resolve_scale_spec(
                    pe2["pos_weight"], t2[joint][1] if has2 else 0, WEIGHT_BOUNDS
                )
            if "rot_weight" in pe2:
                rw2 = _resolve_scale_spec(
                    pe2["rot_weight"], t2[joint][2] if has2 else 0, WEIGHT_BOUNDS
                )

        # --- Apply quaternion changes ---
        if mirror == "off":
            if has1 and new1 is not None:
                t1[joint][4] = new1
            if has2 and new2 is not None:
                t2[joint][4] = new2
        else:
            # auto / strict both reconcile; strict refuses to paper over mismatch.
            if in1 and in2:
                if not _quats_close(new1, new2) and mirror == "strict":
                    raise ValueError(
                        f"table1/table2 quaternions for {joint!r} disagree: "
                        f"{new1} vs {new2}. See SOP §7."
                    )
                if has1 and new1 is not None:
                    t1[joint][4] = new1
                if has2 and new2 is not None:
                    t2[joint][4] = new2
            elif in1:
                if mirror == "strict" and new1 is not None:
                    raise ValueError(
                        f"Joint {joint!r} patched only in ik_match_table1. "
                        "Per SOP §7, both tables must be patched together."
                    )
                if has1 and new1 is not None:
                    t1[joint][4] = new1  # type: ignore[assignment]
                if has2 and new1 is not None:
                    t2[joint][4] = new1  # type: ignore[assignment]
            else:  # in2 only
                if mirror == "strict" and new2 is not None:
                    raise ValueError(
                        f"Joint {joint!r} patched only in ik_match_table2. "
                        "Per SOP §7, both tables must be patched together."
                    )
                if has2 and new2 is not None:
                    t2[joint][4] = new2  # type: ignore[assignment]
                if has1 and new2 is not None:
                    t1[joint][4] = new2  # type: ignore[assignment]

        # --- Apply position-offset changes ---
        if mirror == "off":
            if has1 and new_pos1 is not None:
                t1[joint][3] = new_pos1
            if has2 and new_pos2 is not None:
                t2[joint][3] = new_pos2
        else:
            if in1 and in2:
                if (
                    new_pos1 is not None
                    and new_pos2 is not None
                    and new_pos1 != new_pos2
                    and mirror == "strict"
                ):
                    raise ValueError(
                        f"table1/table2 pos_offsets for {joint!r} disagree: "
                        f"{new_pos1} vs {new_pos2}. See SOP §7."
                    )
                chosen = new_pos1 if new_pos1 is not None else new_pos2
                if has1 and chosen is not None:
                    t1[joint][3] = chosen
                if has2 and chosen is not None:
                    t2[joint][3] = chosen
            elif in1:
                if mirror == "strict" and new_pos1 is not None:
                    raise ValueError(
                        f"Joint {joint!r} patched only in ik_match_table1. "
                        "Per SOP §7, both tables must be patched together."
                    )
                if has1 and new_pos1 is not None:
                    t1[joint][3] = new_pos1
                if has2 and new_pos1 is not None:
                    t2[joint][3] = new_pos1
            else:  # in2 only
                if mirror == "strict" and new_pos2 is not None:
                    raise ValueError(
                        f"Joint {joint!r} patched only in ik_match_table2. "
                        "Per SOP §7, both tables must be patched together."
                    )
                if has2 and new_pos2 is not None:
                    t2[joint][3] = new_pos2
                if has1 and new_pos2 is not None:
                    t1[joint][3] = new_pos2

        # --- Apply weight changes ---
        if mirror == "off":
            if has1:
                _apply_weight_changes(t1[joint], pw1, rw1)
            if has2:
                _apply_weight_changes(t2[joint], pw2, rw2)
        else:
            eff_pw1, eff_pw2 = _mirror_single_weight(pw1, pw2, "pos_weight", joint, mirror)
            eff_rw1, eff_rw2 = _mirror_single_weight(rw1, rw2, "rot_weight", joint, mirror)
            if has1:
                _apply_weight_changes(t1[joint], eff_pw1, eff_rw1)
            if has2:
                _apply_weight_changes(t2[joint], eff_pw2, eff_rw2)

    if "world_rotation" in patch:
        result["world_rotation"] = normalize_quat(list(patch["world_rotation"]))

    # --- human_scale_table ---
    scale_patch: dict = patch.get("human_scale_table") or {}
    if scale_patch:
        scale_table: dict = result.setdefault("human_scale_table", {})
        for bone, spec in scale_patch.items():
            current = scale_table.get(bone, 1.0)
            scale_table[bone] = _resolve_scale_spec(spec, current, SCALE_BOUNDS)

    # --- human_height_assumption ---
    if "human_height_assumption" in patch:
        current = result.get("human_height_assumption", 1.8)
        result["human_height_assumption"] = _resolve_scale_spec(
            patch["human_height_assumption"], current, HEIGHT_BOUNDS
        )

    return result


def _quats_close(q1: Quat | None, q2: Quat | None, tol: float = 1e-6) -> bool:
    """True if two (already-normalized) quaternions represent the same rotation.

    Accounts for double cover: ``q`` and ``-q`` are the same rotation.
    """
    if q1 is None or q2 is None:
        return q1 is q2
    direct = sum(abs(a - b) for a, b in zip(q1, q2, strict=True))
    flipped = sum(abs(a + b) for a, b in zip(q1, q2, strict=True))
    return min(direct, flipped) < tol
