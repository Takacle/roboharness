"""Unit tests for roboharness.alignment.patch.

These tests exercise apply_patch without any GMR / MuJoCo dependency — the
module is pure Python quaternion math. That's precisely why we extracted it:
VLM-loop regressions from Phase 2 should be catchable by ``pytest -q`` in
milliseconds, not by a 60-frame retarget cycle.
"""

from __future__ import annotations

import copy
import math

import pytest

from roboharness._math_utils import normalize_quat as quat_normalize
from roboharness._math_utils import quat_multiply
from roboharness.alignment import apply_patch
from roboharness.alignment.patch import (
    SCALE_BOUNDS,
    _quats_close,
    _resolve_quat_spec,
    _resolve_scale_spec,
)


def _make_config(joints: list[str], quat: list[float] | None = None) -> dict:
    """IK config with both tables populated with identity offsets for ``joints``."""
    q = quat or [1.0, 0.0, 0.0, 0.0]
    entry = lambda: ["human_bone", 1.0, 1.0, [0.0, 0.0, 0.0], list(q)]  # noqa: E731
    return {
        "ik_match_table1": {j: entry() for j in joints},
        "ik_match_table2": {j: entry() for j in joints},
        "world_rotation": [1.0, 0.0, 0.0, 0.0],
    }


# ---------------------------------------------------------------------------
# quaternion helpers
# ---------------------------------------------------------------------------


def test_quat_normalize_unit_preserved() -> None:
    q = quat_normalize([0.5, 0.5, 0.5, 0.5])
    assert math.isclose(sum(v * v for v in q), 1.0, abs_tol=1e-9)
    for a, b in zip(q, [0.5, 0.5, 0.5, 0.5], strict=True):
        assert math.isclose(a, b, abs_tol=1e-9)


def test_quat_normalize_zero_returns_identity() -> None:
    assert quat_normalize([0.0, 0.0, 0.0, 0.0]) == [1.0, 0.0, 0.0, 0.0]


def test_quat_multiply_identity() -> None:
    q = [0.707, 0.0, 0.707, 0.0]
    out = quat_multiply([1.0, 0.0, 0.0, 0.0], q)
    assert out == q


def test_quat_multiply_90x_90x_equals_180x() -> None:
    s = math.sqrt(0.5)
    q90 = [s, s, 0.0, 0.0]
    out = quat_normalize(quat_multiply(q90, q90))
    # 180° about +x → [0, 1, 0, 0] (up to double cover)
    assert _quats_close(out, [0.0, 1.0, 0.0, 0.0])


def test_quats_close_double_cover() -> None:
    assert _quats_close([0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0])
    assert not _quats_close([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0])


def test_resolve_quat_spec_set_and_mul() -> None:
    cur = [0.707, 0.0, 0.707, 0.0]  # 90° about +y
    out_set = _resolve_quat_spec([1.0, 0.0, 0.0, 0.0], cur)
    assert _quats_close(out_set, [1.0, 0.0, 0.0, 0.0])

    # mul 90°+y with current 90°+y → 180° +y
    out_mul = _resolve_quat_spec({"mode": "mul", "quat": [0.707, 0.0, 0.707, 0.0]}, cur)
    assert _quats_close(out_mul, [0.0, 0.0, 1.0, 0.0])


def test_resolve_quat_spec_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown patch mode"):
        _resolve_quat_spec({"mode": "add", "quat": [1.0, 0.0, 0.0, 0.0]}, [1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# apply_patch — basics
# ---------------------------------------------------------------------------


def test_apply_patch_does_not_mutate_input() -> None:
    cfg = _make_config(["j"])
    snapshot = copy.deepcopy(cfg)
    apply_patch(cfg, {"ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]}})
    assert cfg == snapshot


def test_apply_patch_set_mode_bare_list() -> None:
    cfg = _make_config(["j"])
    out = apply_patch(cfg, {"ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]}})
    assert out["ik_match_table1"]["j"][4] == [0.0, 1.0, 0.0, 0.0]


def test_apply_patch_set_mode_dict() -> None:
    cfg = _make_config(["j"])
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"mode": "set", "quat": [0.0, 0.0, 1.0, 0.0]}}},
    )
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 0.0, 1.0, 0.0])


def test_apply_patch_mul_mode_composes() -> None:
    s = math.sqrt(0.5)
    cfg = _make_config(["j"], quat=[s, s, 0.0, 0.0])  # 90°+x currently
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"mode": "mul", "quat": [s, s, 0.0, 0.0]}}},
    )
    # 90°+x * 90°+x = 180°+x
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])


def test_apply_patch_world_rotation() -> None:
    cfg = _make_config(["j"])
    out = apply_patch(cfg, {"world_rotation": [0.0, 2.0, 0.0, 0.0]})  # un-normalized
    assert _quats_close(out["world_rotation"], [0.0, 1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# apply_patch — mirror policy (SOP §7)
# ---------------------------------------------------------------------------


def test_mirror_auto_mirrors_single_table_patch() -> None:
    cfg = _make_config(["j"])
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]}},
        mirror="auto",
    )
    # Default "auto": patch applied to table1 auto-propagates to table2 so the
    # IK solver's strict and relaxed tables stay in agreement.
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])
    assert _quats_close(out["ik_match_table2"]["j"][4], [0.0, 1.0, 0.0, 0.0])


def test_mirror_off_does_not_mirror() -> None:
    cfg = _make_config(["j"])
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]}},
        mirror="off",
    )
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])
    # Untouched
    assert _quats_close(out["ik_match_table2"]["j"][4], [1.0, 0.0, 0.0, 0.0])


def test_mirror_strict_single_table_raises() -> None:
    cfg = _make_config(["j"])
    with pytest.raises(ValueError, match="patched only in ik_match_table1"):
        apply_patch(
            cfg,
            {"ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]}},
            mirror="strict",
        )


def test_mirror_strict_agreeing_both_tables_ok() -> None:
    cfg = _make_config(["j"])
    out = apply_patch(
        cfg,
        {
            "ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]},
            "ik_match_table2": {"j": [0.0, 1.0, 0.0, 0.0]},
        },
        mirror="strict",
    )
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])
    assert _quats_close(out["ik_match_table2"]["j"][4], [0.0, 1.0, 0.0, 0.0])


def test_mirror_strict_disagreeing_both_tables_raises() -> None:
    cfg = _make_config(["j"])
    with pytest.raises(ValueError, match="disagree"):
        apply_patch(
            cfg,
            {
                "ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]},
                "ik_match_table2": {"j": [0.0, 0.0, 1.0, 0.0]},
            },
            mirror="strict",
        )


def test_mirror_auto_tolerates_disagreement() -> None:
    cfg = _make_config(["j"])
    out = apply_patch(
        cfg,
        {
            "ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]},
            "ik_match_table2": {"j": [0.0, 0.0, 1.0, 0.0]},
        },
        mirror="auto",
    )
    # Caller patched both explicitly — honor their choice in auto mode.
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])
    assert _quats_close(out["ik_match_table2"]["j"][4], [0.0, 0.0, 1.0, 0.0])


def test_apply_patch_skips_joint_absent_from_both_tables() -> None:
    cfg = _make_config(["left_shoulder"])
    # "right_shoulder" isn't in either table — patch must be a no-op, not a
    # KeyError or a stealth insertion.
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"right_shoulder": [0.0, 1.0, 0.0, 0.0]}},
    )
    assert "right_shoulder" not in out["ik_match_table1"]
    assert "right_shoulder" not in out["ik_match_table2"]


def test_apply_patch_mul_on_mirror_uses_each_tables_current() -> None:
    """mul mode must compose with the per-table current quat, not mix them."""
    s = math.sqrt(0.5)
    cfg = _make_config(["j"])
    # Start with table1 at 90°+x, table2 at identity — a broken config the
    # mirror-enforcement is meant to rescue. A mul patch of 90°+x in table1
    # should compose against table1's current (→ 180°+x) and (auto) mirror
    # that final value to table2.
    cfg["ik_match_table1"]["j"][4] = [s, s, 0.0, 0.0]
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"mode": "mul", "quat": [s, s, 0.0, 0.0]}}},
        mirror="auto",
    )
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])
    assert _quats_close(out["ik_match_table2"]["j"][4], [0.0, 1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# _resolve_scale_spec
# ---------------------------------------------------------------------------


def test_resolve_scale_spec_set_bare_float() -> None:
    assert _resolve_scale_spec(0.75, 1.0) == 0.75


def test_resolve_scale_spec_set_dict() -> None:
    assert _resolve_scale_spec({"mode": "set", "value": 0.85}, 1.0) == 0.85


def test_resolve_scale_spec_mul() -> None:
    assert _resolve_scale_spec({"mode": "mul", "value": 0.9}, 0.8) == pytest.approx(0.72)


def test_resolve_scale_spec_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown scale patch mode"):
        _resolve_scale_spec({"mode": "add", "value": 0.5}, 1.0)


def test_resolve_scale_spec_bounds_low() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        _resolve_scale_spec(0.05, 1.0, SCALE_BOUNDS)


def test_resolve_scale_spec_bounds_high() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        _resolve_scale_spec(5.0, 1.0, SCALE_BOUNDS)


def test_resolve_scale_spec_invalid_type() -> None:
    with pytest.raises(ValueError, match="Invalid scale spec"):
        _resolve_scale_spec("abc", 1.0)


# ---------------------------------------------------------------------------
# apply_patch — human_scale_table
# ---------------------------------------------------------------------------


def _make_config_with_scales(
    joints: list[str],
    scales: dict[str, float] | None = None,
    height: float = 1.8,
) -> dict:
    """IK config that also carries human_scale_table + human_height_assumption."""
    cfg = _make_config(joints)
    cfg["human_scale_table"] = scales or {"Hips": 0.9, "LeftArm": 0.75}
    cfg["human_height_assumption"] = height
    return cfg


def test_apply_patch_scale_table_set() -> None:
    cfg = _make_config_with_scales(["j"])
    out = apply_patch(
        cfg,
        {"human_scale_table": {"LeftArm": 0.85, "RightLeg": {"mode": "set", "value": 0.92}}},
    )
    assert out["human_scale_table"]["LeftArm"] == 0.85
    assert out["human_scale_table"]["RightLeg"] == 0.92
    assert out["ik_match_table1"]["j"][4] == [1.0, 0.0, 0.0, 0.0]


def test_apply_patch_scale_table_mul() -> None:
    cfg = _make_config_with_scales(["j"], scales={"Hips": 0.9, "LeftArm": 0.75})
    out = apply_patch(
        cfg,
        {"human_scale_table": {"LeftArm": {"mode": "mul", "value": 0.8}}},
    )
    assert out["human_scale_table"]["LeftArm"] == pytest.approx(0.6)
    assert out["human_scale_table"]["Hips"] == 0.9


def test_apply_patch_scale_table_adds_new_bones() -> None:
    cfg = _make_config_with_scales(["j"])
    out = apply_patch(
        cfg,
        {"human_scale_table": {"Spine": 0.88}},
    )
    assert out["human_scale_table"]["Spine"] == 0.88


def test_apply_patch_scale_table_does_not_mutate_input() -> None:
    cfg = _make_config_with_scales(["j"])
    snapshot = copy.deepcopy(cfg)
    apply_patch(cfg, {"human_scale_table": {"Hips": 0.5}})
    assert cfg == snapshot


def test_apply_patch_scale_table_bounds_reject() -> None:
    cfg = _make_config_with_scales(["j"])
    with pytest.raises(ValueError, match="out of bounds"):
        apply_patch(cfg, {"human_scale_table": {"Hips": 0.05}})
    with pytest.raises(ValueError, match="out of bounds"):
        apply_patch(cfg, {"human_scale_table": {"Hips": 5.0}})


def test_apply_patch_scale_table_empty_patch_noop() -> None:
    cfg = _make_config_with_scales(["j"])
    out = apply_patch(cfg, {})
    assert out["human_scale_table"] == {"Hips": 0.9, "LeftArm": 0.75}


# ---------------------------------------------------------------------------
# apply_patch — human_height_assumption
# ---------------------------------------------------------------------------


def test_apply_patch_height_set() -> None:
    cfg = _make_config_with_scales(["j"], height=1.8)
    out = apply_patch(cfg, {"human_height_assumption": 1.75})
    assert out["human_height_assumption"] == 1.75


def test_apply_patch_height_set_bare_float() -> None:
    cfg = _make_config_with_scales(["j"], height=1.8)
    out = apply_patch(cfg, {"human_height_assumption": 1.65})
    assert out["human_height_assumption"] == pytest.approx(1.65)


def test_apply_patch_height_mul() -> None:
    cfg = _make_config_with_scales(["j"], height=1.8)
    out = apply_patch(cfg, {"human_height_assumption": {"mode": "mul", "value": 0.9}})
    assert out["human_height_assumption"] == pytest.approx(1.62)


def test_apply_patch_height_bounds_reject() -> None:
    cfg = _make_config_with_scales(["j"])
    with pytest.raises(ValueError, match="out of bounds"):
        apply_patch(cfg, {"human_height_assumption": 0.1})
    with pytest.raises(ValueError, match="out of bounds"):
        apply_patch(cfg, {"human_height_assumption": 5.0})


# ---------------------------------------------------------------------------
# apply_patch — combined (quaternion + scale + height)
# ---------------------------------------------------------------------------


def test_apply_patch_combined_quat_scale_height() -> None:
    cfg = _make_config_with_scales(["j"], height=1.8)
    out = apply_patch(
        cfg,
        {
            "ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]},
            "human_scale_table": {"LeftArm": 0.7, "RightLeg": 0.92},
            "human_height_assumption": 1.72,
        },
    )
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])
    assert out["human_scale_table"]["LeftArm"] == 0.7
    assert out["human_scale_table"]["RightLeg"] == 0.92
    assert out["human_height_assumption"] == 1.72


def test_apply_patch_combined_mirror_still_works() -> None:
    cfg = _make_config_with_scales(["j"])
    out = apply_patch(
        cfg,
        {
            "ik_match_table1": {"j": [0.0, 1.0, 0.0, 0.0]},
            "human_scale_table": {"Hips": 0.5},
        },
        mirror="auto",
    )
    assert _quats_close(out["ik_match_table2"]["j"][4], [0.0, 1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# apply_patch — pos_weight / rot_weight
# ---------------------------------------------------------------------------


def _make_config_with_weights(
    joints: list[str],
    pos_weight: float = 10.0,
    rot_weight: float = 5.0,
    quat: list[float] | None = None,
) -> dict:
    """IK config with each joint having specified pos_weight / rot_weight."""
    q = quat or [1.0, 0.0, 0.0, 0.0]
    entry = lambda: ["human_bone", pos_weight, rot_weight, [0.0, 0.0, 0.0], list(q)]  # noqa: E731
    return {
        "ik_match_table1": {j: entry() for j in joints},
        "ik_match_table2": {j: entry() for j in joints},
    }


def test_apply_patch_pos_weight_set() -> None:
    cfg = _make_config_with_weights(["j"], pos_weight=10)
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"pos_weight": {"mode": "set", "value": 50}}}},
    )
    assert out["ik_match_table1"]["j"][1] == 50.0  # pos_weight at index 1
    assert out["ik_match_table1"]["j"][2] == 5.0  # rot_weight unchanged


def test_apply_patch_pos_weight_bare_float() -> None:
    cfg = _make_config_with_weights(["j"], pos_weight=10)
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"pos_weight": 50}}},
    )
    assert out["ik_match_table1"]["j"][1] == 50.0


def test_apply_patch_rot_weight_mul() -> None:
    cfg = _make_config_with_weights(["j"], rot_weight=10)
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"rot_weight": {"mode": "mul", "value": 2.0}}}},
    )
    assert out["ik_match_table1"]["j"][2] == pytest.approx(20.0)


def test_apply_patch_weight_both_pos_and_rot() -> None:
    cfg = _make_config_with_weights(["j"], pos_weight=10, rot_weight=5)
    out = apply_patch(
        cfg,
        {
            "ik_match_table1": {
                "j": {
                    "pos_weight": {"mode": "set", "value": 100},
                    "rot_weight": {"mode": "mul", "value": 3.0},
                }
            }
        },
    )
    assert out["ik_match_table1"]["j"][1] == 100.0
    assert out["ik_match_table1"]["j"][2] == pytest.approx(15.0)


def test_apply_patch_weight_does_not_affect_quat() -> None:
    cfg = _make_config_with_weights(["j"])
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"pos_weight": 50}}},
    )
    assert out["ik_match_table1"]["j"][4] == [1.0, 0.0, 0.0, 0.0]


def test_apply_patch_quat_does_not_affect_weight() -> None:
    cfg = _make_config_with_weights(["j"], pos_weight=10, rot_weight=5, quat=[0.5, 0.5, 0.5, 0.5])
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": [1.0, 0.0, 0.0, 0.0]}},
    )
    assert out["ik_match_table1"]["j"][1] == 10.0
    assert out["ik_match_table1"]["j"][2] == 5.0


def test_apply_patch_weight_combined_with_quat() -> None:
    cfg = _make_config_with_weights(["j"])
    out = apply_patch(
        cfg,
        {
            "ik_match_table1": {
                "j": {
                    "quat": {"mode": "set", "quat": [0.0, 1.0, 0.0, 0.0]},
                    "pos_weight": {"mode": "set", "value": 50},
                    "rot_weight": {"mode": "mul", "value": 2.0},
                }
            }
        },
    )
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])
    assert out["ik_match_table1"]["j"][1] == 50.0
    assert out["ik_match_table1"]["j"][2] == pytest.approx(10.0)


def test_apply_patch_pos_offset_mirror_auto() -> None:
    cfg = _make_config_with_weights(["j"])
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"pos_offset": [1, 2, 3]}}},
        mirror="auto",
    )
    assert out["ik_match_table1"]["j"][3] == [1.0, 2.0, 3.0]
    assert out["ik_match_table2"]["j"][3] == [1.0, 2.0, 3.0]


def test_apply_patch_pos_offset_combined_with_quat_and_weight() -> None:
    cfg = _make_config_with_weights(["j"])
    out = apply_patch(
        cfg,
        {
            "ik_match_table1": {
                "j": {
                    "quat": {"mode": "set", "quat": [0.0, 1.0, 0.0, 0.0]},
                    "pos_offset": [0.1, 0.2, 0.3],
                    "pos_weight": 50,
                }
            }
        },
    )
    assert _quats_close(out["ik_match_table1"]["j"][4], [0.0, 1.0, 0.0, 0.0])
    assert out["ik_match_table1"]["j"][3] == [0.1, 0.2, 0.3]
    assert out["ik_match_table1"]["j"][1] == 50.0


def test_apply_patch_weight_mirror_auto() -> None:
    cfg = _make_config_with_weights(["j"], pos_weight=10)
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"pos_weight": 50}}},
        mirror="auto",
    )
    assert out["ik_match_table1"]["j"][1] == 50.0
    assert out["ik_match_table2"]["j"][1] == 50.0


def test_apply_patch_weight_mirror_off() -> None:
    cfg = _make_config_with_weights(["j"], pos_weight=10)
    out = apply_patch(
        cfg,
        {"ik_match_table1": {"j": {"pos_weight": 50}}},
        mirror="off",
    )
    assert out["ik_match_table1"]["j"][1] == 50.0
    assert out["ik_match_table2"]["j"][1] == 10.0  # unchanged


def test_apply_patch_weight_mirror_strict_single_raises() -> None:
    cfg = _make_config_with_weights(["j"])
    with pytest.raises(ValueError, match="patched only in ik_match_table1"):
        apply_patch(
            cfg,
            {"ik_match_table1": {"j": {"pos_weight": 50}}},
            mirror="strict",
        )


def test_apply_patch_weight_bounds_reject() -> None:
    cfg = _make_config_with_weights(["j"])
    with pytest.raises(ValueError, match="out of bounds"):
        apply_patch(cfg, {"ik_match_table1": {"j": {"pos_weight": -1}}})
    with pytest.raises(ValueError, match="out of bounds"):
        apply_patch(cfg, {"ik_match_table1": {"j": {"rot_weight": 2000}}})


def test_apply_patch_weight_does_not_mutate_input() -> None:
    cfg = _make_config_with_weights(["j"])
    snapshot = copy.deepcopy(cfg)
    apply_patch(cfg, {"ik_match_table1": {"j": {"pos_weight": 50}}})
    assert cfg == snapshot
