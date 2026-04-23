"""Unit tests for roboharness.alignment.metrics.

These tests use a minimal two-body MJCF chain, not a real robot XML. The
metric is a pure function of (qpos, xml, spec) and should be validated in
isolation from robot-specific concerns.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from roboharness.alignment.metrics import (  # noqa: E402
    _rotation_matrix_to_axis_angle,
    compute_deviations,
    load_tpose_spec,
    total_deviation,
    worst_k,
)

# A two-body chain: world → arm (rotates around z), with a named body 'arm'
# at qpos[0] radians rotation. nq=1, no freejoint.
_TEST_XML = """
<mujoco model="test">
  <worldbody>
    <body name="arm" pos="0 0 0">
      <joint name="arm_joint" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.02 0.02"/>
      <body name="tip" pos="0.2 0 0">
        <geom type="sphere" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def xml_file(tmp_path: Path) -> Path:
    p = tmp_path / "test.xml"
    p.write_text(_TEST_XML)
    return p


def _spec_from_qpos(xml_file: Path, qpos: np.ndarray, link_names: list[str]) -> dict:
    """Forward-sim, snapshot xmat for each named body, return a spec dict."""
    model = mujoco.MjModel.from_xml_path(str(xml_file))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    links = {}
    for name in link_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        links[name] = {
            "pos": np.asarray(data.xpos[bid]).tolist(),
            "R": np.asarray(data.xmat[bid]).reshape(3, 3).tolist(),
        }
    return {
        "robot": "test",
        "xml_path": str(xml_file),
        "qpos": qpos.tolist(),
        "links": links,
    }


# ---- axis-angle helper ----


def test_axis_angle_identity() -> None:
    axis, angle = _rotation_matrix_to_axis_angle(np.eye(3))
    assert angle == 0.0
    # axis convention: z for identity
    np.testing.assert_allclose(axis, [0.0, 0.0, 1.0])


def test_axis_angle_90deg_z() -> None:
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    axis, angle = _rotation_matrix_to_axis_angle(R)
    assert abs(angle - math.pi / 2) < 1e-9
    np.testing.assert_allclose(axis, [0.0, 0.0, 1.0], atol=1e-9)


def test_axis_angle_180deg_x() -> None:
    # 180° rotations are the numerically tricky case (sin(angle) ~ 0).
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    axis, angle = _rotation_matrix_to_axis_angle(R)
    assert abs(angle - math.pi) < 1e-6
    # Axis can be ±x; magnitude must match.
    assert abs(abs(axis[0]) - 1.0) < 1e-6
    assert abs(axis[1]) < 1e-6
    assert abs(axis[2]) < 1e-6


# ---- identity / perturbation ----


def test_identity_qpos_zero_deviation(xml_file: Path) -> None:
    qpos = np.array([0.0])
    spec = _spec_from_qpos(xml_file, qpos, ["arm", "tip"])
    report = compute_deviations(qpos, xml_file, spec)  # type: ignore[arg-type]
    assert set(report.keys()) == {"arm", "tip"}
    for name, dev in report.items():
        assert dev["angle_deg"] < 1e-6, f"{name} expected 0 deviation, got {dev}"
    assert total_deviation(report) < 1e-6


def test_identity_nonzero_qpos_still_zero_deviation(xml_file: Path) -> None:
    # Spec captured at qpos=0.5 rad; replay at same qpos → identity.
    qpos = np.array([0.5])
    spec = _spec_from_qpos(xml_file, qpos, ["arm"])
    report = compute_deviations(qpos, xml_file, spec)  # type: ignore[arg-type]
    # Floating-point noise from arccos near identity — any residual below
    # 1e-4 degrees (~1.7e-6 rad) is pure rounding, not geometric disagreement.
    assert report["arm"]["angle_deg"] < 1e-4


def test_single_joint_perturbation_localizes(xml_file: Path) -> None:
    # Spec at qpos=0; replay at qpos=pi/2 — arm rotates 90° around z.
    spec = _spec_from_qpos(xml_file, np.array([0.0]), ["arm", "tip"])
    perturbed = np.array([math.pi / 2])
    report = compute_deviations(perturbed, xml_file, spec)  # type: ignore[arg-type]

    arm_dev = report["arm"]
    assert abs(arm_dev["angle_deg"] - 90.0) < 1e-4
    # Axis should be world z (arm's hinge axis). Sign can be either since
    # axis-angle has ± ambiguity — what matters is it's along z, not x/y.
    assert abs(arm_dev["axis"][2]) > 0.999
    assert abs(arm_dev["axis"][0]) < 1e-4
    assert abs(arm_dev["axis"][1]) < 1e-4

    # Tip is a child of arm — it inherits the rotation, so also reports 90°.
    # (This is a property of kinematic chains, not a bug.)
    assert abs(report["tip"]["angle_deg"] - 90.0) < 1e-4


def test_total_deviation_monotone_in_perturbation(xml_file: Path) -> None:
    spec = _spec_from_qpos(xml_file, np.array([0.0]), ["arm"])
    angles = [0.0, 0.1, 0.3, 0.7, 1.2]
    totals = [
        total_deviation(compute_deviations(np.array([a]), xml_file, spec))  # type: ignore[arg-type]
        for a in angles
    ]
    for prev, curr in itertools.pairwise(totals):
        assert curr > prev, f"total_deviation should increase, got {totals}"


# ---- worst_k ----


def test_worst_k_ordering(xml_file: Path) -> None:
    spec = _spec_from_qpos(xml_file, np.array([0.0]), ["arm", "tip"])
    report = compute_deviations(np.array([0.3]), xml_file, spec)  # type: ignore[arg-type]
    top = worst_k(report, k=1)
    assert len(top) == 1
    # Both arm and tip deviate equally in this chain; top entry should be
    # one of them with angle_deg ~ 17.19° (0.3 rad).
    assert abs(top[0][1] - math.degrees(0.3)) < 1e-4


def test_worst_k_excludes_nan() -> None:
    report: dict[str, Any] = {
        "a": {"axis": [0, 0, 1], "angle_deg": 10.0},
        "b": {"axis": [0, 0, 1], "angle_deg": float("nan")},
        "c": {"axis": [0, 0, 1], "angle_deg": 5.0},
    }
    top = worst_k(report, k=5)
    assert [n for n, _ in top] == ["a", "c"]
    assert math.isnan(total_deviation(report)) is False
    assert total_deviation(report) == 15.0


# ---- missing body ----


def test_missing_body_reports_nan(xml_file: Path) -> None:
    spec = _spec_from_qpos(xml_file, np.array([0.0]), ["arm"])
    # Inject a link that doesn't exist in the XML.
    spec["links"]["nonexistent"] = {"pos": [0, 0, 0], "R": np.eye(3).tolist()}
    report = compute_deviations(np.array([0.0]), xml_file, spec)  # type: ignore[arg-type]
    assert math.isnan(report["nonexistent"]["angle_deg"])
    # Total ignores NaN.
    assert total_deviation(report) < 1e-6


# ---- qpos shape guard ----


def test_wrong_qpos_shape_raises(xml_file: Path) -> None:
    spec = _spec_from_qpos(xml_file, np.array([0.0]), ["arm"])
    with pytest.raises(ValueError, match="qpos shape mismatch"):
        compute_deviations(np.array([0.0, 0.0]), xml_file, spec)  # type: ignore[arg-type]


# ---- load_tpose_spec validation ----


def test_load_tpose_spec_roundtrip(tmp_path: Path, xml_file: Path) -> None:
    spec = _spec_from_qpos(xml_file, np.array([0.2]), ["arm"])
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    loaded = load_tpose_spec(path)
    assert loaded["robot"] == "test"
    assert list(loaded["links"].keys()) == ["arm"]


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda s: s.pop("robot"), "missing required key: 'robot'"),
        (lambda s: s.pop("links"), "missing required key: 'links'"),
        (lambda s: s.__setitem__("links", {}), "empty or malformed 'links'"),
    ],
)
def test_load_tpose_spec_rejects_malformed(
    tmp_path: Path,
    xml_file: Path,
    mutate: Any,
    match: str,
) -> None:
    spec = _spec_from_qpos(xml_file, np.array([0.0]), ["arm"])
    mutate(spec)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match=match):
        load_tpose_spec(path)


def test_load_tpose_spec_rejects_bad_link(tmp_path: Path, xml_file: Path) -> None:
    spec = _spec_from_qpos(xml_file, np.array([0.0]), ["arm"])
    spec["links"]["arm"]["R"] = [[1, 0, 0], [0, 1, 0]]  # 2x3, malformed
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="R must be 3x3"):
        load_tpose_spec(path)
