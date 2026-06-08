"""Regression tests for solver import guards and default T-pose spec paths."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from gmr_harness import solver


class _StopAfterDefaultSpec(RuntimeError):
    pass


def _agent_args() -> SimpleNamespace:
    return SimpleNamespace(
        robot="test_robot",
        src="bvh",
        motion_file="motion.bvh",
        bvh_format="auto",
        frames=1,
        output="agent_output",
        model="dummy-model",
        max_iter=1,
        dry_run=True,
        tpose_spec=None,
        tpose_motion=None,
        solve_mode=True,
        tune_mode="scale",
        api_base="",
        api_key="",
    )


def test_retarget_tpose_qpos_calls_ensure_gmr_before_direct_import(monkeypatch):
    calls: list[str] = []

    def fake_ensure(feature: str = "") -> None:
        calls.append(feature)
        raise RuntimeError("stop before importing GMR")

    monkeypatch.setattr(solver, "_ensure_gmr", fake_ensure)

    with pytest.raises(RuntimeError, match="stop before importing GMR"):
        solver._retarget_tpose_qpos("bvh", "motion.bvh", "test_robot", "auto")

    assert calls == ["retarget tpose qpos"]


def test_run_agent_default_tpose_spec_uses_cwd_relative_path(monkeypatch, tmp_path):
    spec_dir = tmp_path / "specs" / "tpose"
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / "test_robot.json"
    spec_file.write_text(json.dumps({"links": {}, "qpos": [], "xml_path": "robot.xml"}))
    config_file = tmp_path / "ik.json"
    config_file.write_text(json.dumps({}))

    gmr_pkg = ModuleType("general_motion_retargeting")
    params_mod = ModuleType("general_motion_retargeting.params")
    params_mod.IK_CONFIG_DICT = {"bvh": {"test_robot": str(config_file)}}
    params_mod.ROBOT_XML_DICT = {"test_robot": str(tmp_path / "robot.xml")}
    params_mod.VIEWER_CAM_DISTANCE_DICT = {"test_robot": 2.5}
    monkeypatch.setitem(sys.modules, "general_motion_retargeting", gmr_pkg)
    monkeypatch.setitem(sys.modules, "general_motion_retargeting.params", params_mod)
    monkeypatch.setattr("gmr_harness.gmr_integration.find_root_body", lambda _xml_path: "pelvis")

    loaded_paths: list[str] = []

    def fake_load_tpose_spec(path):
        loaded_paths.append(str(path))
        raise _StopAfterDefaultSpec

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(solver, "_ensure_gmr", lambda feature="": None)
    monkeypatch.setattr("gmr_harness.alignment.load_tpose_spec", fake_load_tpose_spec)

    with pytest.raises(_StopAfterDefaultSpec):
        solver.run_agent(_agent_args())

    assert loaded_paths == [str(Path("specs/tpose") / "test_robot.json")]


def test_run_agent_solve_mode_dry_run_restores_config_and_skips_backup(monkeypatch, tmp_path):
    spec_file = tmp_path / "spec.json"
    spec_file.write_text(json.dumps({"links": {}, "qpos": [], "xml_path": "robot.xml"}))
    config_file = tmp_path / "ik.json"
    original_config = {"robot_root_name": "base", "ik_match_table1": {}}
    config_file.write_text(json.dumps(original_config, indent=4))

    params_mod = ModuleType("general_motion_retargeting.params")
    params_mod.IK_CONFIG_DICT = {"bvh": {"test_robot": str(config_file)}}
    params_mod.ROBOT_XML_DICT = {"test_robot": str(tmp_path / "robot.xml")}
    params_mod.VIEWER_CAM_DISTANCE_DICT = {"test_robot": 2.5}
    monkeypatch.setitem(
        sys.modules,
        "general_motion_retargeting",
        ModuleType("general_motion_retargeting"),
    )
    monkeypatch.setitem(sys.modules, "general_motion_retargeting.params", params_mod)
    monkeypatch.setattr("gmr_harness.gmr_integration.find_root_body", lambda _xml_path: "pelvis")
    monkeypatch.setattr(solver, "_ensure_gmr", lambda feature="": None)
    monkeypatch.setattr(solver, "_retarget", lambda *_args, **_kwargs: (np.zeros((1, 1)), []))
    monkeypatch.setattr(
        "gmr_harness.alignment.load_tpose_spec",
        lambda _path: json.loads(spec_file.read_text()),
    )
    monkeypatch.setattr(
        solver,
        "solve_direct",
        lambda **_kwargs: {"robot_root_name": "base", "ik_match_table1": {"changed": []}},
    )
    monkeypatch.setattr(solver, "extract_init_qpos", lambda _spec: {})
    monkeypatch.setattr(solver, "_retarget_tpose_qpos", lambda *_args, **_kwargs: [])
    monkeypatch.setattr("gmr_harness.alignment.compute_deviations", lambda *_args, **_kwargs: {})
    monkeypatch.setattr("gmr_harness.alignment.total_deviation", lambda _report: 0.0)
    monkeypatch.setattr("gmr_harness.alignment.worst_k", lambda _report, _k: [])

    args = _agent_args()
    args.tpose_spec = str(spec_file)
    args.tpose_motion = "motion.bvh"
    args.tpose_bvh_format = "auto"
    args.tpose_threshold = 5.0

    assert solver.run_agent(args) == 0
    assert json.loads(config_file.read_text()) == original_config
    assert not config_file.with_suffix(".json.bak").exists()
