"""Unit tests for roboharness.alignment.optimize.

Tests the numerical scale optimisation routine without requiring GMR or MuJoCo
— the retarget callback and deviation function are mocked.
"""

from __future__ import annotations

import numpy as np

from roboharness.alignment.optimize import optimize_scales


def _make_ret_fn(qpos_return: np.ndarray | None = None):

    q = qpos_return if qpos_return is not None else np.zeros(10)

    def _fn(scales: dict, height: float) -> np.ndarray:
        assert isinstance(height, float)
        return q

    return _fn


def _make_dev_fn(err: float = 0.05):

    def _fn(qpos: np.ndarray) -> float:
        return err

    return _fn


def test_optimize_scales_returns_patch() -> None:
    bones = ["Hips", "LeftArm", "RightArm"]
    patch = optimize_scales(
        bones=bones,
        init_scales=[0.9, 0.8, 0.8],
        init_height=1.75,
        retarget_fn=_make_ret_fn(),
        deviation_fn=_make_dev_fn(0.05),
        max_evals=10,
        verbose=False,
    )

    assert "human_scale_table" in patch
    for bone in bones:
        assert bone in patch["human_scale_table"]
        assert 0.2 <= patch["human_scale_table"][bone] <= 2.5
    assert "human_height_assumption" not in patch


def test_optimize_scales_with_height() -> None:
    bones = ["Spine", "LeftLeg"]
    patch = optimize_scales(
        bones=bones,
        init_scales=[0.85, 0.9],
        init_height=1.72,
        retarget_fn=_make_ret_fn(),
        deviation_fn=_make_dev_fn(0.03),
        optimize_height=True,
        max_evals=10,
        verbose=False,
    )

    assert "human_height_assumption" in patch
    assert 0.5 <= patch["human_height_assumption"] <= 3.0


def test_optimize_scales_retarget_error_graceful() -> None:
    bones = ["Hips"]

    def _failing(scales: dict, height: float) -> np.ndarray:
        raise RuntimeError("GMR failed")

    patch = optimize_scales(
        bones=bones,
        init_scales=[0.9],
        init_height=1.8,
        retarget_fn=_failing,
        deviation_fn=_make_dev_fn(0.1),
        max_evals=10,
        verbose=False,
    )

    assert "Hips" in patch["human_scale_table"]


def test_optimize_scales_zero_error_conserves() -> None:
    bones = ["Hips"]
    initial = 0.85
    patch = optimize_scales(
        bones=bones,
        init_scales=[initial],
        init_height=1.75,
        retarget_fn=_make_ret_fn(),
        deviation_fn=_make_dev_fn(0.0),
        max_evals=10,
        verbose=False,
    )

    assert "Hips" in patch["human_scale_table"]
