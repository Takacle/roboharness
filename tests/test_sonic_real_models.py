"""Real-model SONIC validation against the published HuggingFace ONNX artifacts.

These tests download ONNX models from HuggingFace and require network access.
They are skipped by default unless ``ROBOHARNESS_RUN_REAL_SONIC=1`` is set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

_RUN_REAL = os.environ.get("ROBOHARNESS_RUN_REAL_SONIC", "") == "1"
_skip_reason = "Set ROBOHARNESS_RUN_REAL_SONIC=1 to run real SONIC model tests (requires network)"

pytest.importorskip("huggingface_hub", reason="huggingface_hub not installed")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed")

import roboharness.robots.unitree_g1.locomotion as _loc  # noqa: E402


def _yaw_to_sixd(yaw: float) -> np.ndarray:
    """Encode a yaw-only rotation into SONIC's row-wise 6D format."""
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    return np.array([cy, -sy, sy, cy, 0.0, 0.0], dtype=np.float32)


def _make_tracking_clip(num_frames: int = 80) -> _loc.MotionClip:
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)

    joint_positions = np.zeros((num_frames, 29), dtype=np.float32)
    joint_positions[:, 0] = 0.10 * np.sin(2.0 * np.pi * t)
    joint_positions[:, 3] = 0.60 + 0.05 * np.sin(4.0 * np.pi * t)
    joint_positions[:, 4] = -0.30 + 0.02 * np.cos(2.0 * np.pi * t)
    joint_positions[:, 15] = 0.15
    joint_positions[:, 18] = 0.55
    joint_positions[:, 22] = 0.15
    joint_positions[:, 25] = 0.55

    joint_velocities = np.gradient(joint_positions, axis=0).astype(np.float32) * 50.0
    root_height = np.full(num_frames, 0.74, dtype=np.float32)
    root_rotation_6d = np.stack([_yaw_to_sixd(0.15 * float(v)) for v in t], axis=0)

    return _loc.MotionClip(
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        root_height=root_height,
        root_rotation_6d=root_rotation_6d,
        fps=50.0,
        name="deterministic_fixture",
    )


def _make_state(step: int = 0) -> dict[str, np.ndarray]:
    yaw = 0.01 * step
    half_yaw = yaw / 2.0
    qpos = np.zeros(36, dtype=np.float32)
    qpos[2] = 0.74
    qpos[3] = np.cos(half_yaw)
    qpos[6] = np.sin(half_yaw)
    qpos[7:36] = _loc.SONIC_TRACKING_DEFAULT_ANGLES + (
        0.01 * np.sin(0.2 * step + np.arange(29, dtype=np.float32) * 0.1)
    )

    qvel = np.zeros(35, dtype=np.float32)
    qvel[3:6] = np.array([0.02, -0.01, 0.03], dtype=np.float32)
    qvel[6:35] = 0.05 * np.cos(0.15 * step + np.arange(29, dtype=np.float32) * 0.07)
    return {"qpos": qpos, "qvel": qvel}


@pytest.mark.slow
@pytest.mark.skipif(not _RUN_REAL, reason=_skip_reason)
def test_sonic_real_model_signatures_match_repo_contract() -> None:
    encoder = _loc._load_onnx_session(_loc.SONIC_HF_REPO, _loc.SONIC_ENCODER_FILE)
    decoder = _loc._load_onnx_session(_loc.SONIC_HF_REPO, _loc.SONIC_DECODER_FILE)
    planner = _loc._load_onnx_session(_loc.SONIC_HF_REPO, _loc.SONIC_PLANNER_FILE)

    enc_input = encoder.get_inputs()[0]
    enc_output = encoder.get_outputs()[0]
    dec_input = decoder.get_inputs()[0]
    dec_output = decoder.get_outputs()[0]
    planner_inputs = {item.name for item in planner.get_inputs()}
    planner_outputs = [item.name for item in planner.get_outputs()]

    assert enc_input.name == "obs_dict"
    assert enc_input.shape == [1, _loc.SONIC_ENCODER_INPUT_DIM]
    assert enc_output.name == "encoded_tokens"
    assert enc_output.shape == [1, _loc.SONIC_LATENT_DIM]

    assert dec_input.name == "obs_dict"
    assert dec_input.shape == [1, _loc.SONIC_DECODER_INPUT_DIM]
    assert dec_output.name == "action"
    assert dec_output.shape == [1, _loc.SONIC_DECODER_OUTPUT_DIM]

    assert "context_mujoco_qpos" in planner_inputs
    assert "target_vel" in planner_inputs
    assert planner_outputs == ["mujoco_qpos", "num_pred_frames"]


@pytest.mark.slow
@pytest.mark.skipif(not _RUN_REAL, reason=_skip_reason)
def test_sonic_real_planner_compute_runs() -> None:
    controller = _loc.SonicLocomotionController()
    state = _make_state()

    action = controller.compute(
        command={"velocity": [0.3, 0.0, 0.0], "mode": _loc.SonicMode.WALK},
        state=state,
    )

    assert action.shape == (29,)
    assert np.all(np.isfinite(action))


@pytest.mark.slow
@pytest.mark.skipif(not _RUN_REAL, reason=_skip_reason)
def test_sonic_real_tracking_compute_runs_and_advances() -> None:
    controller = _loc.SonicLocomotionController()
    controller.set_tracking_clip(_make_tracking_clip())

    outputs = []
    for step in range(6):
        outputs.append(controller.compute(command={"tracking": True}, state=_make_state(step)))

    assert outputs
    assert all(output.shape == (29,) for output in outputs)
    assert all(np.all(np.isfinite(output)) for output in outputs)
    assert controller._tracking_frame_index > 0
    assert controller._encoder_session is not None
    assert controller._decoder_session is not None
