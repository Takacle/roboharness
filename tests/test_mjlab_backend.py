"""Tests for MjlabBackend.

These tests use mocked mjlab internals so no GPU or real mjlab installation
is required.  The real mjlab dependency is only needed for end-to-end
integration (see examples/mjlab_integration.py).
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roboharness.core.capture import CameraView
from roboharness.core.harness import SimulatorBackend

# ---------------------------------------------------------------------------
# mjlab stub — injected before importing MjlabBackend
# ---------------------------------------------------------------------------


def _make_mjlab_stub() -> ModuleType:
    """Build a minimal stub of the mjlab package tree."""
    mjlab = ModuleType("mjlab")
    mjlab_envs = ModuleType("mjlab.envs")
    mjlab_envs_mgr = ModuleType("mjlab.envs.manager_based_rl_env")
    mjlab_tasks = ModuleType("mjlab.tasks")
    mjlab_tasks_registry = ModuleType("mjlab.tasks.registry")

    mjlab.envs = mjlab_envs
    mjlab.tasks = mjlab_tasks

    sys.modules["mjlab"] = mjlab
    sys.modules["mjlab.envs"] = mjlab_envs
    sys.modules["mjlab.envs.manager_based_rl_env"] = mjlab_envs_mgr
    sys.modules["mjlab.tasks"] = mjlab_tasks
    sys.modules["mjlab.tasks.registry"] = mjlab_tasks_registry

    return mjlab


def _make_fake_sim(qpos: list[float] | None = None, time: float = 0.0) -> MagicMock:
    """Return a fake Simulation object backed by numpy."""
    qpos_arr = np.array(qpos or [0.0, 0.0], dtype=np.float64)
    qvel_arr = np.zeros_like(qpos_arr)
    nq = len(qpos_arr)

    mj_data = MagicMock()
    mj_data.time = time
    mj_data.qpos = qpos_arr
    mj_data.qvel = qvel_arr
    mj_data.qpos.shape = (nq,)
    mj_data.qvel.shape = (nq,)
    mj_model = MagicMock()

    wp_data = MagicMock()
    wp_data.qpos = MagicMock()
    wp_data.qvel = MagicMock()
    wp_data.time = MagicMock()
    wp_data.qpos.numpy.return_value = np.tile(qpos_arr, (2, 1))
    wp_data.qvel.numpy.return_value = np.tile(qvel_arr, (2, 1))
    wp_data.time.numpy.return_value = np.array([time, time], dtype=np.float64)

    sim = MagicMock()
    sim.mj_data = mj_data
    sim.mj_model = mj_model
    sim.wp_data = wp_data
    sim.forward = MagicMock()
    return sim


def _make_fake_env(num_envs: int = 1, qpos: list[float] | None = None) -> MagicMock:
    """Return a fake ManagerBasedRlEnv."""
    import torch

    env = MagicMock()
    env.num_envs = num_envs
    env.device = "cpu"

    sim = _make_fake_sim(qpos=qpos)
    env.sim = sim
    env.scene.sim = sim
    env.scene.sensors = {}  # no camera sensors by default

    env.reset = MagicMock(return_value={"policy": torch.zeros(num_envs, 4)})
    env.step = MagicMock(return_value=({"policy": torch.zeros(num_envs, 4)}, None, None, None, {}))
    return env


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _inject_mjlab_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a mjlab stub into sys.modules before each test."""
    _make_mjlab_stub()
    yield
    # Clean up so stubs don't leak between test runs.
    for mod in list(sys.modules):
        if mod.startswith("mjlab"):
            del sys.modules[mod]


@pytest.fixture()
def fake_env() -> MagicMock:
    return _make_fake_env()


def _make_backend(fake_env: MagicMock) -> Any:
    """Instantiate MjlabBackend with a fake env injected."""
    from roboharness.backends.mjlab_backend import MjlabBackend

    backend = MjlabBackend.__new__(MjlabBackend)
    backend._env_id = 0
    backend._camera_names = ["front"]
    backend._render_width = 64
    backend._render_height = 64
    backend._renderers = {}
    backend._env = fake_env
    return backend


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_implements_simulator_backend_protocol(fake_env: MagicMock) -> None:
    """MjlabBackend must satisfy the SimulatorBackend runtime-checkable Protocol."""
    backend = _make_backend(fake_env)
    assert isinstance(backend, SimulatorBackend)


# ---------------------------------------------------------------------------
# reset / step / get_state
# ---------------------------------------------------------------------------


def test_reset_returns_state_dict(fake_env: MagicMock) -> None:
    backend = _make_backend(fake_env)
    state = backend.reset()
    fake_env.reset.assert_called_once()
    assert "time" in state
    assert "qpos" in state
    assert "qvel" in state


def test_step_advances_and_returns_state(fake_env: MagicMock) -> None:
    backend = _make_backend(fake_env)
    action = np.zeros(4, dtype=np.float32)
    state = backend.step(action)
    fake_env.step.assert_called_once()
    assert "time" in state


def test_step_broadcasts_1d_action_to_batch(fake_env: MagicMock) -> None:
    """A 1-D action must be expanded to (num_envs, action_dim) before env.step."""
    import torch

    num_envs = 3
    fake_env_multi = _make_fake_env(num_envs=num_envs)
    backend = _make_backend(fake_env_multi)
    backend._env = fake_env_multi

    action = np.ones(4, dtype=np.float32)
    backend.step(action)

    call_args = fake_env_multi.step.call_args[0][0]
    assert isinstance(call_args, torch.Tensor)
    assert call_args.shape == (num_envs, 4)


def test_step_accepts_torch_tensor(fake_env: MagicMock) -> None:
    import torch

    backend = _make_backend(fake_env)
    action = torch.zeros(4)
    state = backend.step(action)
    assert "time" in state


def test_get_state_returns_numpy_arrays(fake_env: MagicMock) -> None:
    backend = _make_backend(fake_env)
    state = backend.get_state()
    assert isinstance(state["qpos"], np.ndarray)
    assert isinstance(state["qvel"], np.ndarray)
    assert isinstance(state["time"], float)


# ---------------------------------------------------------------------------
# save_state / restore_state
# ---------------------------------------------------------------------------


def test_save_restore_round_trip(fake_env: MagicMock) -> None:
    """save_state / restore_state must reconstruct the original state."""
    import mujoco as mj

    backend = _make_backend(fake_env)

    state_size = 10
    saved_vec = np.arange(state_size, dtype=np.float64)

    with (
        patch.object(mj, "mj_stateSize", return_value=state_size),
        patch.object(mj, "mj_getState", side_effect=lambda m, d, v, f: np.copyto(v, saved_vec)),
        patch.object(mj, "mj_setState") as mock_set,
    ):
        saved = backend.save_state()
        assert np.array_equal(saved["mjlab_state"], saved_vec)
        assert saved["time"] == pytest.approx(0.0)

        backend.restore_state(saved)
        mock_set.assert_called_once()
        # GPU sync must be triggered after restore.
        fake_env.scene.sim.forward.assert_called_once()


# ---------------------------------------------------------------------------
# capture_camera
# ---------------------------------------------------------------------------


def test_capture_camera_cpu_renderer_fallback(fake_env: MagicMock) -> None:
    """When no scene sensor exists, falls back to CPU mujoco.Renderer."""
    import mujoco as mj

    backend = _make_backend(fake_env)
    fake_rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    mock_renderer = MagicMock()
    mock_renderer.render.return_value = fake_rgb

    with (
        patch.object(mj, "Renderer", return_value=mock_renderer),
        patch.object(mj, "mj_forward"),
    ):
        view = backend.capture_camera("front")

    assert isinstance(view, CameraView)
    assert view.name == "front"
    assert view.rgb.shape == (64, 64, 3)
    assert view.rgb.dtype == np.uint8


def test_capture_camera_uses_scene_sensor_when_present(fake_env: MagicMock) -> None:
    """When a CameraSensor is registered, its data should be used."""
    import torch

    backend = _make_backend(fake_env)
    fake_rgb_t = torch.zeros(64, 64, 3, dtype=torch.uint8)
    sensor = MagicMock()
    # Use a MagicMock for rgb so __getitem__ can be overridden.
    sensor.data.rgb = MagicMock()
    sensor.data.rgb.__getitem__ = MagicMock(return_value=fake_rgb_t)
    fake_env.scene.sensors = {"front": sensor}

    view = backend.capture_camera("front")

    assert isinstance(view, CameraView)
    assert view.name == "front"
    assert view.rgb.dtype == np.uint8


def test_capture_camera_caches_renderer(fake_env: MagicMock) -> None:
    """A second capture_camera call for the same camera must reuse the renderer."""
    import mujoco as mj

    backend = _make_backend(fake_env)
    fake_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = fake_rgb

    with (
        patch.object(mj, "Renderer", return_value=mock_renderer) as mock_cls,
        patch.object(mj, "mj_forward"),
    ):
        backend.capture_camera("front")
        backend.capture_camera("front")
        # Constructor called only once despite two captures.
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# get_sim_time
# ---------------------------------------------------------------------------


def test_get_sim_time_returns_float(fake_env: MagicMock) -> None:
    backend = _make_backend(fake_env)
    t = backend.get_sim_time()
    assert isinstance(t, float)
    assert t == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_requires_task_id_or_env_cfg() -> None:
    from roboharness.backends.mjlab_backend import MjlabBackend

    with pytest.raises(ValueError, match="task_id or env_cfg"):
        MjlabBackend()


def test_constructor_rejects_both_task_id_and_env_cfg() -> None:
    from roboharness.backends.mjlab_backend import MjlabBackend

    with pytest.raises(ValueError, match="not both"):
        MjlabBackend(task_id="Mjlab-Cartpole-Balance", env_cfg=object())
