"""mjlab backend adapter.

Bridges roboharness's SimulatorBackend protocol to mjlab's ManagerBasedRlEnv
(GPU-accelerated MuJoCo-Warp parallel environments).

Requires: pip install roboharness[mjlab]

Usage::

    from roboharness.backends.mjlab_backend import MjlabBackend
    from roboharness.core.harness import Harness

    # Via task registry
    backend = MjlabBackend(task_id="Mjlab-Cartpole-Balance", cameras=["front"])

    # Via direct env config
    from my_task import MyEnvCfg
    backend = MjlabBackend(env_cfg=MyEnvCfg(), cameras=["front", "side"])

    harness = Harness(backend, output_dir="./output")
    harness.reset()
    result = harness.run_to_next_checkpoint(actions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from roboharness.core.capture import CameraView

if TYPE_CHECKING:
    import mujoco


class MjlabBackend:
    """Backend adapter for mjlab GPU-accelerated environments.

    Implements the SimulatorBackend protocol. Wraps a ``ManagerBasedRlEnv``
    and exposes a single environment slice (``env_id``) to the harness.

    GPU tensors are converted to numpy arrays at the boundary so the rest of
    roboharness can remain framework-agnostic.

    Parameters
    ----------
    task_id : str | None
        Task ID from the mjlab registry (e.g. ``"Mjlab-Cartpole-Balance"``).
        Mutually exclusive with ``env_cfg``.
    env_cfg : ManagerBasedRlEnvCfg | None
        Pre-built environment config. Mutually exclusive with ``task_id``.
    cameras : list[str] | None
        Camera names for capture (default: ``["front"]``).
    render_width, render_height : int
        Resolution for CPU off-screen rendering fallback.
    env_id : int
        Which parallel environment to expose (default: 0).
    num_envs : int
        Number of parallel GPU environments to create (default: 1).
    device : str
        Torch/Warp device string (default: ``"cuda"``).
    play : bool
        If True and using ``task_id``, load the *play* config instead of the
        training config (often fewer envs, interactive settings).
    """

    def __init__(
        self,
        task_id: str | None = None,
        env_cfg: Any | None = None,
        cameras: list[str] | None = None,
        render_width: int = 640,
        render_height: int = 480,
        env_id: int = 0,
        num_envs: int = 1,
        device: str = "cuda",
        play: bool = False,
    ) -> None:
        try:
            import mjlab  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "mjlab is required for this backend. Install with: pip install roboharness[mjlab]"
            ) from exc

        if task_id is None and env_cfg is None:
            raise ValueError("Provide either task_id or env_cfg.")
        if task_id is not None and env_cfg is not None:
            raise ValueError("Provide task_id or env_cfg, not both.")

        self._env_id = env_id
        self._camera_names = cameras or ["front"]
        self._render_width = render_width
        self._render_height = render_height
        self._renderers: dict[str, mujoco.Renderer] = {}

        resolved_cfg = self._resolve_cfg(task_id, env_cfg, num_envs, device, play)
        self._env = self._build_env(resolved_cfg, device)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_cfg(
        task_id: str | None,
        env_cfg: Any | None,
        num_envs: int,
        device: str,
        play: bool,
    ) -> Any:
        """Return a fully configured ``ManagerBasedRlEnvCfg``."""
        if task_id is not None:
            from mjlab.tasks.registry import load_env_cfg

            cfg = load_env_cfg(task_id, play=play)
        else:
            import copy

            cfg = copy.deepcopy(env_cfg)

        # Override parallelism and device so the harness controls these.
        cfg.scene.num_envs = num_envs
        cfg.scene.device = device
        return cfg

    @staticmethod
    def _build_env(cfg: Any, device: str) -> Any:
        # Prefer InstinctRlEnv (supports custom terrain types like hacked_generator).
        try:
            from instinct_mj.envs import InstinctRlEnv

            return InstinctRlEnv(cfg, device=device)
        except ImportError:
            pass

        from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

        return ManagerBasedRlEnv(cfg, device=device)

    # ------------------------------------------------------------------
    # SimulatorBackend protocol
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Reset the environment and return initial state."""
        self._env.reset()
        return self._build_state()

    def step(self, action: Any) -> dict[str, Any]:
        """Advance the environment by one policy step.

        ``action`` may be a numpy array or torch Tensor for a single
        environment.  It is automatically broadcast to ``(num_envs, action_dim)``.
        """
        action_t = self._coerce_action(action)
        self._env.step(action_t)
        return self._build_state()

    def get_state(self) -> dict[str, Any]:
        """Get current simulation state for env_id slice."""
        return self._build_state()

    def save_state(self) -> dict[str, Any]:
        """Serialise full MuJoCo physics state for later restoration.

        Uses ``mujoco.mj_getState(mjSTATE_FULLPHYSICS)`` on the CPU-side
        ``mj_data``, which captures all position/velocity/acceleration state.
        """
        import mujoco

        sim = self._env.sim
        state_size = mujoco.mj_stateSize(sim.mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        state_vec = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(
            sim.mj_model, sim.mj_data, state_vec, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        return {"mjlab_state": state_vec, "time": float(sim.mj_data.time)}

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore the simulator to a previously saved state.

        After restoring the CPU-side ``mj_data``, calls ``sim.forward()`` to
        propagate the state to GPU Warp buffers.
        """
        import mujoco

        sim = self._env.sim
        mujoco.mj_setState(
            sim.mj_model,
            sim.mj_data,
            state["mjlab_state"],
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        # Sync GPU wp_data from the restored CPU state.
        sim.forward()

    def capture_camera(self, camera_name: str) -> CameraView:
        """Render an RGB frame from a named camera.

        First tries a ``CameraSensor`` registered in the scene under
        ``camera_name``.  Falls back to a CPU ``mujoco.Renderer`` if the
        sensor is absent.
        """
        scene = self._env.scene

        # Sanitise name for filenames (mjlab prefixes entity names with '/').
        safe_name = camera_name.replace("/", "_")

        # Path 1: GPU camera sensor (CameraSensor registered in the scene).
        sensors = getattr(scene, "sensors", {})
        if camera_name in sensors:
            sensor = sensors[camera_name]
            rgb_tensor = sensor.data.rgb[self._env_id]
            rgb = rgb_tensor.cpu().numpy().astype(np.uint8)
            return CameraView(name=safe_name, rgb=rgb)

        # Path 2: CPU off-screen renderer via mujoco.Renderer.
        # Sync GPU warp state → CPU mj_data before rendering.
        self._sync_gpu_to_cpu()
        import mujoco

        sim = self._env.sim
        mujoco.mj_forward(sim.mj_model, sim.mj_data)
        renderer = self._get_cpu_renderer(camera_name)
        renderer.update_scene(sim.mj_data, camera=camera_name)
        rgb = renderer.render()
        return CameraView(name=safe_name, rgb=rgb)

    def get_sim_time(self) -> float:
        """Return current simulation time from the CPU-side mj_data."""
        self._sync_gpu_to_cpu()
        return float(self._env.sim.mj_data.time)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state(self) -> dict[str, Any]:
        """Build a state dict from the GPU-side warp data (synced to CPU)."""
        self._sync_gpu_to_cpu()
        sim = self._env.sim
        return {
            "time": float(sim.mj_data.time),
            "qpos": sim.mj_data.qpos.copy(),
            "qvel": sim.mj_data.qvel.copy(),
        }

    def _sync_gpu_to_cpu(self) -> None:
        """Copy qpos/qvel/time from GPU warp data to CPU mj_data."""
        sim = self._env.sim
        wp_data = sim.wp_data
        mj_data = sim.mj_data
        env_id = self._env_id

        qpos_np = wp_data.qpos.numpy()
        qvel_np = wp_data.qvel.numpy()
        mj_data.qpos[:] = qpos_np[env_id, : mj_data.qpos.shape[0]]
        mj_data.qvel[:] = qvel_np[env_id, : mj_data.qvel.shape[0]]
        mj_data.time = float(wp_data.time.numpy()[env_id])

    def _coerce_action(self, action: Any) -> Any:
        """Coerce an arbitrary action into a batched torch.Tensor on the env device."""
        import torch

        device = self._env.device
        num_envs = self._env.num_envs

        if isinstance(action, torch.Tensor):
            action_t = action.to(device=device, dtype=torch.float32)
        else:
            action_t = torch.tensor(
                np.asarray(action, dtype=np.float32), device=device, dtype=torch.float32
            )

        # Broadcast 1-D single-env action to (num_envs, action_dim).
        if action_t.dim() == 1:
            action_t = action_t.unsqueeze(0).expand(num_envs, -1)
        return action_t

    def _get_cpu_renderer(self, camera_name: str) -> mujoco.Renderer:
        """Return a cached CPU ``mujoco.Renderer`` for *camera_name*."""
        import mujoco

        if camera_name not in self._renderers:
            self._renderers[camera_name] = mujoco.Renderer(
                self._env.sim.mj_model,
                height=self._render_height,
                width=self._render_width,
            )
        return self._renderers[camera_name]
