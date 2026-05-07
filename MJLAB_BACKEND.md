# MjlabBackend Interface Reference

Quick reference for integrating mjlab environments with roboharness.

## Import path

```python
from roboharness.backends.mjlab_backend import MjlabBackend
```

## Constructor signature

```python
MjlabBackend(
    task_id: str | None = None,          # registered task ID, e.g. "Instinct-Locomotion-Flat-G1-v0"
    env_cfg = None,                       # or pass a ManagerBasedRlEnvCfg directly
    cameras: list[str] | None = None,    # camera names; fallback: CPU mujoco.Renderer
    render_width: int = 640,
    render_height: int = 480,
    env_id: int = 0,                     # which parallel env to expose (slice index)
    num_envs: int = 1,                   # GPU parallelism
    device: str = "cuda",
    play: bool = False,                  # use play_env_cfg instead of env_cfg
)
```

Provide **either** `task_id` (from mjlab's task registry) **or** `env_cfg` — not both.

## Does it need Harness / RobotHarnessWrapper?

Use `Harness` (not `RobotHarnessWrapper`).  
`RobotHarnessWrapper` wraps Gymnasium training loops; `Harness` is the right fit for
checkpoint-based visual testing.

## Minimal example

```python
from roboharness.backends.mjlab_backend import MjlabBackend
from roboharness.core.harness import Harness

backend = MjlabBackend(
    task_id="Instinct-Locomotion-Flat-G1-v0",
    cameras=["front", "side"],
    device="cuda",
)

harness = Harness(backend, output_dir="./harness_output", task_name="g1_loco")
harness.add_checkpoint("init",  cameras=["front", "side"])
harness.add_checkpoint("walk",  cameras=["front", "side"])

harness.reset()

import numpy as np
action_dim = backend._env.action_manager.total_action_dim

# Phase 1: capture initial stance
harness.run_to_next_checkpoint([np.zeros(action_dim)] * 10)

# Phase 2: run 200 steps with your policy actions
policy_actions = [np.random.randn(action_dim).astype("f4") for _ in range(200)]
result = harness.run_to_next_checkpoint(policy_actions)

# Inspect
print(result.views[0].rgb.shape)  # (480, 640, 3)
print(result.state["qpos"])       # numpy array from mj_data.qpos
```

## What's exposed on `result`

| Field | Type | Notes |
|-------|------|-------|
| `result.views[i].rgb` | `np.ndarray (H,W,3) uint8` | one per camera |
| `result.views[i].depth` | `np.ndarray (H,W) float32` or `None` | if renderer supports it |
| `result.state["qpos"]` | `np.ndarray` | CPU-side `mj_data.qpos` |
| `result.state["qvel"]` | `np.ndarray` | CPU-side `mj_data.qvel` |
| `result.state["time"]` | `float` | simulation time in seconds |
| `result.step` | `int` | harness step count |
| `result.sim_time` | `float` | same as `state["time"]` |

## Camera rendering

- If a `CameraSensor` named `camera_name` is registered in the mjlab scene,
  its GPU-rendered RGB tensor is used (converted to numpy).
- Otherwise, falls back to a CPU `mujoco.Renderer` on the scene's `mj_model`/`mj_data`.
  Any camera defined in the MuJoCo model XML is valid.

## State save / restore

The backend uses `mujoco.mj_getState(mjSTATE_FULLPHYSICS)` on the CPU-side `mj_data`
and calls `sim.forward()` after restore to propagate state to GPU Warp buffers.
This is handled automatically by `harness.run_to_next_checkpoint()` and
`harness.restore_checkpoint()`.

## Installation

```bash
# Install mjlab from its local path first (not on PyPI):
pip install /home/user2/mjlab

# Then install roboharness with the mjlab extra:
pip install -e ".[mjlab]"
```
