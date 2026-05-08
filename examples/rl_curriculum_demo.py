"""Roboharness RL Curriculum Learning Demo.

Demonstrates how roboharness checkpoint capture enables curriculum learning
on MuJoCo Reacher-v4:

  Phase 0 (ep 0-49):   train from random initial state
  Phase 1 (ep 50-119): reset from states saved at midpoint checkpoints (arm near target)
  Phase 2 (ep 120+):   same pool, policy should be converging

Key roboharness hooks used:
  - RobotHarnessWrapper: captures RGB frames at step 25 (midpoint) and step 50 (final)
  - info["checkpoint"]: signals when a checkpoint fires → triggers state capture
  - env.unwrapped.set_state(): restores MuJoCo state for curriculum reset

Usage:
    MUJOCO_GL=osmesa python examples/rl_curriculum_demo.py

Requirements:
    pip install -e ".[demo]"
"""

from __future__ import annotations

import random
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from roboharness.wrappers import RobotHarnessWrapper

# ---------------------------------------------------------------------------
# State pool
# ---------------------------------------------------------------------------


class MuJoCoStatePool:
    """In-memory pool of (qpos, qvel) snapshots captured at checkpoints."""

    def __init__(self, maxsize: int = 200) -> None:
        self._pool: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=maxsize)

    def add(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self._pool.append((qpos.copy(), qvel.copy()))

    def sample(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._pool:
            return None
        return random.choice(list(self._pool))

    def size(self) -> int:
        return len(self._pool)


# ---------------------------------------------------------------------------
# Gaussian policy (pure numpy, no external ML library required)
# ---------------------------------------------------------------------------


class GaussianPolicy:
    """Two-layer MLP Gaussian policy with REINFORCE updates."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64, lr: float = 3e-4) -> None:
        self.W1 = np.random.randn(hidden, obs_dim) * np.sqrt(2.0 / obs_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(act_dim, hidden) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(act_dim)
        self.log_std = np.full(act_dim, -0.5)
        self.lr = lr

    def _forward(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (hidden activations h, action mean)."""
        h = np.tanh(obs @ self.W1.T + self.b1)
        mean = np.tanh(h @ self.W2.T + self.b2)
        return h, mean

    def sample(self, obs: np.ndarray) -> tuple[np.ndarray, float]:
        """Sample an action. Returns (clipped_action, log_prob)."""
        _, mean = self._forward(obs)
        std = np.exp(self.log_std)
        eps = np.random.randn(*mean.shape)
        action = mean + std * eps
        log_prob = float((-0.5 * eps**2 - self.log_std - 0.5 * np.log(2 * np.pi)).sum())
        return np.clip(action, -1.0, 1.0), log_prob

    def update(self, trajectories: list[dict[str, Any]]) -> float:
        """REINFORCE gradient update over a batch of trajectories."""
        gamma = 0.99
        dW1 = np.zeros_like(self.W1)
        db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2)
        db2 = np.zeros_like(self.b2)
        total_return = 0.0

        for traj in trajectories:
            # Discounted returns
            G = 0.0
            returns: list[float] = []
            for r in reversed(traj["rewards"]):
                G = r + gamma * G
                returns.insert(0, G)
            ret_arr = np.array(returns)
            if len(ret_arr) > 1:
                ret_arr = (ret_arr - ret_arr.mean()) / (ret_arr.std() + 1e-8)
            total_return += float(traj["rewards"][0]) if traj["rewards"] else 0.0

            for obs, action, G_t in zip(traj["obs"], traj["actions"], ret_arr, strict=True):
                h, mean = self._forward(obs)
                std = np.exp(self.log_std)
                eps = (action - mean) / std

                # d(log π)/d(mean) = eps / std
                d_mean = eps / std

                # Backprop through output tanh
                pre2 = h @ self.W2.T + self.b2
                d2 = d_mean * (1.0 - np.tanh(pre2) ** 2)
                dW2 += G_t * np.outer(d2, h)
                db2 += G_t * d2

                # Backprop through hidden tanh
                d1 = (self.W2.T @ d2) * (1.0 - h**2)
                dW1 += G_t * np.outer(d1, obs)
                db1 += G_t * d1

        n = max(len(trajectories), 1)
        self.W1 += self.lr * dW1 / n
        self.b1 += self.lr * db1 / n
        self.W2 += self.lr * dW2 / n
        self.b2 += self.lr * db2 / n
        return total_return / n


# ---------------------------------------------------------------------------
# Curriculum reset
# ---------------------------------------------------------------------------


def curriculum_reset(wrapper: RobotHarnessWrapper, pool: MuJoCoStatePool, phase: int) -> Any:
    """Reset env, optionally seeding from a saved curriculum state."""
    obs, _ = wrapper.reset()
    if phase >= 1 and pool.size() > 0:
        state = pool.sample()
        if state is not None:
            qpos, qvel = state
            mj_env = wrapper.unwrapped
            mj_env.set_state(qpos, qvel)
            obs = mj_env._get_obs()
    return obs


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    NUM_EPISODES = 200
    MAX_STEPS = 50
    BATCH_SIZE = 8
    OUTPUT_DIR = Path("output/curriculum")

    base_env = gym.make("Reacher-v4", render_mode="rgb_array")
    env = RobotHarnessWrapper(
        base_env,
        checkpoints=[
            {"name": "midpoint", "step": 25},
            {"name": "final", "step": 50},
        ],
        output_dir=str(OUTPUT_DIR),
        task_name="reacher",
    )

    obs_dim: int = env.observation_space.shape[0]  # type: ignore[union-attr]
    act_dim: int = env.action_space.shape[0]  # type: ignore[union-attr]
    policy = GaussianPolicy(obs_dim, act_dim)
    pool = MuJoCoStatePool(maxsize=200)
    batch: list[dict[str, Any]] = []

    print(f"Reacher-v4 | obs={obs_dim} act={act_dim}")
    print(f"Captures  -> {OUTPUT_DIR}/reacher/")
    print(f"{'Ep':>4}  {'phase':>5}  {'return':>8}  {'pool':>5}  {'note'}")
    print("-" * 48)

    for episode in range(NUM_EPISODES):
        phase = 0 if episode < 50 else (1 if episode < 120 else 2)
        obs = curriculum_reset(env, pool, phase)

        traj: dict[str, Any] = {"obs": [], "actions": [], "rewards": []}
        ep_return = 0.0

        for _ in range(MAX_STEPS):
            action, _ = policy.sample(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            traj["obs"].append(obs)
            traj["actions"].append(action)
            traj["rewards"].append(float(reward))
            ep_return += float(reward)

            if "checkpoint" in info and info["checkpoint"]["name"] == "midpoint":
                mj_data = env.unwrapped.data
                pool.add(mj_data.qpos.copy(), mj_data.qvel.copy())

            obs = next_obs
            if terminated or truncated:
                break

        batch.append(traj)

        note = ""
        if len(batch) >= BATCH_SIZE:
            policy.update(batch)
            batch = []
            note = "update"

        phase_label = ["random", "curriculum", "curriculum"][phase]
        print(f"{episode:4d}  {phase_label:>10}  {ep_return:8.2f}  {pool.size():5d}  {note}")

    env.close()
    print(f"\nDone. Captures saved to {OUTPUT_DIR}/reacher/")


if __name__ == "__main__":
    main()
