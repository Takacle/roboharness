#!/usr/bin/env python3
"""mjlab Integration Example — Roboharness + mjlab GPU environments.

Demonstrates the full harness workflow with a mjlab environment:
  1. Load a registered mjlab task via the task registry
  2. Run a reach sequence through REACH_PROTOCOL checkpoints
  3. Capture multi-view screenshots at each checkpoint

Run (requires mjlab + GPU):
    pip install roboharness[mjlab]
    python examples/mjlab_integration.py --task Mjlab-Cartpole-Balance

Output:
    ./harness_output/mjlab_cartpole/trial_001/
        rest/    — initial state
        reach/   — mid-episode
        hold/    — end state
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roboharness.backends.mjlab_backend import MjlabBackend
from roboharness.core.harness import Harness
from roboharness.core.protocol import REACH_PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboharness mjlab integration example")
    parser.add_argument(
        "--task",
        default="Mjlab-Cartpole-Balance",
        help="mjlab task ID (default: Mjlab-Cartpole-Balance)",
    )
    parser.add_argument(
        "--output-dir",
        default="./harness_output",
        help="Output directory (default: ./harness_output)",
    )
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["front"],
        help="Camera names to capture (space-separated)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch/Warp device (default: cuda)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("  Roboharness: mjlab Integration Example")
    print("=" * 60)
    print(f"  Task   : {args.task}")
    print(f"  Device : {args.device}")
    print(f"  Cameras: {args.cameras}")

    # 1. Create backend via task registry.
    print("\n[1/4] Loading mjlab environment ...")
    backend = MjlabBackend(
        task_id=args.task,
        cameras=args.cameras,
        render_width=args.width,
        render_height=args.height,
        device=args.device,
    )
    print("      Environment loaded.")

    # 2. Set up harness with semantic reach protocol.
    print("[2/4] Setting up harness ...")
    task_slug = args.task.lower().replace("-", "_")
    harness = Harness(backend, output_dir=str(output_dir), task_name=task_slug)
    harness.load_protocol(REACH_PROTOCOL, phases=["rest", "reach", "hold"])
    print(f"      Protocol : {harness.active_protocol.name}")
    print(f"      Checkpoints: {harness.list_checkpoints()}")

    # 3. Build a simple action sequence (zero actions — just capture states).
    print("[3/4] Running episode ...")
    harness.reset()

    # Use zero actions as a neutral sequence; real usage would supply policy outputs.
    action_dim = backend._env.action_manager.total_action_dim
    zero_action = np.zeros(action_dim, dtype=np.float32)
    steps_per_phase = 50

    for phase in ["rest", "reach", "hold"]:
        actions = [zero_action] * steps_per_phase
        result = harness.run_to_next_checkpoint(actions)
        if result is None:
            print(f"      WARNING: No checkpoint for phase '{phase}'")
            continue
        print(
            f"      Checkpoint '{phase}': {len(result.views)} view(s)"
            f" | step={result.step} | sim_time={result.sim_time:.3f}s"
        )
        for view in result.views:
            print(f"        camera '{view.name}': rgb {view.rgb.shape} {view.rgb.dtype}")

    # 4. Done.
    print("\n[4/4] Done!")
    trial_dir = output_dir / task_slug / "trial_001"
    total_images = len(list(trial_dir.rglob("*_rgb.png"))) if trial_dir.exists() else 0
    print(f"      {total_images} images saved to: {trial_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
