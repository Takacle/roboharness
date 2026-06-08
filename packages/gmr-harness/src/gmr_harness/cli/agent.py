"""GMR Alignment Agent - AI-driven automatic IK config optimization.

Refactored from ``examples/gmr_alignment_agent.py``.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GMR Alignment Agent - AI-driven IK config auto-tuning",
    )
    parser.add_argument("--robot", required=True)
    parser.add_argument(
        "--motion_file",
        default=None,
        help="Motion file path (not required for --solve_mode --src smplx)",
    )
    parser.add_argument("--src", default="bvh", choices=["bvh", "smplx", "fbx_offline"])
    parser.add_argument("--bvh_format", default="auto", choices=["auto", "lafan1", "soma"])
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--max_iter", type=int, default=8, help="Max optimization iterations")
    parser.add_argument("--output", default="./agent_output", help="Output directory for captures")
    parser.add_argument(
        "--model",
        default="glm-5v-turbo",
        help="Vision model for analysis (default: glm-5v-turbo)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Capture and show analysis but do NOT write config changes",
    )
    parser.add_argument(
        "--tpose_spec",
        default=None,
        help="T-pose spec JSON (default: specs/tpose/{robot}.json)",
    )
    parser.add_argument("--tpose_motion", default=None, help="T-pose source motion file")
    parser.add_argument(
        "--tpose_bvh_format",
        default="auto",
        choices=["auto", "lafan1", "soma"],
    )
    parser.add_argument(
        "--tpose_threshold",
        type=float,
        default=5.0,
        help="Per-link angle_deg threshold for auto-stop",
    )
    parser.add_argument(
        "--regression_tolerance",
        type=float,
        default=0.0,
        help="Auto-revert if post-patch deviation exceeds pre-patch by this amount",
    )
    parser.add_argument(
        "--solve_mode",
        action="store_true",
        help="Directly compute IK config quaternions from T-pose",
    )
    parser.add_argument(
        "--tune_mode",
        default="scale",
        choices=["scale", "weights", "optimize_scale", "quaternion"],
    )
    parser.add_argument(
        "--preserve",
        default="",
        help="Comma-separated joints to preserve in --solve_mode",
    )
    parser.add_argument(
        "--world_rot",
        default="",
        help="Set world_rotation after solve: 'angle,ax,ay,az'",
    )
    parser.add_argument(
        "--smplx_template_model",
        default=None,
        help="Path to SMPLX body model for template calibration",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("OPENAI_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")),
    )
    parser.add_argument(
        "--api_base",
        default="https://open.bigmodel.cn/api/paas/v4",
        help="OpenAI-compatible base URL",
    )
    args = parser.parse_args(argv)

    try:
        from gmr_harness.solver import run_agent
    except ImportError as exc:
        print(f"[agent] ERROR: {exc}", file=sys.stderr)
        print("Install dependencies: pip install gmr-harness[all]", file=sys.stderr)
        sys.exit(1)

    sys.exit(run_agent(args) or 0)
