"""Numeric T-pose alignment validator for a GMR IK config.

Given a canonical T-pose source motion (BVH/SMPL-X/FBX) and a committed
``specs/tpose/<robot>.json`` spec, this script:

1. Runs GMR retargeting on the source motion's first frame → candidate ``qpos``.
2. Calls ``roboharness.alignment.compute_deviations`` to measure per-link
   residual rotation against the spec.
3. Prints ``total_deviation``, ``worst_k``, and a pass/fail verdict.

Exit code is ``0`` iff every link's ``angle_deg`` is below ``--threshold``
(default 5°). This makes the script drop-in for CI gates, pre-commit hooks,
or the ``gmr_alignment_agent.py`` convergence check.

Unlike the VLM loop, this is a **pure numeric sensor**: no model calls, no
images. If this script reports 0°, the config is correct at T-pose; if it
reports 1000°, the VLM is not going to fix that by eyeballing pixels.

Usage:
    python examples/gmr_tpose_validate.py \\
        --robot unitree_g1 \\
        --tpose_motion /path/to/tpose.bvh \\
        --src bvh \\
        [--bvh_format soma] \\
        [--spec specs/tpose/unitree_g1.json] \\
        [--threshold 5.0]

Requires ``pip install -e ".[demo]"`` plus a GMR checkout as a sibling dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from _gmr_shared import load_motion

from roboharness.alignment import (
    compute_deviations,
    load_tpose_spec,
    total_deviation,
    worst_k,
)

_HERE = Path(__file__).resolve().parent


def _retarget_first_frame(src: str, motion_file: str, robot: str, bvh_format: str) -> np.ndarray:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    frames, human_height, _ = load_motion(src, motion_file, bvh_format)

    if not frames:
        raise RuntimeError(f"No frames loaded from {motion_file}")

    retargeter = GMR(
        src_human=src,
        tgt_robot=robot,
        actual_human_height=human_height,
        verbose=False,
    )
    return retargeter.retarget(frames[0]).copy()


def _default_spec_path(robot: str) -> Path:
    return _HERE.parent / "specs" / "tpose" / f"{robot}.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a retargeted T-pose against the committed spec.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--robot", required=True, help="Robot key (e.g. unitree_g1)")
    parser.add_argument(
        "--tpose_motion", required=True, help="Path to a canonical T-pose motion file"
    )
    parser.add_argument("--src", default="bvh", choices=["bvh", "smplx", "fbx_offline"])
    parser.add_argument("--bvh_format", default="auto", choices=["auto", "lafan1", "soma"])
    parser.add_argument(
        "--spec",
        type=Path,
        help="Path to T-pose spec JSON. Defaults to specs/tpose/{robot}.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=5.0,
        help="Per-link angle_deg threshold for pass/fail (default 5°)",
    )
    parser.add_argument("--top_k", type=int, default=5, help="How many worst links to print")
    args = parser.parse_args()

    spec_path = args.spec or _default_spec_path(args.robot)
    if not spec_path.exists():
        print(f"[validate] ERROR: spec not found at {spec_path}", file=sys.stderr)
        print("           Author it with scripts/stage_tpose.py first.", file=sys.stderr)
        return 2

    spec = load_tpose_spec(spec_path)

    # Apply SMPL-X base rotation to spec R matrices (same logic as agent)
    if args.src in ("smplx",):
        from roboharness.alignment.orientation_aligner import apply_smplx_base_rotation

        spec = apply_smplx_base_rotation(spec)

    print(f"[validate] robot     : {args.robot}")
    print(f"[validate] spec      : {spec_path}")
    print(f"[validate] motion    : {args.tpose_motion} ({args.src}/{args.bvh_format})")
    print(f"[validate] threshold : {args.threshold}° per link")
    print()

    qpos = _retarget_first_frame(args.src, args.tpose_motion, args.robot, args.bvh_format)
    report = compute_deviations(qpos, spec["xml_path"], spec)
    total = total_deviation(report)
    top = worst_k(report, args.top_k)
    max_angle = top[0][1] if top else 0.0

    print(f"[validate] total_deviation : {total:7.2f}°  ({len(report)} links)")
    print(f"[validate] max_angle       : {max_angle:7.2f}°")
    print(f"[validate] worst {args.top_k}:")
    for name, angle in top:
        axis = report[name]["axis"]
        axis_str = f"[{axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f}]"
        print(f"           {name:40s} {angle:7.2f}°  axis={axis_str}")

    print()
    if max_angle < args.threshold:
        print(f"[validate] PASS — all links within {args.threshold}° of T-pose.")
        return 0
    print(
        f"[validate] FAIL — {sum(1 for _, a in top if a >= args.threshold)} of top "
        f"{args.top_k} exceed {args.threshold}°. Tune the IK config and re-run."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
