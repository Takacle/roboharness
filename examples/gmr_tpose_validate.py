"""Numeric T-pose alignment validator for a GMR IK config.

Given a canonical T-pose source motion (BVH/SMPL-X/FBX) and a committed
``specs/tpose/<robot>.json`` spec, this script:

1. Runs GMR retargeting on the source motion's first frame → candidate ``qpos``.
2. Calls ``roboharness.alignment.compute_deviations`` to measure per-link
   residual rotation against the spec.
3. Prints ``total_deviation``, ``worst_k``, and a pass/fail verdict.

For SMPL-X sources, a template validation mode is also available via
``--use_smplx_template``. This generates a synthetic SMPL-X frame from the
body model zero-pose (no motion file required) and retargets that through
GMR, comparing the result against the committed spec. This is the recommended
validation path for SMPL-X configs because it avoids the ~180 degree root
orientation deviation present in motion capture walking sequences.

Exit code is ``0`` iff every link's ``angle_deg`` is below ``--threshold``
(default 5°). This makes the script drop-in for CI gates, pre-commit hooks,
or the ``gmr_alignment_agent.py`` convergence check.

Unlike the VLM loop, this is a **pure numeric sensor**: no model calls, no
images. If this script reports 0°, the config is correct at T-pose; if it
reports 1000°, the VLM is not going to fix that by eyeballing pixels.

Usage:
    # Motion-based validation (BVH/SMPL-X/FBX motion file)
    python examples/gmr_tpose_validate.py \\
        --robot unitree_g1 \\
        --tpose_motion /path/to/tpose.bvh \\
        --src bvh \\
        [--bvh_format soma] \\
        [--spec specs/tpose/unitree_g1.json] \\
        [--threshold 5.0]

    # Template-based validation (SMPL-X, no motion file needed)
    python examples/gmr_tpose_validate.py \\
        --robot v11 \\
        --src smplx \\
        --use_smplx_template \\
        --spec specs/tpose/v11.json

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


def _retarget_template_frame(
    robot: str,
    body_model_dir: str | None = None,
) -> np.ndarray:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    from roboharness.alignment.smplx_template import (
        load_smplx_template_tpose,
        resolve_body_model_path,
    )

    body_model_root = resolve_body_model_path(body_model_dir)

    frame, human_height = load_smplx_template_tpose(body_model_root)
    print(f"[validate] Template frame: {len(frame)} joints, height={human_height:.2f}m")

    retargeter = GMR(
        src_human="smplx",
        tgt_robot=robot,
        actual_human_height=human_height,
        verbose=False,
    )
    return retargeter.retarget(frame).copy()


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
        "--tpose_motion",
        required=False,
        default=None,
        help="Path to a canonical T-pose motion file "
        "(not required when --use_smplx_template is set)",
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
    parser.add_argument(
        "--use_smplx_template",
        action="store_true",
        help="Use SMPL-X body model zero-pose as calibration source (no --tpose_motion needed)",
    )
    parser.add_argument(
        "--smplx_template_model",
        default=None,
        help="Path to SMPLX body model *directory* (containing smplx/ subfolder). "
        "Defaults to GMR/assets/body_models.",
    )
    args = parser.parse_args()

    if args.use_smplx_template and args.src != "smplx":
        print(
            "[validate] ERROR: --use_smplx_template requires --src smplx.",
            file=sys.stderr,
        )
        return 2

    spec_path = args.spec or _default_spec_path(args.robot)
    if not spec_path.exists():
        print(f"[validate] ERROR: spec not found at {spec_path}", file=sys.stderr)
        print("           Author it with scripts/stage_tpose.py first.", file=sys.stderr)
        return 2

    spec = load_tpose_spec(spec_path)

    print(f"[validate] robot     : {args.robot}")
    print(f"[validate] spec      : {spec_path}")
    if args.use_smplx_template:
        print("[validate] source    : SMPL-X template (body model zero-pose)")
    else:
        print(f"[validate] motion    : {args.tpose_motion} ({args.src}/{args.bvh_format})")
    print(f"[validate] threshold : {args.threshold}° per link")
    print()

    if args.src in ("smplx",):
        import numpy as np

        root_quat = spec.get("qpos", [0] * 7)[3:7]
        root_quat = np.asarray(root_quat, dtype=np.float64)
        root_quat /= np.linalg.norm(root_quat) + 1e-12
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        dot = float(abs(np.dot(root_quat, identity)))
        angle_deg = float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))
        print(f"[validate] SMPL-X root quaternion deviation from identity: {angle_deg:.2f}°")
        if angle_deg > 10.0:
            print(
                f"[validate] WARNING: SMPL-X root quaternion deviates {angle_deg:.2f}° "
                "from identity. The robot may not be upright. "
                "Re-stage the T-pose spec with stage_tpose.py --src smplx."
            )

    if args.use_smplx_template:
        qpos = _retarget_template_frame(args.robot, args.smplx_template_model)
    else:
        if not args.tpose_motion:
            print(
                "[validate] ERROR: --tpose_motion is required when "
                "--use_smplx_template is not set.",
                file=sys.stderr,
            )
            return 2
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
    if args.src in ("smplx",) and max_angle > 30.0:
        print(
            "[validate] SMPL-X large-angle failure hint:\n"
            "           - Verify staged spec was generated with "
            "`stage_tpose.py --src smplx`\n"
            "           - Re-solve offsets using the template calibration:\n"
            "             python scripts/setup_robot.py --robot <robot> "
            "--src smplx --update_scripts\n"
            "           - Walking .npz files are motion inputs, "
            "NOT calibration sources\n"
            "           - Re-run setup/solve if spec was staged "
            "without SMPL-X root base quat"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
