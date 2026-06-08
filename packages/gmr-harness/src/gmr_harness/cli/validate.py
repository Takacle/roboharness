"""Numeric T-pose alignment validator for a GMR IK config.

Refactored from ``examples/gmr_tpose_validate.py`` to use proper package imports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from gmr_harness.alignment import (
    compute_deviations,
    load_tpose_spec,
    total_deviation,
    worst_k,
)
from gmr_harness.alignment.smplx_coordinate import validate_smplx_runtime_config

_HERE = Path(__file__).resolve().parent


def _ensure_gmr(feature: str = ""):
    from gmr_harness.gmr_integration import _require_gmr

    _require_gmr(feature)


def _retarget_first_frame(src: str, motion_file: str, robot: str, bvh_format: str) -> np.ndarray:
    _ensure_gmr("validation retarget")
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.params import IK_CONFIG_DICT

    from gmr_harness.gmr_integration import load_motion

    if src == "smplx":
        cfg_path = IK_CONFIG_DICT.get(src, {}).get(robot, "")
        if cfg_path:
            p = Path(str(cfg_path))
            if p.exists():
                with p.open() as _f:
                    validate_smplx_runtime_config(json.load(_f), p)

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
    _ensure_gmr("SMPL-X template validation")
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    from gmr_harness.alignment.smplx_template import (
        load_smplx_template_tpose,
        resolve_body_model_path,
    )

    body_model_root = resolve_body_model_path(body_model_dir)

    frame, human_height = load_smplx_template_tpose(body_model_root)
    print(f"[validate] Template frame: {len(frame)} joints, height={human_height:.2f}m")

    from general_motion_retargeting.params import IK_CONFIG_DICT

    cfg_path = IK_CONFIG_DICT.get("smplx", {}).get(robot, "")
    if cfg_path:
        p = Path(str(cfg_path))
        if p.exists():
            with p.open() as _f:
                validate_smplx_runtime_config(json.load(_f), p)

    retargeter = GMR(
        src_human="smplx",
        tgt_robot=robot,
        actual_human_height=human_height,
        verbose=False,
    )
    return retargeter.retarget(frame).copy()


def _resolve_spec_xml(spec: dict) -> str:
    resolved = spec.get("_resolved_xml_path")
    if resolved and Path(resolved).exists():
        return resolved
    return spec["xml_path"]


def _default_spec_path(robot: str) -> Path:
    return Path("specs/tpose") / f"{robot}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a retargeted T-pose against the committed spec.",
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
        help="Per-link angle_deg threshold for pass/fail (default 5deg)",
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
        help="Path to SMPLX body model directory. Defaults to GMR/assets/body_models.",
    )
    args = parser.parse_args(argv)

    if args.use_smplx_template and args.src != "smplx":
        print(
            "[validate] ERROR: --use_smplx_template requires --src smplx.",
            file=sys.stderr,
        )
        return 2

    spec_path = args.spec or _default_spec_path(args.robot)
    if not spec_path.exists():
        print(f"[validate] ERROR: spec not found at {spec_path}", file=sys.stderr)
        print("           Author it with gmr-harness stage first.", file=sys.stderr)
        return 2

    spec = load_tpose_spec(spec_path)

    print(f"[validate] robot     : {args.robot}")
    print(f"[validate] spec      : {spec_path}")
    if args.use_smplx_template:
        print("[validate] source    : SMPL-X template (body model zero-pose)")
    else:
        print(f"[validate] motion    : {args.tpose_motion} ({args.src}/{args.bvh_format})")
    print(f"[validate] threshold : {args.threshold} deg per link")
    print()

    if args.src in ("smplx",):
        import numpy as np

        root_quat = spec.get("qpos", [0] * 7)[3:7]
        root_quat = np.asarray(root_quat, dtype=np.float64)
        root_quat /= np.linalg.norm(root_quat) + 1e-12
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        dot = float(abs(np.dot(root_quat, identity)))
        angle_deg = float(np.degrees(2 * np.arccos(np.clip(dot, 0, 1))))
        print(f"[validate] SMPL-X root quaternion deviation from identity: {angle_deg:.2f}deg")
        if angle_deg > 10.0:
            print(
                f"[validate] WARNING: SMPL-X root quaternion deviates {angle_deg:.2f}deg "
                "from identity. Re-stage with gmr-harness stage --src smplx."
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
    report = compute_deviations(qpos, _resolve_spec_xml(spec), spec)
    total = total_deviation(report)
    top = worst_k(report, args.top_k)
    max_angle = top[0][1] if top else 0.0

    print(f"[validate] total_deviation : {total:7.2f}deg  ({len(report)} links)")
    print(f"[validate] max_angle       : {max_angle:7.2f}deg")
    print(f"[validate] worst {args.top_k}:")
    for name, angle in top:
        axis = report[name]["axis"]
        axis_str = f"[{axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f}]"
        print(f"           {name:40s} {angle:7.2f}deg  axis={axis_str}")

    print()
    if max_angle < args.threshold:
        print(f"[validate] PASS - all links within {args.threshold}deg of T-pose.")
        return 0
    print(
        f"[validate] FAIL - {sum(1 for _, a in top if a >= args.threshold)} of top "
        f"{args.top_k} exceed {args.threshold}deg."
    )
    if args.src in ("smplx",) and max_angle > 30.0:
        print(
            "[validate] SMPL-X large-angle failure hint:\n"
            "           - Verify spec was generated with stage --src smplx\n"
            "           - Re-solve using: gmr-harness setup --robot <robot> --src smplx\n"
            "           - Walking .npz files are motion inputs, NOT calibration sources"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
