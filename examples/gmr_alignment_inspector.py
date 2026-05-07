"""GMR Joint Alignment Inspector — roboharness-powered visual inspection for GMR IK config tuning.

Two-phase workflow:
  A. Retarget: run GMR retargeting (BVH / SMPL-X / FBX-offline) to produce a qpos sequence.
  B. Inspect: replay via roboharness, capturing front/side/back screenshots at 4 key frames.

After each run, inspect the PNGs to identify misaligned joints, edit the IK config JSON,
and re-run to verify the fix.

Usage:
    # BVH (SOMA or LAFAN1, auto-detected)
    python examples/gmr_alignment_inspector.py \\
        --robot unitree_g1 --motion_file /path/to/motion.bvh

    # SMPL-X .npz
    python examples/gmr_alignment_inspector.py \\
        --robot unitree_g1 --motion_file /path/to/motion.npz --src smplx

    # FBX offline .pkl
    python examples/gmr_alignment_inspector.py \\
        --robot unitree_g1 --motion_file /path/to/motion.pkl --src fbx_offline

    Optional flags:
        --bvh_format  auto|lafan1|soma   BVH parser (default: auto)
        --meshcat                         Open Meshcat interactive viewer
        --frames N                        Limit to first N frames (fast checks)
        --output DIR                      Output directory (default: ./alignment_output)

Alignment tuning guide:
    Visual symptom                  → JSON field to edit in <src>_to_<robot>.json
    Arm / wrist wrong direction     → ik_match_table1["<arm_link>"][4]   (quaternion)
    Leg rotated around hip          → ik_match_table1["<hip_link>"][4]   (quaternion)
    Spine / torso tilted            → ik_match_table1[root_link][4]  or  "world_rotation"
    Foot twisted / inverted         → ankle link quaternion offset
    All joints look rotated by 90°  → "world_rotation" at the top level

Quaternion format in configs: [w, x, y, z] (scalar-first).
Common correction values:
    [0.707, 0, 0, 0.707]  = +90° around Z
    [0.707, 0, -0.707, 0] = +90° around Y
    [0.707, -0.707, 0, 0] = +90° around X
    [0, 0, 0, 1]          = 180° around Z
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _gmr_shared import (
    GMR_ROOT,
    GMRReplayBackend,
    find_root_body,
    load_motion,
    scaled_human_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GMR Joint Alignment Inspector — visual config tuning with roboharness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--robot", required=True, help="Robot name (e.g. unitree_g1)")
    parser.add_argument("--motion_file", required=True, help="Path to input motion file")
    parser.add_argument(
        "--src",
        default="bvh",
        choices=["bvh", "smplx", "fbx_offline"],
        help="Motion source format: bvh (default), smplx (.npz), fbx_offline (.pkl)",
    )
    parser.add_argument(
        "--bvh_format",
        default="auto",
        choices=["auto", "lafan1", "soma"],
        help="BVH sub-format (only used when --src bvh). auto-detects SOMA vs LAFAN1.",
    )
    parser.add_argument(
        "--output",
        default="./alignment_output",
        help="Root output directory for captures (default: ./alignment_output)",
    )
    parser.add_argument(
        "--meshcat",
        action="store_true",
        help="Open Meshcat interactive 3D viewer in browser (requires local display)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Limit retargeting to first N frames (faster for quick checks)",
    )
    args = parser.parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.params import ROBOT_XML_DICT, VIEWER_CAM_DISTANCE_DICT
    from tqdm import tqdm

    print("\n=== Phase A: Retargeting ===")
    print(f"Motion file : {args.motion_file}")
    print(f"Source      : {args.src}")
    print(f"Robot       : {args.robot}")

    frames, human_height, _fps = load_motion(args.src, args.motion_file, args.bvh_format)

    print(f"Human height: {human_height:.2f} m")

    if args.frames is not None:
        frames = frames[: args.frames]
        print(f"Truncated to first {len(frames)} frames")

    retargeter = GMR(
        src_human=args.src,
        tgt_robot=args.robot,
        actual_human_height=human_height,
        verbose=False,
    )

    qpos_list: list[np.ndarray] = []
    human_list: list[dict] = []
    for frame in tqdm(frames, desc="retargeting"):
        qpos_list.append(retargeter.retarget(frame).copy())
        human_list.append(scaled_human_reference(retargeter, frame))

    qpos_seq = np.array(qpos_list)
    n = len(qpos_seq)
    print(f"Done. qpos shape: {qpos_seq.shape}")

    from roboharness.core.harness import Harness

    print("\n=== Phase B: Visual Inspection ===")

    xml_path = Path(str(ROBOT_XML_DICT[args.robot]))
    cam_distance = float(VIEWER_CAM_DISTANCE_DICT.get(args.robot, 2.5))
    cameras = ["inspect_front", "inspect_side", "inspect_back"]
    root_body = find_root_body(xml_path)

    print(f"Robot XML    : {xml_path}")
    print(f"Root body    : {root_body}")
    print(f"Cam distance : {cam_distance:.1f} m")

    backend = GMRReplayBackend(
        xml_path=xml_path,
        qpos_seq=qpos_seq,
        cameras=cameras,
        root_body_name=root_body,
        cam_distance=cam_distance,
        use_meshcat=args.meshcat,
        human_seq=human_list,
    )

    harness = Harness(
        backend=backend,
        output_dir=args.output,
        task_name=args.robot,
    )

    checkpoint_defs = [
        ("frame_start", 0),
        ("frame_quarter", n // 4),
        ("frame_half", n // 2),
        ("frame_three_quarter", 3 * n // 4),
    ]

    for name, frame_idx in checkpoint_defs:
        harness.add_checkpoint(
            name=name,
            cameras=cameras,
            trigger_step=frame_idx + 1,
        )

    harness.reset()
    trial_dir = Path(args.output) / args.robot / f"trial_{harness._trial_count:03d}"

    print(f"\nCapturing {len(checkpoint_defs)} checkpoints x {len(cameras)} cameras")
    print(f"Output: {trial_dir.resolve()}\n")

    for _name, frame_idx in checkpoint_defs:
        steps_needed = max(1, (frame_idx + 1) - harness.step_count + 10)
        result = harness.run_to_next_checkpoint([None] * steps_needed)

        if result is not None:
            print(f"  [{result.checkpoint_name}] frame={frame_idx}  step={result.step}")
            for view in result.views:
                png = trial_dir / result.checkpoint_name / f"{view.name}_rgb.png"
                print(f"    {png.resolve()}")

    backend.cleanup()

    print("\n=== Done ===")
    print(f"Captures saved to: {trial_dir.resolve()}")
    ik_configs_dir = GMR_ROOT / "general_motion_retargeting" / "ik_configs"
    config_path = ik_configs_dir / f"{args.src}_to_{args.robot}.json"
    print(f"IK config to edit: {config_path}")
    print("""
Alignment tuning guide:
  Visual symptom                 → config field
  Arm/wrist wrong direction      → ik_match_table1["<arm_link>"][4]  (quaternion [w,x,y,z])
  Leg rotated around hip         → ik_match_table1["<hip_link>"][4]
  Spine/torso tilted             → ik_match_table1[root_link][4]  or  "world_rotation"
  Foot twisted/inverted          → ankle link quaternion
  All joints off by 90°          → "world_rotation" field

Re-run after each edit to compare the new renders.
""")


if __name__ == "__main__":
    main()
