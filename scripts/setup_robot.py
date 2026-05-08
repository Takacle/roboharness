"""One-command setup for a new robot in the GMR / Harness pipeline.

Orchestrates: config generation → params registration → T-pose staging →
quaternion solving → validation.

Usage examples::

    # Minimal (generate config + register in params.py + update scripts)
    python scripts/setup_robot.py --robot my_robot \
        --xml $GMR_ROOT/assets/my_robot/robot.xml \
        --formats smplx bvh \
        --auto_register --update_scripts

    # Full pipeline with T-pose alignment
    python scripts/setup_robot.py --robot unitree_h1 \
        --tpose_motion /path/to/tpose.bvh \
        --auto_register --update_scripts

    # Clone from existing robot (same format only)
    python scripts/setup_robot.py --robot my_robot \
        --clone_from unitree_h1 \
        --xml $GMR_ROOT/assets/my_robot/robot.xml \
        --formats smplx

    # Interactive mode (prompts for unmatched body mappings)
    python scripts/setup_robot.py --robot my_robot \
        --xml $GMR_ROOT/assets/my_robot/robot.xml --interactive

Steps performed:

1. Load MuJoCo XML, extract body names.
2. Match body names to human skeleton roles (heuristic + interactive).
3. Generate IK config JSON(s) for each format in ``--formats``.
4. [optional] Auto-register in ``params.py`` and update script ``choices``.
5. [optional] Stage T-pose and solve quaternions.
6. [optional] Validate alignment.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
sys.path.insert(0, str(_PROJECT / "src"))

from roboharness.alignment._gmr_params import load_gmr_params  # noqa: E402
from roboharness.alignment._gmr_path import find_gmr_root  # noqa: E402
from roboharness.alignment.body_matcher import match_bodies  # noqa: E402
from roboharness.alignment.config_gen import (  # noqa: E402
    clone_ik_config,
    generate_ik_config,
    write_ik_config,
)
from roboharness.alignment.gmr_register import (  # noqa: E402
    register_in_params,
    update_script_choices,
)
from roboharness.alignment.orientation_aligner import (  # noqa: E402
    extract_xml_body_names,
    parse_world_rotation_arg,
)
from roboharness.alignment.skeleton_maps import get_skeleton  # noqa: E402
from roboharness.alignment.smplx_offset_solver import (  # noqa: E402
    solve_smplx_offsets_from_template,
    write_solved_config,
)

GMR_ROOT = find_gmr_root()


def _find_xml(robot: str) -> Path | None:
    params = load_gmr_params(GMR_ROOT)
    xml_dict = getattr(params, "ROBOT_XML_DICT", {})
    if robot in xml_dict:
        return Path(str(xml_dict[robot]))
    asset_dir = GMR_ROOT / "assets" / robot
    if asset_dir.is_dir():
        xmls = sorted(asset_dir.glob("*.xml"))
        if xmls:
            return xmls[0]
    return None


def _resolve_xml(args: argparse.Namespace) -> Path:
    if args.xml:
        xml_path = Path(args.xml).resolve()
        if not xml_path.exists():
            print(f"[setup] ERROR: XML not found: {xml_path}")
            sys.exit(1)
        asset_dir = (GMR_ROOT / "assets" / args.robot).resolve()
        try:
            rel = xml_path.relative_to(asset_dir)
        except ValueError:
            print(
                f"[setup] ERROR: --xml must be inside {asset_dir}/.\n"
                f"  Provided path: {xml_path}\n"
                f"  Move the XML (and its mesh assets) into {asset_dir}/ "
                f"before running setup, or omit --xml to auto-detect."
            )
            sys.exit(1)
        if rel.parent != Path():
            print(
                f"[setup] ERROR: --xml must be directly inside {asset_dir}/, "
                f"not a subdirectory.\n"
                f"  Provided path: {xml_path}\n"
                f"  Nested paths like {asset_dir.name}/subdir/model.xml are not "
                f"supported because params.py registers only the filename."
            )
            sys.exit(1)
        return xml_path

    if args.robot:
        xml_path = _find_xml(args.robot)
        if xml_path is not None:
            return xml_path
        print(
            f"[setup] ERROR: cannot find XML for {args.robot!r}. "
            f"Place XML at {GMR_ROOT}/assets/{args.robot}/*.xml or use --xml."
        )
        sys.exit(1)

    print("[setup] ERROR: --robot or --xml is required.")
    sys.exit(1)


def _interactive_resolve(result, skeleton, robot_body_names):
    """Prompt user to resolve unmatched roles."""

    overrides: dict[str, str] = {}
    for role in result.unmatched_roles:
        joint_name = skeleton.role_to_joint[role]
        print(f"\n  Unmatched role: {role} (human joint: {joint_name})")
        print(f"  Available bodies: {result.unmatched_bodies}")
        answer = input(f"  Enter body name for {role} (or 'skip'): ").strip()
        if answer.lower() == "skip" or not answer:
            print(f"  → skipped {role}")
            continue
        if answer in robot_body_names:
            overrides[role] = answer
            print(f"  → mapped {role} → {answer}")
        else:
            print(f"  → '{answer}' not found in body list, skipped")
    if overrides:
        result = match_bodies(robot_body_names, skeleton, overrides=overrides)
    return result


def _solve_smplx_offsets(
    robot: str,
    tpose_spec_path: Path,
    body_model_root: Path | None = None,
) -> bool:
    params = load_gmr_params(GMR_ROOT)
    ik_dict = getattr(params, "IK_CONFIG_DICT", {})
    config_path = Path(str(ik_dict.get("smplx", {}).get(robot, "")))
    if not config_path.exists():
        offsets_script = GMR_ROOT / "scripts" / "compute_smplx_tpose_offsets.py"
        if offsets_script.exists():
            cmd = [sys.executable, str(offsets_script), "--robot", robot, "--generate"]
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                print("[setup] ERROR: SMPL-X offset computation failed.")
                return False
            params = load_gmr_params(GMR_ROOT)
            ik_dict = getattr(params, "IK_CONFIG_DICT", {})
            config_path = Path(str(ik_dict.get("smplx", {}).get(robot, "")))
        else:
            print(f"[setup] ERROR: IK config not found at {config_path}")
            return False

    if not tpose_spec_path.exists():
        print(f"[setup] ERROR: T-pose spec not found at {tpose_spec_path}")
        return False

    try:
        solved = solve_smplx_offsets_from_template(
            ik_config_path=config_path,
            tpose_spec_path=tpose_spec_path,
            body_model_path=body_model_root,
        )
    except Exception as exc:
        print(f"[setup] ERROR: template offset solving failed: {exc}")
        return False

    write_solved_config(solved, config_path)
    print(f"[setup] Solved SMPL-X offsets from template → {config_path.name}")
    return True


def _solve_via_agent(args: argparse.Namespace, spec_path: Path) -> bool:
    solve_cmd = [
        sys.executable,
        str(_PROJECT / "examples" / "gmr_alignment_agent.py"),
        "--robot",
        args.robot,
        "--src",
        args.tpose_src,
        "--motion_file",
        args.tpose_motion,
        "--tpose_spec",
        str(spec_path),
        "--tpose_motion",
        args.tpose_motion,
        "--tpose_src",
        args.tpose_src,
        "--tpose_bvh_format",
        args.bvh_format,
        "--solve_mode",
    ]
    if args.world_rot:
        solve_cmd.extend(["--world_rot", args.world_rot])
    result = subprocess.run(solve_cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print("[setup] ERROR: solving failed.")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup a new robot in the GMR alignment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--robot", required=True, help="Robot key (e.g. unitree_h1)")
    parser.add_argument("--xml", help="Override XML path (auto-detected from --robot if omitted)")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        choices=["smplx", "bvh", "fbx", "fbx_offline"],
        help="Source formats to generate IK configs for (default: matches --src)",
    )
    parser.add_argument("--base_body", help="Override root body name (auto-detected if omitted)")
    parser.add_argument(
        "--cam_distance", type=float, default=2.5, help="Viewer camera distance (default: 2.5)"
    )
    parser.add_argument("--clone_from", help="Clone IK config from existing robot (same format)")
    parser.add_argument(
        "--mapping_override",
        action="append",
        default=[],
        metavar="ROLE=BODY",
        help="Manual body mapping, repeatable",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for unmatched body mappings (on by default when stdout is a tty)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive prompts for unmatched body mappings",
    )
    parser.add_argument("--auto_register", action="store_true", help="Auto-modify GMR params.py")
    parser.add_argument(
        "--update_scripts", action="store_true", help="Auto-update GMR script --robot choices"
    )
    parser.add_argument("--tpose_motion", help="T-pose source motion (BVH/SMPL-X)")
    parser.add_argument(
        "--tpose_src",
        default=None,
        choices=["bvh", "smplx", "fbx_offline"],
        help="Source format for --tpose_motion (default: auto-detect from extension)",
    )
    parser.add_argument(
        "--tpose_preset",
        default="tpose",
        choices=["home", "tpose"],
        help="Baseline pose for T-pose staging (default: tpose)",
    )
    parser.add_argument(
        "--tpose_joint",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a joint angle for T-pose staging, repeatable",
    )
    parser.add_argument("--src", default="bvh", choices=["bvh", "smplx", "fbx_offline"])
    parser.add_argument("--bvh_format", default="auto", choices=["auto", "lafan1", "soma"])
    parser.add_argument("--world_rot", default="", help="world_rotation (e.g. '90,0,0,1')")
    parser.add_argument(
        "--smplx_template_model",
        default=None,
        help="Path to SMPLX body model *directory* (e.g. .../body_models, "
        "which must contain a smplx/ subfolder with SMPLX_MALE.npz). "
        "Defaults to GMR/assets/body_models when it exists.",
    )
    parser.add_argument("--output_dir", default=str(_PROJECT / "specs" / "tpose"))
    parser.add_argument("--skip_stage", action="store_true", help="Skip T-pose staging")
    parser.add_argument("--skip_solve", action="store_true", help="Skip quaternion solving")
    parser.add_argument(
        "--skip_validate", action="store_true", help="Skip validation after solving"
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.tpose_src is None and args.tpose_motion:
        ext = Path(args.tpose_motion).suffix.lower()
        if ext == ".npz":
            args.tpose_src = "smplx"
        elif ext == ".bvh":
            args.tpose_src = "bvh"
        else:
            args.tpose_src = args.src
    elif args.tpose_src is None:
        args.tpose_src = args.src

    if args.formats is None:
        args.formats = [args.src]

    if args.tpose_src not in args.formats:
        args.formats.append(args.tpose_src)

    _DEFAULT_SMPLX_BODY_MODEL = GMR_ROOT / "assets" / "body_models"
    smplx_body_model_root: Path | None = None
    if args.smplx_template_model:
        smplx_body_model_root = Path(args.smplx_template_model).resolve()
        if not smplx_body_model_root.exists():
            print(f"[setup] ERROR: --smplx_template_model path not found: {smplx_body_model_root}")
            sys.exit(1)
    elif _DEFAULT_SMPLX_BODY_MODEL.exists():
        smplx_body_model_root = _DEFAULT_SMPLX_BODY_MODEL

    smplx_template_available = (
        args.src == "smplx" or args.tpose_src == "smplx"
    ) and smplx_body_model_root is not None

    print(f"\n{'=' * 60}")
    print(f"Setup Robot: {args.robot}")
    print(f"{'=' * 60}")

    # ── Step 0: Resolve XML ──
    xml_path = _resolve_xml(args)
    print(f"[setup] XML: {xml_path}")

    body_names = extract_xml_body_names(xml_path)
    preview = str(body_names[:10])
    suffix = "..." if len(body_names) > 10 else ""
    print(f"[setup] Bodies ({len(body_names)}): {preview}{suffix}")

    params = load_gmr_params(GMR_ROOT)
    base_dict = getattr(params, "ROBOT_BASE_DICT", {})
    root_body = args.base_body or base_dict.get(args.robot)
    if not root_body:
        first_body_match = re.search(r'<body\s+name="([^"]+)"', xml_path.read_text())
        root_body = first_body_match.group(1) if first_body_match else body_names[0]
    print(f"[setup] Root body: {root_body}")

    overrides: dict[str, str] = {}
    for item in args.mapping_override:
        if "=" not in item:
            print(f"[setup] WARN: ignoring malformed --mapping_override: {item!r}")
            continue
        role, _, body = item.partition("=")
        overrides[role.strip()] = body.strip()

    # ── Step 1-2: Match bodies and generate IK config per format ──
    config_paths: list[Path] = []
    for fmt in args.formats:
        print(f"\n[setup] --- Format: {fmt} ---")

        if args.clone_from:
            src_config_path = _find_clone_source(args.clone_from, fmt)
            if src_config_path is None:
                print(
                    f"[setup] WARN: no config to clone for "
                    f"{args.clone_from}/{fmt}, generating fresh"
                )
            else:
                new_body_map = _compute_clone_mapping(args.clone_from, body_names, root_body, fmt)
                ik_dir = GMR_ROOT / "general_motion_retargeting" / "ik_configs"
                clone_dest = ik_dir / f"{fmt}_to_{args.robot}.json"
                if args.dry_run:
                    print(f"[setup] (dry_run — would clone config → {clone_dest})")
                else:
                    path = clone_ik_config(
                        src_config_path,
                        new_body_map,
                        args.robot,
                        fmt,
                        output_dir=ik_dir,
                    )
                    print(f"[setup] Cloned config → {path}")
                    config_paths.append(path)
                continue

        skeleton = get_skeleton(fmt)
        result = match_bodies(
            body_names,
            skeleton,
            root_body_hint=root_body,
            overrides=overrides,
        )

        print(f"[setup] Matched {len(result.mapping)}/{len(skeleton.role_to_joint)} roles")
        for role, body in sorted(result.mapping.items()):
            print(f"  {role:20s} → {body}")

        if result.unmatched_roles:
            print(f"[setup] Unmatched roles: {result.unmatched_roles}")
            if args.dry_run:
                print("[setup] (dry_run — skipping interactive resolution)")
            elif args.no_interactive:
                print("[setup] (--no-interactive — skipping unmatched roles)")
            elif args.interactive or sys.stdin.isatty():
                result = _interactive_resolve(result, skeleton, body_names)
            else:
                print(
                    "[setup] stdin is not a tty — skipping interactive resolution. "
                    "Use --interactive to force prompts or --no-interactive to suppress."
                )

        if len(result.mapping) < 4:
            print("[setup] ERROR: too few matches (< 4) after resolution.")

        config = generate_ik_config(result, skeleton, xml_path=xml_path, src_format=fmt)
        if args.world_rot:
            wr = parse_world_rotation_arg(args.world_rot)
            config["world_rotation"] = wr
        elif args.world_rot == "":
            pass  # auto-detection already ran via xml_path

        ik_dir = GMR_ROOT / "general_motion_retargeting" / "ik_configs"
        dest_path = ik_dir / f"{fmt}_to_{args.robot}.json"
        if args.dry_run:
            print(f"[setup] (dry_run — would write config → {dest_path})")
            _print_config_summary(config)
        else:
            path = write_ik_config(config, args.robot, fmt, output_dir=ik_dir)
            print(f"[setup] Generated config → {path}")
            config_paths.append(path)

    # ── Step 3: Auto-register in params.py ──
    if args.auto_register:
        print("\n[setup] Step 3: Registering in params.py...")
        diffs = register_in_params(
            GMR_ROOT,
            args.robot,
            xml_path.name,
            root_body,
            args.cam_distance,
            args.formats,
            dry_run=args.dry_run,
        )
        for d in diffs:
            print(d)
    elif not args.dry_run:
        _print_missing_params(params, args, xml_path, root_body)

    # ── Step 3b: Update script choices ──
    if args.update_scripts:
        print("\n[setup] Step 3b: Updating script choices...")
        results = update_script_choices(GMR_ROOT, args.robot, dry_run=args.dry_run)
        for r in results:
            print(r)

    if args.dry_run:
        print("\n[setup] Dry run complete — no files modified.")
        return

    # ── Step 4: Stage T-pose (requires tpose_motion) ──
    if args.tpose_motion and not args.skip_stage:
        print("\n[setup] Step 4: Staging T-pose...")
        spec_path = Path(args.output_dir) / f"{args.robot}.json"
        stage_cmd = [
            sys.executable,
            str(_HERE / "stage_tpose.py"),
            "--robot",
            args.robot,
            "--src",
            args.tpose_src,
            "--preset",
            args.tpose_preset,
            "--output_dir",
            args.output_dir,
        ]
        for jt in args.tpose_joint:
            stage_cmd.extend(["--joint", jt])
        result = subprocess.run(stage_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            print("[setup] ERROR: staging failed.")
            return
        print(result.stdout.strip().split("\n")[-3:])

    # ── Step 5: Solve quaternions ──
    spec_path = Path(args.output_dir) / f"{args.robot}.json"
    should_solve = args.tpose_motion or (smplx_template_available and spec_path.exists())
    if should_solve and not args.skip_solve:
        print("\n[setup] Step 5: Solving IK config quaternions...")

        if args.tpose_src == "smplx" or (args.src == "smplx" and smplx_template_available):
            solve_result = _solve_smplx_offsets(
                args.robot, spec_path, body_model_root=smplx_body_model_root
            )
        else:
            solve_result = _solve_via_agent(args, spec_path)

        if not solve_result:
            return

    # ── Step 6: Validate ──
    should_validate = should_solve and not args.skip_solve and not args.skip_validate
    if should_validate:
        params = load_gmr_params(GMR_ROOT)
        ik_dict = getattr(params, "IK_CONFIG_DICT", {})
        ik_config_src = ik_dict.get(args.tpose_src, {})
        if args.robot not in ik_config_src:
            print(
                f"\n[setup] WARNING: robot {args.robot!r} not found in "
                f"IK_CONFIG_DICT[{args.tpose_src!r}] after registration. "
                "Skipping validation — the GMR params module may need a "
                "manual reload or the ik_configs directory may be missing."
            )
        else:
            print("\n[setup] Step 6: Validating...")
            spec_path = Path(args.output_dir) / f"{args.robot}.json"
            vcmd = [
                sys.executable,
                str(_PROJECT / "examples" / "gmr_tpose_validate.py"),
                "--robot",
                args.robot,
                "--src",
                args.tpose_src,
                "--bvh_format",
                args.bvh_format,
                "--spec",
                str(spec_path),
                "--threshold",
                "5.0",
            ]
            if smplx_template_available and not args.tpose_motion:
                vcmd.append("--use_smplx_template")
                if smplx_body_model_root is not None:
                    vcmd.extend(["--smplx_template_model", str(smplx_body_model_root)])
            else:
                vcmd.extend(["--tpose_motion", args.tpose_motion])
            vr = subprocess.run(vcmd, capture_output=True, text=True)
            print(vr.stdout)
            if vr.returncode != 0:
                print("[setup] WARNING: validation threshold not met.")
            else:
                print("[setup] PASSED -- robot is aligned.")

    print("\n[setup] Done.")
    for p in config_paths:
        print(f"  Config: {p}")


def _find_clone_source(source_robot: str, fmt: str) -> Path | None:
    config_dir = GMR_ROOT / "general_motion_retargeting" / "ik_configs"
    path = config_dir / f"{fmt}_to_{source_robot}.json"
    return path if path.exists() else None


def _compute_clone_mapping(
    source_robot: str,
    new_body_names: list[str],
    new_root: str,
    fmt: str,
) -> dict[str, str]:
    source_path = _find_clone_source(source_robot, fmt)
    if source_path is None:
        return {}
    with source_path.open() as f:
        config = json.load(f)

    source_root = config.get("robot_root_name", "")
    old_bodies: set[str] = set()
    for table_key in ("ik_match_table1", "ik_match_table2"):
        old_bodies.update(config.get(table_key, {}).keys())

    skeleton = get_skeleton(fmt)
    old_match = match_bodies(sorted(old_bodies), skeleton, root_body_hint=source_root)
    new_match = match_bodies(new_body_names, skeleton, root_body_hint=new_root)

    mapping: dict[str, str] = {source_root: new_root}
    for role in old_match.mapping:
        if role in new_match.mapping:
            old_body = old_match.mapping[role]
            new_body = new_match.mapping[role]
            mapping[old_body] = new_body

    return mapping


def _print_missing_params(
    params: object, args: argparse.Namespace, xml_path: Path, root_body: str
) -> None:
    base_dict = getattr(params, "ROBOT_BASE_DICT", {})
    cam_dict = getattr(params, "VIEWER_CAM_DISTANCE_DICT", {})
    ik_dict = getattr(params, "IK_CONFIG_DICT", {})
    xml_dict = getattr(params, "ROBOT_XML_DICT", {})

    missing: list[str] = []
    if args.robot not in xml_dict:
        missing.append(f'    "{args.robot}": ASSET_ROOT / "{args.robot}" / "{xml_path.name}",')
    if args.robot not in base_dict:
        missing.append(f'    "{args.robot}": "{root_body}",')
    if args.robot not in cam_dict:
        missing.append(f'    "{args.robot}": {args.cam_distance},')
    for fmt in args.formats:
        if args.robot not in ik_dict.get(fmt, {}):
            entry = f'        "{args.robot}": IK_CONFIG_ROOT / "{fmt}_to_{args.robot}.json",'
            missing.append(entry)

    if missing:
        print("\n[setup] Missing GMR params.py entries (use --auto_register to add automatically):")
        for line in missing:
            print(f"  {line}")
    else:
        print("[setup] All GMR params.py entries present.")


def _print_config_summary(config: dict) -> None:
    """Print a concise summary of the generated config (used in dry-run mode)."""
    t1 = config.get("ik_match_table1", {})
    t2 = config.get("ik_match_table2", {})
    scale = config.get("human_scale_table", {})
    wr = config.get("world_rotation")
    print(f"  Bodies in table1: {len(t1)}, table2: {len(t2)}")
    print(f"  Scale entries: {len(scale)}")
    if wr:
        print(f"  world_rotation: [{wr[0]:.4f}, {wr[1]:.4f}, {wr[2]:.4f}, {wr[3]:.4f}]")
    else:
        print("  world_rotation: (auto or none)")


if __name__ == "__main__":
    main()
