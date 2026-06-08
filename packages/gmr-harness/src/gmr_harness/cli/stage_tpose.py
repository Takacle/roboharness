"""Stage a robot at T-pose and dump its alignment spec.

Authors a ``specs/tpose/{robot}.json`` — the numeric contract every downstream
alignment metric measures against — plus three reference PNGs (front/side/back)
that agents and humans use as the visual ground truth for "what correct looks
like".

Pose is constructed **from the command line**, not from GUI drag. MuJoCo's
passive viewer is view-only (no joint sliders), and GUI staging is not
reproducible across agents anyway. The flow is:

    # 1. Build the pose from a baseline (qpos0 from the XML) + named tweaks.
    #    Baseline 'home' = model.qpos0 (standing upright, arms at sides).
    #    Baseline 'tpose' = qpos0 + shoulder_roll ±π/2 (arms out to sides).
    gmr-harness stage --robot unitree_g1 --preset tpose \\
        --output_dir specs/tpose/

    # 2. Override individual joints by name if the preset needs tweaking.
    gmr-harness stage --robot unitree_g1 --preset tpose \\
        --joint left_wrist_roll_joint=0.1 \\
        --joint right_wrist_roll_joint=-0.1 \\
        --output_dir specs/tpose/

    # 3. Optional: open viewer on the constructed pose for visual confirmation.
    #    Uses mujoco.viewer.launch (managed), gravity disabled, robot frozen.
    #    Close the window to continue and write artifacts.
    gmr-harness stage --robot unitree_g1 --preset tpose \\
        --preview --output_dir specs/tpose/

    # 4. Reuse an existing spec's qpos verbatim (round-trip).
    gmr-harness stage --robot unitree_g1 \\
        --qpos_file specs/tpose/unitree_g1.json --output_dir specs/tpose/

List joint names for a robot with ``--list_joints``.

Requires ``pip install -e ".[demo]"`` (mujoco). GMR is auto-located as a
sibling directory to fetch ``ROBOT_XML_DICT`` and ``IK_CONFIG_DICT``; pass
``--xml`` + ``--link_names`` explicitly to use a robot GMR doesn't know about.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent

from gmr_harness.alignment._gmr_params import load_gmr_params  # noqa: E402
from gmr_harness.alignment._gmr_path import find_gmr_root  # noqa: E402
from gmr_harness.alignment.orientation_aligner import extract_xml_body_names  # noqa: E402

_gmr_root_cache: Path | None = None


def _get_gmr_root() -> Path:
    global _gmr_root_cache
    if _gmr_root_cache is None:
        _gmr_root_cache = find_gmr_root()
    return _gmr_root_cache


def _resolve_robot(robot: str, src: str) -> tuple[Path, list[str], float]:
    """Return (xml_path, controlled_link_names, viewer_cam_distance) from GMR."""
    params = load_gmr_params(_get_gmr_root())
    xml_path = Path(str(params.ROBOT_XML_DICT[robot]))
    cam_distance = float(params.VIEWER_CAM_DISTANCE_DICT.get(robot, 2.5))

    # IK config provides the canonical link list. When missing (new robot),
    # extract body names directly from the XML.
    try:
        ik_config_path = Path(str(params.IK_CONFIG_DICT[src][robot]))
    except KeyError:
        print(f"[stage_tpose] No IK config for {robot}/{src}; extracting body names from XML.")
        link_names = extract_xml_body_names(xml_path)
        return xml_path, link_names, cam_distance

    if not ik_config_path.exists():
        fallback_path = (
            _get_gmr_root()
            / "general_motion_retargeting"
            / "ik_configs"
            / (f"{src}_to_{robot}.json")
        )
        if fallback_path.exists():
            print(
                f"[stage_tpose] WARN: registered IK config missing: {ik_config_path}. "
                f"Using generated config: {fallback_path}"
            )
            ik_config_path = fallback_path
        else:
            print(
                f"[stage_tpose] WARN: IK config not found at {ik_config_path}; "
                "extracting body names from XML."
            )
            link_names = extract_xml_body_names(xml_path)
            return xml_path, link_names, cam_distance

    with ik_config_path.open() as f:
        ik_config = json.load(f)

    link_names = sorted(
        set(ik_config.get("ik_match_table1", {}).keys())
        | set(ik_config.get("ik_match_table2", {}).keys())
    )
    return xml_path, link_names, cam_distance


# T-pose auto-detection: instead of per-robot named presets, scan hinge joints
# by their physical axis to identify shoulder-roll and elbow joints.
_SHOULDER_ROLL_AXIS = (1.0, 0.0, 0.0)  # X-axis → abduction in T-pose
_ELBOW_AXIS = (0.0, 1.0, 0.0)  # Y-axis → flexion in T-pose
_TPOSE_SHOULDER_RADS = 1.5708  # ±π/2
_TPOSE_ELBOW_RADS = 1.5708  # π/2


def _axis_close(axis: tuple[float, ...], target: tuple[float, ...], tol: float = 0.1) -> bool:
    return all(abs(a - t) < tol for a, t in zip(axis, target, strict=False))


def _detect_side(jname_lower: str) -> float:
    if "left" in jname_lower or jname_lower.endswith("_l") or "_l_" in jname_lower:
        return 1.0
    if "right" in jname_lower or jname_lower.endswith("_r") or "_r_" in jname_lower:
        return -1.0
    return 0.0


def _detect_tpose_overrides(model: Any, robot_name: str) -> dict[str, float]:
    """Auto-detect T-pose joint overrides from axis semantics.

    Scans all hinge joints in the MuJoCo model, identifies shoulder-roll
    and elbow joints by their axis convention (not by name pattern), and
    returns {joint_name: rad_value} overrides to raise arms to T-pose.

    Works for any humanoid robot regardless of naming convention.
    Side detection recognises ``left``/``right`` as well as ``_L``/``_R``
    suffixes.  Axis matching uses a tolerance to handle joints with
    slightly tilted axes (e.g. ``axis="0 0.998 0.063"`` ≈ Y-axis).
    """
    import mujoco

    overrides: dict[str, float] = {}
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        jtype = int(model.jnt_type[jid])
        if jtype != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        axis = tuple(float(model.jnt_axis[jid][i]) for i in range(3))
        jname_lower = jname.lower()
        side = _detect_side(jname_lower)

        if (
            _axis_close(axis, _SHOULDER_ROLL_AXIS)
            and "shoulder" in jname_lower
            and "roll" in jname_lower
        ):
            if side != 0.0:
                overrides[jname] = _TPOSE_SHOULDER_RADS * side
        elif _axis_close(axis, _ELBOW_AXIS) and "elbow" in jname_lower and "pitch" in jname_lower:
            overrides[jname] = _TPOSE_ELBOW_RADS

    if not overrides:
        raise ValueError(
            f"No shoulder-roll or elbow joints auto-detected for robot {robot_name!r}. "
            "Verify the XML has hinge joints with axis=(1,0,0) for shoulder roll "
            "and axis=(0,1,0) for elbow. Use --joint flags to specify manually."
        )
    return overrides


def _apply_joint_overrides(
    model: Any,  # mujoco.MjModel
    qpos: np.ndarray,
    overrides: dict[str, float],
    source: str,
) -> np.ndarray:
    """Set named joint qpos entries by looking up qposadr from the model."""
    import mujoco

    out = qpos.copy()
    for joint_name, value in overrides.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(
                f"{source}: joint {joint_name!r} not found in model. "
                "Use --list_joints to see available names."
            )
        jtype = int(model.jnt_type[jid])
        # mjJNT_HINGE=3, mjJNT_SLIDE=2 occupy a single qpos slot; free/ball are
        # multi-DOF and don't fit this scalar-override interface.
        if jtype not in (2, 3):
            raise ValueError(
                f"{source}: joint {joint_name!r} is type {jtype} "
                "(free/ball); use --qpos_file for multi-DOF joints."
            )
        adr = int(model.jnt_qposadr[jid])
        out[adr] = float(value)
    return out


def _parse_joint_flag(spec: str) -> tuple[str, float]:
    """Parse NAME=VALUE. Raises ValueError for malformed input."""
    if "=" not in spec:
        raise ValueError(f"--joint expects NAME=VALUE, got {spec!r}")
    name, _, value_str = spec.partition("=")
    name = name.strip()
    if not name:
        raise ValueError(f"--joint has empty name: {spec!r}")
    try:
        value = float(value_str)
    except ValueError as exc:
        raise ValueError(f"--joint {name!r}: bad float {value_str!r}") from exc
    return name, value


def _build_qpos(
    model: Any,
    preset: str,
    joint_flags: list[str],
    qpos_file: Path | None,
    qpos_raw: str | None,
    robot_name: str,
) -> np.ndarray:
    """Construct the final qpos from baseline + preset + per-joint overrides.

    Priority (each later layer wins): qpos0 → preset → --joint overrides.
    Alternatively, --qpos_file or --qpos short-circuit to a full vector.
    """
    import mujoco

    if qpos_file is not None:
        with qpos_file.open() as f:
            data = json.load(f)
        qpos = np.asarray(data["qpos"], dtype=np.float64)
        if qpos.shape != (model.nq,):
            raise ValueError(
                f"qpos in {qpos_file} has shape {qpos.shape}, model expects ({model.nq},)"
            )
        return qpos
    if qpos_raw is not None:
        qpos = np.fromstring(qpos_raw, dtype=np.float64, sep=" ")
        if qpos.shape != (model.nq,):
            raise ValueError(f"--qpos had {qpos.shape[0]} values, model expects {model.nq}")
        return qpos

    # Start from the XML's default standing pose.
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qpos = np.asarray(data.qpos).copy()

    if preset == "tpose":
        try:
            overrides = _detect_tpose_overrides(model, robot_name)
        except ValueError:
            overrides = {}
            if not joint_flags:
                raise
        qpos = _apply_joint_overrides(model, qpos, overrides, source=f"preset={preset}")
    elif preset != "home":
        raise ValueError(f"Unknown preset {preset!r}; choose from: home, tpose")

    if joint_flags:
        cli_overrides = dict(_parse_joint_flag(s) for s in joint_flags)
        qpos = _apply_joint_overrides(model, qpos, cli_overrides, source="--joint")

    return qpos


def _snapshot_spec(
    robot: str,
    xml_path: Path,
    qpos: np.ndarray,
    link_names: list[str],
) -> dict:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    links: dict[str, dict] = {}
    for name in link_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            print(f"[stage_tpose] WARN: link {name!r} not in XML — skipped")
            continue
        links[name] = {
            "pos": np.asarray(data.xpos[bid]).tolist(),
            "R": np.asarray(data.xmat[bid]).reshape(3, 3).tolist(),
        }

    return {
        "robot": robot,
        "xml_path": str(xml_path),
        "qpos": qpos.tolist(),
        "links": links,
    }


def _guard_qpos_input_not_output(qpos_file: Path | None, spec_path: Path) -> None:
    """Avoid overwriting the user-authored qpos source spec in place."""
    if qpos_file is None:
        return
    try:
        same_path = qpos_file.resolve() == spec_path.resolve()
    except FileNotFoundError:
        same_path = qpos_file.absolute() == spec_path.absolute()
    if same_path:
        raise ValueError(
            "--qpos_file must not be the same path as the output spec. "
            f"Input and output both resolve to {spec_path}. "
            "Use a separate qpos source file or a different --output_dir."
        )


def _render_reference_views(
    xml_path: Path,
    qpos: np.ndarray,
    cam_distance: float,
    out_dir: Path,
    stem: str,
    width: int = 640,
    height: int = 480,
) -> list[Path]:
    """Render front/side/back offscreen PNGs aimed at the robot root."""
    import mujoco
    from PIL import Image

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    lookat = np.asarray(data.xpos[1])  # first non-world body center

    views = {
        "front": (0.0, -10.0),  # azimuth=0, elevation=-10° (facing +X looking toward +Y)
        "side": (90.0, -10.0),
        "back": (180.0, -10.0),
    }
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = cam_distance

    out_paths: list[Path] = []
    for name, (az, el) in views.items():
        cam.azimuth = az
        cam.elevation = el
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        path = out_dir / f"{stem}_{name}.png"
        Image.fromarray(img).save(path)
        out_paths.append(path)
        print(f"[stage_tpose] wrote {path}")
    renderer.close()
    return out_paths


def _preview_in_viewer(xml_path: Path, qpos: np.ndarray) -> np.ndarray:
    """Open a MuJoCo viewer on the constructed qpos for visual confirmation.

    Uses the passive viewer with its own sync loop; gravity is zeroed so the
    robot does not fall and ``mj_forward`` (not ``mj_step``) keeps the pose
    fixed at exactly ``qpos``. The user can orbit the camera and close the
    window when done. This is purely a visual check — joint edits in the
    viewer would require the managed ``launch()`` entry point and its
    physics loop, which defeats the point of a reproducible CLI-built pose.

    Returns ``qpos`` unchanged; keeping the signature ``-> ndarray`` so
    future implementations could allow mutation (e.g. a managed viewer
    behind a flag) without changing callers.
    """
    import time

    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    # Freeze physics — we want the user to see exactly what gets written,
    # not a falling rag-doll.
    model.opt.gravity[:] = 0.0
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_GRAVITY

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    print(
        "\n[stage_tpose] Preview viewer opened. This shows the constructed pose.\n"
        "             Orbit camera with mouse. Close the window to write\n"
        "             artifacts. Joint sliders are NOT available here —\n"
        "             adjust the pose via --joint NAME=VALUE on the CLI.\n"
    )
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Keep qpos pinned; do not step physics.
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)

    return qpos


def _list_joints(xml_path: Path) -> None:
    """Print joint table for a robot, formatted for CLI use."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    jtype_names = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
    print(f"{'idx':>3} {'type':6s} {'range':<24} name")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = jtype_names.get(int(model.jnt_type[i]), "?")
        lo, hi = model.jnt_range[i]
        rng = f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"{i:>3} {jtype:6s} {rng:<24} {name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Stage a robot at T-pose and dump its alignment spec + reference renders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--robot", help="Robot key in GMR ROBOT_XML_DICT (e.g. unitree_g1)")
    parser.add_argument(
        "--src",
        default="bvh",
        choices=["bvh", "smplx", "fbx_offline"],
        help="IK config source — controls which links are in the spec",
    )
    parser.add_argument(
        "--xml",
        help="Override XML path (use instead of --robot for unknown robots)",
    )
    parser.add_argument(
        "--link_names",
        nargs="+",
        help="Explicit link list (use with --xml when not using --robot)",
    )
    parser.add_argument(
        "--preset",
        default="tpose",
        choices=["home", "tpose"],
        help="Baseline pose: 'home' = XML qpos0 (default standing); "
        "'tpose' = qpos0 + shoulder_roll ±π/2 (arms horizontal).",
    )
    parser.add_argument(
        "--joint",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Override a joint's qpos value. Repeatable. Applied after --preset.",
    )
    parser.add_argument(
        "--qpos",
        help="Whitespace-separated full qpos vector (bypasses preset + --joint)",
    )
    parser.add_argument(
        "--qpos_file",
        type=Path,
        help="Path to an existing spec JSON; reuses its qpos field verbatim",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Open MuJoCo viewer on the constructed pose (read-only); close to continue.",
    )
    parser.add_argument(
        "--list_joints",
        action="store_true",
        help="Print the model's joint table and exit.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("specs/tpose"),
        help="Where to write {robot}.json and {robot}_{view}.png",
    )
    parser.add_argument(
        "--cam_distance",
        type=float,
        help="Override camera distance for reference renders",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Skip producing reference PNGs",
    )
    args = parser.parse_args(argv)

    # Resolve XML + link set
    if args.list_joints:
        # list_joints only needs the XML, not IK config
        if args.xml:
            xml_path = Path(args.xml)
        else:
            if not args.robot:
                parser.error("either --robot or --xml is required")
            params = load_gmr_params(_get_gmr_root())
            xml_path = Path(str(params.ROBOT_XML_DICT[args.robot]))
        _list_joints(xml_path)
        return

    if args.xml:
        if not args.link_names:
            parser.error("--xml requires --link_names")
        xml_path = Path(args.xml)
        link_names = list(args.link_names)
        cam_distance = args.cam_distance or 2.5
        robot_name = args.robot or xml_path.stem
    else:
        if not args.robot:
            parser.error("either --robot or (--xml + --link_names) is required")
        xml_path, link_names, cam_distance = _resolve_robot(args.robot, args.src)
        robot_name = args.robot
        if args.cam_distance is not None:
            cam_distance = args.cam_distance

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    qpos = _build_qpos(
        model=model,
        preset=args.preset,
        joint_flags=args.joint,
        qpos_file=args.qpos_file,
        qpos_raw=args.qpos,
        robot_name=robot_name,
    )

    if args.preview:
        qpos = _preview_in_viewer(xml_path, qpos)

    # Snapshot + write
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = args.output_dir / f"{robot_name}.json"
    _guard_qpos_input_not_output(args.qpos_file, spec_path)
    spec = _snapshot_spec(robot_name, xml_path, qpos, link_names)
    with spec_path.open("w") as f:
        json.dump(spec, f, indent=2)
    print(f"[stage_tpose] wrote spec {spec_path} ({len(spec['links'])} links)")

    if not args.no_render:
        _render_reference_views(
            xml_path=xml_path,
            qpos=qpos,
            cam_distance=cam_distance,
            out_dir=args.output_dir,
            stem=robot_name,
        )

    print("\n[stage_tpose] Done. Next: commit the spec and PNGs, review visually,")
    print("             then run alignment metrics against any candidate qpos with")
    print("             gmr_harness.alignment.compute_deviations(...) or")
    print("             gmr-harness validate --robot <name> ...")


if __name__ == "__main__":
    main()
