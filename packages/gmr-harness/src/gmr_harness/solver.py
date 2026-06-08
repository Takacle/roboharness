"""Direct IK config solver — one-shot quaternion offset computation + VLM agent loop.

Extracted from ``examples/gmr_alignment_agent.py`` so it can be tested and
reused independently of the legacy script.

Architecture:
    - ``solve_direct``: one-shot quaternion solve from human bone orientations.
    - ``solve_smplx_template``: SMPL-X template calibration.
    - ``extract_init_qpos``: non-zero joint angles from T-pose spec.
    - ``run_agent``: full VLM-driven iteration loop (scale / weights / quaternion).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gmr_harness.alignment import apply_patch, compute_direct_patch


def _ensure_gmr(feature: str = ""):
    from gmr_harness.gmr_integration import _require_gmr

    _require_gmr(feature)


_SYSTEM_PROMPT = """\
You are a robotics motion-retargeting expert specializing in IK scale tuning.

Your task: inspect rendered images of a humanoid robot replaying retargeted motion, \
identify bone-length mismatches, and return a JSON patch adjusting scale values so \
the retargeted robot posture matches the target T-pose (green translucent overlay).

## Reading the images
Each rendered frame shows TWO overlaid figures:
  1. The SOLID ROBOT — the retargeted output using the current IK config.
  2. A TRANSLUCENT GREEN SKELETON — the TARGET T-pose.

Your primary diagnostic: compare solid robot link endpoints to green target endpoints.
  - If a robot arm/hand extends PAST the green target → scale too LARGE → reduce.
  - If a robot arm/hand falls SHORT → scale too SMALL → increase.
  - If they overlap closely → fine, do not touch.

## Response format
Return ONLY valid JSON. No prose. Choose one:

Case A — changes needed:
{
  "verdict": "needs_fix",
  "analysis": "one sentence",
  "patch": {
    "human_scale_table": {
      "LeftArm": {"mode": "set", "value": 0.82},
      "RightArm": {"mode": "mul", "value": 0.90}
    },
    "human_height_assumption": {"mode": "set", "value": 1.75}
  }
}

Case B — looks well-aligned:
{
  "verdict": "ok",
  "analysis": "robot pose matches target T-pose within tolerance"
}

Rules:
- Target the top-5 worst links per iteration.
- Keep fixes conservative: ±0.05-0.10 per iteration.
- Scale values must be between 0.2 and 2.5. human_height_assumption between 0.5 and 3.0.
"""

_WEIGHTS_PROMPT = """\
You are a robotics motion-retargeting expert specializing in IK weight tuning.

Adjust pos_weight and rot_weight in IK match tables to improve solver convergence.

## Response format
Return ONLY valid JSON. Choose one:

Case A — changes needed:
{
  "verdict": "needs_fix",
  "analysis": "one sentence",
  "patch": {
    "ik_match_table1": {
      "left_elbow": {
        "pos_weight": {"mode": "set", "value": 100},
        "rot_weight": {"mode": "mul", "value": 2.0}
      }
    },
    "ik_match_table2": {
      "left_elbow": {
        "pos_weight": {"mode": "set", "value": 100},
        "rot_weight": {"mode": "mul", "value": 2.0}
      }
    }
  }
}

Case B — converged:
{"verdict": "ok", "analysis": "all joints within acceptable alignment"}

Weight values must be between 0 and 1000. Keep both tables in sync.
"""

_QUATERNION_PROMPT = """\
You are a robotics motion-retargeting expert specializing in quaternion offset tuning.

Diagnose joint rotational misalignments and return a JSON patch.

## Common cardinal corrections (mul mode, scalar-first [w,x,y,z])
  90°  about X → [0.707, 0.707, 0, 0]    180° about X → [0, 1, 0, 0]
  90°  about Y → [0.707, 0, 0.707, 0]    180° about Y → [0, 0, 1, 0]
  90°  about Z → [0.707, 0, 0, 0.707]    180° about Z → [0, 0, 0, 1]

## Response format
Return ONLY valid JSON. Choose one:

Case A — changes needed:
{
  "verdict": "needs_fix",
  "analysis": "one sentence",
  "patch": {
    "ik_match_table1": {
      "left_shoulder": {"mode": "mul", "quat": [0.707, 0, 0.707, 0]}
    },
    "ik_match_table2": {
      "left_shoulder": {"mode": "mul", "quat": [0.707, 0, 0.707, 0]}
    }
  }
}

Case B — converged:
{"verdict": "ok", "analysis": "all joint rotations within acceptable tolerance"}

Keep both tables in sync. Be conservative: 1-3 joints per iteration.
"""

_MODEL_LIMITS: dict[str, dict] = {
    "glm-4v-flash": {"image_limit": 4, "max_tokens": 1024, "json_mode": True},
    "glm-4v-plus": {"image_limit": 5, "max_tokens": 1024, "json_mode": False},
    "glm-5v-turbo": {"image_limit": 8, "max_tokens": 4096, "json_mode": False},
}


def _get_model_limits(model: str) -> dict:
    for prefix, limits in _MODEL_LIMITS.items():
        if model.lower().startswith(prefix):
            return limits
    return {}


def solve_direct(
    src: str,
    robot: str,
    tpose_motion: str,
    tpose_spec: dict,
    config: dict,
    config_path: Path,
    bvh_format: str = "auto",
    preserve: set[str] | None = None,
    world_rot: list[float] | None = None,
) -> dict:
    _ensure_gmr("direct IK solve")
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    from gmr_harness.gmr_integration import load_motion, scaled_human_reference

    preserve = preserve or set()

    tpose_frames, _, _ = load_motion(src, tpose_motion, bvh_format)
    tpose_frame = tpose_frames[0]

    retargeter = GMR(
        src_human=src,
        tgt_robot=robot,
        actual_human_height=None,
        verbose=False,
    )
    human_ref = scaled_human_reference(retargeter, tpose_frame)

    if world_rot:
        from scipy.spatial.transform import Rotation as R

        r_wr = R.from_quat(world_rot, scalar_first=True)
        for name in human_ref:
            pos, quat = human_ref[name]
            new_pos = r_wr.apply(np.asarray(pos))
            new_quat = (r_wr * R.from_quat(np.asarray(quat), scalar_first=True)).as_quat(
                scalar_first=True
            )
            human_ref[name] = (list(new_pos), list(new_quat))

    direct_patch = compute_direct_patch(
        human_data=human_ref,
        config=config,
        tpose_spec=tpose_spec,
        preserve=preserve,
    )
    return apply_patch(config, direct_patch, mirror="auto")


def solve_smplx_template(
    robot: str,
    tpose_spec: dict,
    config_path: Path,
    body_model_path: Path | str | None = None,
) -> dict:
    from gmr_harness.alignment.smplx_offset_solver import solve_smplx_offsets_from_template

    spec_path = tpose_spec.get("_spec_path", "")
    return solve_smplx_offsets_from_template(
        ik_config_path=config_path,
        tpose_spec_path=Path(spec_path) if spec_path else None,
        body_model_path=body_model_path,
    )


def extract_init_qpos(tpose_spec: dict) -> dict[str, float]:
    xml_path = tpose_spec.get("_resolved_xml_path", tpose_spec.get("xml_path", ""))
    qpos_spec = tpose_spec.get("qpos", [])
    if not xml_path or len(qpos_spec) <= 7:
        return {}

    try:
        import mujoco as mj
    except ImportError:
        return {}

    model = mj.MjModel.from_xml_path(str(xml_path))
    init_qpos: dict[str, float] = {}
    for i in range(model.njnt):
        jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if jname is None or model.jnt_type[i] == 0:
            continue
        adr = model.jnt_qposadr[i]
        if adr >= len(qpos_spec):
            continue
        v = qpos_spec[adr]
        if abs(v) > 0.0001:
            init_qpos[jname] = round(float(v), 6)
    return init_qpos


def _retarget(
    src: str, motion_file: str, robot: str, bvh_format: str, max_frames: int | None
) -> tuple[np.ndarray, list[dict]]:
    from tqdm import tqdm

    from gmr_harness.gmr_integration import (
        check_smplx_config_before_retarget,
        load_motion,
        scaled_human_reference,
    )

    check_smplx_config_before_retarget(robot, src)
    frames, human_height, _ = load_motion(src, motion_file, bvh_format)

    _ensure_gmr("retarget motion")
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    if max_frames is not None:
        frames = frames[:max_frames]

    retargeter = GMR(
        src_human=src,
        tgt_robot=robot,
        actual_human_height=human_height,
        verbose=False,
    )

    cfg_path = _get_config_path(src, robot)
    if cfg_path is not None and cfg_path.exists():
        with cfg_path.open() as f:
            _patch_fallback_offsets(retargeter, json.load(f))

    qpos_list: list[np.ndarray] = []
    human_list: list[dict] = []
    for f in tqdm(frames, desc="retargeting", leave=False):
        qpos_list.append(retargeter.retarget(f).copy())
        human_list.append(scaled_human_reference(retargeter, f))
    return np.array(qpos_list), human_list


def _retarget_tpose_qpos(src: str, motion_file: str, robot: str, bvh_format: str) -> np.ndarray:
    _ensure_gmr("retarget tpose qpos")
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    from gmr_harness.gmr_integration import (
        check_smplx_config_before_retarget,
        load_motion,
    )

    check_smplx_config_before_retarget(robot, src)
    frames, human_height, _ = load_motion(src, motion_file, bvh_format)

    retargeter = GMR(
        src_human=src, tgt_robot=robot, actual_human_height=human_height, verbose=False
    )
    cfg_path = _get_config_path(src, robot)
    if cfg_path is not None and cfg_path.exists():
        with cfg_path.open() as f:
            _patch_fallback_offsets(retargeter, json.load(f))
    return retargeter.retarget(frames[0]).copy()


def _get_config_path(src: str, robot: str) -> Path | None:
    _ensure_gmr("GMR params")
    try:
        from general_motion_retargeting.params import IK_CONFIG_DICT

        p = IK_CONFIG_DICT.get(src, {}).get(robot)
        return Path(str(p)) if p else None
    except Exception:
        return None


def _patch_fallback_offsets(retargeter: Any, config: dict) -> None:
    from scipy.spatial.transform import Rotation as R

    fallback = config.get("_fallback_offsets", {})
    if not fallback:
        return
    for human_bone, q_off in fallback.items():
        r = R.from_quat(q_off, scalar_first=True)
        retargeter.rot_offsets1[human_bone] = r
        retargeter.rot_offsets2[human_bone] = r
        retargeter.pos_offsets1[human_bone] = np.zeros(3)
        retargeter.pos_offsets2[human_bone] = np.zeros(3)


def _format_deviation_report(
    report: dict, total: float, max_angle: float, prev_total: float | None, top_k: int = 5
) -> str:
    from gmr_harness.alignment import worst_k

    lines = ["## Rotation deviation report (supplementary)"]
    if prev_total is not None:
        delta = total - prev_total
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"total_deviation = {total:.2f}°   (prev {prev_total:.2f}°, Δ {sign}{delta:.2f}°)"
        )
    else:
        lines.append(f"total_deviation = {total:.2f}°   (no prior iteration)")
    lines.append(f"max_angle       = {max_angle:.2f}°")
    lines.append(f"worst {top_k}:")
    for name, angle in worst_k(report, top_k):
        axis = report[name]["axis"]
        axis_str = f"[{axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f}]"
        lines.append(f"  {name:40s} {angle:7.2f}°  axis={axis_str}")
    return "\n".join(lines)


def _format_position_report(
    pos_report: dict,
    pos_total: float,
    max_pos_err: float,
    top_k: int = 5,
) -> str:
    from gmr_harness.alignment import worst_k_position

    lines = ["## Position deviation report (primary — adjust human_scale_table to reduce these)"]
    lines.append(f"total_position_err = {pos_total:.3f} m  ({pos_total * 100:.1f} cm)")
    lines.append(f"max_position_err   = {max_pos_err:.3f} m  ({max_pos_err * 100:.1f} cm)")
    lines.append(f"worst {top_k}:")
    for name, err_m in worst_k_position(pos_report, top_k):
        direction = pos_report[name]["direction"]
        dir_str = f"[{direction[0]:+.2f}, {direction[1]:+.2f}, {direction[2]:+.2f}]"
        cm = err_m * 100
        lines.append(f"  {name:40s} {cm:6.1f} cm  from→target={dir_str}")
    return "\n".join(lines)


def _ask_vlm(
    client: Any,
    model: str,
    image_paths: list[Path],
    config: dict,
    iteration: int,
    deviation_text: str | None = None,
    pos_deviation_text: str | None = None,
    tune_mode: str = "scale",
) -> dict:
    from gmr_harness._utils import encode_image_base64, select_image_files

    model_limits = _get_model_limits(model)
    image_limit = model_limits.get("image_limit")
    if image_limit and len(image_paths) > image_limit:
        original_count = len(image_paths)
        image_paths = select_image_files(image_paths, image_limit)
        print(
            f"[agent] Reduced images from {original_count} to {len(image_paths)} "
            f"(model {model} image limit: {image_limit})"
        )

    default_max_tokens = 2048
    allowed_max_tokens = model_limits.get("max_tokens", default_max_tokens)
    effective_max_tokens = min(default_max_tokens, allowed_max_tokens)

    if tune_mode == "quaternion":
        system_prompt = _QUATERNION_PROMPT
    elif tune_mode == "weights":
        system_prompt = _WEIGHTS_PROMPT
    else:
        system_prompt = _SYSTEM_PROMPT

    if tune_mode == "quaternion":
        intro_text = (
            f"## Iteration {iteration}\n\n"
            f"Below are renders at the T-pose frame ({len(image_paths)} images). "
            "The translucent GREEN overlay is the TARGET T-pose. "
            "Focus on ROTATIONAL alignment. "
            "After the images I show the current quaternion offsets for deviated joints."
        )
        quat_config = {}
        for table_name in ("ik_match_table1", "ik_match_table2"):
            for joint, entry in (config.get(table_name) or {}).items():
                if joint not in quat_config:
                    quat_config[joint] = {
                        "human_bone": entry[0],
                        "quat_offset": entry[4] if len(entry) > 4 else [1, 0, 0, 0],
                    }
        config_text = (
            f"## Current quaternion offsets\n```json\n{json.dumps(quat_config, indent=2)}\n```"
        )
    elif tune_mode == "weights":
        intro_text = (
            f"## Iteration {iteration}\n\n"
            f"Below are renders ({len(image_paths)} images). "
            "After the images I show the current weight config."
        )
        weight_config = {
            joint: {
                "pos_weight": entry[1],
                "rot_weight": entry[2],
                "human_bone": entry[0],
                "quat_offset": entry[4] if len(entry) > 4 else [1, 0, 0, 0],
            }
            for table_name in ("ik_match_table1", "ik_match_table2")
            for joint, entry in (config.get(table_name) or {}).items()
        }
        config_text = f"## Current IK weights\n```json\n{json.dumps(weight_config, indent=2)}\n```"
    else:
        intro_text = (
            f"## Iteration {iteration}\n\n"
            f"Below are renders ({len(image_paths)} images). "
            "After the images I show the current scale config."
        )
        scale_config = {
            "human_scale_table": config.get("human_scale_table", {}),
            "human_height_assumption": config.get("human_height_assumption", 1.8),
        }
        config_text = f"## Current scale config\n```json\n{json.dumps(scale_config, indent=2)}\n```"

    intro_parts = [intro_text, ""]
    if tune_mode == "quaternion":
        if deviation_text:
            intro_parts.extend([deviation_text, ""])
        if pos_deviation_text:
            intro_parts.extend([pos_deviation_text, ""])
    else:
        if pos_deviation_text:
            intro_parts.extend([pos_deviation_text, ""])
        if deviation_text:
            intro_parts.extend([deviation_text, ""])
    intro_parts.append(
        "Compare the solid robot to the green target overlay. Return a JSON patch, "
        'or return {"verdict":"ok"} if well-aligned.'
    )
    user_text_intro = "\n".join(intro_parts)

    user_content: list[dict] = [{"type": "text", "text": user_text_intro}]
    for img_path in image_paths:
        b64 = encode_image_base64(img_path)
        user_content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
        )
    user_content.append({"type": "text", "text": config_text})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": effective_max_tokens,
        "messages": messages,
    }
    if model_limits.get("json_mode"):
        create_kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**create_kwargs)
    msg = response.choices[0].message
    content = msg.content
    if content is None:
        if hasattr(msg, "refusal") and msg.refusal:
            raise RuntimeError(
                f"VLM refused the request (model={model}): {msg.refusal}"
            )
        thought = getattr(msg, "reasoning_content", None) or ""
        if thought:
            content = thought
            print("[agent] WARNING: message.content was None, using reasoning_content as fallback")
        else:
            raise RuntimeError(
                f"VLM returned empty content (model={model}). "
                f"Finish reason: {response.choices[0].finish_reason!r}"
            )
    raw = content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = raw
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        for trim_char in (",", ":"):
            idx = repaired.rfind(trim_char)
            if idx != -1:
                after = repaired[idx + 1 :].strip()
                if after and after[-1] not in ']}"0123456789':
                    repaired = repaired[:idx]
                    break
        repaired += "]" * max(0, open_brackets) + "}" * max(0, open_braces)
        try:
            result = json.loads(repaired)
            print(f"[agent] Warning: repaired truncated JSON response from {model}")
            return result
        except json.JSONDecodeError:
            print(f"[agent] ERROR: could not parse model response. Raw text:\n{raw[:500]}")
            return {
                "verdict": "needs_fix",
                "analysis": "Model returned unparseable response",
                "patch": {},
            }


def _capture(
    qpos_seq: np.ndarray,
    xml_path: Path,
    root_body: str,
    cam_distance: float,
    output_dir: Path,
    trial: int,
    human_seq: list[dict] | None = None,
    tpose_spec: dict | None = None,
) -> list[Path]:
    from roboharness.core.harness import Harness

    from gmr_harness.replay import GMRReplayBackend

    cameras = ["inspect_front", "inspect_side", "inspect_back"]
    backend = GMRReplayBackend(
        xml_path=xml_path,
        qpos_seq=qpos_seq,
        cameras=cameras,
        root_body_name=root_body,
        cam_distance=cam_distance,
        use_meshcat=False,
        human_seq=human_seq,
        tpose_spec=tpose_spec,
    )
    harness = Harness(backend=backend, output_dir=output_dir, task_name="align")
    checkpoint_defs = [("frame_start", 0)]
    for name, frame_idx in checkpoint_defs:
        harness.add_checkpoint(name=name, cameras=cameras, trigger_step=frame_idx + 1)

    harness.reset()
    trial_dir = output_dir / "align" / f"trial_{harness._trial_count:03d}"

    png_paths: list[Path] = []
    for _name, frame_idx in checkpoint_defs:
        steps_needed = max(1, (frame_idx + 1) - harness.step_count + 10)
        result = harness.run_to_next_checkpoint([None] * steps_needed)
        if result is not None:
            for view in result.views:
                png = trial_dir / result.checkpoint_name / f"{view.name}_rgb.png"
                png_paths.append(png)

    backend.cleanup()
    return sorted(png_paths)


def run_agent(args: Any, remaining: list[str] | None = None) -> int:
    """Entry point for the agent CLI subcommand. Returns exit code."""

    _ensure_gmr("alignment agent")

    from general_motion_retargeting.params import (
        IK_CONFIG_DICT,
        ROBOT_XML_DICT,
        VIEWER_CAM_DISTANCE_DICT,
    )

    from gmr_harness.alignment import (
        apply_patch as _apply_patch,
    )
    from gmr_harness.alignment import (
        compute_deviations,
        compute_position_deviations,
        load_tpose_spec,
        optimize_scales,
        total_deviation,
        total_position_deviation,
        worst_k,
        worst_k_position,
    )
    from gmr_harness.gmr_integration import find_root_body

    api_base = getattr(args, "api_base", "https://open.bigmodel.cn/api/paas/v4")
    api_key = getattr(args, "api_key", "")
    solve_mode = getattr(args, "solve_mode", False)
    model = getattr(args, "model", "glm-5v-turbo")

    if api_base and not solve_mode:
        try:
            import httpx
            from openai import OpenAI
        except ModuleNotFoundError as e:
            print(f"[agent] ERROR: {e}. Install openai/httpx or use --solve_mode only.")
            return 1

        base_url = api_base.rstrip("/")
        client = OpenAI(
            api_key=api_key or "sk-placeholder",
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )
    elif not solve_mode:
        try:
            import anthropic
        except ModuleNotFoundError as e:
            print(f"[agent] ERROR: {e}. Install anthropic or use --solve_mode.")
            return 1

        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    else:
        client = None

    xml_path = Path(str(ROBOT_XML_DICT[args.robot]))
    cam_distance = float(VIEWER_CAM_DISTANCE_DICT.get(args.robot, 2.5))
    root_body = find_root_body(xml_path)

    try:
        config_path = Path(str(IK_CONFIG_DICT[args.src][args.robot]))
    except KeyError:
        config_path = _create_default_ik_config(args.robot, args.src, xml_path)

    output_dir = Path(args.output)

    print(f"\n{'=' * 60}")
    print("GMR Alignment Agent")
    print(f"{'=' * 60}")
    print(f"Robot      : {args.robot}")
    print(f"Motion     : {args.motion_file}")
    print(f"Source     : {args.src}")
    print(f"IK config  : {config_path}")
    print(f"Model      : {model}")
    print(f"Max iters  : {args.max_iter}")
    print(f"Dry run    : {args.dry_run}")
    print(f"{'=' * 60}\n")

    tpose_spec: dict | None = None
    tpose_spec_path = getattr(args, "tpose_spec", None)
    if tpose_spec_path and not isinstance(tpose_spec_path, Path):
        tpose_spec_path = Path(tpose_spec_path)
    if tpose_spec_path is None:
        default_path = Path("specs/tpose") / f"{args.robot}.json"
        if default_path.exists():
            tpose_spec_path = default_path

    tpose_motion = getattr(args, "tpose_motion", None)
    if tpose_spec_path and tpose_motion:
        tpose_spec = load_tpose_spec(tpose_spec_path)
        print(
            f"[agent] Numeric gate ON: spec={tpose_spec_path.name} "
            f"({len(tpose_spec['links'])} links)  tpose_motion={tpose_motion}"
        )
    elif tpose_spec_path and not tpose_motion:
        tpose_spec = load_tpose_spec(tpose_spec_path)
        if solve_mode and args.src == "smplx":
            print(
                f"[agent] Numeric gate (template): spec={tpose_spec_path.name} "
                f"({len(tpose_spec['links'])} links)  SMPL-X template calibration"
            )
        else:
            tpose_spec = None
            print(
                f"[agent] Numeric gate OFF: spec {tpose_spec_path.name} found but "
                "--tpose_motion not supplied. VLM will rely on vision only."
            )
    else:
        print("[agent] Numeric gate OFF: no T-pose spec / motion — VLM-only mode.")

    with config_path.open() as f:
        config = json.load(f)

    backup_path = config_path.with_suffix(".json.bak")
    if not args.dry_run and not backup_path.exists():
        backup_path.write_text(json.dumps(config, indent=4))
        print(f"[agent] Config backed up to {backup_path}")

    smplx_template_solve = solve_mode and args.src == "smplx" and tpose_spec is not None
    if smplx_template_solve:
        from gmr_harness.alignment.smplx_template import resolve_body_model_path

        try:
            _body_model_path = resolve_body_model_path(getattr(args, "smplx_template_model", None))
        except FileNotFoundError:
            _body_model_path = None
        smplx_template_solve = _body_model_path is not None
    else:
        _body_model_path = None

    tune_mode = getattr(args, "tune_mode", "scale")

    if smplx_template_solve:
        print("[agent] Phase A: skipped (SMPL-X template solve does not require motion)")
        print("[agent] Solve mode: using SMPL-X template calibration...")

        new_config = solve_smplx_template(
            robot=args.robot,
            tpose_spec=tpose_spec,
            config_path=config_path,
            body_model_path=_body_model_path,
        )
        n_solved = sum(1 for v in new_config.get("ik_match_table1", {}).values() if len(v) > 4)
        print(f"[agent] Solved quaternions for {n_solved} joints via template calibration")

        init_qpos = extract_init_qpos(tpose_spec)
        if init_qpos:
            new_config["init_qpos"] = init_qpos
            print(f"[agent] Set init_qpos: {list(init_qpos.keys())}")

        original_config_text = config_path.read_text()
        if args.dry_run:
            print("[agent] --dry_run active: config changes will be restored after validation.")
        else:
            with config_path.open("w") as f:
                json.dump(new_config, f, indent=4)
            print(f"[agent] Config written → {config_path}")

        print("[agent] Validating with template frame...")
        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        from gmr_harness.alignment.smplx_template import load_smplx_template_tpose

        try:
            if args.dry_run:
                with config_path.open("w") as f:
                    json.dump(new_config, f, indent=4)

            frame, _ = load_smplx_template_tpose(_body_model_path)
            retargeter_tmpl = GMR(
                src_human="smplx",
                tgt_robot=args.robot,
                actual_human_height=1.66,
                verbose=False,
            )
            tpose_qpos = retargeter_tmpl.retarget(frame).copy()
        finally:
            if args.dry_run:
                config_path.write_text(original_config_text)

        report_new = compute_deviations(tpose_qpos, tpose_spec["xml_path"], tpose_spec)
        total_new = total_deviation(report_new)
        top_new = worst_k(report_new, 5)
        max_new = top_new[0][1] if top_new else 0.0
        print(f"[agent] Post-solve: total={total_new:.2f}°  max={max_new:.2f}°")
        if max_new < getattr(args, "tpose_threshold", 5.0):
            print("[agent] OK — within threshold.")
        else:
            print(f"[agent] Residual max={max_new:.2f}° — may need manual tuning.")
        return 0

    if not smplx_template_solve:
        print("[agent] Phase A: initial retargeting...")
        qpos_seq, human_seq = _retarget(
            args.src, args.motion_file, args.robot, args.bvh_format, args.frames
        )
        print(f"[agent] qpos shape: {qpos_seq.shape}")

    if solve_mode:
        if tpose_spec is None or not tpose_motion:
            print("[agent] ERROR: --solve_mode requires --tpose_spec + --tpose_motion.")
            return 1

        print("[agent] Solve mode: computing direct IK config from human bone orientations...")

        preserve_str = getattr(args, "preserve", "")
        preserve = set(j.strip() for j in preserve_str.split(",") if j.strip())

        world_rot = None
        if getattr(args, "world_rot", ""):
            from gmr_harness.alignment.orientation_aligner import parse_world_rotation_arg

            try:
                world_rot = parse_world_rotation_arg(args.world_rot)
            except ValueError as exc:
                print(f"[agent] ERROR: {exc}")
                return 1

        new_config = solve_direct(
            src=args.src,
            robot=args.robot,
            tpose_motion=tpose_motion,
            tpose_spec=tpose_spec,
            config=config,
            config_path=config_path,
            bvh_format=getattr(args, "tpose_bvh_format", "auto"),
            preserve=preserve,
            world_rot=world_rot,
        )

        init_qpos = extract_init_qpos(tpose_spec)
        if init_qpos:
            new_config["init_qpos"] = init_qpos
            print(f"[agent] Set init_qpos: {list(init_qpos.keys())}")

        if getattr(args, "world_rot", ""):
            from gmr_harness.alignment.orientation_aligner import parse_world_rotation_arg

            try:
                wr_quat = parse_world_rotation_arg(args.world_rot)
            except ValueError as exc:
                print(f"[agent] ERROR: {exc}")
                return 1
            new_config["world_rotation"] = wr_quat
            print(f"[agent] Set world_rotation: {wr_quat}")

        original_config_text = config_path.read_text()
        if args.dry_run:
            print("[agent] --dry_run active: config changes will be restored after validation.")
        else:
            with config_path.open("w") as f:
                json.dump(new_config, f, indent=4)
            print(f"[agent] Config written → {config_path}")

        print("[agent] Retargeting T-pose with new config...")
        try:
            if args.dry_run:
                with config_path.open("w") as f:
                    json.dump(new_config, f, indent=4)
            tpose_qpos = _retarget_tpose_qpos(
                args.src, tpose_motion, args.robot, getattr(args, "tpose_bvh_format", "auto")
            )
        finally:
            if args.dry_run:
                config_path.write_text(original_config_text)
        report_new = compute_deviations(tpose_qpos, tpose_spec["xml_path"], tpose_spec)
        new_total = total_deviation(report_new)
        new_max = worst_k(report_new, 1)[0][1] if report_new else 0.0
        print(f"[agent] Post-solve: total={new_total:.2f}°  max={new_max:.2f}°")

        if new_max < getattr(args, "tpose_threshold", 5.0):
            print(f"[agent] PASSED (max < {getattr(args, 'tpose_threshold', 5.0)}°).")
        else:
            print(f"[agent] residual max={new_max:.2f}° — may need manual tuning.")

        print(f"\n[agent] Final config: {config_path}")
        return 0

    if tune_mode == "optimize_scale":
        if tpose_spec is None or not tpose_motion:
            print(
                "[agent] ERROR: --tune_mode optimize_scale requires --tpose_spec + --tpose_motion."
            )
            return 1

        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        from gmr_harness.gmr_integration import load_motion

        print("[agent] Pre-loading retargeter and MuJoCo model...")

        tpose_frames, actual_h, _ = load_motion(
            args.src, tpose_motion, getattr(args, "tpose_bvh_format", "auto")
        )
        tpose_frame = tpose_frames[0]
        scale_table = config.get("human_scale_table", {})
        bones = sorted(scale_table.keys())
        init_scales = [scale_table[b] for b in bones]
        init_height = config.get("human_height_assumption", 1.8)

        retargeter = GMR(
            src_human=args.src,
            tgt_robot=args.robot,
            actual_human_height=actual_h,
            verbose=False,
        )

        def _retarget_fn(new_scales: dict, height: float) -> np.ndarray:
            retargeter.human_scale_table = dict(new_scales)
            retargeter.actual_human_height = actual_h
            retargeter.human_height_assumption = height
            return retargeter.retarget(tpose_frame).copy()

        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(tpose_spec["xml_path"])
        mj_data = mujoco.MjData(mj_model)
        _spec_links: set[str] = set(tpose_spec["links"].keys())

        def _deviation_fn(qpos: np.ndarray) -> float:
            mj_data.qpos[:] = np.asarray(qpos, dtype=np.float64)
            mujoco.mj_forward(mj_model, mj_data)
            total = 0.0
            for link_name in _spec_links:
                body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
                if body_id < 0:
                    continue
                pos_actual = np.asarray(mj_data.xpos[body_id], dtype=np.float64)
                pos_expected = np.asarray(tpose_spec["links"][link_name]["pos"], dtype=np.float64)
                total += float(np.linalg.norm(pos_actual - pos_expected))
            return total

        print(f"[agent] Optimising {len(bones)} bones + height (max evals=80)...")
        patch = optimize_scales(
            bones=bones,
            init_scales=init_scales,
            init_height=init_height,
            retarget_fn=_retarget_fn,
            deviation_fn=_deviation_fn,
            optimize_height=True,
            max_evals=80,
            verbose=True,
        )

        new_config = _apply_patch(config, patch)
        with config_path.open("w") as f:
            json.dump(new_config, f, indent=4)
        print(f"[agent] Config updated → {config_path}")

        qpos = _retarget_fn(
            new_config.get("human_scale_table", scale_table),
            new_config.get("human_height_assumption", init_height),
        )
        pos_report = compute_position_deviations(qpos, str(tpose_spec["xml_path"]), tpose_spec)
        pos_total = total_position_deviation(pos_report)
        top_pos = worst_k_position(pos_report, 1)
        pos_max = top_pos[0][1] if top_pos else 0.0

        print(
            f"[agent] Post-optimise: pos_err={pos_total * 100:.1f} cm  max={pos_max * 100:.1f} cm"
        )
        for name, err in worst_k_position(pos_report, 5):
            print(f"  {name:40s} {err * 100:6.1f} cm")

        print(f"\n[agent] Final config: {config_path}")
        return 0

    # ── VLM iteration loop ──
    prev_total: float | None = None
    prev_pos_total: float | None = None
    regression_tolerance = getattr(args, "regression_tolerance", 0.0)
    tpose_threshold = getattr(args, "tpose_threshold", 5.0)
    tpose_bvh_format = getattr(args, "tpose_bvh_format", "auto")

    for iteration in range(1, args.max_iter + 1):
        print(f"\n--- Iteration {iteration}/{args.max_iter} ---")

        deviation_text: str | None = None
        pos_deviation_text: str | None = None
        cur_total: float | None = None
        cur_max: float | None = None
        cur_pos_total: float | None = None
        cur_pos_max: float | None = None

        if tpose_spec is not None and tpose_motion:
            print("[agent] Retargeting T-pose for numeric gate...")
            tpose_qpos = _retarget_tpose_qpos(
                args.src,
                tpose_motion,
                args.robot,
                tpose_bvh_format,
            )

            report = compute_deviations(tpose_qpos, tpose_spec["xml_path"], tpose_spec)
            cur_total = total_deviation(report)
            cur_max = worst_k(report, 1)[0][1] if report else 0.0
            deviation_text = _format_deviation_report(report, cur_total, cur_max, prev_total)

            pos_report = compute_position_deviations(tpose_qpos, tpose_spec["xml_path"], tpose_spec)
            cur_pos_total = total_position_deviation(pos_report)
            top_pos = worst_k_position(pos_report, 1)
            cur_pos_max = top_pos[0][1] if top_pos else 0.0
            pos_deviation_text = _format_position_report(pos_report, cur_pos_total, cur_pos_max)

            pos_cm = cur_pos_total * 100
            print(
                f"[agent] position: total={pos_cm:.1f} cm  max={cur_pos_max * 100:.1f} cm  "
                f"(max_angle={cur_max:.2f}°, threshold={tpose_threshold}°)"
            )
            if tune_mode == "quaternion":
                if cur_max < tpose_threshold:
                    print(
                        f"[agent] Numeric gate PASSED (max_angle < {tpose_threshold}°). Stopping."
                    )
                    break
            else:
                if cur_max < tpose_threshold and cur_pos_total < 0.05:
                    print(
                        f"[agent] Numeric gate PASSED "
                        f"(max_angle < {tpose_threshold}°, pos_err < 5 cm). Stopping."
                    )
                    break

        print("[agent] Capturing renders...")
        png_paths = _capture(
            qpos_seq=qpos_seq,
            xml_path=xml_path,
            root_body=root_body,
            cam_distance=cam_distance,
            output_dir=output_dir,
            trial=iteration,
            human_seq=human_seq,
            tpose_spec=tpose_spec,
        )
        print(f"[agent] {len(png_paths)} images captured")
        for p in png_paths:
            print(f"        {p}")

        print(f"[agent] Querying {model} (tune_mode={tune_mode})...")
        response = _ask_vlm(
            client=client,
            model=model,
            image_paths=png_paths,
            config=config,
            iteration=iteration,
            deviation_text=deviation_text,
            pos_deviation_text=pos_deviation_text,
            tune_mode=tune_mode,
        )

        verdict = response.get("verdict", "needs_fix")
        analysis = response.get("analysis", "")
        print(f"[agent] Verdict  : {verdict}")
        print(f"[agent] Analysis : {analysis}")

        if verdict == "ok":
            print("\n[agent] Motion looks aligned. Stopping.")
            break

        patch = response.get("patch", {})
        if not patch:
            print("[agent] No patch provided — stopping.")
            break

        print("[agent] Patch:")
        print(json.dumps(patch, indent=2))

        if args.dry_run:
            print("[agent] --dry_run active: config NOT modified.")
            break

        pre_patch_config = copy.deepcopy(config)
        new_config = _apply_patch(config, patch, mirror="auto")
        with config_path.open("w") as f:
            json.dump(new_config, f, indent=4)
        print(f"[agent] Config updated → {config_path}")

        print("[agent] Re-retargeting with updated config...")
        qpos_seq, human_seq = _retarget(
            args.src, args.motion_file, args.robot, args.bvh_format, args.frames
        )

        if tpose_spec is not None and tpose_motion and regression_tolerance >= 0.0:
            tpose_qpos_post = _retarget_tpose_qpos(
                args.src, tpose_motion, args.robot, tpose_bvh_format
            )
            report_post = compute_deviations(tpose_qpos_post, tpose_spec["xml_path"], tpose_spec)
            pos_report_post = compute_position_deviations(
                tpose_qpos_post, tpose_spec["xml_path"], tpose_spec
            )
            new_total = total_deviation(report_post)
            new_pos_total = total_position_deviation(pos_report_post)

            if tune_mode == "quaternion":
                cur_val = cur_total if cur_total is not None else 0.0
                delta = new_total - cur_val
                if delta > regression_tolerance:
                    print(
                        f"[agent] REGRESSION: rot {cur_val:.2f}° → {new_total:.2f}° "
                        f"(Δ {delta:+.2f}°, tol {regression_tolerance:.2f}°). Reverting."
                    )
                    with config_path.open("w") as f:
                        json.dump(pre_patch_config, f, indent=4)
                    config = pre_patch_config
                    qpos_seq, human_seq = _retarget(
                        args.src, args.motion_file, args.robot, args.bvh_format, args.frames
                    )
                    prev_total = cur_val
                else:
                    print(f"[agent] Regression gate PASS: rot {cur_val:.2f}° → {new_total:.2f}°")
                    config = new_config
                    prev_total = new_total
            elif cur_pos_total is not None:
                delta = new_pos_total - cur_pos_total
                tol_m = regression_tolerance / 100.0
                if delta > tol_m:
                    print(
                        f"[agent] REGRESSION: pos {cur_pos_total * 100:.1f} → "
                        f"{new_pos_total * 100:.1f} cm (Δ {delta * 100:+.1f} cm). Reverting."
                    )
                    with config_path.open("w") as f:
                        json.dump(pre_patch_config, f, indent=4)
                    config = pre_patch_config
                    qpos_seq, human_seq = _retarget(
                        args.src, args.motion_file, args.robot, args.bvh_format, args.frames
                    )
                    prev_pos_total = cur_pos_total
                else:
                    print(
                        f"[agent] Regression gate PASS: pos "
                        f"{cur_pos_total * 100:.1f} → {new_pos_total * 100:.1f} cm"
                    )
                    config = new_config
                    prev_pos_total = new_pos_total
            else:
                config = new_config
        else:
            config = new_config

        if cur_total is not None:
            prev_total = cur_total
        if cur_pos_total is not None and prev_pos_total is None:
            prev_pos_total = cur_pos_total
    else:
        print(
            f"\n[agent] Reached max iterations ({args.max_iter}). Review output and tune manually."
        )

    print(f"\n[agent] Final config: {config_path}")
    print(f"[agent] Captures   : {output_dir}/")
    print(f"[agent] Backup     : {backup_path}")
    print(f"\nTo restore original config:\n  cp {backup_path} {config_path}")
    return 0


def _create_default_ik_config(robot: str, src: str, xml_path: Path) -> Path:
    import re as _re

    from gmr_harness.alignment.body_matcher import match_bodies
    from gmr_harness.alignment.config_gen import generate_ik_config, write_ik_config
    from gmr_harness.alignment.skeleton_maps import get_skeleton
    from gmr_harness.gmr_integration import find_root_body

    skeleton = get_skeleton(src)
    body_names = sorted(set(_re.findall(r'<body\s+name="([^"]+)"', xml_path.read_text())))
    root_body = find_root_body(xml_path)

    match = match_bodies(body_names, skeleton, root_body_hint=root_body)
    config = generate_ik_config(match, skeleton, xml_path=xml_path, src_format=src)
    out_path = write_ik_config(config, robot, src)

    print(f"[agent] Created default IK config: {out_path}")

    try:
        from general_motion_retargeting.params import IK_CONFIG_DICT

        if src not in IK_CONFIG_DICT:
            IK_CONFIG_DICT[src] = {}
        IK_CONFIG_DICT[src][robot] = str(out_path)
    except Exception:
        pass

    return out_path
