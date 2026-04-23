"""GMR Alignment Agent — AI-driven automatic IK config optimization.

Uses Claude's vision capability to inspect retargeted robot renders, diagnose
joint misalignments, and iteratively update the IK config quaternion offsets
until the motion looks correct.

Architecture:
    loop (max --max_iter rounds):
        1. Run GMR retargeting with current config → qpos sequence
        2. Replay via roboharness → capture front/side/back at 4 key frames
        3. Send up to N images to VLM with the current config
           (N depends on the model's per-request image limit)
        4. VLM returns a JSON patch: {table: joint: [w,x,y,z]} or "ok"
        5. Apply patch to config, write back, continue
        6. Stop when VLM says "ok" or max iterations reached

Usage:
    python examples/gmr_alignment_agent.py \\
        --robot unitree_g1 \\
        --motion_file /path/to/motion.bvh \\
        [--src bvh|smplx|fbx_offline] \\
        [--bvh_format auto] \\
        [--frames 60] \\
        [--max_iter 8] \\
        [--dry_run]       # inspect only, do not modify config
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (same as gmr_alignment_inspector.py)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
GMR_ROOT = _HERE.parent.parent / "GMR"
if not GMR_ROOT.exists():
    raise RuntimeError(f"GMR not found at {GMR_ROOT}")

sys.path.insert(0, str(GMR_ROOT))
_ROBOHARNESS_SRC = _HERE.parent / "src"
if _ROBOHARNESS_SRC.exists() and str(_ROBOHARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(_ROBOHARNESS_SRC))

# Re-use all helpers from the inspector script
from gmr_alignment_inspector import (  # noqa: E402
    GMRReplayBackend,
    _find_root_body,
    _load_bvh,
    _load_fbx_offline,
    _load_smplx,
)

# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def _quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    """Multiply two [w,x,y,z] quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _quat_normalize(q: list[float]) -> list[float]:
    norm = sum(v * v for v in q) ** 0.5
    return [v / norm for v in q] if norm > 1e-9 else [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Config patch application
# ---------------------------------------------------------------------------


def apply_patch(config: dict, patch: dict) -> dict:
    """Apply a Claude-generated patch to the IK config.

    Patch format (all fields optional):
    {
        "ik_match_table1": {
            "left_shoulder_yaw_link": [w, x, y, z],   // replaces quat offset
            ...
        },
        "ik_match_table2": { ... },                    // optional
        "world_rotation": [w, x, y, z]                // optional top-level
    }

    Quaternions in patches may be:
      - Absolute:  {"mode": "set", "quat": [w,x,y,z]}
      - Relative:  {"mode": "mul", "quat": [w,x,y,z]}  (applied as q_new = patch * q_old)
      - Shorthand: [w, x, y, z]  (treated as "set")
    """
    config = copy.deepcopy(config)

    for table_name in ("ik_match_table1", "ik_match_table2"):
        if table_name not in patch:
            continue
        table = config.get(table_name, {})
        for joint, quat_spec in patch[table_name].items():
            if joint not in table:
                continue
            entry = table[joint]
            if isinstance(quat_spec, dict):
                mode = quat_spec.get("mode", "set")
                quat = quat_spec["quat"]
            else:
                mode = "set"
                quat = quat_spec

            quat = _quat_normalize(quat)
            if mode == "mul":
                old_quat = entry[4]
                quat = _quat_normalize(_quat_multiply(quat, old_quat))

            entry[4] = quat

    if "world_rotation" in patch:
        config["world_rotation"] = _quat_normalize(patch["world_rotation"])

    return config


# ---------------------------------------------------------------------------
# Claude Vision call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robotics motion-retargeting expert specializing in MuJoCo IK config tuning.

Your task: inspect rendered images of a humanoid robot replaying retargeted motion, \
identify joint axis misalignments, and return a JSON patch to fix them.

## Reading the images
Each rendered frame shows TWO overlaid figures:
  1. The SOLID ROBOT — the retargeted output using the current IK config.
  2. A TRANSLUCENT CYAN SKELETON — the anatomical human reference, scaled to
     the robot's body proportions. This is what the robot SHOULD look like if
     the config quaternions were correct.

Your primary diagnostic: compare the solid robot to the cyan skeleton.
  - If a robot limb points in a different direction than the matching cyan bone,
    that joint's quaternion offset is wrong — patch it.
  - If they overlap closely, that joint is fine — do not touch it.
  - If the whole robot is rotated relative to the cyan skeleton, fix world_rotation.

## Config structure
The IK config has two tables (ik_match_table1, ik_match_table2). Each entry:
  "robot_joint_name": ["human_bone", pos_w, rot_w, [px,py,pz], [w,x,y,z]]
The quaternion [w,x,y,z] (scalar-first) is the rotation offset that maps human \
joint orientation → robot joint orientation. Adjusting this is the primary tuning lever.

## Numeric deviation report (when provided)
If the user message contains a "T-pose deviation report" section, it is the
GROUND TRUTH for which joints are misaligned. Each row is:
    link_name   angle_deg   axis=[x, y, z]
where ``angle_deg`` is the rotation distance from the committed T-pose spec,
and ``axis`` is the world-frame rotation that maps the current pose onto
the target. TRUST THE NUMBERS — patch the worst links first, and stop
when ``max(angle_deg) < 5``.

Interpreting ``angle_deg`` (per the alignment SOP):
    < 1°     : excellent, leave alone
    1-5°     : tolerable, chasing it risks regressions
    5-30°    : likely IK solver tolerance
    30-60°   : axis confusion
    60-120°  : cardinal rotation missing (snap to the nearest 90° along axis)
    > 120°   : 180° flip (sign error on quaternion offset)

Cardinal correction quaternions (apply as "mode": "mul"):
  90°  about axis a  = [cos(45°), a_x·sin(45°), a_y·sin(45°), a_z·sin(45°)]
  180° about axis a  = [0, a_x, a_y, a_z]
  270° about axis a  = 90° about -a

## Common fixes (use ONLY when no numeric report is provided)
Arm points backward instead of out → multiply current by [0, 0, 1, 0]  (180° Y)
Arm points up instead of out       → multiply current by [0.707, 0, 0.707, 0]  (90° Y)
Forearm/wrist twisted 90°          → multiply current by [0.707, 0.707, 0, 0]  (90° X)
Leg bent backward at knee          → multiply current by [0, 1, 0, 0]  (180° X)
Foot twisted outward 90°           → multiply current by [0.707, 0, 0, 0.707]  (90° Z)
Root/torso tilted forward 90°      → set world_rotation to [0.707, -0.707, 0, 0]
All joints systematically off      → adjust world_rotation first

## Response format
Return ONLY valid JSON. No prose. Choose one:

Case A — changes needed:
{
  "verdict": "needs_fix",
  "analysis": "one sentence describing the main problem",
  "patch": {
    "ik_match_table1": {
      "joint_name": {"mode": "mul", "quat": [w, x, y, z]},
      ...
    },
    "ik_match_table2": {
      "joint_name": {"mode": "mul", "quat": [w, x, y, z]}
    }
  }
}

Case B — looks good:
{
  "verdict": "ok",
  "analysis": "motion looks well aligned"
}

Rules:
- When a deviation report is provided, target the top-5 worst links FIRST.
  Ignore the images for joints that the report says are fine (< 5°).
- Only include joints that clearly need fixing. Do not change joints that look correct.
- When fixing ik_match_table1, apply the same fix to ik_match_table2 for consistency.
- Keep fixes conservative: prefer a single axis rotation. Multiple 90° fixes in one \
  iteration cause oscillation.
- Normalize all quaternions (unit length).
"""


# Known per-model limits.
# image_limit: max images per chat-completion request.
# max_tokens:  max output tokens allowed by the model.
_MODEL_LIMITS: dict[str, dict] = {
    "glm-4v-flash": {"image_limit": 4, "max_tokens": 1024, "json_mode": False},
    "glm-4v-plus": {"image_limit": 5, "max_tokens": 1024, "json_mode": False},
    "glm-5v-turbo": {"image_limit": 8, "max_tokens": 4096, "json_mode": True},
}


def _get_model_limits(model: str) -> dict:
    """Return limits dict for *model*, or empty dict (no known limits)."""
    for prefix, limits in _MODEL_LIMITS.items():
        if model.lower().startswith(prefix):
            return limits
    return {}


def _select_images(image_paths: list[Path], max_images: int) -> list[Path]:
    """Select up to *max_images*, prioritising front views across keyframes."""
    if len(image_paths) <= max_images:
        return image_paths
    fronts = sorted(p for p in image_paths if "front" in p.name)
    sides = sorted(p for p in image_paths if "side" in p.name)
    backs = sorted(p for p in image_paths if "back" in p.name)
    selected = (fronts + sides + backs)[:max_images]
    return sorted(selected)


def _encode_image(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def ask_claude(
    client: Any,
    model: str,
    image_paths: list[Path],
    config: dict,
    iteration: int,
    client_type: str = "anthropic",
    max_images: int | None = None,
    deviation_text: str | None = None,
) -> dict:
    """Send images + current config to Claude (or OpenAI-compatible), return parsed JSON."""
    model_limits = _get_model_limits(model)

    # Enforce per-model image limits
    image_limit = max_images or model_limits.get("image_limit")
    if image_limit and len(image_paths) > image_limit:
        original_count = len(image_paths)
        image_paths = _select_images(image_paths, image_limit)
        print(
            f"[agent] Reduced images from {original_count} to {len(image_paths)} "
            f"(model {model} image limit: {image_limit})"
        )

    # Clamp max_tokens to model's allowed maximum
    default_max_tokens = 2048
    allowed_max_tokens = model_limits.get("max_tokens", default_max_tokens)
    effective_max_tokens = min(default_max_tokens, allowed_max_tokens)

    intro_parts = [
        f"## Iteration {iteration}",
        "",
        f"Below are renders at motion keyframes ({len(image_paths)} images). "
        "After the images I show the current IK config.",
        "",
    ]
    if deviation_text:
        intro_parts.extend([deviation_text, ""])
    intro_parts.append(
        "Identify misaligned joints and return a JSON patch, or return "
        '{"verdict":"ok"} if all links are within 5° of the T-pose spec.'
    )
    user_text_intro = "\n".join(intro_parts)
    config_text = f"## Current IK config\n```json\n{json.dumps(config, indent=2)}\n```"

    if client_type == "openai":
        # OpenAI-compatible format (also works with most third-party proxies)
        user_content: list[dict] = [{"type": "text", "text": user_text_intro}]
        for img_path in image_paths:
            b64 = _encode_image(img_path)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        user_content.append({"type": "text", "text": config_text})

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": effective_max_tokens,
            "messages": messages,
        }
        # GLM-5V-Turbo supports structured JSON output via response_format,
        # which prevents the model from emitting prose around the JSON.
        if model_limits.get("json_mode"):
            create_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**create_kwargs)
        raw = response.choices[0].message.content.strip()

    else:
        # Anthropic SDK format
        content: list[dict] = [{"type": "text", "text": user_text_intro}]
        for img_path in image_paths:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": _encode_image(img_path),
                    },
                }
            )
        content.append({"type": "text", "text": config_text})

        response = client.messages.create(
            model=model,
            max_tokens=effective_max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Truncated response — try to salvage by closing open braces/brackets.
        # This handles the common case where max_tokens cuts mid-JSON.
        repaired = raw
        open_braces = repaired.count("{") - repaired.count("}")
        open_brackets = repaired.count("[") - repaired.count("]")
        # Trim trailing incomplete key/value (after last comma or colon)
        for trim_char in (",", ":"):
            idx = repaired.rfind(trim_char)
            if idx != -1:
                # Check if what follows the trim_char is incomplete
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


# ---------------------------------------------------------------------------
# Retargeting helper (shared with inspector)
# ---------------------------------------------------------------------------


def _retarget(
    src: str, motion_file: str, robot: str, bvh_format: str, max_frames: int | None
) -> tuple[np.ndarray, list[dict]]:
    """Retarget and return (qpos_seq, human_seq).

    human_seq[i] is a dict {bone_name: (pos, quat)} from the retargeter's
    internal scaled_human_data — the human reference pose already scaled to
    the robot's body proportions. This is exactly what should be overlaid on
    the rendered robot so the VLM can visually compare 'where the human bone
    is' vs 'where the robot body landed after IK'.
    """
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from tqdm import tqdm

    if src == "bvh":
        frames, human_height, _ = _load_bvh(motion_file, bvh_format)
    elif src == "smplx":
        frames, human_height, _ = _load_smplx(motion_file)
    elif src == "fbx_offline":
        frames, human_height, _ = _load_fbx_offline(motion_file)
    else:
        raise ValueError(f"Unknown src: {src}")

    if max_frames is not None:
        frames = frames[:max_frames]

    retargeter = GMR(
        src_human=src,
        tgt_robot=robot,
        actual_human_height=human_height,
        verbose=False,
    )
    qpos_list: list[np.ndarray] = []
    human_list: list[dict] = []
    for f in tqdm(frames, desc="retargeting", leave=False):
        qpos_list.append(retargeter.retarget(f).copy())
        human_list.append(_scaled_human_reference(retargeter, f))
    return np.array(qpos_list), human_list


def _retarget_tpose_qpos(src: str, motion_file: str, robot: str, bvh_format: str) -> np.ndarray:
    """Retarget the *first frame* of a T-pose source motion → candidate qpos.

    Used to drive the numeric alignment gate. GMR reads the IK config from
    disk on each ``GMR(...)`` construction, so callers must write the
    current config before invoking this helper.
    """
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    if src == "bvh":
        frames, human_height, _ = _load_bvh(motion_file, bvh_format)
    elif src == "smplx":
        frames, human_height, _ = _load_smplx(motion_file)
    elif src == "fbx_offline":
        frames, human_height, _ = _load_fbx_offline(motion_file)
    else:
        raise ValueError(f"Unknown tpose src: {src!r}")

    retargeter = GMR(
        src_human=src, tgt_robot=robot, actual_human_height=human_height, verbose=False
    )
    return retargeter.retarget(frames[0]).copy()


def _format_deviation_report(
    report: dict, total: float, max_angle: float, prev_total: float | None, top_k: int = 5
) -> str:
    """Render a deviation report block for injection into the VLM user message."""
    from roboharness.alignment import worst_k

    lines = ["## T-pose deviation report (numeric ground truth)"]
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


def _scaled_human_reference(retargeter: Any, raw_frame: dict) -> dict:
    """Compute the anatomical human pose at robot scale, before per-joint offsets.

    This is the *reference* the VLM needs: "what the human is doing, rescaled to
    fit the robot." We deliberately skip offset_human_data() because its pos/rot
    offsets come from the IK config we're trying to tune — including them would
    make the reference skeleton move in lockstep with a wrong config and
    defeat the whole purpose of a visual comparison.
    """
    data = retargeter.to_numpy(raw_frame)
    data = retargeter.scale_human_data(
        data, retargeter.human_root_name, retargeter.human_scale_table
    )
    data = retargeter.apply_world_rotation(data)
    return {k: (p.copy(), q.copy()) for k, (p, q) in data.items()}


# ---------------------------------------------------------------------------
# Capture helper
# ---------------------------------------------------------------------------


def _capture(
    qpos_seq: np.ndarray,
    xml_path: Path,
    root_body: str,
    cam_distance: float,
    output_dir: Path,
    trial: int,
    human_seq: list[dict] | None = None,
) -> list[Path]:
    """Run harness replay and return sorted list of saved RGB PNG paths.

    human_seq, when provided, is overlaid on each render as a translucent
    cyan skeleton — this is the VLM's reference for "where the human joint
    actually is" vs the robot's resulting pose.
    """
    from roboharness.core.harness import Harness

    cameras = ["inspect_front", "inspect_side", "inspect_back"]
    backend = GMRReplayBackend(
        xml_path=xml_path,
        qpos_seq=qpos_seq,
        cameras=cameras,
        root_body_name=root_body,
        cam_distance=cam_distance,
        use_meshcat=False,
        human_seq=human_seq,
    )
    harness = Harness(backend=backend, output_dir=output_dir, task_name="align")
    n = len(qpos_seq)
    checkpoint_defs = [
        ("frame_start", 0),
        ("frame_quarter", n // 4),
        ("frame_half", n // 2),
        ("frame_three_quarter", 3 * n // 4),
    ]
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


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GMR Alignment Agent — AI-driven IK config auto-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--robot", required=True)
    parser.add_argument("--motion_file", required=True)
    parser.add_argument("--src", default="bvh", choices=["bvh", "smplx", "fbx_offline"])
    parser.add_argument("--bvh_format", default="auto", choices=["auto", "lafan1", "soma"])
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--max_iter", type=int, default=4, help="Max optimization iterations")
    parser.add_argument(
        "--output", default="/home/user2/GMR/agent_output", help="Output directory for captures"
    )
    parser.add_argument(
        "--model",
        default="glm-5v-turbo",
        help=(
            "Vision model for analysis (default: glm-5v-turbo; "
            "also supports glm-4v-flash, glm-4v-plus)"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Capture and show Claude's analysis but do NOT write config changes",
    )
    parser.add_argument(
        "--tpose_spec",
        type=Path,
        help="T-pose spec JSON (default: specs/tpose/{robot}.json if it exists).",
    )
    parser.add_argument(
        "--tpose_motion",
        help="Canonical T-pose source motion for numeric alignment gate (e.g. "
        "a T-pose BVH). When provided together with a spec, each iteration "
        "computes per-link deviation and injects the worst-k report into "
        "the VLM prompt.",
    )
    parser.add_argument(
        "--tpose_src",
        default="bvh",
        choices=["bvh", "smplx", "fbx_offline"],
        help="Source format for --tpose_motion (default: bvh).",
    )
    parser.add_argument(
        "--tpose_bvh_format",
        default="auto",
        choices=["auto", "lafan1", "soma"],
        help="BVH parser for --tpose_motion when --tpose_src=bvh.",
    )
    parser.add_argument(
        "--tpose_threshold",
        type=float,
        default=5.0,
        help="Per-link angle_deg threshold. When max_angle drops below this, "
        "the agent exits with 'ok' even if the VLM would keep going.",
    )
    parser.add_argument(
        "--api_key",
        default="c26ea0774bf9489db207c863f14b3605.ygnHRDwZMLgRcyYI",
        help="API key (overrides ANTHROPIC_API_KEY / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--api_base",
        default="https://open.bigmodel.cn/api/paas/v4",
        help=(
            "OpenAI-compatible base URL "
            "(e.g. https://node-hk.sssaicode.com/v1). "
            "When set, uses openai SDK instead of anthropic SDK."
        ),
    )
    args = parser.parse_args()

    from general_motion_retargeting.params import (
        IK_CONFIG_DICT,
        ROBOT_XML_DICT,
        VIEWER_CAM_DISTANCE_DICT,
    )

    # Build API client — two paths:
    #   1. api_base set → OpenAI-compatible endpoint (third-party proxy)
    #   2. default      → Anthropic SDK (official API)
    if args.api_base:
        import httpx
        from openai import OpenAI

        # Use a proxy-free httpx client — environment SOCKS proxies break httpx
        # unless httpx[socks] is installed; the remote endpoint is reachable directly.
        # base_url must end with /v1 — the SDK appends /chat/completions to it
        base_url = args.api_base.rstrip("/")
        client = OpenAI(
            api_key=args.api_key or "sk-placeholder",
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )
        client_type = "openai"
    else:
        import anthropic

        client = (
            anthropic.Anthropic(api_key=args.api_key) if args.api_key else anthropic.Anthropic()
        )
        client_type = "anthropic"

    xml_path = Path(str(ROBOT_XML_DICT[args.robot]))
    cam_distance = float(VIEWER_CAM_DISTANCE_DICT.get(args.robot, 2.5))
    root_body = _find_root_body(xml_path)
    config_path = Path(str(IK_CONFIG_DICT[args.src][args.robot]))
    output_dir = Path(args.output)

    print(f"\n{'=' * 60}")
    print("GMR Alignment Agent")
    print(f"{'=' * 60}")
    print(f"Robot      : {args.robot}")
    print(f"Motion     : {args.motion_file}")
    print(f"Source     : {args.src}")
    print(f"IK config  : {config_path}")
    print(f"Model      : {args.model}")
    print(f"Max iters  : {args.max_iter}")
    print(f"Dry run    : {args.dry_run}")
    print(f"{'=' * 60}\n")

    # Load T-pose spec (optional, enables numeric deviation gate)
    tpose_spec: dict | None = None
    tpose_spec_path = args.tpose_spec
    if tpose_spec_path is None:
        default_path = _HERE.parent / "specs" / "tpose" / f"{args.robot}.json"
        if default_path.exists():
            tpose_spec_path = default_path
    if tpose_spec_path and args.tpose_motion:
        from roboharness.alignment import load_tpose_spec

        tpose_spec = load_tpose_spec(tpose_spec_path)
        print(
            f"[agent] Numeric gate ON: spec={tpose_spec_path.name} "
            f"({len(tpose_spec['links'])} links)  tpose_motion={args.tpose_motion}"
        )
    elif tpose_spec_path and not args.tpose_motion:
        print(
            f"[agent] Numeric gate OFF: spec {tpose_spec_path.name} found but "
            "--tpose_motion not supplied. VLM will rely on vision only."
        )
    else:
        print("[agent] Numeric gate OFF: no T-pose spec / motion — VLM-only mode.")

    # Load initial config
    with config_path.open() as f:
        config = json.load(f)

    # Keep a backup of original config
    backup_path = config_path.with_suffix(".json.bak")
    if not backup_path.exists():
        backup_path.write_text(json.dumps(config, indent=4))
        print(f"[agent] Config backed up to {backup_path}")

    # Phase A: retarget once — qpos seq is recomputed each iter because config
    # changes affect GMR's internal setup, so we must re-instantiate each time.
    print("[agent] Phase A: initial retargeting...")
    qpos_seq, human_seq = _retarget(
        args.src, args.motion_file, args.robot, args.bvh_format, args.frames
    )
    print(f"[agent] qpos shape: {qpos_seq.shape}")

    prev_total: float | None = None
    for iteration in range(1, args.max_iter + 1):
        print(f"\n--- Iteration {iteration}/{args.max_iter} ---")

        # Compute T-pose numeric deviation for the current config, if enabled.
        # This drives three things: the early-exit gate, the prev/current
        # delta the VLM sees, and the worst-k injected into the prompt.
        deviation_text: str | None = None
        cur_total: float | None = None
        cur_max: float | None = None
        if tpose_spec is not None and args.tpose_motion:
            from roboharness.alignment import compute_deviations, total_deviation, worst_k

            print("[agent] Retargeting T-pose for numeric gate...")
            tpose_qpos = _retarget_tpose_qpos(
                args.tpose_src,
                args.tpose_motion,
                args.robot,
                args.tpose_bvh_format,
            )
            report = compute_deviations(tpose_qpos, tpose_spec["xml_path"], tpose_spec)
            cur_total = total_deviation(report)
            cur_max = worst_k(report, 1)[0][1] if report else 0.0
            deviation_text = _format_deviation_report(report, cur_total, cur_max, prev_total)
            print(
                f"[agent] deviation: total={cur_total:.2f}°  max={cur_max:.2f}°  "
                f"(threshold={args.tpose_threshold}°)"
            )
            if cur_max < args.tpose_threshold:
                print(f"[agent] Numeric gate PASSED (max < {args.tpose_threshold}°). Stopping.")
                break

        # Capture renders with current config's effect already baked into qpos
        print("[agent] Capturing renders...")
        png_paths = _capture(
            qpos_seq=qpos_seq,
            xml_path=xml_path,
            root_body=root_body,
            cam_distance=cam_distance,
            output_dir=output_dir,
            trial=iteration,
            human_seq=human_seq,
        )
        print(f"[agent] {len(png_paths)} images captured")
        for p in png_paths:
            print(f"        {p}")

        # Ask Claude
        print(f"[agent] Querying {args.model}...")
        response = ask_claude(
            client=client,
            model=args.model,
            image_paths=png_paths,
            config=config,
            iteration=iteration,
            client_type=client_type,
            deviation_text=deviation_text,
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

        # Apply patch and write config
        config = apply_patch(config, patch)
        with config_path.open("w") as f:
            json.dump(config, f, indent=4)
        print(f"[agent] Config updated → {config_path}")

        # Re-retarget with updated config
        print("[agent] Re-retargeting with updated config...")
        qpos_seq, human_seq = _retarget(
            args.src, args.motion_file, args.robot, args.bvh_format, args.frames
        )
        # Roll forward the numeric baseline so next iteration's prompt shows Δ.
        if cur_total is not None:
            prev_total = cur_total

    else:
        print(
            f"\n[agent] Reached max iterations ({args.max_iter})."
            " Review output and tune manually if needed."
        )

    print(f"\n[agent] Final config: {config_path}")
    print(f"[agent] Captures   : {output_dir}/")
    print(f"[agent] Backup     : {backup_path}")
    print("\nTo restore original config:")
    print(f"  cp {backup_path} {config_path}")


if __name__ == "__main__":
    main()
