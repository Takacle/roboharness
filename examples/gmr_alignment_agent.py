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
import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from _gmr_shared import (
    GMRReplayBackend,
    find_root_body,
    load_motion,
    scaled_human_reference,
)

from roboharness._utils import encode_image_base64, select_image_files
from roboharness.alignment import apply_patch, compute_direct_patch

# ---------------------------------------------------------------------------
# Claude Vision call
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robotics motion-retargeting expert specializing in IK scale tuning.

Your task: inspect rendered images of a humanoid robot replaying retargeted motion, \
identify bone-length mismatches, and return a JSON patch adjusting scale values so \
the retargeted robot posture matches the target T-pose (green translucent overlay).

## Reading the images
Each rendered frame shows TWO overlaid figures:
  1. The SOLID ROBOT — the retargeted output using the current IK config.
  2. A TRANSLUCENT GREEN SKELETON — the TARGET T-pose: the robot rendered at
     its canonical (spec) T-pose. This is the pose the robot SHOULD achieve.

Your primary diagnostic: compare solid robot link endpoints to green target endpoints.
  - If a robot arm/hand extends PAST the green target → that bone's scale is too
    LARGE → reduce it.
  - If a robot arm/hand falls SHORT of the green target → that bone's scale is
    too SMALL → increase it.
  - If they overlap closely → that bone is fine, do not touch it.

## Scale config structure
The IK config has two scale-related fields:

  human_scale_table (dict of bone→float):
    Scale factor per human bone in its local frame. This controls the effective
    bone length used by the IK solver.
    1.0 = no scaling, <1.0 = shorter, >1.0 = longer.
    Typical defaults: arms 0.70-0.85, legs 0.80-0.95, trunk 0.85-1.0.

  human_height_assumption (float):
    Overall human reference height in metres (typically 1.6-1.9).
    Increasing this shrinks ALL bones (ratio = actual_height / assumption).
    Use sparingly — it affects everything at once.

## Numeric deviation report
If the user message contains a "Position deviation report" section, it is the
GROUND TRUTH for which links are misaligned. Each row is:
    link_name   pos_err_cm
where ``pos_err_cm`` is the Euclidean distance between the robot link's current
position and the spec T-pose position.

Interpreting pos_err_cm:
    < 3 cm     : excellent, leave alone
    3-10 cm    : mild misalignment — small scale tweak (±0.05) may help
    10-25 cm   : significant mismatch — scale likely off by 0.10-0.20
    > 25 cm    : severe mismatch — scale may need correction of 0.20+

A rotation deviation report (angle_deg per link) may also be present — use it
as supplementary information about orientation errors.

## How scale affects different body regions
  - Hand/wrist link position error > foot/ankle error → arm scale is off
  - Foot/ankle position error > hand/wrist error → leg scale is off
  - Torso link error → spine scale or human_height_assumption is off
  - Both arms equally too long/short → adjust BOTH LeftArm + RightArm by same amount
  - Both legs equally too long/short -> adjust BOTH LeftUpLeg + RightUpLeg
    (and/or LeftLeg + RightLeg)

## Common fixes
  Hand overshoots target     → reduce forearm scale by ~0.10  (mul 0.85)
  Hand undershoots target    → increase arm scale by ~0.10    (set to 0.85)
  Feet above/below target    → adjust leg scale by ~0.05-0.10
  Torso too tall/short       → adjust Spine2 scale or human_height_assumption
  Whole body too large       → increase human_height_assumption
  Whole body too small       → decrease human_height_assumption

## Response format
Return ONLY valid JSON. No prose. Choose one:

Case A — changes needed:
{
  "verdict": "needs_fix",
  "analysis": "one sentence describing the main length mismatch",
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
- When a position deviation report is provided, target the top-5 worst links
  FIRST. Ignore the images for links that the report says are fine (< 5 cm).
- Only include bones that clearly need fixing. Do not change bones that are
  close to the target.
- Keep fixes conservative: small adjustments (0.05-0.10) per iteration to
  avoid oscillation.
- For mul mode, value=0.95 means "shrink to 95% of current". For set mode,
  directly set to the given value.
- Scale values must be between 0.2 and 2.5. human_height_assumption must be
  between 0.5 and 3.0.
"""

_WEIGHTS_PROMPT = """\
You are a robotics motion-retargeting expert specializing in IK weight tuning.

Your task: inspect rendered images comparing the retargeted robot (solid) to
the TARGET T-pose overlay (translucent green), and adjust pos_weight and
rot_weight in the IK match tables to improve IK solver convergence.

## Background
The IK match tables map robot joints to human bones with two priority weights:
  pos_weight (index [1], range 0-1000): how much the IK solver prioritizes
     matching this joint's POSITION against the target bone position.
  rot_weight (index [2], range 0-1000): how much the IK solver prioritizes
     matching this joint's ROTATION against the target bone orientation.

Higher weight = the solver works harder on that joint, potentially at the
expense of other joints. The solver has finite capacity, so balance matters.

## When to adjust weights
- A joint has high position error (large gap to green target) but low pos_weight
  → increase pos_weight to tell the solver this joint matters more.
- A joint has high rotation error but low rot_weight → increase rot_weight.
- A joint is already well-aligned → its weight is fine or could decrease
  slightly to free solver capacity for struggling joints.
- Multiple joints all have errors → raise weights of the WORST few, don't
  boost everything (solver will be overloaded).

## Weight conventions
- Default pos_weight: 10-100 (trunk low, extremities high)
- Default rot_weight: 5-100
- Hand/wrist joints benefit from pos_weight 50-200
- Knee/ankle joints benefit from rot_weight 50-100
- Torso/root joints typically pos_weight 10-50

## Response format
Return ONLY valid JSON. No prose. Choose one:

Case A — changes needed:
{
  "verdict": "needs_fix",
  "analysis": "one sentence describing which joints need weight changes",
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
{
  "verdict": "ok",
  "analysis": "all joints within acceptable alignment"
}

## Optional quaternion corrections
You may also include a "quat" field alongside weight changes when the rotation
deviation report shows a link with > 10° error AND simply adjusting rot_weight
is insufficient (the orientation is fundamentally wrong, not just deprioritised):
{
  "left_elbow": {
    "pos_weight": {"mode": "set", "value": 100},
    "rot_weight": {"mode": "mul", "value": 2.0},
    "quat": {"mode": "mul", "quat": [0.707, 0.707, 0, 0]}
  }
}
Common cardinal corrections (mul mode, scalar-first [w,x,y,z]):
  90°  about X → [0.707, 0.707, 0, 0]    180° about X → [0, 1, 0, 0]
  90°  about Y → [0.707, 0, 0.707, 0]    180° about Y → [0, 0, 1, 0]
  90°  about Z → [0.707, 0, 0, 0.707]    180° about Z → [0, 0, 0, 1]
Only include quat for joints with clear rotational misalignment. The "quat" field
is resolved by Hamilton product: q_new = patch * current.

Rules:
- When a position/rotation deviation report is provided, target the worst-5.
- For mul mode, value=2.0 means double the weight; value=0.5 means halve it.
- For set mode, directly set weight to the given value.
- Keep both tables (ik_match_table1 and ik_match_table2) in sync.
- Weight values must be between 0 and 1000.
"""

_QUATERNION_PROMPT = """\
You are a robotics motion-retargeting expert specializing in quaternion offset tuning.

Your task: inspect rendered images comparing the retargeted robot (solid) to the
TARGET T-pose overlay (translucent green), diagnose joint rotational misalignments,
and return a JSON patch adjusting the quaternion offsets in the IK match tables.

## Reading the images
Each rendered frame shows TWO overlaid figures:
  1. The SOLID ROBOT — the retargeted output using the current IK config.
  2. A TRANSLUCENT GREEN SKELETON — the TARGET T-pose: the robot rendered at
     its canonical (spec) T-pose. This is the pose the robot SHOULD achieve.

For rotation diagnosis:
  - If a robot limb points in a different direction than the green overlay at
    the same joint → that joint's quaternion offset is incorrect.
  - Use the rotation deviation report (axis-angle per link) as the ground truth
    for which joints have orientation errors.

## Rotation deviation report
Each row is:
    link_name   angle_deg   axis=[ax, ay, az]
where angle_deg is the angular distance between the robot link's current
orientation and the spec T-pose orientation, and axis is the rotation axis
(from current toward target).

Interpreting angle_deg:
    < 5°   : excellent, leave alone
    5-15°  : mild misalignment — small correction may help
    15-30° : significant — likely needs a cardinal or near-cardinal correction
    > 30°  : severe — likely off by 90° or 180°

## Quaternion offset semantics
Each entry in ik_match_table1/2 is a list: [human_bone, pos_weight, rot_weight, ?, quat_offset]
The quat_offset at index [4] (scalar-first [w, x, y, z]) maps the human bone's
orientation to the robot joint's orientation:
    R_robot = Q_offset * R_human

Your patch replaces or multiplies into this quat_offset:
  "mode": "mul" → q_new = patch * current  (Hamilton product, apply correction)
  "mode": "set" → q_new = patch             (absolute override)

## Common cardinal corrections (mul mode, scalar-first [w,x,y,z])
  90°  about X → [0.707, 0.707, 0, 0]    180° about X → [0, 1, 0, 0]
  90°  about Y → [0.707, 0, 0.707, 0]    180° about Y → [0, 0, 1, 0]
  90°  about Z → [0.707, 0, 0, 0.707]    180° about Z → [0, 0, 0, 1]
  -90° about X → [0.707, -0.707, 0, 0]   (negate the vector part)

For intermediate angles, compute:
  half = angle_rad / 2
  q = [cos(half), ax*sin(half), ay*sin(half), az*sin(half)]
where [ax, ay, az] is the unit rotation axis from the report.

## Diagnostic strategy
1. Start from the rotation deviation report — sort links by angle_deg descending.
2. For the worst link, determine which axis the correction should be about:
   - The report gives you axis=[ax,ay,az] (direction from current toward target).
   - If the axis is close to a cardinal axis (X/Y/Z), use the corresponding
     cardinal quaternion correction.
   - If the axis is oblique, compute the axis-angle quaternion directly.
3. Consider parent-child chains: if shoulder AND elbow are both wrong, the
   shoulder correction often fixes the elbow too (correction propagates).
   Fix the PARENT first, then re-evaluate children in the next iteration.

## Response format
Return ONLY valid JSON. No prose. Choose one:

Case A — changes needed:
{
  "verdict": "needs_fix",
  "analysis": "one sentence describing the main rotational misalignment",
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
{
  "verdict": "ok",
  "analysis": "all joint rotations within acceptable tolerance"
}

Rules:
- Target the top 3-5 worst joints per iteration. Do not try to fix everything at once.
- Prefer mul mode over set mode (incremental corrections are safer).
- Keep both tables (ik_match_table1 and ik_match_table2) in sync — patch the
  same joints with the same quaternion in both.
- Be conservative: correct 1-3 joints per iteration to avoid oscillation.
- Do NOT include pos_weight or rot_weight — those are handled by weights mode.
- Quaternion values must be valid (approximately unit norm).
"""

_MODEL_LIMITS: dict[str, dict] = {
    "glm-4v-flash": {"image_limit": 4, "max_tokens": 1024, "json_mode": False},
    "glm-4v-plus": {"image_limit": 5, "max_tokens": 1024, "json_mode": False},
    "glm-5v-turbo": {"image_limit": 8, "max_tokens": 4096, "json_mode": True},
}


def _get_model_limits(model: str) -> dict:
    for prefix, limits in _MODEL_LIMITS.items():
        if model.lower().startswith(prefix):
            return limits
    return {}


def ask_claude(
    client: Any,
    model: str,
    image_paths: list[Path],
    config: dict,
    iteration: int,
    client_type: str = "anthropic",
    max_images: int | None = None,
    deviation_text: str | None = None,
    pos_deviation_text: str | None = None,
    tune_mode: str = "scale",
) -> dict:
    model_limits = _get_model_limits(model)

    image_limit = max_images or model_limits.get("image_limit")
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
            "The translucent GREEN overlay is the TARGET T-pose you should match. "
            "Focus on ROTATIONAL alignment — joint orientations, not positions. "
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
            f"Below are renders at the T-pose frame ({len(image_paths)} images). "
            "The translucent GREEN overlay is the TARGET T-pose you should match. "
            "After the images I show the current weight config for deviated joints."
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
            f"Below are renders at the T-pose frame ({len(image_paths)} images). "
            "The translucent GREEN overlay is the TARGET T-pose you should match. "
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
        'or return {"verdict":"ok"} if the poses are well-aligned.'
    )
    user_text_intro = "\n".join(intro_parts)

    if client_type == "openai":
        user_content: list[dict] = [{"type": "text", "text": user_text_intro}]
        for img_path in image_paths:
            b64 = encode_image_base64(img_path)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
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
        raw = response.choices[0].message.content.strip()

    else:
        content: list[dict] = [{"type": "text", "text": user_text_intro}]
        for img_path in image_paths:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": encode_image_base64(img_path),
                    },
                }
            )
        content.append({"type": "text", "text": config_text})

        response = client.messages.create(
            model=model,
            max_tokens=effective_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        raw = response.content[0].text.strip()

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


# ---------------------------------------------------------------------------
# Retargeting helpers
# ---------------------------------------------------------------------------


def _create_default_ik_config(robot: str, src: str, xml_path: Path) -> Path:
    """Auto-generate a default IK config when none exists for a robot.

    Delegates to ``roboharness.alignment.body_matcher`` and
    ``roboharness.alignment.config_gen`` for matching and generation.
    """
    import re as _re

    from roboharness.alignment.body_matcher import match_bodies
    from roboharness.alignment.config_gen import generate_ik_config, write_ik_config
    from roboharness.alignment.skeleton_maps import get_skeleton

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


def _patch_fallback_offsets(retargeter: Any, config: dict) -> None:
    """Register fallback human bone offsets not backed by a robot body."""
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


def _retarget(
    src: str, motion_file: str, robot: str, bvh_format: str, max_frames: int | None
) -> tuple[np.ndarray, list[dict]]:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from tqdm import tqdm

    frames, human_height, _ = load_motion(src, motion_file, bvh_format)

    if max_frames is not None:
        frames = frames[:max_frames]

    retargeter = GMR(
        src_human=src,
        tgt_robot=robot,
        actual_human_height=human_height,
        verbose=False,
    )
    # Patch fallback offsets for bones without dedicated robot bodies
    from pathlib import Path as _Path

    from general_motion_retargeting.params import IK_CONFIG_DICT as _IKD

    cfg_path = _Path(str(_IKD[src][robot]))
    if cfg_path.exists():
        with cfg_path.open() as _f:
            _patch_fallback_offsets(retargeter, json.load(_f))
    qpos_list: list[np.ndarray] = []
    human_list: list[dict] = []
    for f in tqdm(frames, desc="retargeting", leave=False):
        qpos_list.append(retargeter.retarget(f).copy())
        human_list.append(scaled_human_reference(retargeter, f))
    return np.array(qpos_list), human_list


def _retarget_tpose_qpos(src: str, motion_file: str, robot: str, bvh_format: str) -> np.ndarray:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    frames, human_height, _ = load_motion(src, motion_file, bvh_format)

    retargeter = GMR(
        src_human=src, tgt_robot=robot, actual_human_height=human_height, verbose=False
    )
    from pathlib import Path as _Path

    from general_motion_retargeting.params import IK_CONFIG_DICT as _IKD

    cfg_path = _Path(str(_IKD[src][robot]))
    if cfg_path.exists():
        with cfg_path.open() as _f:
            _patch_fallback_offsets(retargeter, json.load(_f))
    return retargeter.retarget(frames[0]).copy()


def _format_deviation_report(
    report: dict, total: float, max_angle: float, prev_total: float | None, top_k: int = 5
) -> str:
    from roboharness.alignment import worst_k

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
    from roboharness.alignment import worst_k_position

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
    tpose_spec: dict | None = None,
) -> list[Path]:
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
        tpose_spec=tpose_spec,
    )
    harness = Harness(backend=backend, output_dir=output_dir, task_name="align")
    checkpoint_defs = [
        ("frame_start", 0),
        # ("frame_quarter", n // 4),
        # ("frame_half", n // 2),
        # ("frame_three_quarter", 3 * n // 4),
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


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def main() -> None:
    _load_dotenv()
    _HERE = Path(__file__).resolve().parent

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
    parser.add_argument("--max_iter", type=int, default=8, help="Max optimization iterations")
    parser.add_argument("--output", default="./agent_output", help="Output directory for captures")
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
        default=None,
        choices=["bvh", "smplx", "fbx_offline"],
        help="Source format for --tpose_motion (default: same as --src).",
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
        "--regression_tolerance",
        type=float,
        default=0.0,
        help="Auto-revert a patch if post-patch total_deviation exceeds "
        "pre-patch total by more than this many degrees. 0 = strictly worse "
        "reverts. Negative disables the gate. Only active when tpose_motion "
        "is provided.",
    )
    parser.add_argument(
        "--solve_mode",
        action="store_true",
        help="Directly compute correct IK config quaternions from T-pose "
        "human bone orientations. Solves Q_offset = Q_human^{-1} * R_expected "
        "for every joint in a single iteration. Requires --tpose_spec and "
        "--tpose_motion.",
    )
    parser.add_argument(
        "--tune_mode",
        default="scale",
        choices=["scale", "weights", "optimize_scale", "quaternion"],
        help="What to tune: 'scale' uses VLM for bone lengths; "
        "'optimize_scale' uses numerical optimisation (scipy, no VLM); "
        "'weights' uses VLM for IK priority (+ optional quat tweaks); "
        "'quaternion' uses VLM for quaternion offsets. "
        "Run solve_mode first to compute quaternion offsets.",
    )
    parser.add_argument(
        "--preserve",
        default="",
        help="Comma-separated joint names to preserve in --solve_mode. "
        "Preserved joints keep their existing quaternion offsets. "
        'Default: "" (recompute all).',
    )
    parser.add_argument(
        "--world_rot",
        default="",
        help="Set world_rotation in config after --solve_mode. "
        "Format: 'angle,axis_x,axis_y,axis_z' (e.g. '90,0,0,1' for 90° around Z). "
        "Useful for fixing overall facing direction.",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("OPENAI_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "")),
        help="API key (reads OPENAI_API_KEY / ANTHROPIC_API_KEY env vars by default)",
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
    if args.tpose_src is None:
        args.tpose_src = args.src

    from general_motion_retargeting.params import (
        IK_CONFIG_DICT,
        ROBOT_XML_DICT,
        VIEWER_CAM_DISTANCE_DICT,
    )

    if args.api_base and not args.solve_mode:
        try:
            import httpx
            from openai import OpenAI
        except ModuleNotFoundError as e:
            print(f"[agent] ERROR: {e}. Install openai/httpx or use --solve_mode only.")
            return

        base_url = args.api_base.rstrip("/")
        client = OpenAI(
            api_key=args.api_key or "sk-placeholder",
            base_url=base_url,
            http_client=httpx.Client(trust_env=False),
        )
        client_type = "openai"
    elif not args.solve_mode:
        try:
            import anthropic
        except ModuleNotFoundError as e:
            print(f"[agent] ERROR: {e}. Install anthropic or use --solve_mode.")
            return

        client = (
            anthropic.Anthropic(api_key=args.api_key) if args.api_key else anthropic.Anthropic()
        )
        client_type = "anthropic"

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
    print(f"Model      : {args.model}")
    print(f"Max iters  : {args.max_iter}")
    print(f"Dry run    : {args.dry_run}")
    print(f"{'=' * 60}\n")

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

    with config_path.open() as f:
        config = json.load(f)

    backup_path = config_path.with_suffix(".json.bak")
    if not backup_path.exists():
        backup_path.write_text(json.dumps(config, indent=4))
        print(f"[agent] Config backed up to {backup_path}")

    print("[agent] Phase A: initial retargeting...")
    qpos_seq, human_seq = _retarget(
        args.src, args.motion_file, args.robot, args.bvh_format, args.frames
    )
    print(f"[agent] qpos shape: {qpos_seq.shape}")

    if args.solve_mode:
        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        if tpose_spec is None or not args.tpose_motion:
            print("[agent] ERROR: --solve_mode requires --tpose_spec + --tpose_motion.")
            return

        print("[agent] Solve mode: computing direct IK config from human bone orientations...")
        tpose_frames, _, _ = load_motion(args.tpose_src, args.tpose_motion, args.tpose_bvh_format)
        tpose_frame = tpose_frames[0]

        retargeter = GMR(
            src_human=args.src,
            tgt_robot=args.robot,
            actual_human_height=None,
            verbose=False,
        )
        human_ref = scaled_human_reference(retargeter, tpose_frame)

        if args.src in ("smplx",):
            from roboharness.alignment.orientation_aligner import apply_smplx_base_rotation

            tpose_spec = apply_smplx_base_rotation(tpose_spec)
            print("[agent] Applied SMPL-X base rotation to tpose_spec")

        if args.world_rot:
            from scipy.spatial.transform import Rotation as R

            from roboharness.alignment.orientation_aligner import parse_world_rotation_arg

            try:
                q_wr = parse_world_rotation_arg(args.world_rot)
            except ValueError as exc:
                print(f"[agent] ERROR: {exc}")
                return
            r_wr = R.from_quat(q_wr, scalar_first=True)
            for name in human_ref:
                pos, quat = human_ref[name]
                new_pos = r_wr.apply(np.asarray(pos))
                new_quat = (r_wr * R.from_quat(np.asarray(quat), scalar_first=True)).as_quat(
                    scalar_first=True
                )
                human_ref[name] = (list(new_pos), list(new_quat))
            wr_angle_deg = float(args.world_rot.split(",")[0])
            print(f"[agent] Applied world_rot {wr_angle_deg}° to human reference")

        direct_patch = compute_direct_patch(
            human_data=human_ref,
            config=config,
            tpose_spec=tpose_spec,
            preserve=set(j.strip() for j in args.preserve.split(",") if j.strip()),
        )
        n_patched = len(direct_patch.get("ik_match_table1", {}))
        print(f"[agent] Computed quaternions for {n_patched} joints")

        new_config = apply_patch(config, direct_patch, mirror="auto")

        # Extract init_qpos from tpose spec (T-pose joint angles)
        qpos_spec = tpose_spec.get("qpos", [])
        if qpos_spec and len(qpos_spec) > 7:
            import mujoco as mj

            xml_path = tpose_spec.get("xml_path", "")
            if xml_path:
                _model = mj.MjModel.from_xml_path(xml_path)
                _init = {}
                for _i in range(_model.njnt):
                    _jname = mj.mj_id2name(_model, mj.mjtObj.mjOBJ_JOINT, _i)
                    if _jname is None or _model.jnt_type[_i] == 0:
                        continue
                    _adr = _model.jnt_qposadr[_i]
                    if _adr >= len(qpos_spec):
                        continue
                    _v = qpos_spec[_adr]
                    if abs(_v) > 0.0001:
                        _init[_jname] = round(float(_v), 6)
                if _init:
                    new_config["init_qpos"] = _init
            if new_config.get("init_qpos"):
                print(f"[agent] Set init_qpos: {list(new_config['init_qpos'].keys())}")

        if args.world_rot:
            from roboharness.alignment.orientation_aligner import parse_world_rotation_arg

            try:
                wr_quat = parse_world_rotation_arg(args.world_rot)
            except ValueError as exc:
                print(f"[agent] ERROR: {exc}")
                return
            new_config["world_rotation"] = wr_quat
            print(
                f"[agent] Set world_rotation: "
                f"[{wr_quat[0]:.4f},{wr_quat[1]:.4f},"
                f"{wr_quat[2]:.4f},{wr_quat[3]:.4f}]"
            )

        with config_path.open("w") as f:
            json.dump(new_config, f, indent=4)
        print(f"[agent] Config written → {config_path}")

        from roboharness.alignment import compute_deviations, total_deviation, worst_k

        print("[agent] Retargeting T-pose with new config...")
        tpose_qpos = _retarget_tpose_qpos(
            args.tpose_src, args.tpose_motion, args.robot, args.tpose_bvh_format
        )
        report_new = compute_deviations(tpose_qpos, tpose_spec["xml_path"], tpose_spec)
        new_total = total_deviation(report_new)
        new_max = worst_k(report_new, 1)[0][1] if report_new else 0.0
        print(f"[agent] Post-solve: total={new_total:.2f}°  max={new_max:.2f}°")

        if new_max < args.tpose_threshold:
            print(f"[agent] PASSED (max < {args.tpose_threshold}°).")
        else:
            print(f"[agent] residual max={new_max:.2f}° — may need manual tuning.")

        print(f"\n[agent] Final config: {config_path}")
        return

    if args.tune_mode == "optimize_scale":
        from roboharness.alignment import (
            compute_position_deviations,
            optimize_scales,
            total_position_deviation,
            worst_k_position,
        )

        if tpose_spec is None or not args.tpose_motion:
            print(
                "[agent] ERROR: --tune_mode optimize_scale requires --tpose_spec + --tpose_motion."
            )
            return

        from general_motion_retargeting import GeneralMotionRetargeting as GMR

        print("[agent] Pre-loading retargeter and MuJoCo model...")

        tpose_frames, actual_h, _ = load_motion(
            args.tpose_src, args.tpose_motion, args.tpose_bvh_format
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

        new_config = apply_patch(config, patch)
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
        return

    prev_total: float | None = None
    prev_pos_total: float | None = None
    for iteration in range(1, args.max_iter + 1):
        print(f"\n--- Iteration {iteration}/{args.max_iter} ---")

        deviation_text: str | None = None
        pos_deviation_text: str | None = None
        cur_total: float | None = None
        cur_max: float | None = None
        cur_pos_total: float | None = None
        cur_pos_max: float | None = None
        if tpose_spec is not None and args.tpose_motion:
            from roboharness.alignment import (
                compute_deviations,
                compute_position_deviations,
                total_deviation,
                total_position_deviation,
                worst_k,
                worst_k_position,
            )

            print("[agent] Retargeting T-pose for numeric gate...")
            tpose_qpos = _retarget_tpose_qpos(
                args.tpose_src,
                args.tpose_motion,
                args.robot,
                args.tpose_bvh_format,
            )
            # Rotation deviations
            report = compute_deviations(tpose_qpos, tpose_spec["xml_path"], tpose_spec)
            cur_total = total_deviation(report)
            cur_max = worst_k(report, 1)[0][1] if report else 0.0
            deviation_text = _format_deviation_report(report, cur_total, cur_max, prev_total)

            # Position deviations (primary for scale tuning)
            pos_report = compute_position_deviations(tpose_qpos, tpose_spec["xml_path"], tpose_spec)
            cur_pos_total = total_position_deviation(pos_report)
            top_pos = worst_k_position(pos_report, 1)
            cur_pos_max = top_pos[0][1] if top_pos else 0.0
            pos_deviation_text = _format_position_report(pos_report, cur_pos_total, cur_pos_max)

            pos_cm = cur_pos_total * 100
            angle_str = f"max_angle={cur_max:.2f}°"
            print(
                f"[agent] position: total={pos_cm:.1f} cm  max={cur_pos_max * 100:.1f} cm  "
                f"({angle_str}, threshold={args.tpose_threshold}°)"
            )
            if args.tune_mode == "quaternion":
                if cur_max < args.tpose_threshold:
                    print(
                        f"[agent] Numeric gate PASSED "
                        f"(max_angle < {args.tpose_threshold}°). Stopping."
                    )
                    break
            else:
                if cur_max < args.tpose_threshold and cur_pos_total < 0.05:
                    print(
                        f"[agent] Numeric gate PASSED "
                        f"(max_angle < {args.tpose_threshold}°, pos_err < 5 cm). Stopping."
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

        print(f"[agent] Querying {args.model} (tune_mode={args.tune_mode})...")
        response = ask_claude(
            client=client,
            model=args.model,
            image_paths=png_paths,
            config=config,
            iteration=iteration,
            client_type=client_type,
            deviation_text=deviation_text,
            pos_deviation_text=pos_deviation_text,
            tune_mode=args.tune_mode,
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
        new_config = apply_patch(config, patch, mirror="auto")
        with config_path.open("w") as f:
            json.dump(new_config, f, indent=4)
        print(f"[agent] Config updated → {config_path}")

        print("[agent] Re-retargeting with updated config...")
        qpos_seq, human_seq = _retarget(
            args.src, args.motion_file, args.robot, args.bvh_format, args.frames
        )

        if tpose_spec is not None and args.tpose_motion and args.regression_tolerance >= 0.0:
            from roboharness.alignment import (
                compute_deviations,
                compute_position_deviations,
                total_deviation,
                total_position_deviation,
            )

            tpose_qpos_post = _retarget_tpose_qpos(
                args.tpose_src, args.tpose_motion, args.robot, args.tpose_bvh_format
            )
            report_post = compute_deviations(tpose_qpos_post, tpose_spec["xml_path"], tpose_spec)
            pos_report_post = compute_position_deviations(
                tpose_qpos_post, tpose_spec["xml_path"], tpose_spec
            )
            new_total = total_deviation(report_post)
            new_pos_total = total_position_deviation(pos_report_post)

            if args.tune_mode == "quaternion":
                cur_val = cur_total if cur_total is not None else 0.0
                delta = new_total - cur_val
                if delta > args.regression_tolerance:
                    print(
                        f"[agent] REGRESSION: post-patch rot={new_total:.2f}° vs "
                        f"pre-patch {cur_val:.2f}° (Δ {delta:+.2f}°, "
                        f"tol {args.regression_tolerance:.2f}°). Reverting."
                    )
                    with config_path.open("w") as f:
                        json.dump(pre_patch_config, f, indent=4)
                    config = pre_patch_config
                    qpos_seq, human_seq = _retarget(
                        args.src, args.motion_file, args.robot, args.bvh_format, args.frames
                    )
                    prev_total = cur_val
                else:
                    print(
                        f"[agent] Regression gate PASS: rot {cur_val:.2f}° → "
                        f"{new_total:.2f}° (Δ {delta:+.2f}°)"
                    )
                    config = new_config
                    prev_total = new_total
            elif cur_pos_total is not None:
                delta = new_pos_total - cur_pos_total
                tol_m = args.regression_tolerance / 100.0
                if delta > tol_m:
                    print(
                        f"[agent] REGRESSION: post-patch pos_err={new_pos_total * 100:.1f} cm vs "
                        f"pre-patch {cur_pos_total * 100:.1f} cm (Δ {delta * 100:+.1f} cm, "
                        f"tol {tol_m * 100:.1f} cm). Reverting."
                    )
                    with config_path.open("w") as f:
                        json.dump(pre_patch_config, f, indent=4)
                    config = pre_patch_config
                    qpos_seq, human_seq = _retarget(
                        args.src, args.motion_file, args.robot, args.bvh_format, args.frames
                    )
                    prev_pos_total = cur_pos_total
                else:
                    rot_delta = new_total - (cur_total or 0)
                    print(
                        f"[agent] Regression gate PASS: pos_err {cur_pos_total * 100:.1f} → "
                        f"{new_pos_total * 100:.1f} cm (Δ {delta * 100:+.1f} cm), "
                        f"rot {cur_total or 0:.1f}° → {new_total:.1f}° (Δ {rot_delta:+.1f}°)"
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
