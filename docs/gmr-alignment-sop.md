# GMR T-pose Alignment — Standard Operating Procedure

Audience: any agent (or human) tuning a GMR IK config so a retargeted humanoid
matches the source motion. Read this **before** editing `bvh_to_<robot>.json`
quaternion offsets. Following the procedure below, 90° misalignments are a
single-iteration fix; without it, they become multi-iteration oscillation.

---

## 1. Why pure visual iteration fails

`examples/gmr_alignment_agent.py` sends rendered PNGs to a VLM and asks "is
this aligned?". That loop stalls for three reasons:

| Limitation | Consequence |
|---|---|
| VLM cannot measure angles | 85° looks like 90° in a 640x480 render |
| No before/after signal | An improving patch and a worsening patch look equally "close" |
| No ground truth | "Aligned" has no numeric definition — only a vibe |

The fix is not a better prompt. The fix is **a numeric contract** the agent
measures against: `roboharness.alignment.compute_deviations`. Use the metric
as your eyes; use vision only for things the metric cannot see (geometry
intersections, limb orientation that doesn't affect `xmat`, cosmetic issues).

---

## 2. Geometric definition of "T-pose aligned"

A robot is at T-pose when, with feet flat on the ground:

```
              +Z  (up)
               |
      +--------+--------+
      |        |        |        arm_L points along -Y (world)
      |        |        |        arm_R points along +Y (world)
   arm_L     torso    arm_R      spine/torso Z-axis = world +Z
               |                 pelvis Z-axis      = world +Z
              leg                palms face down (forearm Y = world -Z)
               |                 feet forward       = world +X
              foot
                                 coordinate frame: +X forward, +Y left, +Z up
```

The T-pose spec (`specs/tpose/<robot>.json`) records, for every body named in
the IK config's `ik_match_table1` / `table2`, the **world-frame 3x3 rotation
matrix** `R_expected` of that body when the robot is physically staged at
T-pose. Alignment is defined by the residual rotation:

    R_err = R_actual @ R_expected.T

Its axis-angle form is the per-link deviation. Angle **is** the alignment
error — a scalar, not a judgement. `angle_deg < 5°` per link is treated as
aligned; `< 1°` is excellent. Axis tells you *which* rotation would fix it.

The canonical axis-angle → quaternion converter lives in
`roboharness._math_utils.axis_angle_to_quat(axis, angle_deg)`. Use it instead
of hand-writing `cos(half) + axis*sin(half)` in agent code.

---

## 3. Authoring the spec (one-time per robot)

```bash
# Option A: interactive viewer. Drag joints until visibly T-pose, press Enter.
python scripts/stage_tpose.py --robot unitree_g1 --interactive \
    --output_dir specs/tpose/

# Option B: headless from a known qpos.
python scripts/stage_tpose.py --robot unitree_g1 \
    --qpos "0 0 0.793  1 0 0 0  ..." \
    --output_dir specs/tpose/
```

Output is two artifacts that ship together as the contract:

1. `specs/tpose/<robot>.json` — the numeric spec. Human-reviewed once, then
   versioned. Fields: `robot`, `xml_path`, `qpos`, `links: {name: {pos, R}}`.
2. `specs/tpose/<robot>_{front,side,back}.png` — reference renders, used by
   the agent as a visual diff target.

**Reviewing the spec**: open the three PNGs. Check arms horizontal, palms
down, legs straight, torso vertical. If the render is wrong, the spec is
wrong — re-stage before committing. Ship no spec whose reference PNG you
have not eyeballed.

---

## 4. Reading the deviation report

```python
from roboharness.alignment import load_tpose_spec, compute_deviations, worst_k

spec = load_tpose_spec("specs/tpose/unitree_g1.json")
report = compute_deviations(candidate_qpos, xml_path, spec)
# {'left_shoulder_yaw_link': {'axis': [0, 1, 0], 'angle_deg': 87.3}, ...}
print(worst_k(report, k=5))
```

Interpret each entry as: "this link's world frame is rotated by `angle_deg`
around `axis` relative to where T-pose says it should be". Both axis and
angle are in world frame.

**Thresholds**:

| angle_deg | Meaning |
|---|---|
| `< 1°` | Excellent, leave alone |
| `1°–5°` | Tolerable; chasing it often causes regressions elsewhere |
| `5°–30°` | Wrong, but not catastrophic — likely an IK solver tolerance issue |
| `30°–60°` | Axis confusion; one of the coordinate components is off |
| `60°–120°` | **Cardinal rotation missing** — snap to 90° along the reported axis |
| `> 120°` | Likely a 180° flip (sign error on quaternion offset) |

---

## 5. Fix workflow (phased convergence)

> Minimum viable Phase 1 only ships the metric. The phased patch policy below
> is Phase 2 — an agent that reads this doc should already follow it manually.

### Phase A — cardinal snap (iterations 1-2)

For each link where `angle_deg > 45°`:

1. Read its `axis`. Round to the nearest basis vector: `[±1,0,0]`, `[0,±1,0]`,
   `[0,0,±1]`. If the axis is diagonal (no component dominates), do **not**
   patch yet — reduce to a single axis by first fixing links whose axis *is*
   cardinal.
2. Choose a correction quaternion for the closest multiple of 90°:
   - `90°` about axis `a`: `[cos(45°), a_x*sin(45°), a_y*sin(45°), a_z*sin(45°)]`
   - `180°` about axis `a`: `[0, a_x, a_y, a_z]`
   - `270°` = `90°` the other way.

   The function `roboharness._math_utils.axis_angle_to_quat(axis, angle_deg)`
   produces the same quaternion. Use it in code; the formula above is for
   mental arithmetic when reading deviation reports.
3. Apply as `"mode": "mul"` to **both** `ik_match_table1` and
   `ik_match_table2` for the same joint — see §7 on table coupling.
4. Re-run retargeting, recompute the report. If `total_deviation` went up,
   revert the patch and try the opposite sign on the axis.

### Phase B — fine alignment (iteration 3+)

Only after `total_deviation < 30°` total and no link exceeds `45°`:

1. Address residual small-angle errors individually.
2. Use non-cardinal quaternions, but still one axis at a time per iteration.
3. Stop when `max(angle_deg) < 5°` or when two consecutive iterations do not
   reduce `total_deviation` by at least 1°.

### Regression gate

Every iteration: if `total_deviation(new) > total_deviation(old)`, the patch
is **wrong**. Revert it. Do not attempt to "compensate" with another patch —
compounding errors are how loops get stuck.

---

## 6. Quaternion pitfalls

These bite agents and humans equally.

**Scalar-first convention**: `[w, x, y, z]`, same as `mj_ref_quat`. The IK
config JSON uses this. If you see `[x, y, z, w]` somewhere, it is not from
this codebase — do not feed it into `apply_patch`.

**Double cover**: `q` and `-q` represent the same rotation. When comparing
two quaternions for equality, normalise by forcing `w >= 0` first, else
"different" patches may be identical.

**Hemisphere consistency across a motion**: consecutive-frame quaternions in
a motion track should share a hemisphere (dot product > 0); a sign flip
between frames produces a fake 360° spin on playback. The retargeter handles
this at load time — do not hand-edit individual frames.

**Axis mixing in mul vs set**: `"mode": "mul"` multiplies the patch
quaternion **on the left** of the existing offset (`q_new = q_patch @ q_old`
in Hamilton convention). `"mode": "set"` replaces outright. Use `mul` for
incremental corrections; use `set` only when starting from a known-good
baseline.

**Left-right symmetry**: a fix for `left_shoulder_yaw_link` does **not**
auto-apply to `right_shoulder_yaw_link`. If both sides are wrong symmetrically
(arms point backward, not just left arm), patch both — but mirror the axis
along the side-dependent component (usually Y for a +X-forward robot).

---

## 7. IK config table coupling — non-negotiable

Every joint in `ik_match_table1` has a twin entry in `ik_match_table2` with
the **same quaternion offset** and looser weights. They are two IK passes
over the same joint, not two different joints. **Patches must mirror.**

```json
// WRONG — table1 patched, table2 left stale
"ik_match_table1": {"left_elbow_link": [..., [0.707, 0.707, 0, 0]]},
"ik_match_table2": {"left_elbow_link": [..., [1, 0, 0, 0]]}

// CORRECT
"ik_match_table1": {"left_elbow_link": [..., [0.707, 0.707, 0, 0]]},
"ik_match_table2": {"left_elbow_link": [..., [0.707, 0.707, 0, 0]]}
```

The ``apply_patch`` in ``roboharness.alignment.patch`` auto-mirrors
single-table patches when called with ``mirror="auto"`` (the default).
Use ``mirror="strict"`` to raise an error if a patch omits one table.
Agents should still prefer symmetric explicit patches when generating
human-readable diffs. When using ``mirror="off"`` (legacy), the tables
can drift — avoid it.

---

## 8. Checklist before asking the VLM

Agents have historically skipped the metric and gone straight to "send images
and hope". Do not. Run this checklist every iteration:

- [ ] Spec exists at `specs/tpose/<robot>.json` and reference PNGs exist.
- [ ] `compute_deviations` ran on the candidate qpos; report in hand.
- [ ] `worst_k(report, 5)` identified; these are the VLM's focus.
- [ ] `total_deviation` from last iteration recorded for regression check.
- [ ] Patches will target `table1` + `table2` symmetrically.
- [ ] Phase A (cardinal) if any link > 45°; Phase B only after.

The VLM's job is *interpreting the report* and looking at the image for
cosmetic issues the report cannot catch — not re-discovering misalignment
the metric already found.

---

## 9. Known failure modes

| Symptom | Likely cause | Check |
|---|---|---|---|
| `total_deviation` oscillates between two values | Patch and its inverse applied alternately | Log the last 3 patch quaternions; if they include `q` and `q.conjugate`, freeze the loop |
| One link always reports 90° no matter what | Missing `table2` mirror — IK pulls joint back | Diff `table1[joint][4]` vs `table2[joint][4]` |
| All links report the same 90° rotation | `world_rotation` needed at top level, not per joint | Patch `world_rotation`, zero out per-joint offsets |
| All links report ~90-120° with SMPL-X source | SMPL-X does not use `world_rotation` — see §10 | Ensure `world_rotation` key is absent from `smplx_to_*.json`; the IK solver's root free joint handles global orientation |
| Non-arm links drift after an arm patch | Root-relative chain propagation (see `test_single_joint_perturbation_localizes`) | Expected: a shoulder rotation moves the whole arm-hand chain in the report |
| Identity check fails (`spec.qpos` replayed gives non-zero deviation) | Spec XML differs from `--xml` used at runtime | Confirm `spec['xml_path']` is what you loaded |

---

## 10. Format-specific world_rotation handling

`world_rotation` 在 IK config JSON 顶层将一个全局四元数旋转应用于所有人体骨骼的位置和朝向。
其作用是：将人体数据从人体惯例的坐标系粗略旋转到机器人惯例的坐标系，使 IK 求解器
在根关节上只需要极小调整。

### Auto-detection (`orientation_aligner.py`)

`setup_robot.py --auto_register` 会通过解析机器人 XML 的默认姿态 body 几何位置自动推算
`world_rotation`。检测基于 hip → spine（up 方向）和 left_hip → right_hip（lateral 方向）
推算机器人的本地坐标轴，然后与人体惯例对齐。

### Per-format policy

| 格式 | world_rotation 策略 | 原因 |
|------|-------------------|------|
| **BVH / FBX** | 自动检测并写入 config | BVH post-loader 惯例 `(X=left, Y=forward, Z=up)`，可与 robot 坐标轴通过纯旋转对齐 |
| **SMPL-X** | 不写入（config 中无此 key） | SMPL-X 惯例 `(X=right, Y=up, Z=forward)` 与 robot `(X=forward, Y=left, Z=up)` 左右手性相反，纯旋转无法对齐三个轴；IK solver 的 root free joint（6-DoF）在求解时自动处理全局朝向 |

SMPL-X 源格式的 T-pose spec 有专用辅助函数：
`from roboharness.alignment.orientation_aligner import apply_smplx_base_rotation`。
该函数将 `_math_utils.SMPLX_BASE_ROTATION_QUAT` 常数应用于 spec 中所有 `R` 矩阵。

### 手动覆盖

`gmr_alignment_agent.py --world_rot "angle,ax,ay,az"` 会覆盖自动检测值。
格式为空格分隔的角度（度）和旋转轴分量。

---

## 11. Next steps (Phase 2)

Once an agent consumes this SOP and the metric module proves it converges
manually, the natural follow-up is to wire `compute_deviations` directly
into `gmr_alignment_agent.py`:

1. Load spec + reference PNGs at agent startup.
2. Include `prev_report`, `current_report`, `worst_k(current, 5)` in each
   VLM prompt.
3. Enforce Phase A cardinal snapping in `apply_patch` for iterations 1-2.
4. Reject patches that increase `total_deviation` automatically.
5. Exit when `max(angle_deg) < 5°` instead of waiting for VLM "ok".

That work is out of scope for the Phase 1 drop this SOP ships with.
