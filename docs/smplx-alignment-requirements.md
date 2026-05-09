# SMPL-X Alignment Requirements

_Created: 2026-05-07_
_Last updated: 2026-05-09_
_Status: Loader-boundary refactor implemented, pending E2E verification_

This document records the SMPL-X robot alignment fix — template calibration,
world-rotation coordinate conversion, and offset solver — for multi-agent
reference, implementation, and review.

## Monitoring / Review State

flag=1

- `flag=0`: Work is not ready for Codex review.
- `flag=1`: All claimed implementation tasks are complete and ready for Codex review.
- `review_status=pending`: Codex has not reviewed the completed work yet.
- `review_status=approved`: Codex review passed.
- `review_status=changes_requested`: Codex found blockers; opencode should fix them.

Current owner: opencode (GLM-5.1)
Last implementation commit: pending (all changes staged, ready to commit)
Last verification command: `pytest -q` (764 passed, 3 skipped, 91.20% coverage)
Codex review status: pending (6 re-review findings addressed, awaiting 2nd re-review; see smplx-refactoring-plan.md)

## Context

SMPL-X retargeting produced ~180° deviation when setting up robots:

```bash
python scripts/setup_robot.py \
  --robot v11 \
  --src smplx \
  --tpose_motion /home/user2/ACCAD/Male1Walking_c3d/Male1_Cal_stageii.npz \
  --update_scripts
```

Two root causes were identified:

1. **No coordinate frame conversion**: SMPL-X uses Y-up convention; MuJoCo
   uses Z-up. The pipeline had no `world_rotation` to bridge the two frames.
2. **Motion file used as calibration source**: `Male1Walking...npz` is a
   walking sequence, not a canonical zero-pose. Offsets were solved from a
   non-standard human pose.

## Goal

Use a SMPL-X body model asset as the canonical zero-pose source for offset
solving and validation, and introduce `world_rotation` to convert SMPL-X
Y-up human data to MuJoCo Z-up at runtime, keeping the robot root upright.

Requirements:

1. Generate SMPL-X offset frames from body model zero-pose (not motion files).
2. Solve per-joint offsets from the canonical template + robot T-pose spec.
3. Convert Y-up → Z-up via `world_rotation` in IK config at runtime.
4. Keep robot T-pose specs at identity root qpos `[1, 0, 0, 0]`.
5. Validate via template retargeting (no motion file required).

## Design

### Coordinate Conversion (updated 2026-05-09 — loader-boundary refactor)

The SMPL-X to MuJoCo conversion now follows the **BVH pattern**: coordinate
conversion happens at the **loader boundary**, not at runtime via world_rotation.

```
Old: SMPLX loader → Y-up frames → runtime world_rotation=[0.5,0.5,0.5,0.5] → Z-up
New: SMPLX loader → Y-up → smpl_to_mujoco_frame() in loader → Z-up frames → runtime
```

The conversion is still the 120-degree rotation about (1,1,1)/sqrt(3):

```
SMPL-X:  Y-up, X=left,  Z=forward
MuJoCo:  Z-up, X=forward, Y=left   (post-smpl_to_mujoco_frame convention)

Mapping: X→Y, Y→Z, Z→X
```

Single source of truth: `src/roboharness/alignment/smplx_coordinate.py`

```python
SMPL_TO_MUJOCO_QUAT: list[float] = [0.5, 0.5, 0.5, 0.5]  # runtime form
```

**Key changes from old architecture:**

1. `smpl_to_mujoco_frame()` is called in `load_smplx()` and
   `load_smplx_template_tpose()` — not in the solver or at runtime.
2. `compute_world_rotation("smplx")` computes R_mat from robot geometry
   (like BVH), using `forward = cross(left, up)` for the SMPLX post-conversion
   convention (X=forward, Y=left, Z=up).  Returns `None` when robot geometry
   matches SMPLX convention.
3. The solver applies any existing `config["world_rotation"]` to the template
   frame before computing offsets, matching the GMR runtime order.
4. The solver does NOT inject `world_rotation` — that is `compute_world_rotation()`'s
   job via `config_gen.py`.
5. `smpl_to_mujoco_world_rotation()` is deprecated.
6. Stale configs with legacy `world_rotation = [0.5, 0.5, 0.5, 0.5]` emit a
   warning via `_check_stale_smplx_config()`.

**Orientation invariant:** After `smpl_to_mujoco_frame()`, zero-pose body joint
orientations are `SMPL_TO_MUJOCO_QUAT = [0.5, 0.5, 0.5, 0.5]`, NOT identity.
The offset for an identity robot T-pose pelvis is therefore
`SMPLX_BASE_ROTATION_QUAT = [0.5, -0.5, -0.5, -0.5]`.

**Direct GMR callers** (scripts importing `general_motion_retargeting.utils.smpl`
directly) are out of scope — they continue to receive Y-up frames and must
handle conversion themselves or use legacy configs.

### GMR Runtime Pipeline (unchanged)

Order of operations in `GMR/general_motion_retargeting/motion_retarget.py:176-211`:

```
1. scale_human_data       — scale positions per bone ratio
2. apply_world_rotation   — apply geometry-based alignment (may be identity)
3. offset_human_data      — apply per-joint rotation + position offsets
4. apply_ground_offset    — adjust vertical position
5. Seed qpos[0:3] and qpos[3:7] from human_root_name pos + quaternion
```

### Offset Math (Pelvis)

For the pelvis at SMPL-X zero-pose (quaternion = I):

```
Step 1: pelvis_after_wr = SMPL_TO_MUJOCO_QUAT * I = SMPL_TO_MUJOCO_QUAT
Step 2: pelvis_after_offset = SMPL_TO_MUJOCO_QUAT * pelvis_rot_offset
Step 3: For root seed = I: pelvis_rot_offset = SMPL_TO_MUJOCO_QUAT.inv()
        = SMPLX_BASE_ROTATION_QUAT = [0.5, -0.5, -0.5, -0.5]
```

Solver formula (`smplx_offset_solver.py:85`):

```python
r_offset = r_human.inv() * r_target
```

Where:

- `r_human` = SMPL_TO_MUJOCO_QUAT (Y→Z converted template pelvis)
- `r_target` = I (robot T-pose spec pelvis, upright)
- `r_offset` = SMPL_TO_MUJOCO_QUAT.inv() = `[0.5, -0.5, -0.5, -0.5]`

**Verified**: `SMPL_TO_MUJOCO_QUAT * SMPLX_BASE_ROTATION_QUAT = I` ✓

### Verified IK Config Values

`smplx_to_unitree_g1.json`:

| Key | Value | Meaning |
|-----|-------|---------|
| `world_rotation` | `[0.5, 0.5, 0.5, 0.5]` | SMPL_TO_MUJOCO_QUAT, Y-up → Z-up |
| pelvis rot_offset | `[0.5, -0.5, -0.5, -0.5]` | SMPLX_BASE_ROTATION_QUAT, cancels wr |

## Required Behavior

### Task A: Add Canonical SMPL-X Calibration Frame Generation

Implement a helper that creates a GMR-compatible SMPL-X frame from the body
model zero pose.

Suggested API:

```python
load_smplx_template_tpose(
    body_model_path: Path,
    gender: str = "male",
    betas: np.ndarray | None = None,
) -> tuple[dict, float]
```

Return:

- `frame`: dict compatible with GMR retarget input:
  `{joint_name: (position, quat_wxyz)}`
- `human_height`: deterministic height estimate for scaling.

Implementation notes:

- Accept either a directory or a file path and resolve the underlying body
  model asset name-agnostically.
- Prefer using `smplx.create(...)` in the form that matches the resolved
  asset location.
- Use `return_full_pose=True`.
- Reuse the joint orientation propagation pattern from
  `GMR/general_motion_retargeting/utils/smpl.py`.
- Joint names must match existing SMPL-X config names, e.g. `pelvis`,
  `spine3`, `left_hip`, `left_shoulder`, `right_wrist`.
- Quaternions must be scalar-first `[w, x, y, z]`.

Acceptance criteria:

- Unit test confirms returned frame contains at least:
  `pelvis`, `spine3`, `left_hip`, `right_hip`, `left_shoulder`,
  `right_shoulder`, `left_foot`, `right_foot`.
- Unit test confirms all returned quaternions are normalized.
- Unit test confirms root/pelvis orientation is identity in the synthetic
  template frame before any robot-specific SMPL-X base rotation handling.
- Unit test copies a real SMPL-X body model `.npz` to an arbitrary filename and
  confirms `load_smplx_template_tpose()` can load that renamed asset. This must
  exercise the full loader, not only the path resolver.

### Task B: Solve SMPL-X Offsets From Template Frame And Robot T-Pose Spec

Replace or extend `GMR/scripts/compute_smplx_tpose_offsets.py` so SMPL-X
offsets are solved from the canonical template frame and the staged robot
T-pose spec, not by assuming every human joint quaternion is identity.

Required equation:

```text
offset = inverse(human_joint_world_quat) * robot_link_expected_world_quat
```

Where:

- `human_joint_world_quat` comes from the synthetic SMPL-X template frame
  (Y-up, then Y→Z converted inside the solver).
- `robot_link_expected_world_quat` comes from `specs/tpose/<robot>.json`
  link `R`, converted to scalar-first quaternion.

This equation should be used for every robot link listed in the SMPL-X IK
config's `ik_match_table1` / `ik_match_table2` when the mapped SMPL-X joint is
available in the template frame and the robot link exists in the T-pose spec.

Acceptance criteria:

- Running the solver for `v11` writes updated offsets into
  `GMR/general_motion_retargeting/ik_configs/smplx_to_v11.json`.
- Both IK tables receive the same solved quaternion for the same robot link.
- The generated config keeps:
  - `world_rotation` = `[0.5, 0.5, 0.5, 0.5]` (SMPL-X Y-up → Z-up),
  - `init_qpos` for robot T-pose joints where required,
  - `robot_root_name` and `human_root_name` as configured.

> **Note**: Task B's original acceptance criteria said "no top-level
> `world_rotation` for SMPL-X". This is **superseded** by Task E / the
> world-rotation fix — SMPL-X configs now **require** `world_rotation`.

### Task C: Wire `setup_robot.py` To Use Template Calibration For SMPL-X

For `--src smplx`, setup should default to template calibration when solving
offsets and validating the result.

Required CLI behavior:

- Add `--smplx_template_model /path/to/smplx_body_model_asset`.
- Accept either the body-model directory or the explicit model asset path.
- Default to the existing GMR body model location when it exists, but do not
  hardcode machine-specific absolute paths.
- Do not require `--tpose_motion` for SMPL-X offset solving if template
  calibration is available.
- Keep `--tpose_motion` usable for normal motion retargeting workflows, but do
  not pass a walking `.npz` into T-pose validation as the calibration source.

Acceptance criteria:

- This command can solve SMPL-X offsets for `v11` without a motion `.npz`:

```bash
python scripts/setup_robot.py \
  --robot v11 \
  --src smplx \
  --smplx_template_model /path/to/smplx_body_model_asset \
  --update_scripts
```

provided the SMPL-X body model asset exists at the resolved location.

### Task D: Add A Synthetic SMPL-X Template Validation Path

`examples/gmr_tpose_validate.py` currently validates by loading a motion file
and retargeting frame 0. Add a template validation mode for SMPL-X.

Required CLI behavior:

```bash
python examples/gmr_tpose_validate.py \
  --robot v11 \
  --src smplx \
  --use_smplx_template \
  --smplx_template_model /path/to/smplx_body_model_asset \
  --spec specs/tpose/v11.json
```

Behavior:

- Generate the synthetic SMPL-X template frame.
- Retarget that frame through GMR.
- Compare resulting robot qpos against `specs/tpose/v11.json`.
- Print the same `total_deviation`, `max_angle`, and `worst_k` report.
- If `--smplx_template_model` is supplied, use the same resolved model path as
  setup/solve.
- If `--use_smplx_template` is set with a non-SMPLX `--src`, fail fast.

Acceptance criteria:

- The template validation path does not require `--tpose_motion`.
- The old motion-based validation path still works for BVH/FBX and for explicit
  SMPL-X motion debugging.
- Documentation and error messages clearly state that `.npz` motion files are
  motion inputs, not the canonical calibration source.

### Task E: Update SMPL-X Coordinate Policy

The SMPL-X coordinate policy has been **updated** by the world-rotation fix.
The new policy is:

- SMPL-X IK configs **now use** top-level `world_rotation` (the SMPL-X base
  runtime rotation only; do not compose `R_mat @ base_mat`).
- SMPL-X robot T-pose specs stage root qpos as **identity** `[1, 0, 0, 0]`
  (robot upright, no pre-rotation).
- `compute_world_rotation` returns the SMPL-X base runtime rotation for SMPL-X
  (was `None`).
- `smplx_offset_solver` preserves `world_rotation` in config (was deleting it).
- Template-frame quaternions are transformed to Z-up inside the solver before
  computing offsets; the solved offsets bridge Z-up human → Z-up robot.

Acceptance criteria:

- Existing tests in `tests/alignment/test_smplx_tpose_coordinate_fix.py`
  continue to pass.
- SMPL-X IK config `smplx_to_v11.json` **contains** `world_rotation` key.
- T-pose spec root qpos is identity `[1, 0, 0, 0]` for SMPL-X.

## Remaining Verification

### Task V1: End-to-End SMPL-X Motion Retargeting

Retarget an actual SMPL-X motion file and confirm the robot stands upright in
MuJoCo. Template validation alone cannot detect axis errors because all human
joints are identity at zero-pose (offsets absorb any wr value).

```bash
python scripts/retarget_motion.py \
    --robot unitree_g1 \
    --src smplx \
    --motion /path/to/smplx_motion.npz \
    --output /tmp/retarget_test/
```

Acceptance criteria:

- Robot root quaternion at frame 0 is within 5° of identity.
- Robot does not exhibit systematic tilt or inversion.
- Joint orientations match expected T-pose alignment.

### Task V2: Multi-Robot SMPL-X Config Generation

Generate SMPL-X IK configs for additional robots and verify correctness.

```bash
for robot in v11 k1 kuavo h1_2; do
    python scripts/setup_robot.py \
        --robot $robot \
        --src smplx \
        --auto_register \
        --update_scripts
done
```

Acceptance criteria:

- Each generated config has `world_rotation = [0.5, 0.5, 0.5, 0.5]`.
- Each config's pelvis/root rot_offset = `[0.5, -0.5, -0.5, -0.5]`.
- Template validation passes (< 5° total deviation) for each robot.

## Non-Requirements

- Do not solve offsets from motion `.npz` files (e.g. `Male1Walking...npz`).
- Do not change `specs/tpose/<robot>.json` merely to match a walking frame.
- Do not relax the 5 degree validation threshold.
- Do not modify the BVH retargeting pipeline (reference implementation).
- Do not change the solver formula `r_offset = r_human.inv() * r_target`.
- Do not add `R_mat @ base_mat` composition (axis-swap bug — see below).

> **Superseded non-requirements**:
> - ~~Do not add top-level `world_rotation` to SMPL-X configs~~ → SMPL-X configs now include `world_rotation`.
> - ~~SMPL-X robot root qpos should be `[0.5, -0.5, -0.5, -0.5]`~~ → Now identity `[1, 0, 0, 0]`.

## Known Issues / Resolved

### R_mat Axis Swap Bug (Resolved)

Previous iterations composed `R_mat @ base_mat` for SMPL-X world_rotation.
`R_mat` is computed against the BVH canonical layout (X=left, Y=forward), but
SMPL-X after `base_mat` has X=forward, Y=left — the left/forward axes are
swapped. Fix: return `base_mat` only (`SMPL_TO_MUJOCO_QUAT`), no R_mat.

### Offset Direction Confusion (Resolved)

An earlier analysis claimed the pelvis offset should be `[0.5, 0.5, 0.5, 0.5]`
(= `base_mat.inv()`). This is incorrect. The solver computes:

```
r_offset = r_human.inv() * r_target
         = SMPL_TO_MUJOCO_QUAT.inv() * I
         = [0.5, -0.5, -0.5, -0.5]
```

At runtime: `wr * offset = SMPL_TO_MUJOCO_QUAT * [0.5, -0.5, -0.5, -0.5] = I`.

The offset `[0.5, -0.5, -0.5, -0.5]` (= `SMPLX_BASE_ROTATION_QUAT`) is correct.

## Key Files Reference

| File | Role |
|------|------|
| `src/roboharness/alignment/smplx_coordinate.py` | Single source of truth for Y→Z conversion |
| `src/roboharness/alignment/smplx_offset_solver.py` | Template-based offset solver pipeline |
| `src/roboharness/alignment/smplx_template.py` | Body model zero-pose frame generation |
| `src/roboharness/alignment/smplx_scale.py` | Human bone scaling |
| `src/roboharness/alignment/orientation_aligner.py` | `compute_world_rotation()` per format |
| `src/roboharness/_math_utils.py` | `SMPLX_BASE_ROTATION_QUAT` (legacy), quaternion utils |
| `scripts/setup_robot.py` | One-command setup orchestration |
| `scripts/stage_tpose.py` | Robot T-pose spec staging |
| `examples/gmr_tpose_validate.py` | T-pose validation (motion + template modes) |
| `examples/gmr_alignment_agent.py` | Interactive alignment agent |
| `GMR/general_motion_retargeting/motion_retarget.py` | GMR runtime: scale → wr → offset → ground → seed root |
| `GMR/general_motion_retargeting/ik_configs/smplx_to_unitree_g1.json` | Reference SMPL-X IK config |
| `tests/alignment/test_smplx_template_calibration.py` | 25+ tests for template/solver/CLI/validator |
| `tests/alignment/test_smplx_tpose_coordinate_fix.py` | 10 tests for coordinate policy |

## Review Checklist

- Template calibration uses a SMPL-X body model asset, not a motion sequence.
- Offset equation is `human.inv() * robot_expected`, with scalar-first
  quaternion IO.
- `setup_robot.py --src smplx` can solve without a `--tpose_motion` when the
  template model exists.
- Template validation path exists and does not require a motion file.
- Walking `.npz` files are not used as calibration sources by default.
- SMPL-X configs now contain top-level `world_rotation`.
- T-pose spec root qpos is identity `[1, 0, 0, 0]` for SMPL-X (robot upright).
- Tests cover the new helper, solver behavior, and CLI surface.

## Implementation Record

Implemented by: opencode (GLM-5.1) on 2026-05-07, verified on 2026-05-09.

### Files Created

| File | Purpose |
|------|---------|
| `src/roboharness/alignment/smplx_coordinate.py` | Single source of truth for Y→Z conversion. `SMPL_TO_MUJOCO_QUAT`, `smpl_to_mujoco_frame()`, `smpl_to_mujoco_world_rotation()`. |
| `src/roboharness/alignment/smplx_template.py` | Task A — `load_smplx_template_tpose()` generates a GMR-compatible SMPL-X frame from the body model zero-pose. Uses hierarchical orientation propagation matching `GMR/utils/smpl.py`. Returns `{joint_name: (position, quat_wxyz)}` + `human_height`. All body-joint quaternions are identity at zero-pose. `resolve_body_model_path()` for name-agnostic model discovery. |
| `src/roboharness/alignment/smplx_offset_solver.py` | Task B — `solve_smplx_offsets_from_template()` pipeline: load → scale → Y→Z convert → compute offsets. `compute_joint_offsets()` is the pure offset computation. Ensures both `ik_match_table1` and `ik_match_table2` receive the same solved quaternion for the same robot link. Preserves `world_rotation` in config. |
| `src/roboharness/alignment/smplx_scale.py` | `apply_human_scale()` scales positions per bone scale factors. |
| `tests/alignment/test_smplx_template_calibration.py` | Tasks A-E — 25+ tests covering frame generation, offset solving, CLI wiring, validator template mode, and coordinate policy preservation. Skipped via `pytestmark` when body model is absent. |
| `tests/alignment/test_smplx_tpose_coordinate_fix.py` | Tasks A-E — 10 tests: stage_tpose identity root, validator identity check, compute_world_rotation SMPL-X axis mapping, BVH non-None check. |

### Files Modified

| File | Change |
|------|--------|
| `src/roboharness/alignment/orientation_aligner.py` | `compute_world_rotation("smplx")` returns `smpl_to_mujoco_world_rotation()` instead of `None`. `apply_smplx_base_rotation()` deprecated. |
| `scripts/stage_tpose.py` | Removed `SMPLX_BASE_ROTATION_QUAT` from robot root qpos. SMPL-X T-pose specs now use identity root. |
| `examples/gmr_tpose_validate.py` | Task D — Added `--use_smplx_template` and `--smplx_template_model` CLI args. SMPL-X root diagnostic checks identity. `--tpose_motion` is now optional (no longer `required=True`). New `_retarget_template_frame()` function. |
| `examples/gmr_alignment_agent.py` | `--solve_mode --src smplx` auto-detects template calibration. `--motion_file` optional for SMPL-X (Phase A retargeting skipped when template is available). |
| `scripts/setup_robot.py` | Task C — Added `--smplx_template_model` CLI arg. SMPL-X offset solving via template, no `--tpose_motion` required. Validation uses template when available. |
| `examples/_gmr_shared.py` | CAM_AZIMUTHS fixed: `front=0, side=90, back=180`. |

### Key Design Decisions

1. **Name-agnostic body model resolution**: `resolve_body_model_path()` accepts
   a directory, a `smplx/` subfolder, or a `.npz` file and resolves to the
   directory expected by `smplx.create()`. When passed `None`, it discovers
   the GMR root via `find_gmr_root()` and uses `GMR_ROOT/assets/body_models`.
   No hardcoded machine-specific absolute paths exist in source code.

2. **Fallback to GMR script**: `_solve_smplx_offsets()` still falls back to
   `GMR/scripts/compute_smplx_tpose_offsets.py --generate` when the IK config
   file doesn't exist yet (first-time generation).

3. **Zero-pose quaternions**: All 22 body joints (pelvis through wrists) have
   identity orientation at zero-pose. Hand joints (indices 25–54) have
   non-identity rest poses due to SMPL-X hand PCA — these are not used in IK
   matching so they don't affect offset solving.

4. **Template vs motion validation**: When `smplx_template_available` is true
   and no `--tpose_motion` is provided, the validation step passes
   `--use_smplx_template` to `gmr_tpose_validate.py` instead of
   `--tpose_motion`. Both paths produce the same deviation report format.

5. **Coordinate policy**: The solver **preserves** `world_rotation` in the
   config. Template-frame quaternions are transformed to Z-up inside the solver
   before computing offsets. T-pose spec root qpos is identity `[1, 0, 0, 0]`
   for SMPL-X. The frame conversion is handled by `world_rotation` at runtime.

### Verification Baseline (updated 2026-05-09)

```text
$ pytest -q
764 passed, 3 skipped in 18.95s
Coverage: 91.20% (>=90% threshold)

$ ruff check .
All checks passed!

$ mypy src/
Success: no issues found in 54 source files
```

## Modification Log

| Date | Agent | Action | Notes |
|------|-------|--------|-------|
| 2026-05-07 | Codex | Requirements capture | Created original document from code/design review findings. |
| 2026-05-07 | opencode (GLM-5.1) | Implementation | Tasks A-E: template calibration, solver, CLI wiring, validation, coordinate policy. 3 files created, 2 files modified. 704 tests. |
| 2026-05-07 | Codex | Review | 3 issues: path passthrough, source gating, directory contract. |
| 2026-05-07 | opencode (GLM-5.1) | Rev 1 | Fix path passthrough, source-gate template to `--src smplx`, clarify directory contract. 708 tests. |
| 2026-05-07 | Codex | Review | Name-agnostic resolution required. |
| 2026-05-07 | opencode (GLM-5.1) | Rev 2 | `resolve_body_model_path()`, removed all hardcoded `/home/` paths. 715 tests, 90.85% coverage. |
| 2026-05-08 | opencode (GLM-5.1) | Rev 3 | Fixed directory contract — always returns parent dir so `smplx.create()` appends `smplx/`. 716 tests. |
| 2026-05-08 | opencode (GLM-5.1) | Rev 4 | True name-agnostic loading — bypasses `smplx.create()` for `.npz` files. 717 tests. |
| 2026-05-08 | opencode (GLM-5.1) | Rev 5 | Runtime fixes: `num_betas` from model, agent solve_mode template calibration. 717 tests. |
| 2026-05-08 | opencode (GLM-5.1) | Rev 6 | `--motion_file` optional for SMPL-X, `tpose_spec_path` resolution fix. 723 tests. |
| 2026-05-09 | opencode (GLM-5.1) | Status verification | Math verified: `SMPL_TO_MUJOCO_QUAT * SMPLX_BASE_ROTATION_QUAT = I`. IK config correct. 760 tests, 91.18% coverage. |
| 2026-05-09 | opencode (GLM-5.1) | Docs merge | Merged `smplx-world-rotation-fix-requirements.md` into this document. Added Design section, remaining V1/V2 tasks, known issues. Deleted the separate world-rotation doc. |
| 2026-05-09 | Codex | Review | Refactoring plan review: 6 findings. Improper frame construction, solver must apply wr before offsets, orientation invariant contradiction, stale config risk, incomplete downstream scope, docs encode old policy. |
| 2026-05-09 | opencode (GLM-5.1) | Loader-boundary refactor | Addressed all 6 Codex findings. SMPLX conversion now at loader boundary (BVH-style). `compute_world_rotation("smplx")` uses geometry. Solver applies wr before offsets. Stale config warning. 764 tests, 91.20% coverage. See `docs/smplx-refactoring-plan.md` for full details. |
| 2026-05-09 | Codex | Re-review | 6 findings: runtime stale-config not covered, solver warning insufficient, Section 2.3 contradiction, downstream docs stale, orientation_aligner docstring obsolete, weak axis-mapping tests. |
| 2026-05-09 | opencode (GLM-5.1) | Re-review fixes | All 6 re-review findings addressed. `validate_smplx_runtime_config()` (fail-fast) at all SMPL-X entry points. Section 2.3, SOP, user guide, docstrings corrected. Axis-mapping test assertions added. 770 tests, 91.26% coverage. |
