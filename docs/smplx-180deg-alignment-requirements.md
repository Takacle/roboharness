# SMPL-X Canonical T-Pose Alignment Requirements

_Created: 2026-05-07_
_Status: Approved by Codex_

This document records the requested fix for SMPL-X robot alignment and is
intended for multiple agents to pick up, implement, and review.

## Monitoring / Review State

flag=1

- `flag=0`: Work is not ready for Codex review.
- `flag=1`: All claimed implementation tasks are complete and ready for Codex review.
- `review_status=pending`: Codex has not reviewed the completed work yet.
- `review_status=approved`: Codex review passed.
- `review_status=changes_requested`: Codex found blockers; opencode should fix them.

Current owner: opencode (GLM-5.1)
Last implementation commit: TBD
Last verification command: `pytest -q` (723 passed, 3 skipped, 90.89% coverage)
Codex review status: pending
Codex review feedback: (awaiting review of Rev 6)

## Context

This document defines the requested fix for SMPL-X robot alignment after seeing
approximately 180 degrees of deviation when setting up `v11` with:

```bash
python scripts/setup_robot.py \
  --robot v11 \
  --src smplx \
  --tpose_motion /home/user2/ACCAD/Male1Walking_c3d/Male1_Cal_stageii.npz \
  --update_scripts
```

## Goal

Use a user-provided SMPL-X body model asset as the canonical zero-pose/template
source for SMPL-X offset solving and SMPL-X validation.

`Male1Walking_c3d/Male1_Cal_stageii.npz` must remain usable as an actual motion
to retarget, but it must not be treated as the canonical calibration source for
solving `smplx_to_v11.json`.

## Current Finding

The robot T-pose spec is not the source of the 180 degree failure.

Local self-check:

```python
spec = load_tpose_spec("specs/tpose/v11.json")
report = compute_deviations(np.asarray(spec["qpos"]), spec["xml_path"], spec)
```

Observed:

```text
total_deviation = 1.61e-05 deg
worst link      = 2.96e-06 deg
```

The staged `v11` spec also has the expected SMPL-X base root quaternion:

```text
qpos[3:7] = [0.5, -0.5, -0.5, -0.5]
```

The bad validation result came from comparing the robot T-pose spec against a
walking sequence frame. The requested fix is not to tune against that walking
frame, but to solve offsets from SMPL-X's own canonical template pose.

## Body Model Facts

The SMPL-X body model asset is not an AMASS motion sequence. It contains model
parameters such as:

```text
v_template      (10475, 3)
J_regressor     (55, 10475)
kintree_table   (2, 55)
shapedirs       (10475, 3, 400)
posedirs        (10475, 3, 486)
weights         (10475, 55)
```

The implementation should use the installed `smplx` package to instantiate this
body model and generate a synthetic canonical calibration frame with:

```text
betas         = zeros
global_orient = zeros
body_pose     = zeros
transl        = zeros
hand/jaw/eye  = zeros
```

Then derive SMPL-X joint world positions and world quaternions from that model
output.

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
  `/home/user2/GMR/general_motion_retargeting/utils/smpl.py`.
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

Replace or extend `/home/user2/GMR/scripts/compute_smplx_tpose_offsets.py` so
SMPL-X offsets are solved from the canonical template frame and the staged robot
T-pose spec, not by assuming every human joint quaternion is identity.

Required equation:

```text
offset = inverse(human_joint_world_quat) * robot_link_expected_world_quat
```

Where:

- `human_joint_world_quat` comes from the synthetic SMPL-X template frame.
- `robot_link_expected_world_quat` comes from `specs/tpose/<robot>.json`
  link `R`, converted to scalar-first quaternion.

This equation should be used for every robot link listed in the SMPL-X IK
config's `ik_match_table1` / `ik_match_table2` when the mapped SMPL-X joint is
available in the template frame and the robot link exists in the T-pose spec.

Acceptance criteria:

- Running the solver for `v11` writes updated offsets into
  `/home/user2/GMR/general_motion_retargeting/ik_configs/smplx_to_v11.json`.
- Both IK tables receive the same solved quaternion for the same robot link.
- The generated config keeps:
  - no top-level `world_rotation` for SMPL-X,
  - `init_qpos` for robot T-pose joints where required,
  - `robot_root_name = "base_link"`,
  - `human_root_name = "pelvis"`.

### Task C: Wire `setup_robot.py` To Use Template Calibration For SMPL-X

For `--src smplx`, setup should default to template calibration when solving
offsets and validating the result.

Required CLI behavior:

- Add an option similar to:

```text
--smplx_template_model /path/to/smplx_body_model_asset
```

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

- This command also works with an explicit model file:

```bash
python scripts/setup_robot.py \
  --robot v11 \
  --src smplx \
  --smplx_template_model /path/to/smplx_body_model_asset \
  --update_scripts
```

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
- Print the same `total_deviation`, `max_angle`, and `worst_k` report as the
  current validator.
- If `--smplx_template_model` is supplied, use the same resolved model path as
  setup/solve.
- If `--use_smplx_template` is set with a non-SMPLX `--src`, fail fast.

Acceptance criteria:

- The template validation path does not require `--tpose_motion`.
- The old motion-based validation path still works for BVH/FBX and for explicit
  SMPL-X motion debugging.
- Documentation and error messages clearly state that `Male1Walking...npz` is a
  motion input, not the canonical calibration source.

### Task E: Preserve SMPL-X Coordinate Policy

The SMPL-X coordinate policy must stay consistent:

- SMPL-X IK configs should not use top-level `world_rotation`.
- SMPL-X robot T-pose specs should continue staging the root qpos with
  `[0.5, -0.5, -0.5, -0.5]`.
- Validation should compare raw spec rotations. Do not reintroduce
  `apply_smplx_base_rotation(spec)` inside validation or the alignment agent.
- Template-frame quaternions should remain in SMPL-X convention; the solved
  offsets bridge SMPL-X template convention to the robot T-pose spec.

Acceptance criteria:

- Existing tests in `tests/alignment/test_smplx_tpose_coordinate_fix.py`
  continue to pass.
- Add or update tests so a top-level `world_rotation` is not generated for
  `smplx_to_v11.json`.

## Non-Requirements

- Do not solve offsets from
  `/home/user2/ACCAD/Male1Walking_c3d/Male1_Cal_stageii.npz` frame 0.
- Do not change `specs/tpose/v11.json` merely to match a walking frame.
- Do not add top-level `world_rotation` to SMPL-X configs to compensate for the
  walking sequence's root orientation.
- Do not relax the 5 degree validation threshold.

## Review Checklist For Codex

Codex should review the opencode implementation against this checklist:

- Template calibration uses a SMPL-X body model asset, not a motion sequence.
- Offset equation is `human.inv() * robot_expected`, with scalar-first
  quaternion IO.
- `setup_robot.py --src smplx` can solve without a `--tpose_motion` when the
  template model exists.
- Template validation path exists and does not require a motion file.
- Walking `.npz` files are not used as calibration sources by default.
- SMPL-X configs remain free of top-level `world_rotation`.
- Tests cover the new helper, solver behavior, and CLI surface.

## Implementation Record

Implemented by: opencode (GLM-5.1) on 2026-05-07

### Files Created

| File | Purpose |
|------|---------|
| `src/roboharness/alignment/smplx_template.py` | Task A — `load_smplx_template_tpose()` generates a GMR-compatible SMPL-X frame from the body model zero-pose using `smplx.create()` with `return_full_pose=True`. Uses hierarchical orientation propagation matching `GMR/utils/smpl.py`. Returns `{joint_name: (position, quat_wxyz)}` + `human_height`. All body-joint quaternions are identity at zero-pose. |
| `src/roboharness/alignment/smplx_offset_solver.py` | Task B — `solve_smplx_offsets_from_template()` computes `offset = inverse(human_quat) * robot_expected_quat` per IK table entry. Ensures both `ik_match_table1` and `ik_match_table2` receive the same solved quaternion for the same robot link. Removes any pre-existing `world_rotation`. |
| `tests/alignment/test_smplx_template_calibration.py` | Tasks A-E — 25 tests covering frame generation, offset solving, CLI wiring, validator template mode, and coordinate policy preservation. Skipped via `pytestmark` when body model is absent. |

### Files Modified

| File | Change |
|------|--------|
| `scripts/setup_robot.py` | Task C — Added `--smplx_template_model` CLI arg (defaults to `GMR/assets/body_models`). Replaced `_solve_smplx_offsets(robot)` with `_solve_smplx_offsets(robot, spec_path, body_model_root)` using the new template solver. SMPL-X offset solving no longer requires `--tpose_motion` when the body model exists. Validation step uses `--use_smplx_template` when no motion file is provided. |
| `examples/gmr_tpose_validate.py` | Task D — Added `--use_smplx_template` and `--smplx_template_model` CLI args. `--tpose_motion` is now optional (no longer `required=True`). New `_retarget_template_frame()` function generates a synthetic SMPL-X frame and retargets it through GMR. Updated failure hints to mention template calibration and that walking `.npz` files are motion inputs, not calibration sources. |

### Verification

```text
$ pytest -q
716 passed, 3 skipped in 15.86s
Coverage: 90.88% (≥90% threshold)

$ ruff check . && ruff format --check .
All checks passed!

$ mypy src/
Success: no issues found in 54 source files
```

### Revision Log

**Rev 1** — Codex review identified 3 issues; all addressed:

1. **setup/validate template path mismatch**: `setup_robot.py` now passes
   `--smplx_template_model <path>` to `gmr_tpose_validate.py` when invoking
   template validation, ensuring both processes use the same body model.

2. **template mode not source-gated**: `gmr_tpose_validate.py` now rejects
   `--use_smplx_template` unless `--src smplx` is also specified.

3. **body-model path contract underspecified**: All CLI help strings and
   docstrings now clarify directory vs file semantics.

**Rev 2** — Codex required name-agnostic resolution, no hardcoded paths:

4. **Hardcoded absolute paths removed**: `smplx_template.py` and
   `smplx_offset_solver.py` no longer contain any `/home/` literals. Added
   `resolve_body_model_path()` that auto-discovers via `find_gmr_root()` when
   no path is given.

5. **Name-agnostic input**: `resolve_body_model_path()` accepts a directory
   (e.g. `body_models/`), a `smplx/` subfolder, or a `.npz` file, resolving
   all to the directory expected by `smplx.create()`. CLI arg
   `--smplx_template_model` passes through the same flexibility.

6. **Tests updated**: Test discovery path uses `resolve_body_model_path(None)`
   instead of a hardcoded literal. New `TestResolveBodyModelPath` class (7
   tests) verifies directory/subfolder/file/None/nonexistent resolution plus
   asserts no hardcoded paths remain in source.

### Key Design Decisions

1. **Name-agnostic body model resolution**: `resolve_body_model_path()` accepts
   a directory, a `smplx/` subfolder, or a `.npz` file and resolves to the
   directory expected by `smplx.create()`. When passed ``None``, it discovers
   the GMR root via `find_gmr_root()` and uses `GMR_ROOT/assets/body_models`.
   No hardcoded machine-specific absolute paths exist in source code.

2. **Fallback to GMR script**: `_solve_smplx_offsets()` still falls back to `GMR/scripts/compute_smplx_tpose_offsets.py --generate` when the IK config file doesn't exist yet (first-time generation).

3. **Zero-pose quaternions**: All 22 body joints (pelvis through wrists) have identity orientation at zero-pose. Hand joints (indices 25–54) have non-identity rest poses due to SMPL-X hand PCA — these are not used in IK matching so they don't affect offset solving.

4. **Template vs motion validation**: When `smplx_template_available` is true and no `--tpose_motion` is provided, the validation step passes `--use_smplx_template` to `gmr_tpose_validate.py` instead of `--tpose_motion`. Both paths produce the same deviation report format.

5. **Coordinate policy**: The solver explicitly removes `world_rotation` from the config if present. Template-frame quaternions stay in SMPL-X convention (Y-up); the solved offsets bridge to the robot T-pose spec convention (Z-up with SMPLX base rotation already baked into spec `R` matrices).

## Modification Log

- 2026-05-07: Converted to collaborative requirements format with
  `flag` / owner / review state markers for multi-agent handoff.
- 2026-05-07: Implemented by opencode (GLM-5.1). All Tasks A-E complete.
  3 files created, 2 files modified. 704 tests passing, 90.86% coverage.
- 2026-05-07: Codex review → changes_requested (3 items). Revised:
  fix 1) setup→validate path passthrough; fix 2) source-gate
  `--use_smplx_template` to `--src smplx`; fix 3) directory-contract
  docs. 708 tests passing, 90.86% coverage. flag=1.
- 2026-05-07: Codex review → changes_requested (name-agnostic resolution).
  Rev 2: added `resolve_body_model_path()`, removed all hardcoded absolute
  paths, tests assert no `/home/` in source. 715 tests, 90.85% coverage. flag=1.
- 2026-05-08: Rev 3: Fixed `resolve_body_model_path()` directory contract —
  always returns parent dir so `smplx.create()` can append `smplx/` internally.
  Updated docstring and tests to match. 716 tests, 90.88% coverage. flag=1.
- 2026-05-08: Rev 4: True name-agnostic model loading — `load_smplx_template_tpose()`
  bypasses `smplx.create()` for `.npz` files, directly instantiates `smplx.SMPLX()`
  to avoid filename-based model type inference. Added test that copies real
  `SMPLX_MALE.npz` to `arbitrary_name.npz` and confirms full loader works.
  717 tests, 90.89% coverage. flag=1.
- 2026-05-08: Rev 5: Fixed two runtime issues:
  1) `load_smplx_template_tpose()` now reads `num_betas` from the model
  (GMR's custom smplx uses 16, not the standard 10), fixing an einsum
  dimension mismatch in `gmr` conda env.
  2) `gmr_alignment_agent.py --solve_mode --src smplx` now auto-detects
  template calibration instead of using motion file frame 0 as human
  reference. Added `--smplx_template_model` CLI arg. When body model is
  available and `--src smplx`, solve_mode uses body model zero-pose
   (no `--tpose_motion` required). 717 tests, 90.89% coverage. flag=1.
- 2026-05-08: Rev 6: Fixed two Codex review blockers from Rev 5:
  1) `gmr_alignment_agent.py --solve_mode --src smplx` no longer requires
  `--motion_file` — Phase A retargeting is skipped when template calibration
  is available; `--motion_file` changed from `required=True` to optional.
  2) Template solve path now uses resolved `tpose_spec_path` (auto-discovered
  or explicit) instead of `Path(args.tpose_spec)` which was `None` for
  auto-discovered specs. Added 6 new tests: source invariants for
  smplx_template_solve guard, tpose_spec_path usage verification, and
  default spec discovery. 723 tests, 90.89% coverage. flag=1.
