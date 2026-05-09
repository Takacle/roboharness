# SMPL-X Path A Revert Plan

_Created: 2026-05-09_
_Status: Ready for execution_

This document specifies the complete revert plan to restore Path A (the old
SMPL-X alignment approach). It is intended for agents to execute and verify.

## Monitoring / Review State

flag=0

- `flag=0`: Revert not yet executed.
- `flag=1`: Revert complete and verified.

Current owner: unassigned
Last verification command: none
Review status: pending

## Context

Path B (world-rotation approach) introduced several changes that need to be
reverted. The old approach (Path A) works as follows:

1. **No `world_rotation`** — `compute_world_rotation("smplx")` returns `None`.
   The IK config contains no `world_rotation` key.
2. **Robot root qpos = SMPLX_BASE_ROTATION_QUAT** — T-pose specs stage the
   robot root at `[0.5, -0.5, -0.5, -0.5]` for SMPL-X.
3. **Identity-based solver** — Offsets are computed by the GMR script
   `GMR/scripts/compute_smplx_tpose_offsets.py --generate`, which assumes
   all human joint quaternions are identity (no body model involved).
4. **GMR runtime** — `scale_human_data` → `offset_human_data` → seed root from
   scaled pelvis. No `apply_world_rotation` call (wr is `None`).
   The robot root converges naturally to q_base via IK solver.

## Revert Tasks

### Task R1: Delete New SMPL-X Files

Delete all files created by the Path B implementation:

| File | Status |
|------|--------|
| `src/roboharness/alignment/smplx_coordinate.py` | untracked (never committed) |
| `src/roboharness/alignment/smplx_scale.py` | untracked (never committed) |
| `src/roboharness/alignment/smplx_template.py` | committed |
| `src/roboharness/alignment/smplx_offset_solver.py` | committed (has uncommitted modifications) |
| `tests/alignment/test_smplx_coordinate.py` | untracked (never committed) |
| `tests/alignment/test_smplx_scale.py` | untracked (never committed) |
| `tests/alignment/test_smplx_template_calibration.py` | committed (has uncommitted modifications) |
| `tests/alignment/test_smplx_tpose_coordinate_fix.py` | committed (has uncommitted modifications) |

Command:

```bash
# Untracked files — simple delete
rm src/roboharness/alignment/smplx_coordinate.py
rm src/roboharness/alignment/smplx_scale.py
rm tests/alignment/test_smplx_coordinate.py
rm tests/alignment/test_smplx_scale.py

# Committed files — git rm
git rm src/roboharness/alignment/smplx_template.py
git rm src/roboharness/alignment/smplx_offset_solver.py
git rm tests/alignment/test_smplx_template_calibration.py
git rm tests/alignment/test_smplx_tpose_coordinate_fix.py
```

### Task R2: Revert `orientation_aligner.py`

Restore `compute_world_rotation("smplx")` to return `None`.

Baseline (b49163c) state (`orientation_aligner.py:145-146`):

```python
    if src_format in ("smplx",):
        return None
```

Current (HEAD) state returns `smpl_to_mujoco_world_rotation()` with an import
from `smplx_coordinate`. Need to revert to baseline.

```bash
git checkout b49163c -- src/roboharness/alignment/orientation_aligner.py
```

Also deprecate `apply_smplx_base_rotation` — it was already deprecated in Path B
but is needed in Path A. The baseline version has the function. Verify after
checkout that `apply_smplx_base_rotation` is present and functional.

### Task R3: Revert `stage_tpose.py`

Restore SMPL-X base root quaternion application.

Baseline state: `stage_tpose.py` did NOT have `SMPLX_BASE_ROTATION_QUAT`.
The current working tree also does NOT have it (uncommitted removal).

**For Path A, stage_tpose.py needs `SMPLX_BASE_ROTATION_QUAT` applied when
`--src smplx`.** This was added in commit `f1bbc8f` but is being removed
locally.

Action: Check out the HEAD version (which has the block), or manually add:

```python
# At top of main(), after import block:
from roboharness._math_utils import SMPLX_BASE_ROTATION_QUAT

# After qpos construction, before _guard_qpos_input_not_output:
if (
    args.src == "smplx"
    and args.qpos is None
    and args.qpos_file is None
    and model.nq >= 7
    and int(model.jnt_type[0]) == 0
):
    qpos[3:7] = SMPLX_BASE_ROTATION_QUAT
    print(f"[stage_tpose] Applied SMPL-X base root quaternion: {SMPLX_BASE_ROTATION_QUAT}")
```

Alternatively, use `git checkout HEAD -- scripts/stage_tpose.py` to get the
committed version, then remove any unrelated local-only changes carefully.

### Task R4: Revert `setup_robot.py`

Restore the old `_solve_smplx_offsets` function that calls the GMR script.

Baseline `_solve_smplx_offsets`:

```python
def _solve_smplx_offsets(robot: str) -> bool:
    offsets_script = GMR_ROOT / "scripts" / "compute_smplx_tpose_offsets.py"
    if not offsets_script.exists():
        print(f"[setup] ERROR: {offsets_script} not found.")
        return False
    cmd = [sys.executable, str(offsets_script), "--robot", robot, "--generate"]
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print("[setup] ERROR: SMPL-X offset computation failed.")
        return False
    params = load_gmr_params(GMR_ROOT)
    ik_dict = getattr(params, "IK_CONFIG_DICT", {})
    config_path = Path(str(ik_dict.get("smplx", {}).get(robot, "")))
    computed_path = config_path.parent / config_path.name.replace(".json", "_computed.json")
    if computed_path.exists() and config_path.exists():
        config_path.write_text(computed_path.read_text())
        computed_path.unlink()
        print(f"[setup] Replaced {config_path.name} with computed offsets.")
    return True
```

Action:

```bash
git checkout b49163c -- scripts/setup_robot.py
```

This restores:
- The old `_solve_smplx_offsets` (calls GMR script, not template solver)
- `--tpose_src` argument (was `--src` in Path B)
- Old import list (no `smplx_offset_solver`)
- Step 6: "Skipping (SMPL-X offsets computed via MuJoCo FK -- already precise)"

**Verify** after checkout that the file does NOT import from `smplx_offset_solver`
or `smplx_template`.

### Task R5: Revert `gmr_tpose_validate.py`

Restore the baseline validator — motion-based validation only.

Baseline state:
- `--tpose_motion` is `required=True`
- No `--use_smplx_template` or `--smplx_template_model` args
- SMPL-X path calls `apply_smplx_base_rotation(spec)` before computing deviations
- No `_retarget_template_frame()` function

```bash
git checkout b49163c -- examples/gmr_tpose_validate.py
```

**Verify** the file does NOT import from `smplx_template`.

### Task R6: Revert `gmr_alignment_agent.py`

Restore baseline agent — no template calibration.

Baseline state:
- `--motion_file` is `required=True`
- `--solve_mode --src smplx` calls `apply_smplx_base_rotation(tpose_spec)`
- No `--smplx_template_model` arg
- No `smplx_template_solve` branch in `main()`

```bash
git checkout b49163c -- examples/gmr_alignment_agent.py
```

**Verify** the file does NOT import from `smplx_offset_solver` or `smplx_template`.

### Task R7: Revert Docs

Remove merged requirements doc and restore originals (if needed):

```bash
# Remove the merged doc (superseded by this revert plan)
git rm docs/smplx-alignment-requirements.md

# Restore original smplx-180deg requirements from baseline
git checkout b49163c -- docs/smplx-180deg-alignment-requirements.md
```

Alternatively, leave `docs/smplx-alignment-requirements.md` and add a note
that it has been superseded by the Path A revert.

### Task R8: Regenerate T-Pose Specs

Existing SMPL-X T-pose specs (`specs/tpose/v11.json`, `specs/tpose/unitree_g1.json`)
were staged with identity root qpos `[1, 0, 0, 0]`. For Path A, they need root
qpos = `[0.5, -0.5, -0.5, -0.5]`.

```bash
python scripts/stage_tpose.py --robot unitree_g1 --src smplx --preset tpose
```

This will re-stage the spec with `SMPLX_BASE_ROTATION_QUAT` as root qpos.

### Task R9: Regenerate SMPL-X IK Configs

Existing IK configs (`smplx_to_unitree_g1.json`) have `world_rotation = [0.5, 0.5, 0.5, 0.5]`.
For Path A, this key must not be present, and offsets must be solved via the
old GMR script (identity-based).

```bash
python scripts/setup_robot.py --robot unitree_g1 --src smplx --tpose_motion /dev/null --update_scripts --auto_register
```

Or manually delete the `world_rotation` key from existing configs and re-run
the GMR offset solver:

```bash
python /home/user2/GMR/scripts/compute_smplx_tpose_offsets.py --robot unitree_g1 --generate
```

## Verification

After all tasks are complete:

```bash
# 1. Verify no smplx template/coordinate files remain
ls src/roboharness/alignment/smplx_* tests/alignment/test_smplx_*

# 2. Verify imports are clean
grep -r "from roboharness.alignment.smplx_offset_solver" src/ scripts/ examples/
grep -r "from roboharness.alignment.smplx_template" src/ scripts/ examples/
grep -r "from roboharness.alignment.smplx_coordinate" src/ scripts/ examples/

# 3. Verify compute_world_rotation returns None for smplx
python -c "
from roboharness.alignment.orientation_aligner import compute_world_rotation
# This should return None (mocked match not needed since smplx returns None early)
"

# 4. Run full test suite
pytest -q

# 5. Run lint
ruff check . && ruff format --check .
```

Acceptance criteria:

- No smplx template/solver/coordinate/scale files exist.
- `compute_world_rotation("smplx")` returns `None`.
- `stage_tpose.py --src smplx` applies `SMPLX_BASE_ROTATION_QUAT` to root qpos.
- `setup_robot.py` calls old GMR script for SMPL-X offset solving.
- `gmr_tpose_validate.py` uses `apply_smplx_base_rotation`.
- `gmr_alignment_agent.py` uses `apply_smplx_base_rotation` in solve_mode.
- All existing tests pass (any tests from deleted files are naturally gone).
- IK configs have no `world_rotation` key.
- T-pose specs have root qpos = `[0.5, -0.5, -0.5, -0.5]` for SMPL-X.

## Files NOT Affected

These files had changes between b49163c and HEAD, but the changes are
SMPL-X-unrelated (formatting, cleanups, bug fixes):

| File | Change | Action |
|------|--------|--------|
| `src/roboharness/alignment/config_gen.py` | Line wrapping formatting | Keep as-is |
| `src/roboharness/alignment/gmr_register.py` | Type annotation + line wrapping | Keep as-is |
| `src/roboharness/alignment/skeleton_maps.py` | Skeleton data additions | Keep as-is |
| `src/roboharness/_math_utils.py` | `SMPLX_BASE_ROTATION_QUAT` was always present | Keep as-is |
| Various test files | Test additions for gmr-register etc. | Keep as-is |

## Key Constants Reference

```python
# From src/roboharness/_math_utils.py (exists at baseline, keep)
SMPLX_BASE_ROTATION_QUAT: list[float] = [0.5, -0.5, -0.5, -0.5]
```

## Execution Order

Recommended order to minimize breakage:

1. **R1** — Delete all new smplx files (removes imports that would break on revert)
2. **R4** — Revert `setup_robot.py` (main entry point)
3. **R5** — Revert `gmr_tpose_validate.py`
4. **R6** — Revert `gmr_alignment_agent.py`
5. **R2** — Revert `orientation_aligner.py`
6. **R3** — Revert `stage_tpose.py`
7. **R7** — Revert docs
8. **R8** — Regenerate T-pose specs
9. **R9** — Regenerate IK configs
10. Run full verification

## Modification Log

| Date | Agent | Action | Notes |
|------|-------|--------|-------|
| 2026-05-09 | opencode (GLM-5.1) | Plan creation | Created revert plan from baseline (b49163c) analysis. |
