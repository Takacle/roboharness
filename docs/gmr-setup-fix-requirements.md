# GMR Setup Pipeline Fix Requirements

_Created: 2026-05-07_
_Status: Ready for implementation_

This document records the design-review findings for the GMR-harness setup
pipeline. It is intended for agents to read, claim a task, implement the fix,
and append a short modification record.

## Monitoring / Review State

flag=1

- `flag=0`: Work is not ready for Codex review.
- `flag=1`: All claimed implementation tasks are complete and ready for Codex review.
- `review_status=pending`: Codex has not reviewed the completed work yet.
- `review_status=approved`: Codex review passed.
- `review_status=changes_requested`: Codex found blockers; opencode should fix them.

Current owner: opencode (GLM-5.1)
Last implementation commit: b49163c
Last verification command: pytest tests/alignment/ -q --no-cov (73 passed)
Codex review status: pending
Codex review feedback: TBD

## Context

The GMR alignment docs describe a one-command flow:

```bash
python scripts/setup_robot.py \
  --robot my_robot \
  --xml /path/to/robot.xml \
  --tpose_motion /path/to/tpose.bvh \
  --auto_register --update_scripts
```

Expected behavior:

1. Resolve XML.
2. Match robot bodies to human skeleton roles.
3. Generate IK config.
4. Register GMR `params.py` entries.
5. Update GMR script `--robot` choices.
6. Stage T-pose.
7. Solve quaternion offsets.
8. Validate the result.

The core `roboharness.alignment` modules mostly follow this architecture, but
the orchestration contract in `scripts/setup_robot.py` has several mismatches
with `docs/gmr-alignment-guide.md` and `docs/gmr-alignment-sop.md`.

## Task A: Make `--dry_run` Truly Read-Only

**Problem:** `docs/gmr-alignment-guide.md` says `--dry_run` is preview mode and
does not modify files. Current code still writes IK configs before checking
`args.dry_run`.

Relevant code:

- `scripts/setup_robot.py`: config generation writes via `write_ik_config()`
  before the dry-run return.
- `scripts/setup_robot.py`: clone path writes via `clone_ik_config()` before
  the dry-run return.
- `src/roboharness/alignment/config_gen.py`: `write_ik_config()` always writes.
- `src/roboharness/alignment/config_gen.py`: `clone_ik_config()` always writes.

Required behavior:

- `--dry_run` must not create, overwrite, or back up any file.
- Dry-run output should still show the planned config path and a concise summary
  of what would be written.
- Cloning and fresh generation must both honor dry-run.

Suggested implementation:

- Add a dry-run branch in `setup_robot.py` before calling write functions.
- Optionally add pure helper functions that compute output paths without
  writing files.
- Keep `write_ik_config()` as the actual write primitive.

Acceptance criteria:

- A test proves `setup_robot.py --dry_run` leaves the target `ik_configs`
  directory unchanged for fresh generation.
- A test proves `--clone_from ... --dry_run` also leaves it unchanged.
- Existing `register_in_params(..., dry_run=True)` and
  `update_script_choices(..., dry_run=True)` behavior remains unchanged.

## Task B: Make `--xml` Registration Match the Actual XML Location

**Problem:** The docs allow `--xml /path/to/robot.xml`, but registration writes
only `ASSET_ROOT / "{robot}" / "{xml_path.name}"`. If the file is not actually
under `GMR/assets/{robot}/`, later stages load a non-existent or wrong XML.

Relevant code:

- `scripts/setup_robot.py`: passes only `xml_path.name` into
  `register_in_params()`.
- `src/roboharness/alignment/gmr_register.py`: always formats XML as
  `ASSET_ROOT / "{subdir}" / "{xml_filename}"`.
- `scripts/stage_tpose.py`: resolves registered robots from
  `ROBOT_XML_DICT`, so bad registration breaks staging.

Required behavior:

- A new robot setup must not register an XML path that does not exist.
- The implementation must choose and document one policy:
  - Copy the XML into `GMR/assets/{robot}/` before registration.
  - Or require `--xml` to already be inside `GMR/assets/{robot}/`.
  - Or register an absolute/pathlib path intentionally, if GMR supports that.
- Error messages must tell users exactly how to fix invalid XML placement.

Recommended policy:

- Require `--xml` to be inside `GMR/assets/{robot}/` unless a future
  `--copy_xml` option is added. This avoids silently copying meshes
  incorrectly, because XML files often reference nearby mesh assets.

Acceptance criteria:

- A test covers an external `--xml` path and verifies the command fails before
  writing misleading params.
- A test covers a valid `GMR/assets/{robot}/model.xml` path and verifies
  `ROBOT_XML_DICT` points to that asset path.
- Documentation is updated if the policy differs from the current guide.

## Task C: Parse `--world_rot` Consistently

**Problem:** Docs define `--world_rot "90,0,0,1"` as `angle,axis_x,axis_y,axis_z`
in degrees. `gmr_alignment_agent.py --solve_mode` parses it this way, but
`setup_robot.py` writes `[90, 0, 0, 1]` directly into config during generation.
That is not a normalized scalar-first quaternion.

Relevant code:

- `docs/gmr-alignment-guide.md`: `--world_rot` format is angle-axis.
- `scripts/setup_robot.py`: parses comma-separated floats and writes them
  directly as `config["world_rotation"]`.
- `examples/gmr_alignment_agent.py`: converts angle-axis to quaternion.
- `src/roboharness/alignment/config_gen.py`: accepts
  `world_rotation_override` but expects a quaternion list.

Required behavior:

- `--world_rot` means the same thing everywhere: angle-axis input,
  normalized scalar-first quaternion in config.
- Invalid inputs must fail fast:
  - wrong number of fields
  - zero-length axis
  - non-float values

Suggested implementation:

- Add one shared parser, for example
  `roboharness.alignment.orientation_aligner.parse_world_rotation_arg()`.
- Reuse it from both `setup_robot.py` and `gmr_alignment_agent.py`.
- Pass the resulting quaternion into `generate_ik_config(...,
  world_rotation_override=...)`.

Acceptance criteria:

- Unit test: `"90,0,0,1"` becomes approximately
  `[0.70710678, 0.0, 0.0, 0.70710678]`.
- Unit test: invalid input raises `ValueError` with actionable messages.
- Integration test or focused script test verifies generated config stores a
  normalized quaternion, not raw angle-axis.

## Task D: Ensure One-Command Flow Actually Validates New Robots

**Problem:** `setup_robot.py` loads `params` before auto-registration. Step 6
then uses this stale module snapshot to decide whether the robot exists in
`IK_CONFIG_DICT`, so validation can be skipped for newly registered robots.

Relevant code:

- `scripts/setup_robot.py`: initial params load occurs before registration.
- `scripts/setup_robot.py`: validation checks `args.robot in ik_config_src`
  using the old `params` object.

Required behavior:

- If setup reaches solve mode for a T-pose motion, validation must run unless
  the user explicitly disables it.
- New robots registered earlier in the same command must be visible to
  validation.

Suggested implementation:

- Reload params after `register_in_params()` before validation.
- Or avoid the stale params gate and run validation whenever
  `args.tpose_motion and not args.skip_solve`.

Acceptance criteria:

- A test proves the validation command is constructed for a robot registered
  during the same setup invocation.
- If validation cannot run because dependencies or files are missing, the
  command must print a clear warning rather than silently skipping.

## Task E: Clean Up Stale SOP Wording

**Problem:** `docs/gmr-alignment-sop.md` says the existing `apply_patch` in
`examples/gmr_alignment_agent.py` does not enforce table mirroring. That was
true historically, but current code uses `roboharness.alignment.patch.apply_patch`
with mirror support.

Relevant code:

- `src/roboharness/alignment/patch.py`: mirror policy implemented.
- `examples/gmr_alignment_agent.py`: imports and calls `apply_patch`.
- `tests/test_alignment_patch.py`: covers mirror behavior.
- `docs/gmr-alignment-sop.md`: stale warning remains.

Required behavior:

- SOP should describe the current implementation:
  - `apply_patch(..., mirror="auto")` auto-mirrors single-table patches.
  - `mirror="strict"` is available when callers want hard enforcement.
  - Agents should still prefer symmetric explicit patches when generating
    human-readable diffs.

Acceptance criteria:

- SOP no longer claims the current `apply_patch` fails to enforce mirroring.
- Documentation still preserves the warning that table drift is invalid.

## Verification Plan

Before claiming completion, run the repository-mandated preflight and checks:

```bash
uv --version
uv sync --dev
python -c "import pytest_cov; print('pytest-cov ok')"
pytest -q
ruff check .
ruff format --check .
mypy src/
```

If environment limits prevent any command, record the exact blocker in the
modification record.

For focused development, useful starting points:

```bash
pytest tests/alignment/test_config_gen.py tests/alignment/test_gmr_register.py -q
pytest tests/test_alignment_patch.py -q
pytest tests/test_alignment_metrics.py -q
```

New tests should be added for `scripts/setup_robot.py` orchestration behavior.
Prefer subprocess-free unit tests around extracted helper functions where
possible; use subprocess only for end-to-end command behavior.

## Claiming Work

Agents should claim one or more tasks by editing the Modification Record below
before making code changes. Keep claims scoped. If one task requires another,
note the dependency.

Recommended order:

1. Task C, because consistent `world_rot` parsing is small and isolated.
2. Task A, because dry-run safety prevents accidental file writes.
3. Task B, because XML registration policy affects user-facing docs.
4. Task D, because it depends on setup flow structure.
5. Task E can be done any time after confirming current code behavior.

## Modification Record

Append entries here. Keep each entry short and factual.

| Date | Agent | Claimed Tasks | Status | Notes |
|------|-------|---------------|--------|-------|
| 2026-05-07 | Codex | Requirements capture | Done | Created this document from code/design review findings. |
| 2026-05-07 | opencode (GLM-5.1) | Tasks A–E | Done | All tasks implemented and tested. 13 new tests in `tests/alignment/test_setup_robot_tasks.py`. All 73 alignment tests pass. 8 pre-existing failures in mjlab/sonic tests unrelated. |

## Codex Review Feedback

Append one review entry per `flag=1` review cycle. `flag=1` means the whole
GMR setup fix package is ready for review, not that an individual task is done.

opencode responsibilities:

- Keep `flag=0` while implementation is in progress.
- Record claimed work and verification results in the Modification Record.
- Set `flag=1` only after all claimed tasks are complete and self-verified.
- Fill `Last implementation commit` with a commit hash or `uncommitted`.
- Fill `Last verification command` with the final relevant check.
- Leave `Codex review status` for Codex to update.

Codex responsibilities:

- Review only when `flag=1`.
- Check the implementation against Task A-E acceptance criteria.
- Inspect diffs and test evidence; run non-destructive verification as needed.
- Set `Codex review status` to `approved` or `changes_requested`.
- Append concrete review feedback below for opencode.

| Date | Reviewer | Result | Feedback | Required Follow-up |
|------|----------|--------|----------|--------------------|
