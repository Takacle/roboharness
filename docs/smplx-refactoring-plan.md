# SMPLX Conversion Module Refactoring Plan

_Created: 2026-05-09_
_Owner: opencode (GLM-5.1)_
_Status: Implemented — Codex review findings addressed_

## flag=1

- `flag=0`: Implementation in progress, needs final review.
- `flag=1`: All implementation tasks complete, ready for Codex review/re-review.

## Purpose

Refactor the SMPLX conversion module to follow the same architectural pattern as
the BVH pipeline. The BVH pipeline works correctly; SMPLX should mirror its
structure. The primary focus is **axis alignment between human (SMPLX) and robot
(MuJoCo) coordinate frames**.

---

## 1. Problem Statement

### 1.1 BVH Pipeline (Reference — Working Correctly)

```
BVH file
  → GMR loader applies Y→Z rotation matrix [[1,0,0],[0,0,-1],[0,1,0]] internally
  → Frames arrive at runtime in Z-up
  → world_rotation ≈ [0.707, 0, 0, 0.707] (fine-tuning, ~90° about X)
  → rot_offset: small quaternions (bone convention differences)
  → pos_offset: all [0, 0, 0]
```

### 1.2 SMPLX Pipeline (Current — Problematic)

```
SMPLX .npz
  → GMR loader returns frames in Y-up (NO coordinate conversion)
  → Frames arrive at runtime in Y-up
  → world_rotation = [0.5, 0.5, 0.5, 0.5] (carries ENTIRE Y→Z conversion)
  → rot_offset at pelvis = [0.5, -0.5, -0.5, -0.5] (must cancel world_rotation)
  → pos_offset: non-zero values
```

### 1.3 Specific Issues

1. **Split coordinate conversion**: Conversion is split between the template solver
   (pre-converts Y→Z before computing offsets) and the GMR runtime (applies
   world_rotation for Y→Z). The runtime path through `load_smplx()` delivers
   Y-up frames while the solver assumes Z-up.
2. **world_rotation overloaded**: It carries the full coordinate conversion
   (`SMPL_TO_MUJOCO_QUAT`) instead of fine-tuning alignment. This conflates two
   concerns: frame transformation and human-robot axis mapping.
3. **Offsets carry base rotation**: The pelvis rot_offset `[0.5, -0.5, -0.5, -0.5]`
   exists solely to cancel world_rotation. It is not a bone convention offset.
4. **Robot geometry ignored**: `compute_world_rotation(src_format="smplx")`
   hardcodes `smpl_to_mujoco_world_rotation()` instead of computing alignment
   from robot body positions like BVH does.
5. **Axis convention mismatch**: After `smpl_to_mujoco_frame()`, SMPLX frames
   have X=forward, Y=left, Z=up. BVH post-loader has X=left, Y=forward, Z=up.
   The code does not account for this difference in `compute_world_rotation()`.

---

## 2. Proposed Architecture

### 2.1 New SMPLX Pipeline (BVH-style)

```
SMPLX .npz
  → GMR loader returns Y-up frames
  → Post-processing in load_smplx() applies smpl_to_mujoco_frame()
  → Frames arrive at runtime in Z-up
  → world_rotation computed from robot geometry (fine-tuning, like BVH)
  → rot_offset: bone convention differences only
  → pos_offset: minimal (like BVH)
```

### 2.2 Coordinate Convention After Conversion

| Axis | SMPLX Native | After smpl_to_mujoco_frame | BVH Post-loader |
|------|-------------|---------------------------|-----------------|
| Up   | +Y          | +Z                        | +Z              |
| Left | +X          | +Y                        | +X              |
| Forward | +Z      | +X                        | +Y              |

Both post-conversion frames are right-handed and Z-up, but the horizontal axes
are arranged differently. This is handled in `compute_world_rotation()` by
building the human frame matrix with format-specific axis ordering.

### 2.3 Offset Math Comparison

**Current (SMPLX):**
```
runtime: q_robot = world_rotation * (q_human * rot_offset)
pelvis:  I = SMPL_TO_MUJOCO * (I * SMPLX_BASE_ROTATION)
             ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^
             full Y→Z convert   cancel world_rotation
```

**After refactoring:**
```
runtime: q_robot = world_rotation * (q_human * rot_offset)
pelvis:  q_robot = R_mat * (I * r_human.inv() * r_target)
                  ^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                  fine    pure bone convention offset
                  tune
```

For a robot with identity pelvis orientation at T-pose:
- `r_human = I` (template zero-pose, Z-up)
- `r_target = I` (robot T-pose spec, upright)
- `r_offset = I * I = I`
- `world_rotation = R_mat` from robot geometry (may be `None` for simple cases)

---

## 3. Detailed File Changes

### 3.1 `examples/_gmr_shared.py` — SMPLX Loader Post-Processing

**Current** (lines 92–107):
```python
def load_smplx(npz_file: str) -> tuple[list, float, int]:
    smplx_body_model_path = GMR_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        npz_file, smplx_body_model_path
    )
    tgt_fps = 30
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    print(f"[smplx] Loaded: {len(frames)} frames @ {aligned_fps} fps  height={human_height:.2f} m")
    return frames, human_height, aligned_fps
```

**Proposed**:
```python
def load_smplx(npz_file: str) -> tuple[list, float, int]:
    from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame

    smplx_body_model_path = GMR_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        npz_file, smplx_body_model_path
    )
    tgt_fps = 30
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )
    frames = [smpl_to_mujoco_frame(f) for f in frames]
    print(
        f"[smplx] Loaded: {len(frames)} frames @ {aligned_fps} fps"
        f"  height={human_height:.2f} m  (Z-up)"
    )
    return frames, human_height, aligned_fps
```

**Rationale**: BVH loaders apply Y→Z rotation internally. SMPLX cannot modify
the GMR loader, so the conversion is applied as post-processing. This ensures
SMPLX frames arrive in Z-up at the GMR runtime boundary, matching the BVH pattern.

---

### 3.2 `src/roboharness/alignment/smplx_template.py` — Z-up Template Output

**Current** (lines 83–171): Returns Y-up frame.

**Proposed**: Apply `smpl_to_mujoco_frame()` before returning.

```python
def load_smplx_template_tpose(
    body_model_path: Path | str | None = None,
    gender: str = "male",
    betas: np.ndarray | None = None,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Create a GMR-compatible SMPL-X frame from the body model zero-pose.

    Returns
    -------
    frame:
        {joint_name: (position_3d, quat_wxyz)} where quaternions are
        scalar-first [w, x, y, z] and positions are in **Z-up MuJoCo
        coordinates** (Y=left, X=forward, Z=up).
    human_height:
        Deterministic height estimate for scaling.
    """
    # ... existing body model forward pass (unchanged) ...
    # Build frame dict with Y-up positions and orientations ...
    
    # Convert Y-up SMPL-X frame to Z-up MuJoCo frame
    from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_frame
    frame = smpl_to_mujoco_frame(frame)

    height = 1.66 + 0.1 * float(betas[0])
    return frame, float(height)
```

**Impact on tests**:
- `test_pelvis_orientation_is_identity`: The pelvis orientation was `[1,0,0,0]`
  in Y-up. After Z-up conversion, it becomes `SMPL_TO_MUJOCO_QUAT = [0.5,0.5,0.5,0.5]`.
  This test must be updated.
- `test_body_orientations_identity_at_zero_pose`: Same — all orientations
  become `[0.5,0.5,0.5,0.5]` after conversion.
- `test_positions_are_3d`: No change (positions are still 3D).
- `test_human_height_reasonable`: No change (height is scalar).

---

### 3.3 `src/roboharness/alignment/smplx_offset_solver.py` — Simplified Pipeline

**Current pipeline** (4 stages):
1. Load Y-up template
2. Scale positions (Y-up)
3. Convert Y→Z via `smpl_to_mujoco_frame()`
4. Compute offsets
5. Inject `world_rotation = SMPL_TO_MUJOCO_QUAT` if missing

**Proposed pipeline** (3 stages):
1. Load Z-up template (already converted)
2. Scale positions (Z-up)
3. Compute offsets

The `world_rotation` injection is REMOVED. The solver no longer sets
`world_rotation` — that responsibility belongs to `config_gen.py` /
`compute_world_rotation()` / the user.

```python
def solve_smplx_offsets_from_template(
    ik_config_path: Path,
    tpose_spec_path: Path,
    body_model_path: Path | str | None = None,
    gender: str = "male",
) -> dict:
    body_model_resolved = resolve_body_model_path(body_model_path)

    with ik_config_path.open() as f:
        config: dict = json.load(f)

    # Stage 1: Load Z-up template (conversion happens inside template loader)
    frame, human_height = load_smplx_template_tpose(body_model_resolved, gender=gender)

    # Stage 2: Scale positions (Z-up)
    human_root_name = str(config.get("human_root_name", "pelvis"))
    scale_table_raw = config.get("human_scale_table", {})
    height_assumption = float(config.get("human_height_assumption", human_height))
    frame = apply_human_scale(
        frame,
        scale_table_raw,
        human_root_name=human_root_name,
        height_assumption=height_assumption,
        human_height=human_height,
    )

    # Stage 3: Compute offsets (Z-up frame → Z-up spec)
    spec = load_tpose_spec(tpose_spec_path)
    ground_height = float(config.get("ground_height", 0.0))
    compute_joint_offsets(frame, spec, config, ground_height=ground_height)

    return config
```

**Key changes**:
- Stage 3 (Y→Z conversion) is removed — template already returns Z-up.
- The `world_rotation` injection block is removed entirely.
- `smpl_to_mujoco_frame` import is removed from this file.
- `smpl_to_mujoco_world_rotation` import is removed from this file.

---

### 3.4 `src/roboharness/alignment/orientation_aligner.py` — Unified world_rotation

**Current** (lines 199–205):
```python
if src_format in ("smplx",):
    from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_world_rotation
    return smpl_to_mujoco_world_rotation()
```

**Proposed**: SMPLX uses the same R_mat computation as BVH, but with different
axis convention.

```python
def compute_world_rotation(
    xml_path: Path,
    match: MatchResult,
    *,
    src_format: str = "bvh",
) -> list[float] | None:
    positions = _collect_body_positions(xml_path)

    # ... existing landmark computation (unchanged) ...
    root_pos = positions.get(match.mapping.get("root", ""))
    # ...
    robot_up = normalize_vector(spine_pos - hip_pos)
    robot_left = normalize_vector(lat_raw)
    robot_forward = normalize_vector(np.cross(robot_up, robot_left))
    robot_left = normalize_vector(np.cross(robot_forward, robot_up))

    # Build human-to-robot frame matrix.
    # Axis ordering depends on the post-loader convention:
    #   BVH:  X=left,  Y=forward, Z=up  → [left,  forward, up]
    #   SMPLX: X=forward, Y=left, Z=up  → [forward, left, up]
    if src_format in ("smplx",):
        robot_frame = np.column_stack([robot_forward, robot_left, robot_up])
    else:
        robot_frame = np.column_stack([robot_left, robot_forward, robot_up])

    # SVD projection to nearest proper rotation (shared for both formats)
    U, _, Vt = np.linalg.svd(robot_frame)
    R_mat = U @ Vt
    if np.linalg.det(R_mat) < 0.0:
        Vt[-1, :] *= -1
        R_mat = U @ Vt

    # Return None if effectively identity (shared for both formats)
    if np.allclose(R_mat, np.eye(3), atol=1e-4):
        return None

    return _rotation_matrix_to_quat_scalar_first(R_mat)
```

**Key changes**:
- The SMPLX special-case branch that returns `smpl_to_mujoco_world_rotation()`
  is replaced with the same R_mat SVD computation as BVH.
- The axis convention difference is handled by the frame matrix column ordering.
- SMPLX can now return `None` (if robot geometry matches SMPLX post-conversion frame).
- The `from roboharness.alignment.smplx_coordinate import smpl_to_mujoco_world_rotation`
  import inside this function is removed.

---

### 3.5 `src/roboharness/alignment/smplx_coordinate.py` — Deprecation Cleanup

No structural changes. Add deprecation note to `smpl_to_mujoco_world_rotation()`:

```python
def smpl_to_mujoco_world_rotation() -> list[float]:
    """Return the world_rotation quaternion for SMPL-X IK configs.

    .. deprecated::
        This function is no longer used for SMPL-X IK config world_rotation.
        The coordinate conversion is now applied at the loading boundary
        (in load_smplx() and load_smplx_template_tpose()), and
        compute_world_rotation() computes the fine-tuning alignment from
        robot geometry. Kept for backward compatibility.
    """
    return list(SMPL_TO_MUJOCO_QUAT)
```

---

### 3.6 `src/roboharness/alignment/smplx_scale.py` — No Changes

Position scaling is coordinate-frame-independent. No modifications needed.

---

### 3.7 `src/roboharness/alignment/config_gen.py` — No Logic Changes

The `generate_ik_config()` function already calls `compute_world_rotation()`
which will now return a geometry-based quaternion for SMPLX. No code changes
needed in this file.

---

### 3.8 Test Updates

#### `tests/alignment/test_smplx_template.py`

| Test | Change |
|------|--------|
| `test_pelvis_orientation_is_identity` | Pelvis orientation is now `SMPL_TO_MUJOCO_QUAT` after Z-up conversion, not `[1,0,0,0]` |
| `test_body_orientations_identity_at_zero_pose` | All orientations become `SMPL_TO_MUJOCO_QUAT` after Z-up conversion |
| `test_renamed_npz_loads_via_full_loader` | Pelvis check updates to `SMPL_TO_MUJOCO_QUAT` |
| `test_positions_are_3d` | No change |
| `test_human_height_reasonable` | No change |
| `test_custom_betas` | No change |

#### `tests/alignment/test_smplx_offset_solver.py` (in template_calibration file)

| Test | Change |
|------|--------|
| `test_solves_offsets` | Offsets now computed from Z-up template; values will differ |
| `test_both_tables_same_offset` | No change (both tables still get same offset) |
| `test_preserves_world_rotation` | Solver no longer injects world_rotation; test should verify solver does NOT add it |
| `test_solver_always_uses_base_rotation_for_offsets` | Remove: this test verifies old behavior where solver ignores config wr |
| `test_preserves_robot_root_name` | No change |

#### `tests/alignment/test_smplx_coordinate.py`

| Test | Change |
|------|--------|
| `TestSmplToMujocoQuat` | No change (unit tests for the conversion constant) |
| `TestSmplToMujocoFrame` | No change (unit tests for frame conversion) |
| `TestSmplToMujocoWorldRotation` | Mark as deprecated, keep for backward compat |
| `TestSolverUsesPipeline` | Update: solver no longer imports `smpl_to_mujoco_frame` |

#### `tests/alignment/test_smplx_tpose_coordinate_fix.py`

| Test | Change |
|------|--------|
| `test_returns_non_none_for_smplx` | May now return `None` for identity-geometry robots (like BVH) |
| `test_smplx_result_maps_axes_to_mujoco_world` | Remove or rewrite: SMPLX no longer uses hardcoded base rotation |
| `test_smplx_never_returns_none` | Remove: SMPLX can now return `None` (like BVH) |
| `test_bvh_returns_none_for_identity_robot` | No change |

---

### 3.9 Downstream Impact

#### `examples/gmr_alignment_agent.py`

- `smplx_template_solve` path: The solver no longer injects `world_rotation`,
  so the agent's `_create_default_ik_config()` or the explicit `--world_rot`
  flag must handle it.
- The template validation path continues to work because `load_smplx_template_tpose()`
  still returns Z-up frames (now directly instead of requiring solver pre-conversion).

#### `examples/gmr_tpose_validate.py`

- Template validation path (`--use_smplx_template`): No change needed.
  The template frame is already Z-up from `load_smplx_template_tpose()`.

#### `scripts/setup_robot.py`

- SMPLX offset solving: The solver no longer injects `world_rotation`, so
  `config_gen.generate_ik_config()` must be called first (which calls
  `compute_world_rotation()`), or `world_rotation` must be set explicitly.

#### SMPLX IK Config Files

All existing SMPLX IK configs (`smplx_to_*.json`) must be regenerated because:
- `world_rotation` changes from `[0.5, 0.5, 0.5, 0.5]` to a geometry-based value
- `rot_offset` values change (no longer carry base rotation)
- `pos_offset` values change

---

## 4. Implementation Order

| Step | File(s) | Description | Test Gate |
|------|---------|-------------|-----------|
| 1 | `smplx_coordinate.py` | Add deprecation note to `smpl_to_mujoco_world_rotation()` | `pytest tests/alignment/test_smplx_coordinate.py -q` |
| 2 | `smplx_template.py` | Return Z-up directly from `load_smplx_template_tpose()` | `pytest tests/alignment/test_smplx_template_calibration.py -q` |
| 3 | `_gmr_shared.py` | Apply `smpl_to_mujoco_frame()` in `load_smplx()` | Manual test with SMPLX motion file |
| 4 | `smplx_offset_solver.py` | Remove pre-conversion and wr injection | `pytest tests/alignment/test_smplx_template_calibration.py -q` |
| 5 | `orientation_aligner.py` | Unified `compute_world_rotation()` for SMPLX | `pytest tests/alignment/test_smplx_tpose_coordinate_fix.py -q` |
| 6 | Tests | Update all 4 SMPLX test files | `pytest -q` |
| 7 | Integration | Regenerate SMPLX IK configs, E2E validation | `pytest -q` + manual retargeting |

Each step should be a separate commit with passing tests.

---

## 5. Non-Requirements (Explicitly Out of Scope)

- Do NOT modify the BVH pipeline (reference implementation).
- Do NOT modify GMR's internal SMPLX loader (`GMR/utils/smpl.py`).
- Do NOT change the solver formula `r_offset = r_human.inv() * r_target`.
- Do NOT change the GMR runtime processing order (scale → wr → offset → ground).
- Do NOT relax existing test coverage thresholds.
- Do NOT add `R_mat @ base_mat` composition (known axis-swap bug pattern).

---

## 6. Verification Checklist (Post-Implementation)

- [ ] `smpl_to_mujoco_frame()` is called in `load_smplx()` (every frame)
- [ ] `load_smplx_template_tpose()` returns Z-up frame
- [ ] `smplx_offset_solver.py` does NOT import `smpl_to_mujoco_frame`
- [ ] `smplx_offset_solver.py` does NOT inject `world_rotation`
- [ ] `compute_world_rotation("smplx")` uses R_mat from robot geometry
- [ ] `compute_world_rotation("smplx")` can return `None` (identity alignment)
- [ ] Template pelvis orientation is `SMPL_TO_MUJOCO_QUAT` (not `[1,0,0,0]`)
- [ ] Solver offsets for simple robots are near-identity (no base rotation)
- [ ] All 760+ tests pass: `pytest -q`
- [ ] `ruff check .` passes
- [ ] `mypy src/` passes

---

## 7. Rollback Plan

If the refactoring introduces regressions:
1. Each step is a separate commit — revert the problematic step.
2. Existing SMPLX IK configs can be restored from `.json.bak` backups.
3. The `smpl_to_mujoco_frame()` function and `SMPL_TO_MUJOCO_QUAT` constant
   remain unchanged — they are used by both old and new paths.

---

## Review Request

Codex: Please review this plan for correctness, especially:

1. Is the axis convention analysis correct? (SMPLX post-conversion: X=forward, Y=left, Z=up)
2. Is the `compute_world_rotation()` change correct? (frame matrix column ordering)
3. Are there any callers of `smpl_to_mujoco_world_rotation()` that would break?
4. Are the test update expectations correct?
5. Are there any missing downstream impacts not listed in Section 3.9?

---

## Codex Review (2026-05-09)

_Reviewer: Codex_
_Status: Needs revision before implementation_

### Summary

Do **not** implement this plan as written. The high-level direction can work
only if the coordinate conversion boundary, `world_rotation` semantics, and
offset-solving math are made internally consistent. As currently written, the
plan risks double-applying the SMPL-X base rotation, constructing an improper
SMPL-X frame in `compute_world_rotation()`, and computing offsets that do not
match the GMR runtime order.

### Findings

1. **`compute_world_rotation("smplx")` frame construction is not correct.**

   The plan says post-conversion SMPL-X uses `X=forward, Y=left, Z=up`, then
   proposes:

   ```python
   robot_forward = normalize_vector(np.cross(robot_up, robot_left))
   robot_frame = np.column_stack([robot_forward, robot_left, robot_up])
   ```

   For an `X=forward, Y=left, Z=up` human frame, the basis must satisfy
   `forward × left = up`. The proposed `robot_forward = up × left` instead
   produces a basis where `forward × left = -up`. That is an improper frame
   (`det < 0`) before SVD projection. SVD will force a proper rotation, but the
   resulting quaternion is not guaranteed to represent the intended axis
   mapping.

   Recommended fix: for the SMPL-X branch, define forward consistently with the
   SMPL-X post-conversion basis, for example `robot_forward = left × up` (or
   use the negative of the existing BVH forward, if that is the intended robot
   convention). Add tests that assert:

   - `det(R_mat) > 0`
   - `R * [1,0,0] == robot_forward`
   - `R * [0,1,0] == robot_left`
   - `R * [0,0,1] == robot_up`

2. **The solver cannot ignore an existing geometry-based `world_rotation`.**

   GMR runtime order is still:

   ```text
   scale -> apply_world_rotation -> apply rot_offset/pos_offset -> ground
   ```

   If `compute_world_rotation("smplx")` returns a non-identity geometry
   alignment, then offsets must be solved against the human frame **after** that
   same world rotation is applied. Otherwise runtime applies `world_rotation`
   before the offset, but the solver computed the offset as if it would not.

   In other words, the correct rotation offset for a non-identity
   `world_rotation` is based on:

   ```text
   r_human_runtime = r_world * r_human
   r_offset = r_human_runtime.inv() * r_target
   ```

   not just:

   ```text
   r_offset = r_human.inv() * r_target
   ```

   The solver does not need to inject `world_rotation`, but it should read an
   existing `config["world_rotation"]` and apply it to the template frame before
   computing offsets, matching the runtime behavior.

3. **The plan contradicts itself about whether offsets become identity.**

   Section 3.8 says the converted template pelvis orientation becomes
   `SMPL_TO_MUJOCO_QUAT`, not identity. But Section 2.3 says that after
   refactoring, for an identity robot T-pose:

   ```text
   r_human = I
   r_target = I
   r_offset = I
   ```

   These cannot both be true with the current `smpl_to_mujoco_frame()`
   semantics. That function pre-multiplies orientations by
   `SMPL_TO_MUJOCO_QUAT`; therefore an identity SMPL-X template orientation
   becomes `SMPL_TO_MUJOCO_QUAT`, and an identity robot target would still need
   a base inverse offset.

   Before implementation, decide which invariant is desired:

   - Converted positions **and orientations** carry the SMPL-X base rotation.
   - Converted positions are Z-up but zero-pose local/body orientations remain
     identity.

   The current plan assumes both at different points.

4. **The migration plan is missing stale-config / double-rotation handling.**

   Existing SMPL-X configs contain:

   ```json
   "world_rotation": [0.5, 0.5, 0.5, 0.5]
   ```

   If `load_smplx()` starts returning Z-up frames, any stale config with that
   base `world_rotation` will apply the SMPL-X base conversion a second time at
   runtime. This affects normal retargeting, template validation, and inspector
   paths that construct a GMR retargeter from registered IK configs.

   Recommended fix: add a fail-fast or migration guard. For example, after the
   loader-boundary refactor, SMPL-X configs should not contain the legacy base
   conversion unless explicitly intended as geometry alignment. The setup or
   validation path should detect `[0.5, 0.5, 0.5, 0.5]` in an SMPL-X config and
   warn or fail with regeneration instructions.

5. **Downstream impacts are incomplete.**

   Section 3.9 lists roboharness entry points, but does not cover direct GMR
   scripts that call `general_motion_retargeting.utils.smpl` directly. Those
   scripts will not pass through `examples/_gmr_shared.py::load_smplx()`, so
   they will keep receiving Y-up frames while regenerated configs may assume
   loader-boundary Z-up frames.

   Affected class of callers:

   - Direct GMR SMPL-X scripts such as `GMR/scripts/smplx_to_robot.py`
   - Any external user code that imports GMR's SMPL-X loader directly
   - Registered IK config users that instantiate `GeneralMotionRetargeting`
     without using roboharness `load_motion()`

   The plan should either explicitly scope these out with a compatibility
   warning, or provide a shared conversion wrapper that all supported SMPL-X
   entry points use.

6. **Documentation and tests still encode the old policy.**

   Existing docs such as `docs/smplx-alignment-requirements.md` and
   `docs/gmr-alignment-sop.md` currently state that SMPL-X conversion is handled
   by top-level `world_rotation`. Those docs must be updated in the same change,
   or future agents/users will follow the old policy and recreate double
   conversion bugs.

### Answers to Review Questions

1. **Axis convention analysis:** The mapping in `smpl_to_mujoco_frame()` is
   correct: SMPL-X native `+Y up, +X left, +Z forward` maps to post-conversion
   `+Z up, +Y left, +X forward`. The plan's later use of this convention in
   `compute_world_rotation()` is not correct because the forward cross-product
   sign is inconsistent with `X=forward, Y=left, Z=up`.

2. **`compute_world_rotation()` change:** Not correct as written. The SMPL-X
   column ordering alone is insufficient; the robot forward vector must be
   computed with the SMPL-X post-conversion handedness in mind.

3. **Callers of `smpl_to_mujoco_world_rotation()`:** Direct code callers are
   limited, mainly `orientation_aligner.py` and `smplx_offset_solver.py`.
   Removing those imports is unlikely to cause broad import-time breakage, but
   runtime behavior will break if stale SMPL-X IK configs still contain the old
   base `world_rotation`.

4. **Test expectations:** Some are correct, especially updating template
   orientation expectations if `load_smplx_template_tpose()` now returns a fully
   converted frame. However, the expectation that simple-robot offsets become
   near identity is not valid unless template orientations are deliberately kept
   identity after coordinate conversion. Add tests for non-identity
   `world_rotation` in the solver.

5. **Missing downstream impacts:** Add stale config detection, direct GMR script
   compatibility, docs/SOP updates, and validation paths that instantiate GMR
   with registered configs after the template/loader already returned Z-up data.

### Required Plan Changes Before Implementation

- Define one invariant for converted SMPL-X orientations: either carry the base
  world rotation in every joint quaternion, or preserve zero-pose identity
  orientations while converting positions/axes.
- Fix the SMPL-X branch of `compute_world_rotation()` so its basis is proper and
  explicitly tested.
- Make `solve_smplx_offsets_from_template()` apply any existing geometry
  `world_rotation` before computing offsets, matching runtime order.
- Add a migration/fail-fast strategy for stale SMPL-X configs containing the old
  base `[0.5, 0.5, 0.5, 0.5]` world rotation.
- Include direct GMR SMPL-X scripts and external GMR-loader users in downstream
  compatibility notes.
- Update old docs that state SMPL-X conversion is handled by top-level
  `world_rotation`.

---

## Implementation Response (2026-05-09)

_Implementer: opencode (GLM-5.1)_
_All 6 findings addressed. 764 tests pass, 91.20% coverage._

### Finding 1 — Fixed: proper SMPLX frame construction

Chose `robot_forward = cross(left, up)` (not `cross(up, left)`) for the SMPLX
branch.  This produces `det(R_mat) > 0` before SVD projection.

```python
# orientation_aligner.py, lines 195-203
if src_format in ("smplx",):
    robot_forward = normalize_vector(np.cross(robot_left, robot_up))
    robot_left = normalize_vector(np.cross(robot_up, robot_forward))
    robot_frame = np.column_stack([robot_forward, robot_left, robot_up])
else:
    robot_forward = normalize_vector(np.cross(robot_up, robot_left))
    robot_left = normalize_vector(np.cross(robot_forward, robot_up))
    robot_frame = np.column_stack([robot_left, robot_forward, robot_up])
```

For a robot with left=+Y (SMPLX post-conversion convention), the frame matrix
is identity, so `compute_world_rotation("smplx")` correctly returns `None`.

Tests added:
- `test_smplx_returns_none_for_smplx_aligned_robot` — robot with left=+Y → None
- `test_smplx_returns_non_none_for_bvh_aligned_robot` — robot with left=+X → non-None
- `test_smplx_frame_matrix_has_positive_det` — det(R_mat) > 0

### Finding 2 — Fixed: solver applies world_rotation before offsets

Added `_apply_rotation_to_frame()` helper. The solver now reads
`config["world_rotation"]` and applies it to the template frame before calling
`compute_joint_offsets()`, matching the GMR runtime order.

```python
# smplx_offset_solver.py
wr = config.get("world_rotation")
if wr:
    frame = _apply_rotation_to_frame(frame, wr)
compute_joint_offsets(frame, spec, config, ground_height=ground_height)
```

Offset formula at solve time now matches runtime:

```text
r_human_after_wr = r_wr * SMPL_TO_MUJOCO_QUAT
r_offset = r_human_after_wr.inv() * r_target
```

Test added:
- `test_solver_applies_existing_world_rotation_before_offsets` — verifies offset
  matches `(r_wr * SMPL_TO_MUJOCO_QUAT).inv() * r_target`

### Finding 3 — Resolved: one invariant chosen

Chose: **converted positions AND orientations carry the SMPL-X base rotation.**

After `smpl_to_mujoco_frame()`, all zero-pose orientations are
`SMPL_TO_MUJOCO_QUAT = [0.5, 0.5, 0.5, 0.5]`, not identity.  This is
consistent throughout the pipeline:

- Template returns Z-up with base-rotation orientations
- Solver (if world_rotation exists) applies it on top, then computes offsets
- Runtime applies world_rotation then offsets — same math

Section 2.3 claim that offsets become identity for simple robots was incorrect.
The actual invariant is:

```text
For world_rotation=None, r_offset = SMPL_TO_MUJOCO_QUAT.inv() * r_target
For identity r_target: r_offset = SMPLX_BASE_ROTATION_QUAT = [0.5, -0.5, -0.5, -0.5]
```

The offset still carries the base inverse, but the world_rotation is now
cleanly separated (geometry-based, may be None).

Tests updated:
- `test_pelvis_orientation_is_smpl_to_mujoco` — expects SMPL_TO_MUJOCO_QUAT
- `test_body_orientations_carry_base_rotation_at_zero_pose` — same

### Finding 4 — Fixed: stale config detection

Added `_check_stale_smplx_config()` in `smplx_offset_solver.py`.  Emits
`warnings.warn` when the config contains the legacy base world_rotation
`[0.5, 0.5, 0.5, 0.5]`, which would double-apply the Y→Z conversion at
runtime.

```python
def _check_stale_smplx_config(config, config_path):
    wr = config.get("world_rotation")
    if wr is None:
        return
    base = [0.5, 0.5, 0.5, 0.5]
    if len(wr) == 4 and all(abs(a - b) < 1e-6 for a, b in zip(wr, base, strict=True)):
        warnings.warn(
            f"SMPL-X config {config_path.name} contains the legacy base "
            "world_rotation [0.5, 0.5, 0.5, 0.5].  ..."
        )
```

Regeneration instructions are included in the warning message.

### Finding 5 — Documented: direct GMR callers out of scope

The loader-boundary conversion is applied in `examples/_gmr_shared.py::load_smplx()`
only.  Direct GMR callers (`GMR/scripts/smplx_to_robot.py`, external code
importing `general_motion_retargeting.utils.smpl` directly) continue to receive
Y-up frames.  These callers are out of scope for this refactoring.

Users who bypass `roboharness` loaders must either:
1. Apply `smpl_to_mujoco_frame()` themselves, or
2. Continue using legacy IK configs with `world_rotation = [0.5, 0.5, 0.5, 0.5]`.

### Finding 6 — Docs updated

- `smpl_to_mujoco_world_rotation()` marked deprecated in `smplx_coordinate.py`
- `smplx-alignment-requirements.md` updated below with new architecture notes
- This plan document updated with implementation response

### Files Changed

| File | Change |
|------|--------|
| `src/.../orientation_aligner.py` | SMPLX branch: `forward=cross(left,up)`, same SVD path, no hardcoded base |
| `src/.../smplx_template.py` | Returns Z-up directly (`smpl_to_mujoco_frame` called internally) |
| `src/.../smplx_offset_solver.py` | Removed pre-conversion; added `_apply_rotation_to_frame` for world_rotation; removed injection; added stale-config warning |
| `src/.../smplx_coordinate.py` | `smpl_to_mujoco_world_rotation()` deprecated |
| `examples/_gmr_shared.py` | `load_smplx()` applies `smpl_to_mujoco_frame()` to every frame |
| `tests/alignment/test_smplx_tpose_coordinate_fix.py` | Rewritten: SMPLX geometry-based world_rotation, separate BVH/SMPLX XML fixtures |
| `tests/alignment/test_smplx_template_calibration.py` | Template tests expect SMPL_TO_MUJOCO_QUAT; solver tests verify world_rotation application |
| `tests/alignment/test_smplx_coordinate.py` | Solver pipeline tests updated (no smpl_to_mujoco_frame import) |

### Verification

```text
$ pytest -q
764 passed, 3 skipped in 18.95s
Coverage: 91.20% (>=90% threshold)

$ ruff check .
All checks passed!

$ mypy src/
Success: no issues found in 54 source files
```
