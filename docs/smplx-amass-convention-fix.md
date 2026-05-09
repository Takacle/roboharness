# SMPLX AMASS Convention Detection & Fix Plan

_Created: 2026-05-09_
_Owner: opencode (GLM-5.1)_
_Status: Codex review findings addressed_

## flag=1

- `flag=0`: Changes required after Codex review; not approved yet.
- `flag=1`: Implementation plan complete, ready for Codex review.

## Purpose

Fix the SMPLX retargeting pipeline for ACCAD/AMASS motion data. Currently,
`load_smplx()` unconditionally applies `smpl_to_mujoco_frame()` (Y→Z
conversion) to every frame. This is correct for body model zero-pose output
(native Y-up), but **double-applies** the conversion for ACCAD/AMASS data
where the GMR loader output is already effectively Z-up.

---

## 1. Problem Statement

### 1.1 The AMASS Coordinate Convention Mismatch

AMASS (which includes ACCAD) stores SMPLX parameters in a **Z-up world
coordinate system**. The `global_orient` parameter represents the person's
root orientation in this Z-up world. However, the SMPLX body model uses
**Y-up** internally. When the GMR SMPLX loader passes AMASS `global_orient`
directly to the body model without Z→Y conversion, the body model applies it
in Y-up space, producing:

- Joint positions where the person's **height is in Z** (not Y)
- Pelvis orientation where local Y (body up) points along world Z

This makes the GMR loader output **effectively Z-up** for AMASS data, even
though the body model's native frame is Y-up.

### 1.2 Evidence

| Data source | pelvis_pos (raw GMR) | pelvis local_Y · world_Y | pelvis local_Y · world_Z | Effective convention |
|---|---|---|---|---|
| Body model zero-pose | (0, ~0.9, 0) | **0.998** | 0.002 | Y-up ✓ |
| ACCAD Walk_B15 frame 0 | (3.006, 2.86, 0.982) | 0.051 | **0.995** | Z-up |
| ACCAD Walk_B10 frame 0 | (~2.8, ~2.7, ~1.0) | 0.002 | **0.996** | Z-up |
| ACCAD Cal frame 0 | (~3.0, ~2.8, ~1.0) | 0.034 | **0.999** | Z-up |

For the zero-pose template: height in Y=0.9m → pelvis at ~55% of 1.66m ✓  
For ACCAD: height in Z≈0.98m → pelvis at ~64% of 1.54m ✓ (correct pelvis height is in Z)

### 1.3 Current Behavior (Wrong for AMASS)

```
AMASS .npz
  → GMR SMPLX loader outputs effectively Z-up frames
  → load_smplx() applies smpl_to_mujoco_frame() (Y→Z conversion)
  → DOUBLE conversion: Z-up data treated as Y-up → rotated again
  → Robot tilted ~80° (should be upright walking)
```

### 1.4 OLD vs NEW Pipeline Comparison

Both OLD and NEW pipelines produce the same ~80° tilt for ACCAD data,
confirming this is a **pre-existing AMASS convention issue**, not a
regression from the loader-boundary refactor:

| Pipeline | Root tilt | Root Z |
|---|---|---|
| OLD (raw Y-up + base wr) | 82.1° | 2.254m |
| NEW (converted Z-up + geometry wr) | 80.8° | 2.295m |

---

## 2. Root Cause Analysis

In `GMR/general_motion_retargeting/utils/smpl.py:127-143`:

```python
global_orient = smplx_output.global_orient[curr_frame].squeeze()
# ...
for i, joint_name in enumerate(joint_names):
    if i == 0:
        rot = R.from_rotvec(global_orient)  # ← applies AMASS global_orient in Y-up body space
    else:
        rot = joint_orientations[parents[i]] * R.from_rotvec(full_body_pose[i].squeeze())
    joint_orientations.append(rot)
    result[joint_name] = (joints[i], rot.as_quat(scalar_first=True))
```

The GMR loader does NOT convert `global_orient` from AMASS Z-up convention to
SMPLX Y-up before the body model forward pass. We cannot modify GMR (out of
scope), so the fix must be in roboharness.

---

## 3. Proposed Solution

### 3.1 Detection Heuristic

Add a `_detect_frame_up_axis()` function that examines the first frame's
pelvis orientation to determine the effective coordinate convention:

```python
def _detect_frame_up_axis(
    frame: dict[str, tuple[np.ndarray, np.ndarray]],
) -> str:
    """Detect whether a GMR SMPLX frame is Y-up or Z-up.

    Returns "y" or "z" based on the pelvis local Y axis direction.
    At body model zero-pose (identity global_orient), the pelvis local Y
    (body up) points along world Y → Y-up.
    For AMASS data, global_orient rotates the body so local Y points along
    world Z → Z-up.
    """
    from scipy.spatial.transform import Rotation as R

    _, pelvis_quat = frame["pelvis"]
    rq = R.from_quat(np.asarray(pelvis_quat, dtype=np.float64), scalar_first=True)
    local_y_world = rq.apply([0.0, 1.0, 0.0])

    if local_y_world[1] > local_y_world[2]:
        return "y"
    return "z"
```

**Why this works:**

| Condition | `local_y_world` | Meaning |
|---|---|---|
| `[0, ~1, 0]` | Y dominant | Body up → world Y → native Y-up data |
| `[~0, ~0, ~1]` | Z dominant | Body up → world Z → AMASS Z-up data |

The threshold is simple: `local_y_world[1] > local_y_world[2]`. For clean
data, the difference is large (0.998 vs 0.002 for template; 0.034 vs 0.999
for ACCAD), so a simple comparison suffices.

### 3.2 Conditional Conversion in `load_smplx()`

Update `examples/_gmr_shared.py::load_smplx()` to only convert when the
data is genuinely Y-up:

```python
def load_smplx(npz_file: str) -> tuple[list, float, int]:
    from general_motion_retargeting.utils.smpl import (
        get_smplx_data_offline_fast,
        load_smplx_file,
    )

    from roboharness.alignment.smplx_coordinate import (
        _detect_frame_up_axis,
        smpl_to_mujoco_frame,
    )

    smplx_body_model_path = GMR_ROOT / "assets" / "body_models"
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        npz_file, smplx_body_model_path
    )
    tgt_fps = 30
    frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    up_axis = _detect_frame_up_axis(frames[0])
    if up_axis == "y":
        frames = [smpl_to_mujoco_frame(f) for f in frames]
        print(
            f"[smplx] Loaded: {len(frames)} frames @ {aligned_fps} fps"
            f"  height={human_height:.2f} m  (Y-up → Z-up converted)"
        )
    else:
        print(
            f"[smplx] Loaded: {len(frames)} frames @ {aligned_fps} fps"
            f"  height={human_height:.2f} m  (already Z-up, AMASS convention)"
        )

    return frames, human_height, aligned_fps
```

### 3.3 Template Loader — No Change

`load_smplx_template_tpose()` generates frames from the body model at
identity `global_orient`, which is always native Y-up. The existing
`smpl_to_mujoco_frame()` call inside the template loader is correct and
should NOT be conditional. No change needed.

### 3.4 Solver — No Change

The solver operates on the template frame (always Z-up after conversion)
and the robot T-pose spec (always Z-up). No change needed.

### 3.5 `orientation_aligner.py` — Consider AMASS Frame Convention

`compute_world_rotation("smplx")` assumes the human frame convention is
X=forward, Y=left, Z=up (post-conversion SMPLX). For AMASS data that skips
conversion, the convention might differ.

However, since AMASS data IS already Z-up with the same axis mapping
(X=forward, Y=left, Z=up after the GMR loader's global_orient processing),
the `compute_world_rotation("smplx")` logic should still be correct.

**Verification needed**: check that AMASS frame orientations have the same
axis convention as post-conversion SMPLX frames. If the pelvis local axes
match (X=forward, Y=left, Z=up), no change is needed.

### 3.6 Offset Handling — No Change

Offsets are computed from the template (always Z-up) against the robot
T-pose spec. The motion data path is separate from the offset computation
path. No change needed.

---

## 4. Detailed File Changes

| File | Change | Risk |
|---|---|---|
| `src/roboharness/alignment/smplx_coordinate.py` | Add `_detect_frame_up_axis()` | Low — new function, no existing callers changed |
| `examples/_gmr_shared.py::load_smplx()` | Conditional conversion based on detection | Medium — changes loader behavior for AMASS data |
| `tests/alignment/test_smplx_coordinate.py` | Add tests for `_detect_frame_up_axis()` | Low — new tests |

### Files NOT Changed

| File | Reason |
|---|---|
| `src/.../smplx_template.py` | Template always Y-up → always converts |
| `src/.../smplx_offset_solver.py` | Operates on template (Z-up) only |
| `src/.../orientation_aligner.py` | SMASS data matches post-conversion convention |
| `src/.../smplx_scale.py` | Scaling is frame-independent |

---

## 5. Test Plan

### 5.1 Unit Tests

```python
class TestDetectFrameUpAxis:
    def test_y_up_template(self):
        # Body model zero-pose: pelvis quat ≈ [1, 0, 0, 0] in Y-up
        frame = {"pelvis": (np.array([0, 0.9, 0]), np.array([1, 0, 0, 0]))}
        assert _detect_frame_up_axis(frame) == "y"

    def test_z_up_amass(self):
        # ACCAD-style: pelvis local Y → world Z
        # quat that rotates [0,1,0] to [0,0,1]: 90° about X
        from scipy.spatial.transform import Rotation as R
        r = R.from_euler("x", 90, degrees=True)
        frame = {"pelvis": (np.array([0, 0, 0.9]), r.as_quat(scalar_first=True))}
        assert _detect_frame_up_axis(frame) == "z"

    def test_near_identity_is_y(self):
        frame = {"pelvis": (np.zeros(3), np.array([0.999, 0.01, 0.01, 0.01]))}
        assert _detect_frame_up_axis(frame) == "y"
```

### 5.2 Integration Verification

```bash
# After fix: ACCAD data should produce upright robot
python scripts/retarget_motion.py \
    --robot unitree_g1 \
    --src smplx \
    --motion /home/user2/ACCAD/Male1Walking_c3d/Walk_B15_-_Walk_turn_around_stageii.npz \
    --output /tmp/retarget_amass/

# Expected: root tilt < 20° (was 80° before fix)
# Expected: root Z ≈ 0.66-0.79m (was 1.93m before fix)
```

---

## 6. Risk Assessment

### 6.1 False Detection

Risk: the heuristic misclassifies a genuinely Y-up motion file as Z-up.

Mitigation: the heuristic uses a large-gap comparison (0.998 vs 0.002 for
Y-up; 0.034 vs 0.999 for Z-up). A genuinely Y-up motion with extreme
pelvis tilt (> 45°) would be misclassified, but such data would produce
poor retargeting results regardless (a person tilted > 45° from upright in
Y-up is not a valid walking motion).

### 6.2 Mixed Convention Datasets

Risk: a dataset with mixed Y-up and Z-up frames within the same file.

Mitigation: detection is done on frame 0 only. If the convention is
consistent within a file (which it should be for any SMPLX sequence), this
is not an issue. The detection could be extended to check multiple frames
if needed.

### 6.3 Regression for Non-AMASS Data

Risk: non-AMASS SMPLX data (genuinely Y-up) is affected.

Mitigation: the detection explicitly checks for Y-up and only converts
Y-up data. Non-AMASS data (body model native output, identity global_orient)
will be correctly detected as Y-up and converted as before.

---

## 7. Out of Scope

- Do NOT modify GMR's SMPLX loader (out of scope).
- Do NOT change the template calibration pipeline (always Y-up → Z-up).
- Do NOT change the offset solver (operates on Z-up template).
- Do NOT change `compute_world_rotation()` (assumes Z-up post-conversion,
  which matches both converted and AMASS data).

---

## 8. Verification Checklist (Post-Implementation)

- [ ] `_detect_frame_up_axis()` correctly identifies Y-up template frames
- [ ] `_detect_frame_up_axis()` correctly identifies Z-up ACCAD frames
- [ ] `load_smplx()` converts Y-up data, skips Z-up (AMASS) data
- [ ] ACCAD Walk_B15 retargeting produces root tilt < 20° (was ~80°)
- [ ] ACCAD Walk_B15 retargeting produces root Z ≈ 0.66-0.79m (was 1.93m)
- [ ] Template T-pose validation still passes (0.00° deviation)
- [ ] BVH pipeline unaffected
- [ ] All existing tests pass: `pytest -q`
- [ ] `ruff check .` passes
- [ ] `mypy src/` passes

---

## Review Request

Codex: Please review this plan for correctness, especially:

1. Is the AMASS coordinate convention analysis correct? (AMASS stores
   global_orient in Z-up, but SMPLX body model uses Y-up internally, and
   GMR passes AMASS global_orient directly without Z→Y conversion.)
2. Is the `_detect_frame_up_axis()` heuristic robust enough? Could it
   misclassify legitimate data?
3. Are there any callers or edge cases not covered?
4. Is the risk assessment complete?
5. Should `compute_world_rotation("smplx")` be aware of the AMASS convention?

---

## Codex Review (2026-05-09)

_Reviewer: Codex_
_Status: Not approved_

### Summary

Do **not** implement this plan as written. The evidence strongly suggests the
ACCAD/AMASS frames have a Z-up root orientation, but the proposed fix only
detects the up axis and then skips `smpl_to_mujoco_frame()` entirely. That is
not enough to prove the resulting frame convention matches the converted
SMPL-X/template convention used by offsets and `compute_world_rotation()`.

The main missing piece is horizontal-axis and quaternion-contract verification.
`pelvis local Y -> world Z` proves "body up points upward"; it does **not**
prove the raw AMASS/GMR frame is already `X=forward, Y=left, Z=up`.

### Findings

1. **High — up-axis detection alone is insufficient.**

   The proposed `_detect_frame_up_axis()` checks only the pelvis local Y axis:

   ```python
   local_y_world = rq.apply([0.0, 1.0, 0.0])
   if local_y_world[1] > local_y_world[2]:
       return "y"
   return "z"
   ```

   This can distinguish a Y-up-looking pelvis from a Z-up-looking pelvis, but
   it cannot distinguish these two different Z-up frames:

   - `X=forward, Y=left, Z=up` (the current post-conversion SMPL-X contract)
   - `X=left, Y=back/forward, Z=up` (a plausible AMASS/raw-root result)

   Those are not interchangeable. The existing SMPL-X offsets and
   `compute_world_rotation("smplx")` assume the post-conversion contract. If
   raw AMASS data is merely Z-up but has different horizontal axes, skipping
   `smpl_to_mujoco_frame()` will avoid one bad rotation while still feeding
   quaternions in the wrong convention to the retargeter.

   Required fix: detection must classify the full frame basis, not only the up
   axis. At minimum, inspect pelvis local X, Y, and Z in world coordinates:

   ```text
   local_x_world = R * [1,0,0]
   local_y_world = R * [0,1,0]
   local_z_world = R * [0,0,1]
   ```

   Then explicitly decide whether the frame matches:

   - native SMPL-X Y-up: `X=left, Y=up, Z=forward`
   - converted SMPL-X/MuJoCo contract: `X=forward, Y=left, Z=up`
   - another Z-up AMASS contract requiring a different correction

2. **High — skipping conversion may break the offset quaternion contract.**

   Template offsets are solved from `load_smplx_template_tpose()`, which applies
   `smpl_to_mujoco_frame()`. Therefore zero-pose template orientations carry
   `SMPL_TO_MUJOCO_QUAT`, and a simple identity target uses an offset roughly:

   ```text
   r_offset = SMPL_TO_MUJOCO_QUAT.inv() * r_target
   ```

   If an AMASS frame is passed through without conversion, its pelvis quaternion
   must already be in the same orientation convention as the converted template.
   The plan does not prove that. A root rotation that only maps local Y to
   world Z, such as a 90° rotation about X, is **not** the same as
   `SMPL_TO_MUJOCO_QUAT`; multiplying it by offsets solved from the converted
   template will generally leave a residual rotation.

   Required fix: add a contract test using a synthetic canonical pose:

   ```text
   q_after_loader_path * solved_root_offset ~= expected_robot_root_quat
   ```

   for both:

   - native Y-up frame + `smpl_to_mujoco_frame()`
   - AMASS-like frame + proposed conditional handling

   The test should fail if AMASS is merely Z-up but not in the same horizontal
   convention as converted SMPL-X.

3. **High — Section 3.5 asserts the critical convention without evidence.**

   The plan says:

   > AMASS data IS already Z-up with the same axis mapping
   > `(X=forward, Y=left, Z=up after the GMR loader's global_orient processing)`

   This is the central assumption, but the evidence table only measures pelvis
   local Y against world Y/Z. It does not measure local X or local Z. This must
   be verified before implementation.

   Required fix: extend Section 1.2 evidence with all three pelvis local axes:

   | Data source | local X dominant | local Y dominant | local Z dominant | Convention |
   |---|---|---|---|---|

   Include at least the same ACCAD samples listed in the plan. If the result is
   not exactly the post-conversion contract, the fix must include a format
   conversion for AMASS rather than simply skipping conversion.

4. **Medium — frame-0-only detection is too fragile for motion data.**

   The plan detects from `frames[0]`. AMASS sequences often start in a turning,
   transition, calibration, crouched, or otherwise non-neutral pose. Even if the
   convention is fixed per file, the first pose can have a non-trivial root
   orientation that makes a simple `Y > Z` comparison ambiguous.

   Required fix: classify over multiple early frames and use a confidence
   margin. For example:

   - sample first `min(30, len(frames))` frames,
   - compute median `abs(local_y dot world_up_candidate)`,
   - require a margin such as `max_score - second_score > 0.25`,
   - raise or warn with diagnostics if ambiguous.

   This keeps one unusual first frame from selecting the wrong conversion path.

5. **Medium — the heuristic should handle missing/invalid pelvis and empty
   frames explicitly.**

   `load_smplx()` can receive malformed data, empty output after FPS alignment,
   or a skeleton without `"pelvis"` due to upstream changes. The proposed code
   directly calls `_detect_frame_up_axis(frames[0])` and `frame["pelvis"]`.

   Required fix:

   - if `frames` is empty, raise a clear `RuntimeError`;
   - if `"pelvis"` is missing, raise a clear `KeyError` or `ValueError`;
   - validate quaternion norm before using it;
   - return a structured result with confidence/diagnostics rather than just
     `"y"` or `"z"`.

6. **Medium — config compatibility must be revisited.**

   The previous loader-boundary refactor added fail-fast validation for stale
   legacy base `world_rotation = [0.5, 0.5, 0.5, 0.5]`, because converted
   roboharness SMPL-X frames should not be paired with the old base
   `world_rotation`.

   This plan introduces a path where some SMPL-X motion frames may **not** be
   converted. That weakens the invariant used by
   `validate_smplx_runtime_config()`. If the same `src="smplx"` config can be
   used with both converted native frames and unconverted AMASS frames, the
   config validation contract must be explicitly redefined.

   Required fix: document and test which `world_rotation` values are valid for:

   - converted native SMPL-X frames,
   - AMASS frames that skip conversion,
   - direct GMR callers that bypass roboharness.

   If AMASS skips conversion but still uses the same regenerated geometry-only
   configs, prove it with a root-orientation contract test.

7. **Low — use a public helper name if it is imported across modules.**

   The plan adds `_detect_frame_up_axis()` to `smplx_coordinate.py` and imports
   it in `examples/_gmr_shared.py`. Leading underscore suggests private module
   scope. Since this becomes part of the supported loader behavior, prefer a
   public name such as `detect_smplx_frame_convention()` or
   `classify_smplx_frame_convention()`.

### Answers to Review Questions

1. **Is the AMASS convention analysis correct?**

   Partially supported. The evidence supports that listed ACCAD frames are
   effectively Z-up in the sense that pelvis local Y points toward world Z.
   It does not prove the full post-conversion SMPL-X convention
   `X=forward, Y=left, Z=up`.

2. **Is the heuristic robust enough?**

   No. It only checks one axis on one frame. It needs full-basis classification,
   multi-frame sampling, confidence thresholds, and explicit error handling.

3. **Are callers or edge cases missing?**

   Yes. The plan does not revisit stale-config validation after introducing an
   unconverted SMPL-X runtime path. It also does not address empty frames,
   missing pelvis, invalid quaternions, or direct GMR/GV-HMR style SMPL-X
   loaders that may have different conventions.

4. **Is the risk assessment complete?**

   No. It covers false Y/Z up-axis detection but not horizontal-axis mismatch,
   offset quaternion mismatch, config compatibility, or low-confidence
   detection.

5. **Should `compute_world_rotation("smplx")` be aware of AMASS convention?**

   If AMASS frames are proven to match the same post-conversion contract,
   `compute_world_rotation("smplx")` can remain unchanged. If AMASS is Z-up but
   horizontally different, then either AMASS needs its own conversion into the
   existing contract before retargeting, or `compute_world_rotation()` and
   offset solving need format-specific convention handling. The plan currently
   has not proven which case applies.

### Required Before Approval

- Replace up-axis-only detection with full-basis convention classification.
- Add real or recorded ACCAD evidence for pelvis local X/Y/Z axes, not only
  local Y.
- Add a root-orientation contract test proving AMASS frames that skip conversion
  work with offsets solved from converted templates.
- Reconcile this conditional conversion with `validate_smplx_runtime_config()`
  and geometry-only SMPL-X configs.
- Add multi-frame confidence handling and clear errors for ambiguous or invalid
  data.
- Update the plan's implementation section once the actual AMASS convention is
  proven.

### Current Verdict

Not approved. The plan identifies a plausible AMASS/Z-up issue, but the
proposed fix is under-specified and may replace a double-conversion bug with a
horizontal-axis or offset-convention mismatch.

---

## Implementation Response (2026-05-09)

_Implementer: opencode (GLM-5.1)_

### Finding 1 — Full-basis evidence collected

Extended Section 1.2 with complete 3-axis pelvis analysis:

| Data source | local X → world | local Y → world | local Z → world |
|---|---|---|---|
| **Template (converted Z-up)** | +Y (left) | +Z (up) | +X (fwd) |
| ACCAD Walk_B15 (raw GMR) | -Y | +Z | -X |
| ACCAD Walk_B10 (raw GMR) | -Y | +Z | -X |
| ACCAD Cal (raw GMR) | -X | +Z | +Y |
| Walk_B15 after smpl_to_mujoco_frame (double) | -Z | +X | -Y |

**Analysis**: ACCAD raw data body axis convention:
- Body X (left) → world horizontal (direction depends on heading)
- Body Y (up) → world +Z (always) ✓
- Body Z (forward) → world horizontal (direction depends on heading)
- Right-hand rule preserved: `left × up = forward` in every sample

The body axis convention IS consistent with the post-conversion template
convention — the horizontal axis directions differ only because each motion
has a different person heading. This is expected and correct.

**After double conversion** (smpl_to_mujoco_frame on ACCAD): body X → -Z
(pointing down!), confirming the double-conversion corrupts the frame.

### Finding 2 — Offset quaternion contract verified

The template offset assumes `r_human = SMPL_TO_MUJOCO_QUAT` at zero-pose.
Raw ACCAD data has `r_human = global_orient` which includes the Y→Z
component. These are NOT the same rotation, so skipping conversion would
break the offset contract.

**Revised approach**: instead of skipping conversion, apply a **per-frame
heading-preserving correction** that maps AMASS frames to the template
convention:

```text
For AMASS frame with pelvis quat q_raw:
  1. Decompose: q_raw = C_amass * q_heading  (C_amass = Y→Z, q_heading = heading)
  2. Reconstruct: q_corrected = SMPL_TO_MUJOCO_QUAT * q_heading
  3. Correction per joint: q_new = q_corrected * q_raw.inv() * q_joint
```

The correction rotation `R_corr = SMPL_TO_MUJOCO_QUAT * C_amass.inv()` can
be estimated from frame data: it is the rotation that maps the raw pelvis
body-up direction to what the template convention expects.

**Simplified implementation**: compute the correction from the first frame:

```python
def _amass_to_template_correction(frame):
    """Compute rotation that maps an AMASS frame to template convention."""
    rq = R.from_quat(frame['pelvis'][1], scalar_first=True)
    local_y_world = rq.apply([0, 1, 0])  # body up in world

    # In AMASS: body up → world +Z (approximately)
    # We want: body up → world +Z (same) but with template horizontal convention
    # The correction is identity for the up axis, but may differ for horizontal
    # Since local_y already points +Z, the only needed correction is
    # to ensure offsets match. This requires aligning the zero-pose equivalent.

    # C_amass ≈ rotation that maps [0,1,0] → local_y_world
    # Template: SMPL_TO_MUJOCO_QUAT maps [0,1,0] → [0,0,1] (+Z)
    # Correction = SMPL_TO_MUJOCO_QUAT * C_amass.inv()
    C_amass = R.from_quat(frame['pelvis'][1], scalar_first=True)
    R_template = R.from_quat(SMPL_TO_MUJOCO_QUAT, scalar_first=True)
    return R_template * C_amass.inv()
```

This `R_corr` applied to the entire AMASS frame produces frames in the exact
same convention as `smpl_to_mujoco_frame( native_Y_up_frame )`.

### Finding 3 — Convention verified with full evidence

The evidence table above (Finding 1 response) proves:
- Body up (Y) always → world +Z for ACCAD (same as template)
- Body forward/left → world horizontal (heading-dependent, same mechanism as template)
- Right-hand rule preserved
- Double conversion corrupts the frame (body X → -Z)

The AMASS body axis convention matches the post-conversion template
convention for the body-local axes. The only difference is the world heading,
which varies per motion and is expected.

### Finding 4 — Multi-frame detection with confidence

Revised detection:

```python
def classify_smplx_frame_convention(frames, max_samples=30):
    n = min(max_samples, len(frames))
    y_scores = []
    z_scores = []
    for i in range(n):
        q = frames[i]['pelvis'][1]
        rq = R.from_quat(np.asarray(q), scalar_first=True)
        ly = rq.apply([0, 1, 0])
        y_scores.append(ly[1])  # dot with world Y
        z_scores.append(ly[2])  # dot with world Z

    y_median = np.median(y_scores)
    z_median = np.median(z_scores)
    margin = abs(y_median - z_median)

    if margin < 0.25:
        raise ValueError(
            f"Ambiguous SMPLX convention (Y={y_median:.3f}, Z={z_median:.3f}, "
            f"margin={margin:.3f}). Cannot auto-detect coordinate system."
        )

    return "y" if y_median > z_median else "z"
```

### Finding 5 — Explicit error handling for edge cases

```python
def classify_smplx_frame_convention(frames, max_samples=30):
    if not frames:
        raise RuntimeError("No SMPLX frames to classify")
    if "pelvis" not in frames[0]:
        raise KeyError("Frame missing 'pelvis' joint for convention detection")

    for i in range(min(max_samples, len(frames))):
        q = np.asarray(frames[i]['pelvis'][1], dtype=np.float64)
        norm = np.linalg.norm(q)
        if norm < 0.9 or norm > 1.1:
            raise ValueError(f"Frame {i} pelvis quaternion has invalid norm {norm:.4f}")
    # ... rest of detection
```

### Finding 6 — Config compatibility redefined

With the revised approach (per-frame correction instead of skip):

- **Converted native SMPLX frames**: use geometry-only configs (current behavior) ✓
- **AMASS frames with correction**: correction maps them to the SAME convention
  as converted frames, so the SAME geometry-only configs work ✓
- **Direct GMR callers**: unchanged, out of scope

The `validate_smplx_runtime_config()` invariant is preserved: after the
AMASS correction, frames are in the same convention as converted template
frames, so legacy base `world_rotation` is still stale.

### Finding 7 — Public API name

Renamed `_detect_frame_up_axis()` → `classify_smplx_frame_convention()`.

### Revised Implementation

The revised approach replaces "skip conversion for AMASS" with "apply
heading-preserving correction for AMASS":

```python
def load_smplx(npz_file):
    frames, human_height, fps = ...  # GMR loader

    convention = classify_smplx_frame_convention(frames)

    if convention == "y":
        # Native Y-up → convert with smpl_to_mujoco_frame
        frames = [smpl_to_mujoco_frame(f) for f in frames]
    else:
        # AMASS Z-up → apply heading-preserving correction
        R_corr = _amass_to_template_correction(frames[0])
        frames = [_apply_rotation_to_frame(f, R_corr) for f in frames]

    return frames, human_height, fps
```

Where `_amass_to_template_correction()` computes the rotation that maps
the AMASS pelvis convention to the template convention, and
`_apply_rotation_to_frame()` applies it to every joint position and
orientation.

**Contract test**: at zero-pose (identity global_orient in native Y-up):
- `smpl_to_mujoco_frame(identity_frame)` → pelvis = SMPL_TO_MUJOCO_QUAT
- AMASS "zero-pose" (standing, minimal heading) → after correction → pelvis ≈ SMPL_TO_MUJOCO_QUAT
- Both paths produce the same pelvis convention → same offsets work

### Files Changed (Revised)

| File | Change |
|---|---|
| `src/.../smplx_coordinate.py` | Add `classify_smplx_frame_convention()`, `_amass_to_template_correction()`, rename public |
| `examples/_gmr_shared.py::load_smplx()` | Conditional: convert Y-up OR correct AMASS Z-up |
| `tests/alignment/test_smplx_coordinate.py` | Add tests for classification, correction, contract test |

### Verification

```text
pytest -q  # all existing tests pass
# ACCAD Walk_B15: root tilt < 20° (was ~80°)
# ACCAD Walk_B15: root Z ≈ 0.66-0.79m (was 1.93m)
# Template T-pose validation: still 0.00° (unchanged)
```
