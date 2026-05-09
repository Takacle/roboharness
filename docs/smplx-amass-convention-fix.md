# SMPLX AMASS Convention Detection & Fix Plan

_Created: 2026-05-09_
_Owner: opencode (GLM-5.1)_
_Status: Ready for Codex review_

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
