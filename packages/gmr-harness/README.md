# GMR-Harness

Alignment toolchain for **General Motion Retargeting (GMR)** + Robot Simulation.

Provides CLI commands and a Python library for SMPL-X template calibration, BVH motion loading, skeleton matching, IK config generation, quaternion offset solving, and VLM-based alignment agent iteration.

---

## Installation

```bash
pip install gmr-harness
```

### Optional dependencies

| Extra | Purpose |
|-------|---------|
| `[smplx]` | SMPL-X template calibration |
| `[mujoco]` | MuJoCo rendering and T-pose staging |
| `[vlm]` | VLM-based alignment agent |
| `[harness]` | VLM agent loop (pull in `roboharness`) |
| `[all]` | Everything above |

```bash
pip install gmr-harness[all]
```

### External dependency: GMR

[GMR (general_motion_retargeting)](https://github.com/Takacle/GMR) is **not** on PyPI. Clone it and point `GMR_ROOT`:

```bash
git clone <GMR_URL> /path/to/GMR
export GMR_ROOT=/path/to/GMR
```

Verify:

```bash
test -f "$GMR_ROOT/general_motion_retargeting/params.py"
```

`gmr-harness` also auto-discovers GMR placed at `../GMR` relative to your project.

---

## Quick start

```bash
# Setup a new robot
gmr-harness setup \
  --robot my_robot \
  --xml $GMR_ROOT/assets/my_robot/robot.xml \
  --formats bvh smplx

# Stage T-pose spec
gmr-harness stage \
  --robot my_robot \
  --src bvh \
  --preset tpose \
  --output_dir specs/tpose

# Direct IK quaternion solve
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --tpose_motion /path/to/tpose.bvh \
  --solve_mode

# Validate alignment
gmr-harness validate \
  --robot my_robot \
  --src bvh \
  --tpose_motion /path/to/tpose.bvh \
  --spec specs/tpose/my_robot.json
```

---

## Workflow

```
Prepare GMR repo + robot XML
        ↓
gmr-harness setup      Generate IK config, optionally register to GMR params.py
        ↓
gmr-harness stage      Generate specs/tpose/<robot>.json + 3-view reference PNGs
        ↓
gmr-harness agent      Direct IK quaternion offset solve (--solve_mode)
             │
             ├── gmr-harness validate    Numerical deviation check
             │
             └── gmr-harness agent       VLM-driven iterative tuning (optional)
```

Default spec path: `specs/tpose/<robot>.json` (relative to working directory). Check the PNGs into version control as visual ground truth.

---

## Detailed usage

### 1. Add a new robot

Place the MuJoCo XML at `$GMR_ROOT/assets/<robot>/robot.xml` (no nesting):

```bash
mkdir -p "$GMR_ROOT/assets/my_robot"
cp /path/to/robot.xml "$GMR_ROOT/assets/my_robot/robot.xml"
```

Dry-run to preview:

```bash
gmr-harness setup \
  --robot my_robot \
  --xml "$GMR_ROOT/assets/my_robot/robot.xml" \
  --formats bvh smplx \
  --dry_run
```

Generate IK config:

```bash
gmr-harness setup \
  --robot my_robot \
  --xml "$GMR_ROOT/assets/my_robot/robot.xml" \
  --formats bvh smplx
```

Configs land at:

```
$GMR_ROOT/general_motion_retargeting/ik_configs/bvh_to_my_robot.json
$GMR_ROOT/general_motion_retargeting/ik_configs/smplx_to_my_robot.json
```

Register to GMR `params.py` (required in non-TTY environments):

```bash
gmr-harness setup \
  --robot my_robot \
  --xml "$GMR_ROOT/assets/my_robot/robot.xml" \
  --formats bvh smplx \
  --auto_register \
  --yes
```

### 2. Stage T-pose spec

```bash
gmr-harness stage \
  --robot my_robot \
  --src bvh \
  --preset tpose \
  --output_dir specs/tpose
```

Inspecting the generated PNGs is **required** — a bad spec image means bad validation downstream.

List available joints:

```bash
gmr-harness stage --robot my_robot --src bvh --list_joints
```

Override specific joints:

```bash
gmr-harness stage \
  --robot my_robot \
  --src bvh \
  --preset tpose \
  --joint left_wrist_roll_joint=0.1 \
  --joint right_wrist_roll_joint=-0.1 \
  --output_dir specs/tpose
```

### 3. Direct IK quaternion solve

```bash
# Dry-run first
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --tpose_motion /path/to/tpose.bvh \
  --solve_mode \
  --dry_run

# Apply
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --tpose_motion /path/to/tpose.bvh \
  --solve_mode
```

First run creates a `.json.bak` backup of the existing IK config.

Preserve existing offsets on specific links:

```bash
--preserve "left_shoulder_yaw_link,right_shoulder_yaw_link"
```

Override world rotation:

```bash
--world_rot "90,0,0,1"    # angle_deg,axis_x,axis_y,axis_z
```

### 4. Validate alignment

```bash
gmr-harness validate \
  --robot my_robot \
  --src bvh \
  --tpose_motion /path/to/tpose.bvh \
  --spec specs/tpose/my_robot.json \
  --threshold 5.0
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | PASS — all deviations within threshold |
| 1 | FAIL — deviations exceed threshold |
| 2 | Parameter / spec / dependency error |

Deviation interpretation:

| Max deviation | Assessment |
|--------------|------------|
| < 1° | Excellent |
| 1–5° | Acceptable |
| 5–30° | Needs tuning — re-solve, check spec |
| 30–120° | Likely coordinate / quaternion misalignment |
| > 120° | Flipped — check 180° rotation, body mapping, source motion |

### 5. SMPL-X template calibration

Prefer template-based calibration for SMPL-X `.npz` (raw motion may have root orientation).

Default body model path:

```
$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz
```

Stage with SMPL-X:

```bash
gmr-harness stage \
  --robot my_robot \
  --src smplx \
  --preset tpose \
  --output_dir specs/tpose
```

Validate with template:

```bash
gmr-harness validate \
  --robot my_robot \
  --src smplx \
  --use_smplx_template \
  --spec specs/tpose/my_robot.json
```

Custom body model path:

```bash
gmr-harness validate \
  --robot my_robot \
  --src smplx \
  --use_smplx_template \
  --smplx_template_model /path/to/body_models \
  --spec specs/tpose/my_robot.json
```

### 6. VLM-driven iterative tuning

When direct solve is insufficient, the VLM agent iteratively adjusts parameters.

```bash
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/motion.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --tpose_motion /path/to/tpose.bvh \
  --tune_mode scale \
  --max_iter 8
```

Tuning modes:

| Mode | Purpose |
|------|---------|
| `scale` | VLM adjusts `human_scale_table` (default) |
| `weights` | VLM adjusts IK match table weights |
| `quaternion` | VLM adjusts quaternion offset |
| `optimize_scale` | Numerical optimization without VLM |

Default VLM model: `glm-5v-turbo`. Override:

```bash
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/motion.bvh \
  --model glm-5v-turbo \
  --api_base https://api.example.com/v1 \
  --api_key sk-xxx
```

Or via environment variable:

```bash
export OPENAI_API_KEY=sk-xxx
```

---

## CLI reference

| Command | Description |
|---------|-------------|
| `gmr-harness --help` | Top-level help |
| `gmr-harness setup --help` | Robot setup options |
| `gmr-harness stage --help` | T-pose staging options |
| `gmr-harness validate --help` | Validation options |
| `gmr-harness agent --help` | Agent / solve options |

---

## FAQ

### GMR not found

```
FileNotFoundError: GMR not found. Set GMR_ROOT env var or place GMR/ next to your project.
```

```bash
export GMR_ROOT=/path/to/GMR
test -f "$GMR_ROOT/general_motion_retargeting/params.py"
```

### XML path rejected

`--xml` must resolve to a file under `$GMR_ROOT/assets/<robot>/`. Move the XML and retry.

### params.py not modified in non-TTY

Explicitly pass `--auto_register --yes` to `gmr-harness setup`.

### ˜180° deviation

Common causes:
- Using a walking `.npz` as SMPL-X T-pose standard
- Incorrect T-pose spec
- Wrong `world_rotation`
- Left/right body mapping error

Fix order: check spec PNGs → re-solve BVH path with `--dry_run` → prefer `--use_smplx_template` for SMPL-X → set `--world_rot` explicitly.

### SMPL-X body model not found

```bash
ls "$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz"
```

Or pass `--smplx_template_model /path/to/body_models`.

---

## License

MIT
