# GMR-Harness

[中文文档](./README.zh-CN.md)

**GMR-Harness** is an alignment toolchain for [GMR](https://github.com/YanjieZe/GMR) (General Motion Retargeting) + Robot Simulation. It provides CLI commands and a Python library for SMPL-X template calibration, BVH motion loading, skeleton matching, IK config generation, quaternion offset solving, and VLM-based alignment agent iteration.

---

## Why GMR-Harness?

Standalone GMR is a powerful retargeting framework, but using it in production requires additional tooling to bridge the gap between retargeted output and physically plausible robot motion.

### Key features

1. **Automated IK config generation** — Given a new robot's MuJoCo XML, fuzzy-matches joint names across 14 semantic body roles and generates a complete IK config JSON automatically.

2. **T-pose spec staging** — Generate a T-pose specification (JSON + 3-view PNGs) for any robot as the alignment ground truth.

3. **Direct IK quaternion offset solve** — One-shot solve from human bone orientations to correct body-part rotation offsets.

4. **Numerical alignment validation** — Per-link deviation metrics with configurable thresholds and detailed worst-offender reporting.

5. **VLM-driven iterative tuning** — Automatic scale / weights / quaternion adjustment via vision-language model agent, without manual parameter tweaking.

6. **SMPL-X template calibration** — Root-orientation-independent calibration for SMPL-X `.npz` motion data.

7. **Visual replay** — Pause → capture → resume visual inspection of retargeted motion.

---

## Quick Start

### Installation

```bash
pip install gmr-harness[all]
```

Optional extras:

| Extra | Purpose |
|-------|---------|
| `[smplx]` | SMPL-X template calibration |
| `[mujoco]` | MuJoCo rendering and T-pose staging |
| `[vlm]` | VLM-based alignment agent |
| `[harness]` | VLM agent loop (requires `roboharness`) |

### GMR setup

Clone GMR (not on PyPI) and set `GMR_ROOT`:

```bash
git clone <GMR_URL> /path/to/GMR
export GMR_ROOT=/path/to/GMR
```

Verify:

```bash
test -f "$GMR_ROOT/general_motion_retargeting/params.py"
```

`gmr-harness` also auto-discovers GMR at `../GMR` relative to your project.

### Run your first alignment

```bash
# 1. Setup a new robot
gmr-harness setup --robot my_robot --xml $GMR_ROOT/assets/my_robot/robot.xml --formats bvh smplx

# 2. Stage T-pose spec
gmr-harness stage --robot my_robot --src bvh --preset tpose --output_dir specs/tpose

# 3. Direct IK quaternion solve
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json --tpose_motion /path/to/tpose.bvh --solve_mode

# 4. Validate
gmr-harness validate --robot my_robot --src bvh --tpose_motion /path/to/tpose.bvh \
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

---

## Usage

### Add a new robot

Place the MuJoCo XML at `$GMR_ROOT/assets/<robot>/robot.xml`:

```bash
mkdir -p "$GMR_ROOT/assets/my_robot"
cp /path/to/robot.xml "$GMR_ROOT/assets/my_robot/robot.xml"
```

Dry-run first:

```bash
gmr-harness setup --robot my_robot --xml "$GMR_ROOT/assets/my_robot/robot.xml" --formats bvh smplx --dry_run
```

Generate IK configs:

```bash
gmr-harness setup --robot my_robot --xml "$GMR_ROOT/assets/my_robot/robot.xml" --formats bvh smplx
```

Configs land at:

| Source format | Output path |
|---------------|-------------|
| BVH | `$GMR_ROOT/.../ik_configs/bvh_to_my_robot.json` |
| SMPL-X | `$GMR_ROOT/.../ik_configs/smplx_to_my_robot.json` |

Register to GMR `params.py` (required in non-TTY environments):

```bash
gmr-harness setup --robot my_robot --xml "$GMR_ROOT/assets/my_robot/robot.xml" --formats bvh smplx --auto_register --yes
```

### Stage T-pose spec

```bash
gmr-harness stage --robot my_robot --src bvh --preset tpose --output_dir specs/tpose
```

Common flags:

| Flag | Effect |
|------|--------|
| `--list_joints` | Print available joint names |
| `--joint name=value` | Override specific joint angles |
| `--qpos_file path` | Reuse existing qpos from a JSON file |

**Always inspect the generated PNGs** — a bad spec image means bad validation downstream.

### Solve IK quaternion offset

```bash
# Dry-run first
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json --tpose_motion /path/to/tpose.bvh \
  --solve_mode --dry_run

# Apply
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json --tpose_motion /path/to/tpose.bvh \
  --solve_mode
```

Common flags:

| Flag | Effect |
|------|--------|
| `--preserve "link1,link2"` | Preserve existing offsets on specific links |
| `--world_rot "90,0,0,1"` | Override world rotation (angle_deg,axis_x,axis_y,axis_z) |

First run creates a `.json.bak` backup.

### Validate alignment

```bash
gmr-harness validate --robot my_robot --src bvh \
  --tpose_motion /path/to/tpose.bvh --spec specs/tpose/my_robot.json --threshold 5.0
```

Exit codes:

| Code | Meaning |
|------|---------|
| 0 | PASS — all deviations within threshold |
| 1 | FAIL — deviations exceed threshold |
| 2 | Parameter / spec / dependency error |

Deviation guide:

| Max deviation | Assessment |
|--------------|------------|
| < 1° | Excellent |
| 1–5° | Acceptable |
| 5–30° | Needs tuning — re-solve, check spec |
| 30–120° | Likely coordinate / quaternion misalignment |
| > 120° | Flipped — check 180° rotation, body mapping, source motion |

### SMPL-X template calibration

Prefer template-based calibration for SMPL-X `.npz` (raw motion may have root orientation).

Default body model: `$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz`

```bash
# Stage with SMPL-X
gmr-harness stage --robot my_robot --src smplx --preset tpose --output_dir specs/tpose

# Validate with template
gmr-harness validate --robot my_robot --src smplx \
  --use_smplx_template --spec specs/tpose/my_robot.json

# Custom body model path
gmr-harness validate --robot my_robot --src smplx \
  --use_smplx_template --smplx_template_model /path/to/body_models \
  --spec specs/tpose/my_robot.json
```

### VLM-driven iterative tuning

When direct solve isn't enough, the VLM agent iteratively adjusts parameters.

```bash
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/motion.bvh \
  --tpose_spec specs/tpose/my_robot.json --tpose_motion /path/to/tpose.bvh \
  --tune_mode scale --max_iter 8
```

Tuning modes:

| Mode | Purpose |
|------|---------|
| `scale` | VLM adjusts `human_scale_table` (default) |
| `weights` | VLM adjusts IK match table weights |
| `quaternion` | VLM adjusts quaternion offset |
| `optimize_scale` | Numerical optimization without VLM |

Default VLM model: `glm-5v-turbo`. Override via flags or `OPENAI_API_KEY` env var:

```bash
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/motion.bvh \
  --model glm-5v-turbo --api_base https://api.example.com/v1 --api_key sk-xxx
```

---

## CLI Reference

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

The XML must be at `$GMR_ROOT/assets/<robot>/`. Move it and retry.

### params.py not modified

Pass `--auto_register --yes` (required in non-TTY environments).

### ~180° deviation

Common causes:
- Using a walking `.npz` as SMPL-X T-pose standard
- Incorrect T-pose spec
- Wrong `world_rotation`
- Left/right body mapping error

Fix order: check spec PNGs → re-solve with `--dry_run` → use `--use_smplx_template` for SMPL-X → set `--world_rot` explicitly.

### SMPL-X body model not found

```bash
ls "$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz"
```

Or pass `--smplx_template_model /path/to/body_models`.

---

## License

[MIT](LICENSE)


