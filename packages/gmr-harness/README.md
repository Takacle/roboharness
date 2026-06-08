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

---

# GMR-Harness（中文）

**General Motion Retargeting (GMR) + 机器人仿真** 的对齐工具链。

提供 CLI 命令和 Python 库，支持 SMPL-X 模板校准、BVH 运动数据加载、骨骼匹配、IK 配置生成、四元数偏移求解和 VLM 驱动的对齐智能体。

---

## 为什么需要 GMR-Harness？

GMR 本身是强大的重定向框架，但在生产环境中使用需要额外工具来弥合输出与物理合理机器人运动之间的差距。

### 核心功能

1. **自动 IK 配置生成** — 给定新机器人的 MuJoCo XML，跨 14 个语义身体角色模糊匹配关节名称，自动生成完整 IK 配置 JSON。

2. **T-pose 规格制作** — 为任何机器人生成 T-pose 规格（JSON + 三视图 PNG），作为对齐的数值基准。

3. **直接 IK 四元数偏移求解** — 从人体骨骼朝向一步求解，修正身体部位的旋转偏移。

4. **数值对齐验证** — 每个关节的偏差指标，可配置阈值，详细报告最差关节。

5. **VLM 驱动的迭代调优** — 通过视觉语言模型智能体自动调整 scale / weights / quaternion，无需手动调参。

6. **SMPL-X 模板校准** — 对 SMPL-X `.npz` 运动数据进行与根节点朝向无关的校准。

7. **视觉回放** — 暂停→捕获→恢复的视觉检查流程。

---

## 快速开始

### 安装

```bash
pip install gmr-harness[all]
```

可选依赖：

| 额外包 | 用途 |
|--------|------|
| `[smplx]` | SMPL-X 模板校准 |
| `[mujoco]` | MuJoCo 渲染和 T-pose 制作 |
| `[vlm]` | VLM 视觉对齐智能体 |
| `[harness]` | 智能体循环（依赖 `roboharness`） |

### GMR 配置

克隆 GMR（不在 PyPI 上）并设置 `GMR_ROOT`：

```bash
git clone <GMR_URL> /path/to/GMR
export GMR_ROOT=/path/to/GMR
```

验证：

```bash
test -f "$GMR_ROOT/general_motion_retargeting/params.py"
```

`gmr-harness` 也会自动检测项目同级目录 `../GMR`。

### 运行第一次对齐

```bash
# 1. 配置新机器人
gmr-harness setup --robot my_robot --xml $GMR_ROOT/assets/my_robot/robot.xml --formats bvh smplx

# 2. 制作 T-pose 规格
gmr-harness stage --robot my_robot --src bvh --preset tpose --output_dir specs/tpose

# 3. 直接求解 IK quaternion offset
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json --tpose_motion /path/to/tpose.bvh --solve_mode

# 4. 验证
gmr-harness validate --robot my_robot --src bvh --tpose_motion /path/to/tpose.bvh \
  --spec specs/tpose/my_robot.json
```

---

## 工作流总览

```
准备 GMR 仓库 + 机器人 XML
        ↓
gmr-harness setup      生成 IK config，可选注册到 GMR params.py
        ↓
gmr-harness stage      生成 specs/tpose/<robot>.json 和三视图参考图
        ↓
gmr-harness agent      直接求解 IK quaternion offset（--solve_mode）
             │
             ├── gmr-harness validate    数值偏差检查
             │
             └── gmr-harness agent       VLM 迭代调优（可选）
```

---

## 详细用法

### 添加新机器人

将 MuJoCo XML 放在 `$GMR_ROOT/assets/<robot>/robot.xml`：

```bash
mkdir -p "$GMR_ROOT/assets/my_robot"
cp /path/to/robot.xml "$GMR_ROOT/assets/my_robot/robot.xml"
```

先 dry-run：

```bash
gmr-harness setup --robot my_robot --xml "$GMR_ROOT/assets/my_robot/robot.xml" --formats bvh smplx --dry_run
```

生成 IK 配置：

```bash
gmr-harness setup --robot my_robot --xml "$GMR_ROOT/assets/my_robot/robot.xml" --formats bvh smplx
```

配置文件路径：

| 源格式 | 输出路径 |
|--------|----------|
| BVH | `$GMR_ROOT/.../ik_configs/bvh_to_my_robot.json` |
| SMPL-X | `$GMR_ROOT/.../ik_configs/smplx_to_my_robot.json` |

注册到 GMR `params.py`（非 TTY 环境必须显式指定）：

```bash
gmr-harness setup --robot my_robot --xml "$GMR_ROOT/assets/my_robot/robot.xml" --formats bvh smplx --auto_register --yes
```

### 制作 T-pose 规格

```bash
gmr-harness stage --robot my_robot --src bvh --preset tpose --output_dir specs/tpose
```

常用参数：

| 参数 | 作用 |
|------|------|
| `--list_joints` | 打印可用关节名称 |
| `--joint name=值` | 覆盖指定关节角度 |
| `--qpos_file 路径` | 复用已有 qpos JSON |

**务必肉眼检查生成的 PNG**——规格图错误会导致后续验证全部错误。

### 求解 IK quaternion offset

```bash
# 先用 dry-run 验证
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json --tpose_motion /path/to/tpose.bvh \
  --solve_mode --dry_run

# 正式写入
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json --tpose_motion /path/to/tpose.bvh \
  --solve_mode
```

常用参数：

| 参数 | 作用 |
|------|------|
| `--preserve "link1,link2"` | 保留指定关节的已有偏移 |
| `--world_rot "90,0,0,1"` | 覆盖 world rotation（角度,轴x,轴y,轴z） |

首次运行会自动创建 `.json.bak` 备份。

### 验证对齐结果

```bash
gmr-harness validate --robot my_robot --src bvh \
  --tpose_motion /path/to/tpose.bvh --spec specs/tpose/my_robot.json --threshold 5.0
```

退出码：

| 退出码 | 含义 |
|--------|------|
| 0 | 通过——所有偏差在阈值内 |
| 1 | 未通过——偏差超过阈值 |
| 2 | 参数/规格/依赖错误 |

偏差解读：

| 最大偏差 | 判断 |
|----------|------|
| < 1° | 很好 |
| 1–5° | 可接受 |
| 5–30° | 需要调优——重新 solve，检查 spec |
| 30–120° | 坐标或四元数可能错误 |
| > 120° | 翻转——检查 180° 旋转、body mapping、源运动 |

### SMPL-X 模板校准

SMPL-X `.npz` 优先使用模板校准（原始运动可能带有根节点朝向）。

默认 body model 路径：`$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz`

```bash
# 使用 SMPL-X 制作 T-pose
gmr-harness stage --robot my_robot --src smplx --preset tpose --output_dir specs/tpose

# 使用模板验证
gmr-harness validate --robot my_robot --src smplx \
  --use_smplx_template --spec specs/tpose/my_robot.json

# 自定义 body model 路径
gmr-harness validate --robot my_robot --src smplx \
  --use_smplx_template --smplx_template_model /path/to/body_models \
  --spec specs/tpose/my_robot.json
```

### VLM 迭代调优

当直接求解无法满足要求时，VLM 智能体可以迭代调整参数。

```bash
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/motion.bvh \
  --tpose_spec specs/tpose/my_robot.json --tpose_motion /path/to/tpose.bvh \
  --tune_mode scale --max_iter 8
```

调优模式：

| 模式 | 用途 |
|------|------|
| `scale` | VLM 调整 `human_scale_table`（默认） |
| `weights` | VLM 调整 IK match table 权重 |
| `quaternion` | VLM 调整四元数偏移 |
| `optimize_scale` | 无 VLM，使用数值优化 |

默认 VLM 模型：`glm-5v-turbo`。可通过参数或 `OPENAI_API_KEY` 环境变量覆盖：

```bash
gmr-harness agent --robot my_robot --src bvh --motion_file /path/to/motion.bvh \
  --model glm-5v-turbo --api_base https://api.example.com/v1 --api_key sk-xxx
```

---

## CLI 速查

| 命令 | 说明 |
|------|------|
| `gmr-harness --help` | 总帮助 |
| `gmr-harness setup --help` | 机器人配置参数 |
| `gmr-harness stage --help` | T-pose 制作参数 |
| `gmr-harness validate --help` | 验证参数 |
| `gmr-harness agent --help` | 智能体/求解参数 |

---

## 常见问题

### GMR 找不到

```
FileNotFoundError: GMR not found. 请设置 GMR_ROOT 环境变量或将 GMR/ 放在项目同级目录。
```

```bash
export GMR_ROOT=/path/to/GMR
test -f "$GMR_ROOT/general_motion_retargeting/params.py"
```

### XML 路径不被接受

XML 必须在 `$GMR_ROOT/assets/<robot>/` 下。移动后重试。

### params.py 没有更新

传递 `--auto_register --yes`（非 TTY 环境必需）。

### 接近 180° 偏差

常见原因：
- 使用普通行走 `.npz` 当作 SMPL-X T-pose 标准
- T-pose spec 本身错误
- `world_rotation` 方向不对
- 左右 body mapping 错误

处理顺序：检查 spec PNG → 重新 `--dry_run` → SMPL-X 优先 `--use_smplx_template` → 显式设置 `--world_rot`。

### SMPL-X body model 找不到

```bash
ls "$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz"
```

或传入 `--smplx_template_model /path/to/body_models`。

---

## 许可证

[MIT](LICENSE)
