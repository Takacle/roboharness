# GMR-Harness 从零开始使用指南

本指南面向第一次使用 `gmr-harness` 的用户，目标是从一个可用的 MuJoCo 机器人 XML 开始，完成 GMR 注册、IK 配置生成、T-pose 规格制作、姿态偏移求解和验证。

`gmr-harness` 是独立 Python 包。新流程优先使用 `gmr-harness ...` 命令；仓库根目录下的 `scripts/*.py` 和 `examples/gmr_*.py` 只作为旧入口兼容保留。

---

## 1. 你需要准备什么

### 必需条件

- Python 3.10 或更高版本。
- 一个外部 GMR 仓库，目录内必须有 `general_motion_retargeting/params.py`。
- 一个机器人 MuJoCo XML，建议直接放在 `$GMR_ROOT/assets/<robot>/` 下。
- 至少一种人类运动源：BVH、SMPL-X `.npz`，或 GMR 支持的 offline FBX 数据。

### 常用可选依赖

| 功能 | 需要安装 |
|------|----------|
| T-pose stage / MuJoCo 渲染 | `gmr-harness[mujoco]` |
| SMPL-X 模板校准 | `gmr-harness[smplx]` |
| VLM 视觉迭代 agent | `gmr-harness[vlm]` |
| 一次安装全部可选能力 | `gmr-harness[all]` |

---

## 2. 安装

### 2.1 从 PyPI 安装

```bash
pip install gmr-harness[all]
```

如果你只需要基础 CLI 和配置生成，可以先安装最小包：

```bash
pip install gmr-harness
```

### 2.2 在本仓库开发环境安装

```bash
cd /home/user2/roboharness/packages/gmr-harness
pip install -e ".[dev]"
```

如需完整运行 MuJoCo、SMPL-X 和 VLM 流程：

```bash
pip install -e ".[all,dev]"
```

---

## 3. 配置 GMR_ROOT

`gmr-harness` 不把 GMR 当作 pip 依赖安装。你需要让它能找到 GMR 仓库。

推荐显式设置：

```bash
export GMR_ROOT=/path/to/GMR
```

检查：

```bash
test -f "$GMR_ROOT/general_motion_retargeting/params.py"
```

如果你把 GMR 放在当前项目或 `roboharness` 的同级目录下，`gmr-harness` 也会尝试自动发现：

```text
parent/
  GMR/
    general_motion_retargeting/params.py
  your-project/
```

---

## 4. 验证 CLI 可用

任何外部依赖未准备好时，`--help` 也应该能运行：

```bash
gmr-harness --help
gmr-harness setup --help
gmr-harness stage --help
gmr-harness validate --help
gmr-harness agent --help
```

在源码树内也可以用模块方式验证：

```bash
cd /home/user2/roboharness/packages/gmr-harness
PYTHONPATH=src python -m gmr_harness.cli.main --help
```

---

## 5. 推荐工作流总览

典型流程如下：

```text
准备 GMR + robot XML
        ↓
gmr-harness setup      生成 IK config，可选写入 GMR params.py
        ↓
gmr-harness stage      生成 specs/tpose/<robot>.json 和三视图参考图
        ↓
gmr-harness agent      solve_mode 直接求解 IK quaternion offset
        ↓
gmr-harness validate   数值验证 T-pose 偏差
```

默认规格路径统一为当前工作目录下的 `specs/tpose/<robot>.json`。建议在你的项目根目录运行这些命令，并把 `specs/tpose/*.json` 和参考 PNG 纳入版本控制。

---

## 6. 从零添加一个新机器人

下面以机器人名 `my_robot` 为例。

### 6.1 放置 XML

XML 应直接位于 `$GMR_ROOT/assets/<robot>/` 下，不能嵌套到更深目录：

```bash
mkdir -p "$GMR_ROOT/assets/my_robot"
cp /path/to/robot.xml "$GMR_ROOT/assets/my_robot/robot.xml"
```

正确：

```text
$GMR_ROOT/assets/my_robot/robot.xml
```

不推荐：

```text
$GMR_ROOT/assets/my_robot/variants/robot.xml
```

### 6.2 先 dry-run 查看将会做什么

```bash
gmr-harness setup \
  --robot my_robot \
  --xml "$GMR_ROOT/assets/my_robot/robot.xml" \
  --formats bvh smplx \
  --dry_run
```

`--dry_run` 不写入文件，适合检查 body matching、root body、输出路径和缺失的 GMR params 项。

### 6.3 生成 IK config

```bash
gmr-harness setup \
  --robot my_robot \
  --xml "$GMR_ROOT/assets/my_robot/robot.xml" \
  --formats bvh smplx
```

生成的配置通常位于：

```text
$GMR_ROOT/general_motion_retargeting/ik_configs/bvh_to_my_robot.json
$GMR_ROOT/general_motion_retargeting/ik_configs/smplx_to_my_robot.json
```

### 6.4 注册到 GMR params.py

`setup` 默认不会在非 TTY 环境静默写 `params.py`。确认无误后使用 `--auto_register --yes`：

```bash
gmr-harness setup \
  --robot my_robot \
  --xml "$GMR_ROOT/assets/my_robot/robot.xml" \
  --formats bvh smplx \
  --auto_register \
  --yes
```

如需同步更新 GMR 脚本里的 robot choices，可加：

```bash
--update_scripts
```

---

## 7. 制作 T-pose 规格

T-pose 规格是后续验证的数值基准。每个机器人至少应该有一份：

```text
specs/tpose/<robot>.json
specs/tpose/<robot>_front.png
specs/tpose/<robot>_side.png
specs/tpose/<robot>_back.png
```

### 7.1 自动生成 T-pose

```bash
gmr-harness stage \
  --robot my_robot \
  --src bvh \
  --preset tpose \
  --output_dir specs/tpose
```

### 7.2 查看关节列表

如果自动 T-pose 不理想，先列出可用关节：

```bash
gmr-harness stage --robot my_robot --src bvh --list_joints
```

### 7.3 手动覆盖关节角度

```bash
gmr-harness stage \
  --robot my_robot \
  --src bvh \
  --preset tpose \
  --joint left_wrist_roll_joint=0.1 \
  --joint right_wrist_roll_joint=-0.1 \
  --output_dir specs/tpose
```

### 7.4 从已有 qpos 复用

```bash
gmr-harness stage \
  --robot my_robot \
  --src bvh \
  --qpos_file specs/tpose/my_robot.json \
  --output_dir specs/tpose
```

生成后务必肉眼检查三张 PNG。参考图错误意味着规格错误，后续数值验证也会跟着错误。

---

## 8. 直接求解 IK quaternion offset

如果你有一段标准 T-pose motion，推荐先用 `agent --solve_mode` 做一次直接求解。

### 8.1 dry-run 验证求解链路

```bash
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --tpose_motion /path/to/tpose.bvh \
  --solve_mode \
  --dry_run
```

`--dry_run` 会验证 retargeting 和数值 gate，但不会持久修改 IK config，也不会创建 `.bak`。

### 8.2 正式写入求解结果

确认 dry-run 通过后去掉 `--dry_run`：

```bash
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --tpose_motion /path/to/tpose.bvh \
  --solve_mode
```

第一次正式写入时会为 IK config 创建 `.json.bak` 备份。

### 8.3 保留部分已有偏移

```bash
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --tpose_motion /path/to/tpose.bvh \
  --solve_mode \
  --preserve "left_shoulder_yaw_link,right_shoulder_yaw_link"
```

### 8.4 手动设置 world_rotation

```bash
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --tpose_motion /path/to/tpose.bvh \
  --solve_mode \
  --world_rot "90,0,0,1"
```

格式为：`angle_deg,axis_x,axis_y,axis_z`。

---

## 9. 验证 alignment

### 9.1 BVH / motion 验证

```bash
gmr-harness validate \
  --robot my_robot \
  --src bvh \
  --tpose_motion /path/to/tpose.bvh \
  --spec specs/tpose/my_robot.json \
  --threshold 5.0
```

退出码：

| 退出码 | 含义 |
|--------|------|
| `0` | PASS，所有检查通过 |
| `1` | FAIL，运行成功但偏差超过阈值 |
| `2` | 参数、spec 或依赖错误 |

### 9.2 解读输出

典型输出：

```text
[validate] total_deviation :   3.42deg  (12 links)
[validate] max_angle       :   1.87deg
[validate] worst 5:
           left_shoulder_yaw_link                 1.87deg  axis=[+0.00, +1.00, +0.00]
[validate] PASS - all links within 5.0deg of T-pose.
```

经验判断：

| 最大偏差 | 判断 | 建议 |
|----------|------|------|
| `< 1deg` | 很好 | 不需要调整 |
| `1-5deg` | 可接受 | 通常可以使用 |
| `5-30deg` | 有偏差 | 先重新 solve，再检查 T-pose spec |
| `30-120deg` | 坐标或 quaternion 可能错 | 检查 `world_rotation`、body mapping 和偏移方向 |
| `> 120deg` | 可能翻转 | 检查 180 度旋转、左右侧映射或源 motion 坐标系 |

---

## 10. SMPL-X 模板校准

SMPL-X motion `.npz` 可能携带根节点朝向，不适合直接当 T-pose 标准。对于 SMPL-X，优先使用模板校准。

### 10.1 准备 body model

默认路径：

```text
$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz
```

检查：

```bash
ls "$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz"
```

### 10.2 stage SMPL-X T-pose spec

```bash
gmr-harness stage \
  --robot my_robot \
  --src smplx \
  --preset tpose \
  --output_dir specs/tpose
```

### 10.3 使用模板验证

```bash
gmr-harness validate \
  --robot my_robot \
  --src smplx \
  --use_smplx_template \
  --spec specs/tpose/my_robot.json
```

如果 body model 不在默认位置：

```bash
gmr-harness validate \
  --robot my_robot \
  --src smplx \
  --use_smplx_template \
  --smplx_template_model /path/to/body_models \
  --spec specs/tpose/my_robot.json
```

---

## 11. VLM 视觉迭代优化

当直接求解无法满足姿态要求时，可以用 VLM agent 迭代调参。

### 11.1 基本命令

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

### 11.2 调优模式

| 模式 | 用途 |
|------|------|
| `scale` | VLM 调整 `human_scale_table`，默认模式 |
| `weights` | VLM 调整 IK match table 权重 |
| `quaternion` | VLM 调整 quaternion offset |
| `optimize_scale` | 无 VLM，使用数值优化调整 scale |

### 11.3 API 配置

默认模型参数为 `glm-5v-turbo`，默认 API base 为 OpenAI-compatible 地址。可通过参数覆盖：

```bash
gmr-harness agent \
  --robot my_robot \
  --src bvh \
  --motion_file /path/to/motion.bvh \
  --model glm-5v-turbo \
  --api_base https://api.example.com/v1 \
  --api_key sk-xxx
```

也可以用环境变量：

```bash
export OPENAI_API_KEY=sk-xxx
```

---

## 12. 真实 E2E 验收

`packages/gmr-harness/scripts/verify_e2e.sh` 用于确认安装式 CLI、GMR_ROOT、spec、motion 和 solve-mode dry-run 能一起工作。

在源码树中运行：

```bash
cd /home/user2/roboharness/packages/gmr-harness
GMR_HARNESS_E2E_ROBOT=engineai_pm01 \
GMR_HARNESS_E2E_SPEC=/home/user2/roboharness/specs/tpose/engineai_pm01.json \
GMR_HARNESS_E2E_MOTION=/home/user2/soma-retargeter/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh \
GMR_HARNESS_E2E_GMR_ROOT=/home/user2/GMR \
bash scripts/verify_e2e.sh
```

通过时会看到：

```text
===== E2E Summary =====
All tests passed.
```

如果没有 motion 文件，脚本会跳过 solve-mode dry-run；这种情况只能证明 help 和路径解析可用，不能证明 retargeting 链路可用。

---

## 13. 常用命令速查

| 场景 | 命令 |
|------|------|
| 查看总帮助 | `gmr-harness --help` |
| 查看 setup 参数 | `gmr-harness setup --help` |
| dry-run 新机器人配置 | `gmr-harness setup --robot X --xml $GMR_ROOT/assets/X/robot.xml --formats bvh --dry_run` |
| 生成并注册配置 | `gmr-harness setup --robot X --xml $GMR_ROOT/assets/X/robot.xml --formats bvh --auto_register --yes` |
| 生成 T-pose spec | `gmr-harness stage --robot X --src bvh --preset tpose --output_dir specs/tpose` |
| 查看关节列表 | `gmr-harness stage --robot X --src bvh --list_joints` |
| solve dry-run | `gmr-harness agent --robot X --src bvh --motion_file tpose.bvh --tpose_spec specs/tpose/X.json --tpose_motion tpose.bvh --solve_mode --dry_run` |
| 正式 solve | `gmr-harness agent --robot X --src bvh --motion_file tpose.bvh --tpose_spec specs/tpose/X.json --tpose_motion tpose.bvh --solve_mode` |
| 数值验证 | `gmr-harness validate --robot X --src bvh --tpose_motion tpose.bvh --spec specs/tpose/X.json` |
| SMPL-X 模板验证 | `gmr-harness validate --robot X --src smplx --use_smplx_template --spec specs/tpose/X.json` |
| VLM 调 scale | `gmr-harness agent --robot X --src bvh --motion_file motion.bvh --tune_mode scale` |

---

## 14. 常见问题

### 14.1 `GMR not found`

错误示例：

```text
FileNotFoundError: GMR not found. Set GMR_ROOT env var or place GMR/ next to roboharness/.
```

处理：

```bash
export GMR_ROOT=/path/to/GMR
test -f "$GMR_ROOT/general_motion_retargeting/params.py"
```

### 14.2 XML 路径不被接受

`gmr-harness setup --xml` 要求 XML 直接在 `$GMR_ROOT/assets/<robot>/` 下。移动 XML 后重试。

### 14.3 非 TTY 环境没有写 params.py

这是预期行为。非交互环境写 `params.py` 必须显式加：

```bash
--auto_register --yes
```

### 14.4 `--dry_run` 成功但正式运行失败

`setup --dry_run` 不执行完整 stage/solve/validate。分别验证每一步：

```bash
gmr-harness setup --robot X --xml ... --formats bvh --dry_run
gmr-harness stage --robot X --src bvh --preset tpose --output_dir specs/tpose
gmr-harness agent --robot X --src bvh --motion_file tpose.bvh --tpose_spec specs/tpose/X.json --tpose_motion tpose.bvh --solve_mode --dry_run
gmr-harness validate --robot X --src bvh --tpose_motion tpose.bvh --spec specs/tpose/X.json
```

### 14.5 SMPL-X body model 找不到

检查默认路径：

```bash
ls "$GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz"
```

或显式传入：

```bash
--smplx_template_model /path/to/body_models
```

### 14.6 出现接近 180 度偏差

常见原因：

- 用普通行走 `.npz` 当 SMPL-X T-pose 标准。
- T-pose spec 本身姿态错误。
- `world_rotation` 方向不对。
- 左右 body mapping 错误。

优先处理顺序：

1. 重新检查 `specs/tpose/<robot>_front/side/back.png`。
2. 对 BVH 路径重新跑 `agent --solve_mode --dry_run`。
3. 对 SMPL-X 路径优先使用 `--use_smplx_template`。
4. 必要时用 `--world_rot` 明确指定源到机器人坐标变换。

---

## 15. 旧入口迁移说明

以下旧命令仍保留兼容，但会发出弃用警告：

| 旧入口 | 新入口 |
|--------|--------|
| `python scripts/setup_robot.py ...` | `gmr-harness setup ...` |
| `python scripts/stage_tpose.py ...` | `gmr-harness stage ...` |
| `python examples/gmr_alignment_agent.py ...` | `gmr-harness agent ...` |
| `python examples/gmr_tpose_validate.py ...` | `gmr-harness validate ...` |

新脚本、新文档和自动化验收都应使用 `gmr-harness` CLI。
