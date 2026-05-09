# GMR-Harness 用户使用指南

本指南覆盖从注册新机器人到完成姿态校准的完整流程。所有命令均可在终端中直接复制运行。

---

## 1. 概述

GMR-Harness 将 General Motion Retargeting (GMR) 与 roboharness 的对齐工具链集成，提供从注册到校准的一站式流程：

```
注册机器人 → 生成 IK 配置 → T-pose 采集 → 偏移求解 → 数值验证
   └── setup_robot.py ──────────────────────────────────────────┘
```

### 支持的源格式

| 格式 | 配置生成 | T-pose 采集 | 偏移求解 | 验证 |
|------|:--------:|:-----------:|:--------:|:----:|
| **BVH** | `bvh` | `--src bvh` | 运动 + 直接求解 | 运动 / 模板 |
| **SMPL-X** | `smplx` | `--src smplx` | 模板校准（推荐） | 模板 / 运动 |
| **FBX** | `fbx`, `fbx_offline` | `--src fbx_offline` | 运动 + 直接求解 | 运动 |

---

## 2. 环境准备

### 2.1 安装

```bash
cd roboharness
pip install -e ".[demo,dev]"
```

### 2.2 GMR 仓库

GMR 需与 roboharness 放在同一父目录下，或设置环境变量：

```bash
# 方式一：同级目录（自动检测）
ls ../GMR/general_motion_retargeting/params.py  # 应存在

# 方式二：环境变量
export GMR_ROOT=/path/to/GMR
```

### 2.3 SMPL-X 身体模型（SMPL-X 格式需要）

将身体模型放到 GMR 资产目录：

```
GMR/assets/body_models/smplx/SMPLX_MALE.npz
```

验证：

```bash
python -c "from roboharness.alignment.smplx_template import resolve_body_model_path; print(resolve_body_model_path(None))"
# 应输出类似 .../GMR/assets/body_models
```

---

## 3. 新机器人注册（一条命令）

### 3.1 最小注册（仅生成配置 + 写入 params.py）

```bash
python scripts/setup_robot.py \
    --robot my_robot \
    --xml $GMR_ROOT/assets/my_robot/robot.xml \
    --formats smplx bvh \
    --auto_register --update_scripts
```

> XML 文件须放在 `$GMR_ROOT/assets/<robot>/` 目录下。

### 3.2 完整流程（含 T-pose 采集 + 求解 + 验证）

```bash
python scripts/setup_robot.py \
    --robot my_robot \
    --src bvh \
    --tpose_motion /path/to/tpose.bvh \
    --auto_register --update_scripts
```

自动执行：配置生成 → 注册 →  采集 → 偏移求解 → 验证。

### 3.3 从已有机器人克隆

```bash
python scripts/setup_robot.py \
    --robot my_robot \
    --clone_from unitree_h1 \
    --xml $GMR_ROOT/assets/my_robot/robot.xml \
    --formats bvh
```

### 3.4 预览模式（不修改任何文件）

```bash
python scripts/setup_robot.py \
    --robot my_robot \
    --xml $GMR_ROOT/assets/my_robot/robot.xml \
    --formats bvh \
    --dry_run
```

### 生成的文件

| 步骤 | 产物 | 路径 |
|------|------|------|
| 配置生成 | IK 配置 JSON | `$GMR_ROOT/.../ik_configs/bvh_to_my_robot.json` |
| 注册 | `params.py` 修改 | `ROBOT_XML_DICT`, `ROBOT_BASE_DICT`, `IK_CONFIG_DICT` 等 |
| T-pose 采集 | 规格文件 + 参考图 | `specs/tpose/my_robot.json` + `*_front.png` 等 |
| 求解 | 更新 IK 配置 | 偏移 quaternion 写入配置 |
| 验证 | 控制台输出 | PASS / FAIL + 逐环节偏差角度 |

---

## 4. SMPL-X 模板校准

### 为什么用模板校准

运动捕捉序列（如行走 `.npz`）携带根节点朝向，会导致约 180° 偏差。模板校准使用身体模型的零姿态作为标准源，一次求解即正确。

### 一条命令（自动发现身体模型）

```bash
python scripts/setup_robot.py \
    --robot v11 \
    --src smplx \
    --update_scripts
```

### 指定身体模型路径

```bash
# 目录（含 smplx/ 子文件夹）
python scripts/setup_robot.py \
    --robot v11 --src smplx \
    --smplx_template_model /path/to/body_models \
    --update_scripts

# 直接指定 .npz 文件（支持任意文件名）
python scripts/setup_robot.py \
    --robot v11 --src smplx \
    --smplx_template_model /path/to/my_model.npz \
    --update_scripts
```

### 身体模型路径解析规则

| 输入 | 解析结果 | 说明 |
|------|----------|------|
| `None`（省略） | `GMR/assets/body_models` | 自动发现 |
| `body_models/`（含 `smplx/` 子目录） | `body_models/` | `smplx.create()` 内部拼接 `smplx/` |
| `body_models/smplx/` | 返回父目录 `body_models/` | 自动向上收缩 |
| 任意 `.npz` 文件 | 原样返回 | 名称无关，直接加载 |

### 单独验证（无需运动文件）

```bash
python examples/gmr_tpose_validate.py \
    --robot v11 \
    --src smplx \
    --use_smplx_template \
    --spec specs/tpose/v11.json
```

---

## 5. T-pose 规格制作

T-pose 规格是后续所有数值校准的基准。**每台机器人只须制作一次。**

### 5.1 基本采集

```bash
python scripts/stage_tpose.py \
    --robot unitree_g1 \
    --preset tpose \
    --output_dir specs/tpose/
```

### 5.2 交互式预览

```bash
python scripts/stage_tpose.py \
    --robot unitree_g1 \
    --preset tpose \
    --preview \
    --output_dir specs/tpose/
```

### 5.3 手动指定关节角度

```bash
python scripts/stage_tpose.py \
    --robot unitree_g1 \
    --preset tpose \
    --joint left_wrist_roll_joint=0.1 \
    --joint right_wrist_roll_joint=-0.1 \
    --output_dir specs/tpose/
```

### 5.4 SMPL-X 源（自动应用根节点四元数）

```bash
python scripts/stage_tpose.py \
    --robot my_robot \
    --src smplx \
    --output_dir specs/tpose/
```

SMPL-X 源自动设置 `qpos[3:7] = [0.5, -0.5, -0.5, -0.5]`。

### 5.5 查看关节列表

```bash
python scripts/stage_tpose.py --robot unitree_g1 --list_joints
```

### 产出物

```
specs/tpose/my_robot.json          ← 数值规格（版本控制）
specs/tpose/my_robot_front.png     ← 参考渲染
specs/tpose/my_robot_side.png
specs/tpose/my_robot_back.png
```

> **重要：** 提交前务必肉眼检查三张参考图。渲染错误 = 规格错误。

---

## 6. 偏移求解与数值验证

### 6.1 直接求解（一次迭代，无需 VLM）

```bash
python examples/gmr_alignment_agent.py \
    --robot unitree_g1 \
    --motion_file /path/to/motion.bvh \
    --src bvh \
    --tpose_spec specs/tpose/unitree_g1.json \
    --tpose_motion /path/to/tpose.bvh \
    --solve_mode
```

可保留部分关节的已有偏移：

```bash
python examples/gmr_alignment_agent.py ... --solve_mode \
    --preserve "left_shoulder_yaw_link,right_shoulder_yaw_link"
```

### 6.2 数值验证

```bash
# BVH 源
python examples/gmr_tpose_validate.py \
    --robot unitree_g1 \
    --tpose_motion /path/to/tpose.bvh \
    --src bvh \
    --threshold 5.0

# SMPL-X 模板源（无需运动文件）
python examples/gmr_tpose_validate.py \
    --robot v11 \
    --src smplx \
    --use_smplx_template
```

退出码：`0` = PASS，`1` = FAIL，`2` = 错误。

### 6.3 解读偏差报告

```
[validate] total_deviation :   3.42°  (12 links)
[validate] max_angle       :   1.87°
[validate] worst 5:
           left_shoulder_yaw_link                      1.87°  axis=[0, 1, 0]
           right_elbow_link                             0.92°  axis=[0, 0, 1]
           ...
[validate] PASS — all links within 5.0° of T-pose.
```

| 偏差角度 | 含义 | 建议 |
|----------|------|------|
| `< 1°` | 优秀 | 无需调整 |
| `1°–5°` | 可接受 | 通常不值得再调 |
| `5°–30°` | 有偏差 | 检查 IK 求解器精度 |
| `30°–60°` | 坐标轴混淆 | 检查 `world_rotation` 或偏移方向 |
| `60°–120°` | 缺少 90° 旋转 | 沿报告轴补 90° 偏移 |
| `> 120°` | 可能 180° 翻转 | 检查偏移四元数符号 |

---

## 7. VLM 迭代优化

当数值求解无法一步到位时，可用 VLM 迭代调整 IK 配置。

### 7.1 基本用法

```bash
python examples/gmr_alignment_agent.py \
    --robot unitree_g1 \
    --motion_file /path/to/motion.bvh \
    --src bvh \
    --max_iter 8
```

### 7.2 调优模式

```bash
# 骨骼长度缩放（默认）
--tune_mode scale

# 数值优化缩放（无 VLM，scipy）
--tune_mode optimize_scale

# IK 权重调整
--tune_mode weights

# 四元数偏移调整
--tune_mode quaternion
```

### 7.3 搭配数值门

```bash
python examples/gmr_alignment_agent.py \
    --robot unitree_g1 \
    --motion_file /path/to/motion.bvh \
    --src bvh \
    --tpose_spec specs/tpose/unitree_g1.json \
    --tpose_motion /path/to/tpose.bvh \
    --tune_mode scale \
    --max_iter 8
```

每轮迭代自动计算 `total_deviation`，若上升则自动回滚。

### 7.4 模型与 API 配置

```bash
# 默认使用 GLM-4V
--model glm-5v-turbo

# 使用自定义 API
--api_base https://api.example.com/v1 \
--api_key sk-xxx
```

---

## 8. 完整流程示例

### 流程 A：新机器人（BVH 源）

```bash
# 1. 放置 XML
cp robot.xml $GMR_ROOT/assets/my_robot/robot.xml

# 2. 一条命令完成注册 + 采集 + 求解 + 验证
python scripts/setup_robot.py \
    --robot my_robot \
    --src bvh \
    --tpose_motion /path/to/tpose.bvh \
    --auto_register --update_scripts

# 3. 若验证失败，手动调优
python examples/gmr_alignment_agent.py \
    --robot my_robot \
    --motion_file /path/to/motion.bvh \
    --src bvh \
    --tpose_spec specs/tpose/my_robot.json \
    --tpose_motion /path/to/tpose.bvh \
    --solve_mode
```

### 流程 B：新机器人（SMPL-X 模板校准）

```bash
# 1. 放置 XML + 确认身体模型存在
ls $GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz

# 2. 一条命令
python scripts/setup_robot.py \
    --robot v11 \
    --src smplx \
    --update_scripts

# 3. 验证（无需运动文件）
python examples/gmr_tpose_validate.py \
    --robot v11 --src smplx --use_smplx_template
```

### 流程 C：已有机器人迭代优化

```bash
# 1. 确认规格存在
ls specs/tpose/unitree_g1.json

# 2. VLM 迭代
python examples/gmr_alignment_agent.py \
    --robot v11 \
    --motion_file /path/to/motion.bvh \
    --src bvh \
    --tpose_spec specs/tpose/unitree_g1.json \
    --tpose_motion /path/to/tpose.bvh \
    --tune_mode scale \
    --max_iter 8

# 3. 最终验证
python examples/gmr_tpose_validate.py \
    --robot unitree_g1 \
    --tpose_motion /path/to/tpose.bvh \
    --src bvh
```

---

## 9. 常见问题

### `GMR not found`

```
FileNotFoundError: GMR not found. Set GMR_ROOT env var or place GMR/ next to roboharness/.
```

**解决：** 将 GMR 放在 roboharness 同级目录，或设置 `export GMR_ROOT=/path/to/GMR`。

### `SMPLX body model not found`

**解决：** 确认文件存在：

```bash
ls $GMR_ROOT/assets/body_models/smplx/SMPLX_MALE.npz
```

或通过 `--smplx_template_model` 显式指定路径。

### 验证出现约 180° 偏差

| 原因 | 检查 |
|------|------|
| 使用行走 `.npz` 作为校准源 | 改用模板校准（`--use_smplx_template`） |
| SMPL-X 配置缺少 `world_rotation` 或使用旧的组合旋转 | 重新执行 `setup_robot.py --src smplx --auto_register --update_scripts`，确认 `smplx_to_*.json` 中包含新的 base runtime `world_rotation` |
| T-pose 规格未用 SMPL-X 源采集 | 重新执行 `stage_tpose.py --src smplx` |

### `setup_robot.py` 报 XML 不在正确位置

XML 须直接位于 `$GMR_ROOT/assets/<robot>/` 下（不能嵌套子目录）：

```
正确: $GMR_ROOT/assets/my_robot/model.xml
错误: $GMR_ROOT/assets/my_robot/variants/model.xml
```

### 偏差报告振荡

连续迭代偏差在两个值之间交替 → patch 与其逆被交替施加。检查最近 3 次 patch 的四元数是否包含 `q` 和 `q.conjugate`。

### `--dry_run` 显示正常但实际运行失败

dry-run 不执行 T-pose 采集和求解。单独验证各步骤：

```bash
python scripts/setup_robot.py ... --skip_solve --skip_validate  # 仅注册
python scripts/setup_robot.py ... --skip_stage --skip_solve     # 仅验证
```

---

## 10. 命令速查

| 场景 | 命令 |
|------|------|
| 注册新机器人 | `setup_robot.py --robot X --xml ... --formats smplx bvh --auto_register --update_scripts` |
| 完整 BVH 流程 | `setup_robot.py --robot X --src bvh --tpose_motion tpose.bvh --auto_register --update_scripts` |
| SMPL-X 模板校准 | `setup_robot.py --robot X --src smplx --update_scripts` |
| 预览不修改 | `setup_robot.py ... --dry_run` |
| T-pose 采集 | `stage_tpose.py --robot X --preset tpose --output_dir specs/tpose/` |
| 直接求解 | `gmr_alignment_agent.py --robot X --src bvh --solve_mode --tpose_spec ... --tpose_motion ...` |
| 数值验证 | `gmr_tpose_validate.py --robot X --src bvh --tpose_motion tpose.bvh --threshold 5.0` |
| SMPL-X 模板验证 | `gmr_tpose_validate.py --robot X --src smplx --use_smplx_template` |
| VLM 迭代 | `gmr_alignment_agent.py --robot X --src bvh --motion_file ... --tune_mode scale` |
