# GMR Alignment Guide

为新机器人/人体数据组合对齐 IK config 的完整流程。

## 一键编排（推荐）

```bash
# 完整流程。新机器人必须通过 --xml 或 GMR/assets/{robot}/*.xml 提供模型。
python scripts/setup_robot.py \
  --robot my_robot \
  --xml /path/to/robot.xml \
  --tpose_motion /path/to/tpose.bvh \
  --auto_register --update_scripts

# 如果 XML 已放置在 GMR/assets/my_robot/ 下，可省略 --xml：
python scripts/setup_robot.py \
 --robot my_robot \
--tpose_motion /path/to/tpose.bvh \
--auto_register --update_scripts
```

自动完成：XML 发现 → body 名匹配 → IK config 生成 → params.py 注册 →
脚本 choices 更新 → T-pose 舞台 → 四元数求解 → 验证。

## 前提条件

```bash
# 目录结构
roboharness/           # 本项目
GMR/                   # GeneralMotionRetargeting
  assets/{robot}/      # 机器人 MuJoCo XML + STL 网格（需手动放置）
  general_motion_retargeting/
    params.py           # 机器人注册表
    ik_configs/         # IK config JSON（setup_robot 自动生成）
```

## setup_robot.py 参数速览

| 参数 | 说明 |
|------|------|
| `--robot NAME` | 机器人名（在 `ROBOT_XML_DICT` 中注册） |
| `--xml PATH` | 直接指定 XML 路径（未注册机器人用） |
| `--src FORMAT` | 人体数据格式：`bvh` / `smplx` / `fbx_offline`（默认 `bvh`） |
| `--formats FMT...` | 为哪些格式生成 IK config（默认跟随 `--src`） |
| `--auto_register` | 自动修改 `params.py`（四个字典） |
| `--update_scripts` | 自动更新脚本 `--robot` choices 列表 |
| `--tpose_motion PATH` | T-pose 源动作文件 |
| `--tpose_preset home\|tpose` | T-pose 基准姿态（默认 `tpose`） |
| `--tpose_joint NAME=VALUE` | 覆盖 T-pose 关节角度（可重复） |
| `--mapping_override ROLE=BODY` | 手动指定 body→role 映射（可重复） |
| `--no-interactive` | 跳过未匹配 body 的交互提示 |
| `--clone_from ROBOT` | 从已有机器人克隆 IK config |
| `--base_body NAME` | 覆盖自动检测的 root body 名 |
| `--cam_distance FLOAT` | 覆盖相机距离（默认 2.5） |
| `--world_rot "90,0,0,1"` | **覆盖**人体→机器人朝向（BVH 格式默认自动检测，见 § 朝向自动检测） |
| `--bvh_format auto\|soma\|lafan1` | BVH 解析器选择 |
| `--skip_stage` | 跳过 T-pose 舞台 |
| `--skip_solve` | 跳过四元数求解 |
| `--dry_run` | 预览模式，不修改任何文件 |

---

## 场景 1：已有注册表的常规机器人

```bash
python scripts/setup_robot.py \
  --robot my_robot \
  --tpose_motion /path/to/tpose.bvh \
 
```

## 场景 2：全新机器人（尚未在 params.py 注册）

```bash
# 步骤 A：先仅生成 IK config（dry_run 预览）
python scripts/setup_robot.py \
  --robot my_robot \
  --xml assets/my_robot/model.xml \
  --formats smplx bvh \
  --dry_run

# 步骤 B：确认映射无误后，写入 params.py 和脚本
python scripts/setup_robot.py \
  --robot my_robot \
  --xml assets/my_robot/model.xml \
  --formats smplx bvh \
  --auto_register --update_scripts
```

## 场景 3：body 名自动匹配不完整

当启发式匹配无法覆盖某些 body 时（如 `AL2`、`leg_l3_link` 等非标准命名），
终端会自动进入交互提示：

```
Unmatched role: left_shoulder (human joint: left_shoulder)
Available bodies: ['AL2', 'Left_Arm_2', 'zarm_l2_link']
Enter body name for left_shoulder (or 'skip'): AL2
→ mapped left_shoulder → AL2
```

跳过交互：
```bash
python scripts/setup_robot.py ... --no-interactive
```

或预定义映射：
```bash
python scripts/setup_robot.py ... \
  --mapping_override left_shoulder=AL2 \
  --mapping_override right_shoulder=AR2
```

## 场景 4：从已有机器人克隆

```bash
python scripts/setup_robot.py \
  --robot my_robot_v2 \
  --xml assets/my_robot_v2/model.xml \
  --clone_from unitree_h1 \
  --formats smplx
```

仅支持同格式克隆（smplx→smplx、bvh→bvh）。

---

## 朝向自动检测（world_rotation）

`setup_robot.py --auto_register` 会根据 robot XML 默认姿态下的 body 几何位置自动推算
`world_rotation` 四元数，无需手动指定。检测逻辑位于 `orientation_aligner.py`。

| 人体格式 | world_rotation | 原因 |
|---------|---------------|------|
| BVH / FBX | **自动检测** | BVH post-loader 惯例 `(X=left, Y=forward, Z=up)` 与 robot 可对齐 |
| SMPL-X | **不写入** | SMPL-X 惯例 `(X=right, Y=up, Z=forward)` 与 robot 左右手性相反，IK solver 通过 root free joint 处理 |

手动覆盖：`--world_rot "90,0,0,1"` 会覆盖自动检测值（格式为 `"angle,axis_x,axis_y,axis_z"` 度）。

检测依赖以下 body landmark：
- **up** 方向：`spine` − `hip_mid`（若无 spine 则假设 Z-up）
- **left** 方向：`left_hip` − `right_hip`（fallback：shoulder）
- **forward** 方向：`cross(up, left)`
- 无头/无臂/无腿 robot 通过 fallback 自动降级

## 进阶：自定义 T-pose

自动 T-pose 检测（`shoulder_roll = ±π/2`、`elbow = π/2`）基于
Unitree 风格的关节轴约定。不同机器人可能需要不同的关节角度。

```bash
# 查看机器人关节名
python scripts/stage_tpose.py --robot my_robot --list_joints

# 使用 home 基准姿态 + 手动覆盖
python scripts/setup_robot.py \
  --robot my_robot \
  --tpose_motion /path/to/tpose.bvh \
  --tpose_preset home \
  --tpose_joint left_shoulder_roll_joint=1.57 \
  --tpose_joint right_shoulder_roll_joint=-1.57 \
  --tpose_joint left_elbow_joint=0 \
  --tpose_joint right_elbow_joint=0 \
  --auto_register --update_scripts
```

## 两步法：先确认 T-pose，再求解

```bash
# 1. 生成 T-pose spec + 参考 PNG，不求解
python scripts/setup_robot.py --robot my_robot \
  --tpose_motion tpose.bvh --skip_solve

# 2. 视觉确认 specs/tpose/my_robot_{front,side,back}.png
#    若不满意，回到 1 调整 --tpose_joint 或 --tpose_preset

# 3. 重新运行，跳过舞台，仅求解四元数
python scripts/setup_robot.py --robot my_robot \
  --tpose_motion tpose.bvh --skip_stage
```

---

## 分步手动流程

### Step 1: 生成 IK config（无对齐）

```bash
python scripts/setup_robot.py --robot my_robot --xml model.xml \
  --formats smplx bvh --auto_register --update_scripts \
  --skip_stage --skip_solve
```

### Step 2: 制作 T-pose Spec

```bash
python scripts/stage_tpose.py \
  --robot my_robot \
  --preset tpose \
  --output_dir specs/tpose/
```

### Step 3: 求解四元数

```bash
python examples/gmr_alignment_agent.py \
  --robot my_robot \
  --motion_file /path/to/motion.bvh \
  --tpose_motion /path/to/tpose.bvh \
  --tpose_spec specs/tpose/my_robot.json \
  --solve_mode \
  [--world_rot "90,0,0,1"] \
  [--src smplx]          # --tpose_src 默认跟随 --src，无需单独指定
```

`--world_rot` 格式：`"angle,axis_x,axis_y,axis_z"`，覆盖自动检测的朝向。
`--tpose_src` 如未指定则自动跟随 `--src`。

### Step 4: 验证

```bash
python examples/gmr_tpose_validate.py \
  --robot my_robot \
  --tpose_motion /path/to/tpose.bvh \
  --threshold 5.0
```

---

## 新增机器人的完整清单

### 手动准备

| 资源 | 路径 | 说明 |
|------|------|------|
| MuJoCo XML + STL | `GMR/assets/{robot}/` | 机器人模型文件 |

### 自动配置（setup_robot.py --auto_register --update_scripts）

| 配置项 | 位置 | 说明 |
|------|------|------|
| `ROBOT_XML_DICT` | `params.py` | 机器人名 → XML 路径 |
| `ROBOT_BASE_DICT` | `params.py` | root body 名 |
| `VIEWER_CAM_DISTANCE_DICT` | `params.py` | 相机距离 |
| `IK_CONFIG_DICT` | `params.py` | 各格式 IK config 注册 |
| `--robot choices` | `scripts/*.py` | argparse 参数列表 |
| IK config JSON | `ik_configs/{fmt}_to_{robot}.json` | body→role 映射 + 四元数偏移 |
| T-pose spec | `specs/tpose/{robot}.json` + PNG | 对齐基准 |

---

## 恢复原始配置

```bash
cp /path/to/ik_config.json.bak /path/to/ik_config.json
cp /path/to/params.py.bak /path/to/params.py
```

---

## 架构概览

新机器人配置流程由以下模块驱动：

```
_math_utils.py ─────────── 四元数/向量/旋转矩阵共享运算
     │                       (normalize_quat, quat_multiply,
     │                        rotation_matrix_to_axis_angle,
     │                        axis_angle_to_quat, IDENTITY_QUAT)
     ├──→ orientation_aligner.py
     └──→ patch.py

_utils.py ──────────────── JSON/图片/编码共享工具
     │                       (save_json, save_image, to_float,
     │                        encode_image_base64, select_image_files)
     └──→ 各模块

skeleton_maps.py        (人体骨骼名定义：SMPL-X / BVH，含 skeleton edges)

       ↓
body_matcher.py         (XML body 名 → 人体骨骼 role 启发式匹配)

       ↓
orientation_aligner.py  (XML 默认姿态 → world_rotation 自动推算；
                         extract_xml_body_names;
                         apply_smplx_base_rotation)

       ↓
config_gen.py           (IK config JSON 生成)

       ↓ (完全独立)
gmr_register.py         (params.py + 脚本 choices 自动修改)

       ↓
setup_robot.py          (CLI 编排层, 引用 _gmr_params 和 _gmr_path)
```

`_gmr_params.py` 和 `_gmr_path.py` 为内部辅助模块，分别提供 GMR `params.py` 加载和
GMR 根目录自动发现，被 `setup_robot.py` 和 `stage_tpose.py` 共用。

所有模块均可独立测试，`gmr_register.py` 向上游模块无任何依赖。
