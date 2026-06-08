# GMR-Harness Refactor Review

> 日期: 2026-06-03
> 输入方案: `docs/gmr-harness-refactor-plan.md`
> 目标读者: 后续负责修复的 coding agents

## 当前结论

`pip install gmr-harness` 后 `--help` 和 CLI 可用，build/install 已通过验收。GMR harness refactor 的功能性阻塞已清理，完整仓库 gate 已全绿；conda `gmr` 环境下的真实 GMR dry-run E2E 也已通过。

- 全量测试：`pytest -q` 通过，`800 passed, 3 skipped`，覆盖率 `94.76%`（超过 90% 要求）。
- 格式检查：`ruff format --check .` 通过，`164 files already formatted`。
- lint：`ruff check .` 通过，`All checks passed!`。
- type：`mypy src/` 通过，`Success: no issues found in 56 source files`。
- 独立包 tests：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q` 通过，`21 passed in 1.33s`。
- 构建：`python -m build packages/gmr-harness` 通过，生成 wheel/sdist；构建产物已清理。
- 真实 E2E：conda `gmr` + `/home/user2/GMR` 下 `packages/gmr-harness/scripts/verify_e2e.sh` 通过；`agent --solve_mode --dry_run` 前后 `bvh_to_engineai_pm01.json` SHA256 相同。
- 可用性结论：当前代码已经可以使用；无需再安排其他 agents 修复。后续如进入 PR/发布流程，只需要做常规代码审查、提交整理和发布前复验，不属于功能阻塞。

本轮复查（2026-06-04）确认：

- `packages/gmr-harness/tests/` 已恢复通过：`cd packages/gmr-harness && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q` 当前结果为 `21 passed in 1.33s`。此前 `scripts/setup_robot.py` re-export 不存在的 `GMR_ROOT` 导致 `test_setup_robot_help` 失败，现已由其他 agent 移除该 re-export。
- 四个旧入口 `scripts/setup_robot.py`、`scripts/stage_tpose.py`、`examples/gmr_alignment_agent.py`、`examples/gmr_tpose_validate.py` 的 `--help` 均可在 `PYTHONPATH=packages/gmr-harness/src` 下运行，不需要 `GMR_ROOT`。
- `roboharness.alignment.*` 子模块已改为 `gmr_harness.alignment.*` 的 re-export shim（Phase A 完成），`src/roboharness/alignment/__init__.py` 的 docstring 也已更新为当前 shim 状态。
- `packages/gmr-harness/scripts/verify_e2e.sh` 端到端验收脚本已创建，支持 `GMR_HARNESS_E2E_ROBOT/SPEC/MOTION/GMR_ROOT` 环境变量。
- `_gmr_path.py` 的 `find_gmr_root()` 已改为树目录遍历，兼容源树和 site-packages 位置。


## 长期独立运行设计方案

目标状态：`gmr-harness` 应能作为 `packages/gmr-harness/` 下的独立 Python 包长期维护、安装、测试和验收。独立包不能依赖仓库根目录的 `examples/`、`scripts/` 或 `src/roboharness/alignment` 运行；旧入口只保留为兼容 wrapper，并且所有真实实现、测试和文档入口都收敛到 `gmr_harness.*`。

### 设计原则

1. **单一真实实现**
   - `packages/gmr-harness/src/gmr_harness/` 是 GMR harness 的唯一业务实现位置。
   - 顶层 `scripts/` 和 `examples/` 不再复制业务逻辑，只做参数透传、弃用警告和兼容退出码处理。
   - `setup` 内部流程继续调用 `python -m gmr_harness.cli.agent` / `python -m gmr_harness.cli.validate`，不得回退到仓库根 `examples/`。

2. **安装后可用**
   - 用户通过 wheel/sdist 或 editable install 后，应只依赖 console script `gmr-harness` 和 `gmr_harness` 包内模块。
   - `--help`、dry-run、spec 解析、默认路径等不应要求 `GMR_ROOT`。
   - 只有真实 retargeting、MuJoCo 渲染、VLM 调用等运行时路径才检查外部 GMR/MuJoCo/VLM 依赖，并通过 `_ensure_gmr()` / `_deps.require()` 给出明确错误。

3. **兼容有截止线**
   - `roboharness.alignment` 短期保留为兼容 shim，面向已有 tests 和外部调用者给出 `DeprecationWarning`。
   - 新代码、新 examples、新 tests 默认导入 `gmr_harness.alignment`。
   - shim 只保证旧 API 过渡，不再新增功能；新增能力只进 `gmr_harness.alignment`。

4. **验收可复现**
   - 无 GPU/无 GMR 环境下，必须能跑独立包 focused tests、CLI help、dry-run 级别 smoke。
   - 有 GMR 环境下，必须有一条固定命令验证 `agent --solve_mode --dry_run`，覆盖 spec、motion、config 和 retargeting 入口装配。
   - 真实 VLM/retargeting 不纳入默认单元测试，但要有脚本或 marked slow test 记录命令、前置条件和预期产物。

### 阻塞项 1：端到端 agent CLI 验收

修复方案：

- 新增一个可复现验收入口，优先放在 `packages/gmr-harness/scripts/verify_e2e.sh`；如果需要 pytest 管理，则新增 `packages/gmr-harness/tests/test_e2e_agent.py` 并标记 `@pytest.mark.slow`。
- 脚本只做真实环境验收，不混入单元测试：
  - 检查 `GMR_ROOT` 是否存在，不存在则明确退出并说明如何设置。
  - 检查 `specs/tpose/<robot>.json`、motion 文件、GMR ik config 是否存在。
  - 先跑 `gmr-harness agent --help`。
  - 再跑 `gmr-harness agent --robot <robot> --solve_mode --dry_run --tpose_spec <spec> --tpose_motion <motion>`。
  - 最后可选跑真实 retargeting / VLM 模式，但必须单独开关，避免把外部 API 或长耗时步骤变成默认验收。
- 文档中记录一组当前仓库可用的默认参数，例如 `engineai_pm01` 或 `tienkung`，但脚本应允许通过环境变量覆盖：
  - `GMR_HARNESS_E2E_ROBOT`
  - `GMR_HARNESS_E2E_SPEC`
  - `GMR_HARNESS_E2E_MOTION`
  - `GMR_HARNESS_E2E_GMR_ROOT`

验收标准：

```bash
cd /home/user2/roboharness/packages/gmr-harness
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python -m gmr_harness.cli.main agent --help
GMR_HARNESS_E2E_ROBOT=<robot> \
GMR_HARNESS_E2E_SPEC=<spec> \
GMR_HARNESS_E2E_MOTION=<motion> \
GMR_HARNESS_E2E_GMR_ROOT=<gmr-root> \
bash scripts/verify_e2e.sh
```

完成标志：

- 文档能说明无 GMR 时哪些测试必须通过、有 GMR 时如何跑真实验收。
- E2E 脚本失败时错误指向缺失前置条件或真实运行失败，而不是 Python import/path 问题。

### 阻塞项 2：旧 `examples/` / `scripts/` 迁移

修复方案：

- 将下列旧入口改为薄 wrapper：
  - `scripts/setup_robot.py` -> 调用 `gmr_harness.cli.setup_robot.main`
  - `scripts/stage_tpose.py` -> 调用 `gmr_harness.cli.stage_tpose.main`
  - `examples/gmr_alignment_agent.py` -> 调用 `gmr_harness.cli.agent.main`
  - `examples/gmr_tpose_validate.py` -> 调用 `gmr_harness.cli.validate.main`
  - `examples/gmr_alignment_inspector.py` 若仍需保留，则迁到独立包 CLI 或明确标记为 legacy-only
- wrapper 保留：
  - 原入口文件名
  - `DeprecationWarning`
  - `if __name__ == "__main__": raise SystemExit(main())`
  - 最小 `sys.path` 兼容逻辑仅用于 editable repo 开发；安装后路径不得依赖仓库根。
- wrapper 删除：
  - 大量重复业务逻辑
  - 对 `roboharness.alignment` 的直接业务导入
  - 对 `_gmr_shared` 的直接依赖；共享逻辑已由 `gmr_harness.gmr_integration` 承接

测试方案：

- 增加 wrapper smoke tests，至少覆盖旧入口 `--help`：

```bash
cd /home/user2/roboharness
python scripts/setup_robot.py --help
python scripts/stage_tpose.py --help
python examples/gmr_alignment_agent.py --help
python examples/gmr_tpose_validate.py --help
```

- 测试断言不需要 `GMR_ROOT`，且 stderr 里出现弃用警告或等效提示。
- 迁移后运行搜索：

```bash
rg -n "from _gmr_shared|import _gmr_shared|sys.path.insert|roboharness\.alignment" examples scripts packages/gmr-harness tests
```

预期结果：

- `examples/` / `scripts/` 只剩 wrapper 级别的兼容导入或弃用提示。
- `packages/gmr-harness/` 内不得命中 `roboharness.alignment`。
- 旧 tests 若必须覆盖兼容层，可以继续命中 `roboharness.alignment`，但应集中在兼容测试文件中。

### 阻塞项 3：`roboharness.alignment` 迁移策略

修复方案采用“两阶段 shim”。

阶段 A：短期兼容，保证现有主仓库测试和外部调用者不立刻断裂。

- `src/roboharness/alignment/__init__.py` 继续发出 `DeprecationWarning`。
- 子模块逐步改为 re-export shim，例如：
  - `src/roboharness/alignment/body_matcher.py` re-export `gmr_harness.alignment.body_matcher`
  - `src/roboharness/alignment/config_gen.py` re-export `gmr_harness.alignment.config_gen`
  - 其他 alignment 子模块同理
- shim 不再保留 forked implementation，避免双份代码长期漂移。
- `roboharness` 主包若不依赖 `gmr-harness` 安装，需要在主仓库 packaging 层明确依赖关系，或把兼容 shim 放到 optional extra 中；不能让默认 import `roboharness` 因缺 `gmr_harness` 崩溃。

阶段 B：内部迁移，减少旧命名空间使用面。

- 将仓库内非兼容测试、非 legacy wrapper 的导入从 `roboharness.alignment` 改为 `gmr_harness.alignment`。
- 保留少量专门测试验证旧 namespace：
  - 导入旧模块会发出 `DeprecationWarning`
  - 旧模块导出的函数与新模块对象一致或行为一致
  - 缺少 `gmr_harness` 时错误信息明确，不出现半初始化模块
- 在 review 文档中记录移除计划：旧 namespace 可在后续 minor/major 版本删除，删除前至少经过一个发布周期。

风险与决策点：

- 如果主包 `roboharness` 不能默认依赖 `gmr-harness`，则不能直接把所有子模块改成无条件 re-export；应保留最小旧实现或把 GMR alignment 从主包 release 中拆出，并在 import 时提示安装 `gmr-harness`。
- 如果允许主包依赖独立包，则推荐无条件 shim，因为这是消除双实现漂移的最短路径。
- 当前 refactor 的长期目标是独立运行，因此推荐决策是：`gmr_harness.alignment` 为唯一实现，`roboharness.alignment` 为临时兼容层。

完成标志：

```bash
cd /home/user2/roboharness
rg -n "roboharness\.alignment" src examples scripts packages/gmr-harness tests
ruff check .
ruff format --check .
mypy src/
pytest -q
```

其中 `rg` 结果应只剩兼容 shim、legacy wrapper、兼容测试和历史说明文档。最终收口前必须运行 `pytest -q`，不能用 `packages/gmr-harness` focused tests 替代主仓库 UT。

## 已改善项

- 独立包当前路径为 `packages/gmr-harness/`，不再是过期的 `src/gmr_harness/`。
- `packages/gmr-harness/README.md`、独立 `pyproject.toml`、`solver.py`、`vlm.py` 已存在。
- 顶层 `gmr-harness` CLI dispatcher 已能把子命令参数转发给子 parser。
- `setup` / `stage` help 路径已不再因缺少 `GMR_ROOT` 直接崩溃。
- `validate` 已使用 `_resolve_spec_xml(spec)` 调用 `compute_deviations()`，相对 XML 解析问题已改善。
- `setup_robot.py` 已新增 `--from_step {0..6}`，并开始在跳过阶段时检查部分前置产物。
- `setup_robot.py` 当前只保留 `--yes`；默认写入会先展示 proposed changes 并在 TTY 下询问确认，非 TTY 需要 `--yes`。
- `packages/gmr-harness/tests/` 已创建，包含 CLI dispatch/help 和 `_deps.require()` 友好错误测试。
- `packages/gmr-harness/src/gmr_harness/solver.py` 已提取真实 agent 逻辑：`run_agent()` 实现完整的 VLM iteration loop（scale / weights / quaternion / optimize_scale / solve_mode），不再有占位提示。
- `packages/gmr-harness/src/gmr_harness/cli/agent.py` 已扩展为完整 argparse，与 legacy `examples/gmr_alignment_agent.py` 参数对齐。
- `src/roboharness/alignment/__init__.py` 已加 `DeprecationWarning`，提示使用 `gmr_harness.alignment`。
- `packages/gmr-harness/scripts/run_tests.sh` 已创建，封装 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` + `PYTHONPATH=src` + `-p pytest_cov`。
- `verify_e2e.sh` 验收脚本已创建，支持环境变量覆盖 `GMR_HARNESS_E2E_ROBOT/SPEC/MOTION/GMR_ROOT`。
- `roboharness.alignment.*` 全部 14 个子模块改为 `gmr_harness.alignment` 的 re-export shim（含 `DeprecationWarning`）。
- `scripts/setup_robot.py`、`scripts/stage_tpose.py`、`examples/gmr_alignment_agent.py`、`examples/gmr_tpose_validate.py`、`examples/gmr_alignment_inspector.py` 改为调用 `gmr_harness.cli.*` 的薄 wrapper。
- `packages/gmr-harness/tests/test_wrapper_smoke.py` 新增 5 个 wrapper smoke tests。
- `_gmr_path.py` 的 `find_gmr_root()` 改为目录树遍历，兼容源树和 site-packages 两种位置。
- build/install 验收已通过：

```bash
python -m build packages/gmr-harness
# Successfully built gmr_harness-0.1.0.tar.gz and gmr_harness-0.1.0-py3-none-any.whl

pip install packages/gmr-harness/dist/gmr_harness-0.1.0-py3-none-any.whl --force-reinstall
python -c "import gmr_harness; print(gmr_harness.__version__)"  # 0.1.0
gmr-harness --help  # OK
env -u GMR_ROOT gmr-harness agent --help  # OK, shows full argparse
```

- 最近监控命令中，独立包 focused tests 通过：

```bash
cd /home/user2/roboharness/packages/gmr-harness
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q
# 13 passed
```

- 2026-06-04 复查：新增 `packages/gmr-harness/tests/test_solver_regressions.py` 后，独立包 focused tests 通过，`15 passed in 0.29s`。

## 待修复项

### 1. Agent CLI 已提取真实逻辑，缺端到端验收

证据：

- `solver.py` 的 `run_agent()` 已实现完整 VLM iteration loop (solve_mode / optimize_scale / VLM scale/weights/quaternion)。
- `cli/agent.py` 已包含所有 legacy 参数 (--solve_mode, --tune_mode, --model 等)。
- `env -u GMR_ROOT gmr-harness agent --help` 已验证通过，输出完整 argparse 帮助。
- 尚未做 --solve_mode dry-run 或真实 VLM/retargeting 的端到端验收（需 GMR + MuJoCo + 运动文件）。

影响：

占位实现已替换。端到端验收需要 GMR 环境。

建议验收：

```bash
cd /home/user2/roboharness/packages/gmr-harness
PYTHONPATH=src python -m gmr_harness.cli.main agent --help
PYTHONPATH=src python -m gmr_harness.cli.agent --robot <robot> --solve_mode --dry_run --tpose_spec <spec> --tpose_motion <motion>
```

### 2. `params.py` 写入确认已默认启用（opt-out 策略已完成）

证据：

- `packages/gmr-harness/src/gmr_harness/cli/setup_robot.py` 当前只保留 `--yes`，`--confirm` 参数已移除。
- dry_run `pass-through` 路径已删除：无 `--yes` + TTY → 展示 proposed changes + 询问确认；无 `--yes` + 非 TTY → 拒绝写入报错；`--yes` → 直接写入。
- 默认行为符合预期：非交互环境不写 params.py，除非用户传 `--yes`。

影响：

确认策略已完成。如需要可补 `register_in_params()` monkeypatch 测试覆盖三条路径。

### 3. 旧入口和旧子模块依赖 — wrapper / shim 已落地，主仓库旧测试待迁移

证据：

- `scripts/setup_robot.py` 已是薄 wrapper：只发出 `DeprecationWarning`，并调用 `gmr_harness.cli.setup_robot.main`。
- `scripts/stage_tpose.py` 已是薄 wrapper：只发出 `DeprecationWarning`，并调用 `gmr_harness.cli.stage_tpose.main`。
- `examples/gmr_alignment_agent.py` 已是薄 wrapper：调用 `gmr_harness.cli.agent.main`。
- `examples/gmr_tpose_validate.py` 已是薄 wrapper：调用 `gmr_harness.cli.validate.main`。
- `packages/gmr-harness/tests/test_wrapper_smoke.py` 新增 5 个 smoke tests，覆盖旧入口 `--help` 和弃用警告。
- `src/roboharness/alignment/*.py` 子模块已改为 `gmr_harness.alignment.*` 的 re-export shim，并发出模块级 `DeprecationWarning`。
- 本轮复查命令：`PYTHONPATH=packages/gmr-harness/src python scripts/setup_robot.py --help`、`scripts/stage_tpose.py --help`、`examples/gmr_alignment_agent.py --help`、`examples/gmr_tpose_validate.py --help` 均退出 0。

影响：

长期独立运行的 wrapper/shim 代码路径已落地。当前剩余风险是主仓库旧 tests 与兼容 shim 的边界：部分 tests 已改为 mock `gmr_harness.cli.setup_robot.*`，但 `tests/test_alignment_patch.py` 仍直接依赖旧 namespace 的私有 helper `_quats_close`。

待清理：

- `scripts/setup_robot.py` 曾从 `gmr_harness.cli.setup_robot` re-export 不存在的 `GMR_ROOT`，导致 wrapper smoke test 失败；该 re-export 已被移除，focused tests 已恢复通过。
- `src/roboharness/alignment/__init__.py` docstring 已被其他 agents 更新为“子模块均为 shim”，该过期项已清理；后续需要更新文档顶部状态。
- `examples/_gmr_shared.py` 和 `examples/gmr_alignment_inspector.py` 仍有旧共享模块/import 信号；若 inspector 继续保留，需要单独迁到独立包 CLI 或标记为 legacy-only。

### 4. 独立包测试脚本已封装，当前 focused tests 已恢复通过

证据：

- `packages/gmr-harness/pyproject.toml` 的 `[tool.pytest.ini_options]` 已注释说明 ROS 插件问题。
- `packages/gmr-harness/scripts/run_tests.sh` 已创建，封装 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` + `PYTHONPATH=src` + `-p pytest_cov`。
- 2026-06-03 16:27 CST 监控复查：`cd packages/gmr-harness && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q` 结果为 `13 passed in 0.29s`。
- `packages/gmr-harness/tests/test_from_step.py` fixture 的重复 mkdir 问题已修复：`params_dir.mkdir(parents=True, exist_ok=True)`。

影响：

独立包 focused tests 已恢复通过并增加到 13 个用例。剩余风险转移到未覆盖路径和完整仓库检查。

建议修复：

- 保持 `bash packages/gmr-harness/scripts/run_tests.sh` / 等效命令在后续修复后持续通过。
- 继续补更完整的 `--from_step` 前置产物存在/缺失组合。

### 5. `--from_step` 已新增行为测试，13 passed

证据：

- `packages/gmr-harness/tests/test_from_step.py` 现有 5 个测试：缺 config warning、注册验证、缺 spec warning、已有 spec 验证、from_step=6 跳过前序步骤。
- 所有测试使用 `contextlib.suppress(SystemExit)`，无 unused import，ruff 通过。
- fixture 已修复 params.py 中 `HERE` 定义，创建完整的 mock GMR 目录结构。

### 6. 构建和安装验收已通过

证据：

```bash
python -m build packages/gmr-harness
# Successfully built gmr_harness-0.1.0.tar.gz and gmr_harness-0.1.0-py3-none-any.whl

pip install packages/gmr-harness/dist/gmr_harness-0.1.0-py3-none-any.whl --force-reinstall
python -c "import gmr_harness; print(gmr_harness.__version__)"  # 0.1.0
gmr-harness --help  # OK
env -u GMR_ROOT gmr-harness agent --help  # OK
```


### 7. `setup` 的 solve/validate 已改为 `gmr-harness` CLI 调用

证据：

- `packages/gmr-harness/src/gmr_harness/cli/setup_robot.py:213` 的 `_solve_via_agent()` 构造命令由 `str(_PROJECT / "examples" / "gmr_alignment_agent.py")` 改为 `["python", "-m", "gmr_harness.cli.agent", ...]`。
- `packages/gmr-harness/src/gmr_harness/cli/setup_robot.py:668` 验证阶段由 `str(_PROJECT / "examples" / "gmr_tpose_validate.py")` 改为 `["python", "-m", "gmr_harness.cli.validate", ...]`。
- `grep -n "examples/gmr" packages/gmr-harness/src/gmr_harness/cli/setup_robot.py` 结果：无匹配，不再引用仓库根 `examples/`。

影响：

独立包安装后 `gmr-harness setup` 的 solve/validate 阶段不会再尝试执行不存在的旧示例脚本。

### 8. GMR_ROOT-only 安装路径已覆盖 `_retarget_tpose_qpos()`

证据：

- `packages/gmr-harness/src/gmr_harness/solver.py:25` 已新增 `_ensure_gmr(feature)`，调用 `_require_gmr()` 后再执行 direct import。
- `solver.py:168` (`solve_direct`)、`:263` (`_retarget`)、`:311` (`_get_config_path`)、`:580` (`run_agent`) 已添加 `_ensure_gmr()`。
- `packages/gmr-harness/src/gmr_harness/cli/validate.py:28` 新增 `_ensure_gmr()`，并在 `_retarget_first_frame` (`:35`) 和 `_retarget_template_frame` (`:67`) 中使用。
- 2026-06-04 复查：`packages/gmr-harness/src/gmr_harness/solver.py:290` 的 `_retarget_tpose_qpos()` 已在 direct import 前调用 `_ensure_gmr("retarget tpose qpos")`。
- `_retarget_tpose_qpos()` 会在 `run_agent()` 的 solve_mode post-solve 验证路径 (`solver.py:818`) 和 numeric gate 路径中被调用；这些路径现在不会绕过 `_require_gmr()`。

影响：

GMR_ROOT-only 的 T-pose retarget 路径已闭环。`packages/gmr-harness/tests/test_solver_regressions.py` 已补回归测试，断言 `_retarget_tpose_qpos()` 会在 direct GMR import 前调用 `_ensure_gmr("retarget tpose qpos")`。


### 9. `stage` / `setup` / `validate` / `agent` 默认 spec 路径已统一为 `specs/tpose`

证据：

- `stage_tpose.py:480` 默认 `--output_dir` 为 `Path("specs/tpose")`（cwd 相对路径）。
- `setup_robot.py:316` 默认 `--output_dir` 已改为 `"specs/tpose"`。
- `validate.py:106` `_default_spec_path()` 已改为 `Path("specs/tpose") / f"{robot}.json"`。
- `solver.py:662` agent 默认 spec 已改为 `Path("specs/tpose") / f"{args.robot}.json"`，`_HERE = get_gmr_root()` 已移除。
- 四者现在都使用 cwd 相对路径 `specs/tpose/`，help 文案与实际一致。

影响：

用户用 `gmr-harness stage` 在当前工作目录生成 `specs/tpose/<robot>.json` 后，直接跑 `gmr-harness agent` 时，默认 spec 查找语义已与 stage/setup/validate 一致。`packages/gmr-harness/tests/test_solver_regressions.py` 已补回归测试，断言未传 `--tpose_spec` 时 agent 默认查 cwd-relative `specs/tpose/<robot>.json`。



### 10. Focused ruff 已通过

证据：

- 2026-06-04 复查命令：`ruff check scripts/setup_robot.py scripts/stage_tpose.py packages/gmr-harness/src/gmr_harness/cli/setup_robot.py packages/gmr-harness/tests/test_from_step.py packages/gmr-harness/tests/test_solver_regressions.py`。
- 结果：`All checks passed!`。
- `packages/gmr-harness/tests/test_from_step.py` 已清理 unused import、长行和 `try/except SystemExit: pass`。

影响：

focused lint gate 已恢复。仍需在最终收口时跑仓库要求的完整 `ruff check .`、`ruff format --check .`、`mypy src/`。

建议修复：

- 后续每次改 wrapper/setup/tests 后继续跑 focused ruff。
- 发布/合并前按仓库要求跑完整 lint/type/test。



### 11. `scripts/setup_robot.py --help` 回归已修复

证据：

- 回归原因：`scripts/setup_robot.py` 为兼容旧 tests re-export 了不存在的 `GMR_ROOT`，导致顶层 import 失败。
- 当前代码已移除 `GMR_ROOT` re-export，保留 `_find_clone_source`、`_get_gmr_root`、`_solve_smplx_offsets`、`_solve_via_agent`、`find_gmr_root`、`load_gmr_params`、`main` 等兼容导出。
- 2026-06-04 复查命令：

```bash
cd /home/user2/roboharness/packages/gmr-harness
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q
# 20 passed in 1.31s
```

影响：

独立包 focused tests 和旧 wrapper `--help` smoke 已恢复。仍需关注主仓库完整 `pytest -q`、`ruff check .`、`ruff format --check .`、`mypy src/`。



### 12. `roboharness.alignment.patch` shim 私有 helper 导出已修复

证据：

- `src/roboharness/alignment/patch.py` 除 `from gmr_harness.alignment.patch import *` 外，已显式 re-export `SCALE_BOUNDS`、`_quats_close`、`_resolve_quat_spec`、`_resolve_scale_spec`。
- `packages/gmr-harness/src/gmr_harness/alignment/patch.py` 保持这些 helper 的真实实现。
- `tests/test_alignment_patch.py` 继续从旧 namespace 导入私有 helper，作为兼容层覆盖。

本轮 review 验证：

```bash
cd /home/user2/roboharness
PYTHONPATH=packages/gmr-harness/src:src pytest --no-cov tests/test_alignment_patch.py -q
# 53 passed, 4 warnings in 1.35s

cd /home/user2/roboharness/packages/gmr-harness
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q
# 20 passed in 1.34s

cd /home/user2/roboharness
ruff check packages/gmr-harness/
# All checks passed!
```

影响：

第 12 项 collection blocker 已关闭。仍建议最终收口 agent 运行并记录完整 `pytest -q`、`ruff check .`、`ruff format --check .`、`mypy src/`，不要仅以 focused tests 替代全量 gate。



### 13. 完整 lint/type/test gate 已通过

证据（2026-06-04 本轮收口复查）：

```bash
cd /home/user2/roboharness
ruff check .
# All checks passed!

mypy src/
# Success: no issues found in 56 source files

ruff format --check .
# 164 files already formatted

PYTHONPATH=packages/gmr-harness/src:src pytest --no-cov tests/test_alignment_patch.py -q
# 53 passed, 4 warnings in 1.32s

cd /home/user2/roboharness/packages/gmr-harness
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q
# 21 passed in 1.33s

cd /home/user2/roboharness
pytest -q
# 800 passed, 3 skipped, 15 warnings in 18.28s
# Total coverage: 94.76%

python -m build packages/gmr-harness
# Successfully built gmr_harness-0.1.0.tar.gz and gmr_harness-0.1.0-py3-none-any.whl
```

修复内容：

- `scripts/setup_robot.py`、`examples/_gmr_shared.py` 和 `src/roboharness/alignment/*` 兼容 shim 在保留弃用警告语义的前提下，为 re-export import 增加 `E402` noqa。
- 多个 `roboharness.alignment.*` shim 的长弃用字符串已拆分，清理 `E501`。
- `src/roboharness/_math_utils.py::normalize_vector()` 返回值改为显式 `np.asarray(..., dtype=float)` 并用 quoted `typing.cast()` 收敛 numpy typing，清理 `mypy no-any-return`。

影响：

第 13 项完整仓库 gate blocker 已关闭。2026-06-04 后续 review 发现并修复了 `agent --solve_mode --dry_run` 仍会持久写入 IK config / 创建 backup 的行为 bug：当前 dry-run 不创建 backup，验证时如需临时写入 config 会在 `finally` 中恢复原文件。`packages/gmr-harness/tests/test_solver_regressions.py` 已新增回归测试覆盖该语义。conda `gmr` 环境真实 E2E 已通过，且 `/home/user2/GMR/general_motion_retargeting/ik_configs/bvh_to_engineai_pm01.json` 前后 hash 均为 `4031de0088b738721e8ad7ad9f8d2869d2a2bc907cc1e5c89cfd6914f6a08b79`。


## 建议修复顺序

1. ~~修复 `test_from_step.py` fixture，恢复独立包测试~~ → 已完成，13 passed。
2. ~~修复 `setup` 内部仍调用旧 `examples/` 路径~~ → 已完成，改为 `python -m gmr_harness.cli.agent/validate`。
3. ~~统一 GMR runtime import / `GMR_ROOT` 解析策略的主入口部分~~ → 已完成，`_ensure_gmr()` 保护所有入口路径包括 `_retarget_tpose_qpos()`。
4. ~~统一 `stage` / `setup` / `validate` 的默认 spec 路径语义~~ → 已完成，三者均用 `Path("specs/tpose")`。
5. ~~修复 agent 默认 spec 路径与 `stage/setup/validate` 不一致的问题~~ → 已完成，agent 使用 `Path("specs/tpose")`，与三者统一。
6. ~~修复 `_retarget_tpose_qpos()` 缺 `_ensure_gmr()` 的 GMR_ROOT-only 漏洞~~ → 已完成。
7. ~~补 `gmr-harness agent` 的最小端到端验收~~ → 已完成，创建 `packages/gmr-harness/scripts/verify_e2e.sh`，支持环境变量覆盖。
8. ~~完成 `roboharness.alignment` 迁移策略：补齐子模块 shim~~ → 已完成，所有 14 个子模块改为 `gmr_harness.alignment` 的 re-export shim。
9. ~~补 `--from_step` step 6 基础覆盖~~ → 已完成；继续补更多前置产物缺失/存在组合。
10. ~~清理旧 examples/scripts 入口，改为薄 wrapper~~ → 已完成，5 个入口（`scripts/setup_robot.py`、`scripts/stage_tpose.py`、`examples/gmr_alignment_agent.py`、`examples/gmr_tpose_validate.py`、`examples/gmr_alignment_inspector.py`）改为调用 `gmr_harness.cli.*` 的薄 wrapper；新增 5 个 wrapper smoke tests 覆盖 --help。

## 最小验收清单

修复 agent 或测试入口后，请至少运行：

```bash
cd /home/user2/roboharness/packages/gmr-harness
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q
PYTHONPATH=src python -m gmr_harness.cli.main agent --help
PYTHONPATH=src python -m gmr_harness.cli.main setup --help
PYTHONPATH=src python -m gmr_harness.cli.main validate --help
```

修复独立包路径/GMR_ROOT 后，请补充：

```bash
cd /home/user2/roboharness/packages/gmr-harness
PYTHONPATH=src python - <<PY
from gmr_harness.cli import setup_robot, validate
print(setup_robot._PROJECT)
print(validate._default_spec_path("dummy"))
PY
rg -n "examples/gmr_alignment_agent|examples/gmr_tpose_validate|from general_motion_retargeting" src/gmr_harness -S
```

迁移旧入口后，请确认：

```bash
cd /home/user2/roboharness
rg -n "from _gmr_shared|import _gmr_shared|sys.path.insert|roboharness\.alignment" examples scripts packages/gmr-harness tests
```

除兼容 shim 和专门测试旧 API 的用例外，不应再命中。

发布前请补充：

```bash
cd /home/user2/roboharness
python -m build packages/gmr-harness
```

## Agent 同步约定

本文件是跨 agent 的修复同步位。为避免多个 agents 同时轮询、重复跑测试和互相覆盖结论，请遵守以下约定：

- 只维护这一份 review 文档：`docs/gmr-harness-refactor-review.md`。
- 不要恢复本文件此前的过期 `src/gmr_harness/` 初始 review 章节；当前独立包路径以 `packages/gmr-harness/` 为准。
- 同一时间只保留一个 monitor/review owner。其他 agents 不做周期轮询；完成或部分完成一个待修复项后，主动更新对应章节。
- 更新时写清楚证据来源：文件路径、行号、命令和结果。不要只写“已修复”。
- 如果某项只是策略决策而非代码修复，请在对应章节记录最终决策和理由。
- 如果新增阻塞项，请添加到“待修复项”，并说明如何复现。
- 如果删除阻塞项，请同时删除或改写其证据，避免保留互相矛盾的旧结论。
- 多 agent 并行时优先编辑具体章节，不要整篇重写，除非是在清理过期内容。
- monitor/review owner 采用事件驱动检查：只有在 `git status`、本文档、或当前阻塞项相关文件发生变化时才复跑对应聚焦测试。没有实质变化时不更新时间戳、不刷写文档、不重复跑全量命令。
- 当前阻塞项连续两轮无变化时，monitor/review owner 应停止周期轮询，在文档中保持现有 blocker 和复现命令，等待修复 agent 主动更新。

当前状态：无开放 blocker。monitor/review owner 可以停止周期轮询；其他 agents 不需要继续修复本 refactor。若后续进入 PR/发布流程，按常规发布前复验即可。

可选发布前复验命令：

```bash
cd /home/user2/roboharness
ruff check .
ruff format --check .
mypy src/
pytest -q
python -m build packages/gmr-harness

cd /home/user2/roboharness/packages/gmr-harness
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python -m pytest tests -q
```

有 GMR 环境时可选复验真实 dry-run E2E：

```bash
source /home/user2/miniconda3/etc/profile.d/conda.sh
conda activate gmr
cd /home/user2/roboharness/packages/gmr-harness
GMR_HARNESS_E2E_ROBOT=engineai_pm01 GMR_HARNESS_E2E_SPEC=/home/user2/roboharness/specs/tpose/engineai_pm01.json GMR_HARNESS_E2E_MOTION=/home/user2/soma-retargeter/assets/motions/bvh/Neutral_walk_forward_002__A057.bvh GMR_HARNESS_E2E_GMR_ROOT=/home/user2/GMR bash scripts/verify_e2e.sh
```
