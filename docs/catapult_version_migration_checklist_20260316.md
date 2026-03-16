# Catapult 版本迁移检查清单（2026-03-16）

## 目标
在切换到新 Catapult 版本时，最小化语法不兼容与行为漂移风险，保证 prefill-only 流程可复现、可对比、可回滚。

## 0. 迁移前冻结
1. 确认当前基线提交号。
2. 记录当前版本工具信息（Catapult/Oasys/库路径）。
3. 保留当前可运行 Tcl、日志目录与关键错误样本。

## 1. 环境与工具探测
1. 先验证可执行路径
- `command -v catapult`
- `catapult -version`（若支持）

2. 探测新版本安装目录下示例 flow
- 搜索 `go compile/go libraries/go assembly/go architect/go extract`。
- 对照本项目 Tcl 的阶段顺序。

3. 验证许可证与基础库
- 检查 license checkout 是否正常。
- 检查 `nangate-45nm_beh`、`ccs_sample_mem`、`amba` 是否可用。

## 2. 语法兼容优先检查（高优先）
1. pragma 兼容
- `#pragma hls_design ccore`
- `#pragma hls_ccore_type sequential`
- `#pragma hls_unroll yes`
- `#pragma hls_pipeline_init_interval N`

2. 类型与库接口兼容
- `ac_std_float`
- `ccs_dw_fp_lib` 的 `fp_add/fp_sub/fp_mult/fp_div/fp_mac/fp_sqrt/fp_cmp`
- `data_ac_int()/set_data()` 位级访问

3. 头文件与 include 路径兼容
- 重点检查 `qwen_catapult_fp.h`、catapult shims、`CompilerFlags` 与 `SearchPath`。

## 3. 编译流程迁移顺序（建议固定）
1. 先跑 analyze，只修编译语法错误。
2. 再跑 compile，优先修接口边界错误（CIN-84/178/236）。
3. compile 通过后再看 libraries/assembly/architect。
4. 最后再做 extract/report。

## 4. 错误分型与处置
1. 接口连接类（CIN-84/CIN-178/CIN-236）
- 优先检查切片指针跨 ccore 边界。
- 必要时取消局部 helper 的独立 ccore。

2. 循环/展开类（LOOP-2）
- 若要求“无迭代循环痕迹”，要改源码形态（显式展开），不要只加 pragma。

3. 数值与近似类
- 先保证能综合，再回归数值行为（exp/rsqrt/sincos 等近似）。

## 5. 回归检查点（每次迁移都打卡）
1. top 与参数一致
- `llm_accel::qwen_prefill_top_catapult`
- `llm_accel::qwen_prefill_top_catapult_fine`

2. 阶段 marker 一致
- `QWEN_STAGE ANALYZE_DONE`
- `QWEN_STAGE COMPILE_BEGIN/COMPILE_DONE`
- `QWEN_STAGE LIBRARIES_BEGIN/ASSEMBLY_BEGIN/ARCHITECT_BEGIN`

3. 错误回归
- `rg -n "Error:|Failed|CIN-|LOOP-2" <log>`

4. 资源趋势
- 记录 compile elapsed time、memory usage、peak memory。

## 6. 提交策略
1. 每类问题单独提交
- 语法兼容提交
- 接口边界提交
- 循环展开提交
- Tcl/流程提交

2. 禁止“边迁移边大重构”
- 避免难以定位回归来源。

3. 每次提交附带文档增量
- 更新尝试成败和交接说明。

## 7. 交接最小包
1. 代码基线提交号。
2. 可复现 Tcl 与 rerun 目录。
3. 最近一次完整日志。
4. 本清单与以下文档：
- `docs/pending_changes_inventory_20260316.md`
- `docs/catapult_attempts_summary_20260316.md`
- `docs/catapult_handover_notes_20260316.md`
- `docs/catapult_cin84_qtoken_kproj_guide_20260316.md`

## 8. 当前状态备注
- rerun36 日志显示 compile 曾持续推进到约 7050 秒量级；未见你指定的那组 LOOP-2 模式再次命中。
- 迁移到新版本时，请优先复现该日志路径，再做语法兼容修补。
