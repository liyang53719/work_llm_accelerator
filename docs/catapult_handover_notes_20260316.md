# Catapult 交接说明（经验教训 / 工具使用 / 注意事项）

## 一、交接目标
给下一位接手同学提供可落地的最短路径：
- 知道先看什么
- 知道怎么复现
- 知道报错怎么归因
- 知道哪些改法高风险

## 二、推荐工作流
1. 先确认仓库基线
- 在 `work_llm_accelerator` 子仓库执行：
  - `git status --short`
  - `git log --oneline -n 10`

2. 先用已有 Tcl 复现，不先改代码
- 先确认 `go new/analyze/compile/libraries/assembly/architect/extract` 步骤完整。
- 观察 `QWEN_STAGE` 标记判断卡点，而不是凭主观感觉。

3. 先分类型定位
- 接口连接错误：优先看 ccore 边界、参数形态（切片指针/固定数组）。
- LOOP 报告：优先看源码循环形态，必要时改为显式展开。
- compile 进度慢：先确认是否真的超时截断，再做热点优化。

## 三、工具使用建议
1. 日志筛选命令（高频）
- `rg -n "QWEN_STAGE|Error:|Failed|CIN-|LOOP-2" <log>`
- `tail -n 80 <log>`

2. 运行管理
- 长跑一律不设 timeout（避免误判流程中断）。
- 新实验用新 rerun 目录，禁止覆盖旧日志，保证可追溯。

3. 变更策略
- 单次只改一类问题（接口 or 循环 or 数值近似），避免问题耦合。
- 每次改动后都立即 rerun，对比上一条日志关键模式。

## 四、常见坑
1. `#pragma hls_unroll yes` 并不保证日志没有 LOOP-2
- 如果用户要求“不要迭代循环”，要改代码形态，不只是加 pragma。

2. 切片指针 + 独立 ccore 容易触发接口尺寸不匹配
- 典型报错：CIN-84 / CIN-178 / CIN-236。

3. 过早把问题归因给 Tcl
- 先确认 compile 是否真正完成，再判断有没有进入后续阶段。

## 五、版本切换特别注意（你提到后续要换 Catapult 版本）
1. 语法兼容优先级
- 优先检查 pragma 语法与接口约束是否被新版本收紧。
- 优先检查 `ac_std_float` / `ccs_dw_fp_lib` 相关行为变化。

2. 升级策略
- 先以当前 checkpoint 作为基线跑通。
- 再按“最小 diff”修兼容，不要在升级时叠加结构重构。

3. 回归标准
- 至少保证：
  - 无新接口连接错误
  - compile 能推进到既定阶段
  - 关键 top 的日志模式与旧版本可对照

## 六、建议保留文档
- `docs/pending_changes_inventory_20260316.md`
- `docs/catapult_attempts_summary_20260316.md`
- `docs/catapult_handover_notes_20260316.md`
- `docs/catapult_cin84_qtoken_kproj_guide_20260316.md`

这些文档可作为版本切换后的问题定位起点。
