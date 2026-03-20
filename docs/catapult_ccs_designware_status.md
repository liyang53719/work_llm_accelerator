# Catapult DesignWare 状态与实践

## 当前状态

截至 2026-03-20，`work_llm_accelerator` 里与 prefill attention 相关的 Catapult 流程已经从“尝试在 `nangate-45nm_beh + OasysRTL` 上硬插 `ccs_designware`”切换为“直接对齐官方可工作的 DesignCompiler + SAED DW behavioral flow”。

当前确认有效的路线是：

1. C++ 侧在综合路径下直接调用 `ccs_dw_fp_add/sub/mult/div/mac/sqrt/cmp`。
2. Tcl 侧显式启用 `flow package require /DesignCompiler`。
3. solution library 使用 `saed32rvt_tt0p78v125c_dw_beh`，并在 `go libraries` 后 source `setup_saedlib.tcl`。
4. 正式脚本不再把头文件当作独立 design file 加入 Catapult，只保留真正的 translation unit。

这条路线已经验证过可以让 score-only context 导出路径通过 `libraries -> assembly -> architect -> allocate -> extract`，并生成包含 `ccs_dw_fp_*_v1` 依赖的 RTL。

## 已修正的认知

旧文档里“当前应跳过 `ccs_designware`”的说法已经不再准确。更准确的描述是：

1. 不能在 `nangate-45nm_beh + OasysRTL` 这条流上直接加 `ccs_designware`。
2. 但可以参考官方 `ccs_dw_fp_ops_numeric` 示例，改为 SAED DW behavioral library + DesignCompiler flow。
3. 仅仅切换库还不够，C++ 前端也要避免继续依赖 generic `fp_*` 包装，否则仍可能落回 `FPOPS.ccs_fp_*` 组件选择失败。

换句话说，问题不是“DesignWare 不能用”，而是“必须用对它的 flow 和前端调用方式”。

## 根因总结

这轮验证后，根因可以归纳为两层：

### 1. 库流不匹配

以下组合是已证实不可靠的：

- `nangate-45nm_beh`
- `-rtlsyntool OasysRTL`
- generic `fp_*` 运算包装

它常见的结果不是 `go analyze` 失败，而是在 `go architect` 阶段出现：

- `Couldn't find library component for operator 'FPOPS.ccs_fp_*'`
- `Error: incomplete component selection`

### 2. 前端运算包装不匹配

即使头文件已经换成官方 `ccs_dw_fp_lib.h`，如果综合路径里继续通过 `fp_add/fp_mult/fp_mac/fp_cmp` 这类 generic 接口发起运算，Catapult 仍可能回到 generic FP operator binding 路径，而不是稳定落到 `ccs_dw_fp_*` 实例化路径。

因此需要把综合态浮点包装统一收敛为 direct DW 调用。

## 当前代码收敛结果

本轮已经按这个方向做了两类收敛。

### 1. prefill_only 浮点包装统一

`hls/prefill_only/qwen_catapult_fp.h` 现在提供统一的综合态 direct DW 包装，供以下文件共用：

- `qwen_prefill_attention_context_stage_catapult.cpp`
- `qwen_prefill_attention_kernel.cpp`
- `qwen_prefill_attention_kv_cache_stage.cpp`
- `qwen_prefill_mlp_kernel.cpp`

这样做的目的不是重构风格，而是避免同一仓库里一部分 stage 走 direct DW，一部分 stage 仍走 generic `fp_*`，导致行为再次分叉。

### 2. 正式脚本并入 SAED/DW 配置

以下正式脚本已经并入 SAED DesignCompiler flow：

- `script/run_catapult_prefill_attention_context.tcl`
- `script/run_catapult_prefill_attention_kv_cache.tcl`
- `script/run_catapult_prefill_attention_q_context_output.tcl`

关键变化包括：

- `CppStandard` 切到 `c++11`
- 增加 `flow package require /DesignCompiler`
- library 从 `nangate-45nm_beh` 改为 `saed32rvt_tt0p78v125c_dw_beh`
- 在 `go libraries` 后执行 `setup_saedlib.tcl`
- 移除把 `.h` 当作 design file 的写法
- 统一改成先 `project new`，再创建并配置 solution，避免 `go analyze` 实际丢失 `SearchPath` / `CompilerFlags`

## 当前验证结果

截至本轮实际重跑，三条正式脚本的状态已经分化得比较清楚：

1. `run_catapult_prefill_attention_context.tcl`
	- 默认切到已验证的 score-only export top 后，可通过 `assembly -> architect -> allocate -> extract`。
2. `run_catapult_prefill_attention_kv_cache.tcl`
	- 修正脚本初始化顺序后，已越过 `analyze/compile/libraries/assembly`，当前首个阻塞点收敛到 `architect` 的 `SCHD-3 / SCHD-20` 调度反馈路径过长。
3. `run_catapult_prefill_attention_q_context_output.tcl`
	- 同样在修正初始化顺序后越过了前端和库绑定阶段，当前首个阻塞点也是 `architect` 的调度失败，而不是头文件或 DesignWare 绑定失败。

这说明本轮脚本收敛已经把问题边界前推到了真正的 HLS 调度层；剩余工作不应再回退到头链或库流排查。

## 非阻塞现象

当前流里仍可能看到两类 warning：

1. `SAED32_EDK` 相关环境提示
2. 对某些 SAED liberty 文件的可读性提示

在本轮验证中，这些 warning 没有阻止 `extract` 产出 RTL，因此当前按非阻塞处理。

## 最佳实践

后续在这个仓库里继续做 Catapult + DesignWare 时，建议固定遵守下面几条。

### 1. 不要把“是否能 include 头文件”当成成功标准

`go analyze` 或 `go compile` 能过，只说明前端语法和 include 链基本可用。真正决定能否出 RTL 的，往往是 `go architect` 之后的 component selection 和 library binding。

### 2. 一旦失败点进入 `FPOPS.ccs_fp_*`，不要再回头折腾头文件链

这是 operator library / technology binding 问题，不是 `limits.h`、`cmath`、`ac_*` shim 问题。头文件链只需要保证前端能稳定进到 compile；再继续在这条线上投入时间，收益很低。

### 3. 对 isolated stage flow，`design_files` 只放真正的 `.cpp` 翻译单元

不要把 `.h` 单独 `solution file add` 进去。这个仓库已经反复验证过，Catapult 对 standalone header 输入非常敏感，容易在 analyze 阶段引入不必要的 EDG 问题。

### 4. direct DW 包装要集中维护，不要在每个 stage 各写一套

综合态浮点包装如果分散在多个源文件里，很容易出现：

- 一个 stage 改成 direct DW
- 另一个 stage 还保留 generic `fp_*`
- 结果正式脚本和子模块行为重新分叉

统一放在共享头里，后续迁移或回退才可控。

### 5. 正式脚本和临时脚本必须尽快收敛

`work/tmp/*.tcl` 可以用于快速试验，但一旦某条 flow 被证明有效，就应该尽快并入 `script/` 下的正式入口。否则很容易出现：

- 临时脚本能出 RTL
- 正式脚本仍停留在旧配置
- 后续验证、handover、复现都失真

### 6. 同类 stage 要成组检查，不要只修一个入口

一旦 `prefill_only` 中某个 stage 需要 direct DW + SAED flow，通常同类使用 `prefill_catapult_fp_t` 的 attention/mlp/kv-cache stage 都要复核。不要只修当前失败点，否则下一轮会在相邻 stage 复现同样问题。

### 7. 优先保留可验证的最小差异

这类修改应尽量限制在：

- 浮点包装
- solution library / flow package
- design_files 输入集合

不要顺手重构算法主体。否则如果 Catapult 结果变化，就很难判断是 library flow 变化还是功能代码变化导致。

### 8. `project new` 必须先于 solution 配置

这轮验证再次确认，若脚本先 `solution new` / `solution options set`，再 `project new`，Catapult 可能在 `go analyze` 实际只把源文件本身传给前端，而忽略已经设置的 `SearchPath` 和 `CompilerFlags`。症状通常是：

- 前端命令行里只有 `-- source.cpp`
- `cstdint` / `ac_*` / 其他标准头在 `analyze` 直接报 `cannot open source file`

因此正式脚本应固定采用以下顺序：

1. `options defaults`
2. `project new`
3. `flow package require ...`
4. `solution new`
5. `solution options set ...`
6. `solution file add ...`

## 下一步建议

当前最自然的后续动作是：

1. 以正式脚本为基准，分别重跑 context、kv-cache、q-context-output 三条 flow。
2. 记录每条 flow 的最远阶段与首个失败点。
3. 如果某条 flow 仍失败，优先比较该 stage 是否还残留 generic FP 包装或旧 library 约束，而不是再回到 Nangate/OasysRTL 路线。