# Catapult `ccs_designware` 问题排查记录

## 结论

当前 `work_llm_accelerator` 的 prefill attention / MLP Catapult 流程不再尝试真正执行 `solution library add ccs_designware`。这不是绕过问题，而是基于官方示例、官方随包文档和本机环境验证后的明确结论：

1. `ccs_designware` 的官方用法依赖 DesignCompiler-backed library flow，不是给当前这条 `OasysRTL + nangate-45nm_beh` 流程直接插入就能工作的通用库。
2. 本机虽然存在 Synopsys DesignWare 资产，但 `dc_shell` 在当前主机上因运行时依赖不兼容而无法启动，因此不能作为 Catapult 所需的 DesignCompiler backend。
3. 在这种环境下继续硬加 `ccs_designware` 只会稳定触发 `LIB-223`，不会推进到真正的 FP operator binding 问题。

因此，当前脚本中的处理策略是：

- 允许保留环境变量 `QWEN_HLS_ADD_CCS_DESIGNWARE=1` 用于表达实验意图。
- 但脚本只打印说明并跳过 `ccs_designware`，避免在一个已确认不成立的库配置上直接失败。

## 现象

之前 attention 流程在 `go compile` 完成后，执行到库加载阶段会报：

```text
Warning: File '.../pkgs/siflibs/dware/ccs_designware.lib': invalid library selection (LIB-223)
Error: solution library add: Could not locate library file for library name ccs_designware for current configuration
```

这一点说明问题不在 `go analyze` 或 `go compile`，而在库选择阶段。

## 官方资料给出的信息

这次排查使用了 Catapult 2022.2 安装目录内随包示例和文档，核心结论如下。

### 1. 官方示例要求 DesignCompiler flow

Catapult 自带的 `ccs_designware` 示例 TCL 会要求：

- 有效的 `SYNOPSYS`
- 有效的 `SAED32_EDK`
- `flow package require /DesignCompiler`

也就是说，官方示例默认前提不是 OasysRTL，而是 Synopsys DesignCompiler 体系。

### 2. `ccs_designware.lib` 不是和任意 base library 自动兼容

官方 PDF 与脚本都强调：

- 随包提供的 `ccs_designware.lib` 是面向特定技术库条件准备的示例库。
- 如果用户自己的 base library / 工艺环境不同，需要先加载自己的 base library，再通过 `ccs_dw_char_setup.tcl` 重新生成或重建适配的 `ccs_designware.lib`。

这意味着把 `ccs_designware` 直接叠加到当前 `nangate-45nm_beh -- -rtlsyntool OasysRTL` 配置上，本身就不符合官方使用方式。

### 3. 老接口是 legacy 路线

Catapult 随包头文件里对旧版 `ccs_dw_lib.h` 有明确的 deprecated 提示，建议优先使用新的头和更新后的 base library 支持方式。这进一步说明：

- `ccs_designware` 不是“只要头文件能编译就一定能绑定”的纯前端问题。
- 它和底层 library characterization / flow backend 是绑定在一起的。

## 本机环境验证结果

### 1. Synopsys 安装路径里确实有 DW 资产

检查路径：

- `/home/yang/tools/synopsys/syn/O-2018.06-SP1`

已确认其中存在：

- `dw/`
- `packages/dware/`
- `libraries/syn/dw_foundation.sldb`
- 多份 DW 浮点相关文档与资源

所以问题不是“机器上完全没有 DesignWare 文件”。

### 2. 真正的阻塞是 `dc_shell` 无法运行

在设置：

- `SYNOPSYS=/home/yang/tools/synopsys/syn/O-2018.06-SP1`

后执行最小探测，`dc_shell` 在本机上没有进入 license 阶段就失败，报错包含：

- `GLIBC_PRIVATE`
- `PNG12_0`

这说明当前主机运行时环境与该版 DC 二进制不兼容。也就是说：

- 不是先卡在 license。
- 而是连 DesignCompiler 进程本身都无法正常启动。

在这种前提下，官方 `ccs_designware` 路线当前不可用。

## 当前代码里的处理

以下两个脚本已经改成“说明并跳过”：

- `hls/prefill_only/run_catapult_prefill_attention.tcl`
- `hls/prefill_only/run_catapult_prefill_mlp.tcl`

行为是：

- 仍然先加载 `nangate-45nm_beh`
- 仍然加载 `ccs_sample_mem`
- 若检测到 `QWEN_HLS_ADD_CCS_DESIGNWARE=1`
  - 打印 `QWEN_NOTE`
  - 不再实际执行 `solution library add ccs_designware`

这么做的目的只有一个：

- 把故障点从一个已经确认无效的 library selection 问题，推进到真正还未解决的 FP operator / architect 阶段问题。

## 这不代表 FP operator 问题已经解决

当前跳过 `ccs_designware` 之后，只是说明：

- `LIB-223` 这条分支不该再成为阻塞点。

真正还需要继续确认的是：

- 在当前 OasysRTL / Nangate 路线下，attention 是否会继续进入 `go libraries`、`go assembly`、`go architect`
- 如果失败，新的首个阻塞点是：
  - `FPOPS.ccs_fp_*` component selection
  - 还是其他更靠后的资源、时序、建模问题

## 后续启用 `ccs_designware` 的必要条件

如果后面仍然要恢复真正的 `solution library add ccs_designware`，至少要先满足下面几条：

1. 机器上可运行的 DesignCompiler 环境
2. 正常的 `SYNOPSYS` 指向
3. 对应工艺/基础库可用，而不是只靠当前的 OasysRTL base library
4. 必要时基于自己的 base library 重新生成适配的 `ccs_designware.lib`

在这些条件没满足之前，继续强加 `ccs_designware` 没有工程价值。

## 当前建议

这条线当前的正确推进顺序是：

1. 保持跳过 `ccs_designware`
2. 重跑 attention Catapult
3. 确认 `LIB-223` 是否彻底消失
4. 记录新的首个阻塞点
5. 再决定是继续做 FP operator mapping，还是继续走 logic expansion 路线