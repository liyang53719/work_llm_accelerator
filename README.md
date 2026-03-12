# LLM Accelerator Project

该目录用于规划并实现一个全新的独立 LLM 加速器工程，目标覆盖 Qwen2.5-1.5B 的 prefill 与 decode 两条路径，并使用 Catapult 生成 RTL。

## 固定约束

- 模型：`module/qwen_model` 中的 Qwen2.5-1.5B
- 片上 SRAM：1 MB
- 计算资源：2K MAC
- 时钟：1 GHz
- 目标性能：至少 10 token/s
- 输出目录：本目录 `work_llm_accelerator/`

## 工程原则

- 允许参考 `work/` 中已有工程的方法论、预算口径、验证层级和 Catapult sweep 组织方式。
- 不允许直接调用、复制或依赖 `work/` 中的源码、脚本、共享库、综合产物或中间结果。
- Prefill 与 decode 必须都被纳入架构预算、验证计划和 Catapult 设计空间探索。
- 优先保证验证链和指标口径清晰，再推进更激进的并行化或 unified kernel。

## 目录分层

- `docs/`：规划文档、阶段报告、设计决策
- `python/`：模型抽取、case 生成、分析脚本
- `performance_model/`：prefill/decode 联合预算模型
- `hls/common/`：共享类型、近似算子、公共接口
- `hls/decode_only/`：decode 专用 kernel 与 Catapult 脚本
- `hls/prefill_only/`：prefill 专用 kernel 与 Catapult 脚本
- `hls/prefill_decode/`：后续 unified 方案，仅在两条单独路径稳定后推进
- `verification/`：分层 testbench、参考结果、指标对比
- `rtl/`：Catapult 导出的 RTL 与交付物整理
- `tmp/`：临时日志、实验输出、设计点中间结果

## 当前主线

1. 先建立覆盖 prefill + decode 的联合可行性预算。
2. 再分别打通 decode-only 与 prefill-only 的实现和验证链。
3. 完成 Catapult architect 级可行性后，再进行定向设计点收敛。
4. 只有在两条路径都稳定后，才考虑统一成 prefill+decode 共享内核。

## 当前阶段说明

- 当前仓库已经具备 layer0 级别的 reference wrapper 与 host 侧验证路径。
- 这些 reference wrapper 的职责是固定单层数学口径，不是最终 RTL 接口。
- 下一步实现会切到 `descriptor + DDR/AXI + 1 MB SRAM working-set` 的边界上，避免继续扩展 layer-specific 参数表。
- decode 和 prefill 都已经有 top-level wrapper 骨架，host 侧也增加了 descriptor/layout 校验脚本，用于锁定多层复用和地址空间口径。

详细计划见 `docs/project_plan.md`。