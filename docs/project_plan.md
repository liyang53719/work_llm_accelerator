# Qwen2.5-1.5B Prefill + Decode Accelerator 规划

## 1. 目标

在 `work_llm_accelerator/` 下建立一个新的独立工程，面向 Qwen2.5-1.5B 的 LLM 加速器设计与验证，覆盖两类工作负载：

- Prefill：处理 prompt 输入阶段，重点是 blocked attention、多 token 并行和 KV 写回。
- Decode：处理单 token 生成阶段，重点是 weight streaming、KV 读取和低延时调度。

最终目标是使用 Catapult 生成 RTL，并形成可持续迭代的性能、面积、时序和验证报告体系。

## 2. 设计约束

- 模型：Qwen2.5-1.5B，来源 `module/qwen_model`
- On-chip SRAM：1 MB
- Compute：2K MAC units
- Clock：1 GHz
- 目标：至少 10 token/s
- 输出位置：全部文件保存在 `work_llm_accelerator/`

## 3. 独立性要求

- 可以参考 `work/` 的总体思路、预算口径、验证分层、日志组织方式。
- 不得直接调用 `work/` 中的源码、脚本、共享库、测试二进制、Catapult 工程和综合结果。
- 所有 kernel、testbench、脚本、报告和 RTL 产物都必须在本目录独立生成。

## 4. 总体判断

这个项目不能直接沿用 decode-only 假设。Prefill 和 decode 的最优数据流、tile 形态、buffer 组织和 bottleneck 都不同，因此应按两条主线推进：

- Decode-only：优先追求单 token 延时，倾向 `M=1`、weight streaming、KV 流式读取。
- Prefill-only：优先追求吞吐与 SRAM 复用，必须引入 blocked attention、动态 tile-M、多阶段 softmax 与 partial-sum banking。

统一内核不是第一阶段目标。过早追 unified 设计会把接口、banking 和时序问题耦合在一起，降低收敛效率。

## 5. 架构主线

### 5.1 系统层

- 片外存储保存完整 INT4 权重与长上下文 KV cache。
- 1 MB SRAM 作为 staging、ping-pong buffer、partial sum scratch、softmax/reduction scratch 和小范围 KV working set。
- 软件或 firmware 负责任务描述、layer 调度、KV metadata 和 tile issue。
- 硬件负责 matmul、norm、rope、softmax/reduction、KV read/write datapath 和片上 buffer 调度。

### 5.2 Decode 路线

- 以单 token 为基本调度单位。
- 采用 GEMV 或 skinny GEMM 形态。
- 重点关注权重流式供给、KV 命中模式、单层 attention/MLP 时延分解。
- 需要明确 final_norm + lm_head 是否独立建核或与前序路径做有限融合。

### 5.3 Prefill 路线

- 以 blocked attention 为核心，不采用 decode-only 的固定 `M=1` 假设。
- 需要动态 tile-M 与 tile-N 组合，兼顾 QK、softmax、AV 和 MLP 路径。
- 需要多阶段 reduction、partial-sum banking、attention write-back/reuse 和 KV paging/descriptor。
- 需要明确 prefill 的吞吐目标与与 decode 的切换边界。

## 6. 建议目录与输出职责

- `docs/`
  - 规划文档、阶段总结、架构决策记录
- `python/`
  - 模型配置抽取
  - reference case 生成
  - 定量误差分析
- `performance_model/`
  - prefill/decode MAC、DDR、KV、SRAM 联合预算
  - token/s 与 tile 参数敏感性分析
- `hls/common/`
  - 公共类型定义
  - 量化/定点近似
  - 通用 buffer 与 helper
- `hls/decode_only/`
  - decode attention kernel
  - decode MLP kernel
  - Catapult 脚本与 metric 提取
- `hls/prefill_only/`
  - prefill attention kernel
  - prefill MLP kernel
  - blocked attention 子路径
- `hls/prefill_decode/`
  - 后续 unified kernel 预研，不作为第一阶段依赖
- `verification/`
  - micro-kernel testbench
  - block testbench
  - decode step / prefill block reference compare
- `rtl/`
  - Catapult 导出 RTL
  - 版本化交付快照与摘要
- `tmp/`
  - 实验日志、中间报表、设计点输出

## 7. 阶段门控

### 阶段 A：Feasibility

目标：先把系统级预算做完整。

完成标准：

- 有统一的模型规格与 workload facts 抽取脚本
- 有 prefill + decode 的 MAC 预算
- 有 DDR 带宽、KV cache、SRAM working set 联合预算
- 能回答 10 token/s 目标在 decode 和 mixed workload 下的瓶颈来源

### 阶段 B：Architecture

目标：把功能与 HLS 基础流打通。

完成标准：

- decode-only 与 prefill-only 各自至少有一条 kernel 主线
- host 侧 testbench 可以对齐软件参考
- Catapult 至少跑到 architect
- 指标采集方式固定下来

### 阶段 C：Explore

目标：围绕真实瓶颈做设计空间探索。

完成标准：

- decode 有 attention/MLP 的分项时延和瓶颈解释
- prefill 有 blocked attention 的 tile 扫描与银行冲突解释
- sweep 重点不只限于 unroll，也包含 memory banking、buffer 分配、pipeline II 和 tile 形状

### 阶段 D：Timing

目标：开始向可交付实现收敛。

完成标准：

- 主要候选点 slack 开始接近收敛
- 吞吐、带宽、面积三者有成体系的折中表
- 能说明为什么某条路线被淘汰，为什么候选点保留

### 阶段 E：RTL

目标：形成可交付成果。

完成标准：

- Catapult 生成 RTL
- 验证、性能、面积、时序结论成文
- 交付版本可追溯到脚本、参数和日志

## 8. 第一阶段优先任务

1. 建立 `performance_model/` 下的联合预算脚本，覆盖 prefill + decode。
2. 固化模型配置抽取和 workload facts 生成逻辑。
3. 定义 decode-only 的 baseline kernel 划分。
4. 定义 prefill-only 的 blocked attention baseline kernel 划分。
5. 约定 Catapult 输出格式、metric 命名和日志归档路径。
6. 定义验证分层，避免把 full-model、kernel-level、quant-level 误差混在一起。

## 9. 关键风险

- 1 MB SRAM 对 prefill 的限制比 decode 更严，attention 中间态和 softmax scratch 很可能先成为主瓶颈。
- 2K MAC 在 decode 路线上未必先受算力限制，但在 prefill blocked attention 中可能转化为 tile 选择和 feed 效率问题。
- 如果没有明确的 banking 模型，Catapult 的 architect 成功并不等于真实实现可落地。
- 如果过早把 prefill 与 decode 共用接口，可能导致两边都无法收敛。

## 10. 推荐推进策略

1. 先完成 decode baseline，用来打通公共类型、量化接口、Catapult 工具链和指标采集。
2. 并行建立 prefill attention 骨架，不等 decode 完全收敛后才启动。
3. 以 workload 和 memory-system 解释驱动设计，而不是只做广义 unroll sweep。
4. 在两条路径各自稳定前，不把 unified kernel 作为短期交付目标。

## 11. 当前结论

`work_llm_accelerator/` 的定位应是一个从零开始、覆盖 prefill + decode 的独立 LLM 加速器工程。它参考 `work/` 的经验，但不继承 `work/` 的实现资产。短期重点是建立联合预算、分离 prefill/decode 主线、打通 Catapult 到 architect 的可验证链路；中期重点是围绕 memory banking、tile 形状和 schedule 收敛；最终目标是形成可交付 RTL 与对应报告。