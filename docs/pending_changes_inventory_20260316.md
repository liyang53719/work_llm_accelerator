# work_llm_accelerator 未提交改动盘点（2026-03-16）

## 2026-03-18 当前增量
- 当前实际未提交代码改动已收敛到 `hls/prefill_only/qwen_prefill_attention_kernel.cpp` 一处，主要是 `score/context` 路径的 helper 分层继续收窄。
- 本轮新增方向：
	- `per-key` score/max/value helper 下沉，减少 tile loop 内直接混合的 load/compute/update 逻辑。
	- `head group` 级包装块收口，令 query 级入口只保留 head-group 调度。
- 当前验证状态：`catapult_prefill_attention_context` 正在长时间 `compile`，已观察到持续推进且暂未出现新的 `Error:` / `Failed`。
- 额外未跟踪文件仍有 `docs/arc_talk.md`，不应混入这轮最小提交。

## 仓库状态
- 仓库路径：`/home/yang/Documents/github/tvm/work_llm_accelerator`
- 分支：`main`
- 基线提交：`9286a0a`（Checkpoint Catapult header shims and DW fallback）

## 当前未提交改动（tracked）
1. `hls/prefill_only/qwen_prefill_attention_kernel.cpp`
- 规模：大改（attention catapult 化、RMSNorm/投影/rope/context 路径重构、helper 展开/接口处理）

2. `hls/prefill_only/qwen_prefill_attention_kernel.h`
- 规模：大改（新增 catapult prefill attention stage 及 kernel 声明）

3. `hls/prefill_only/qwen_prefill_mlp_kernel.cpp`
- 规模：大改（MLP catapult 相关路径补齐与数值近似/投影风格统一）

4. `hls/prefill_only/qwen_prefill_mlp_kernel.h`
- 规模：中改（对应 catapult 接口声明补齐）

5. `hls/prefill_only/qwen_prefill_top_wrapper.cpp`
- 规模：中改（top wrapper 适配）

6. `hls/prefill_only/run_catapult_prefill_attention.tcl`
- 规模：中改（run flow/参数与阶段脚本调整）

## 当前未提交改动（untracked）
1. `hls/prefill_only/qwen_catapult_fp.h`
- 新增 catapult 类型桥接头

2. `hls/prefill_only/qwen_prefill_top_catapult.cpp`
- 新增 prefill top（含 fine/coarse 相关 orchestration）

3. `hls/prefill_only/qwen_prefill_top_catapult.h`
- 新增 top 声明

4. `hls/prefill_only/qwen_prefill_top_core.cpp`
- 新增 top core 实现

5. `hls/prefill_only/qwen_prefill_top_core.h`
- 新增 top core 声明

6. `docs/catapult_cin84_qtoken_kproj_guide_20260316.md`
- 新增问题处理文档（CIN-84/CIN-178/CIN-236）

## 结论
- 当前改动不是单点修补，而是“prefill-only catapult 路径的一组阶段化重构 + 多轮验证脚本迭代”。
- 建议作为一个 checkpoint 提交，便于后续切换 Catapult 版本后做兼容修补与回归对比。
