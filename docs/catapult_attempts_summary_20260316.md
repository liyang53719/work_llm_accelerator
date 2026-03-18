# Catapult 尝试成败总结（截至 2026-03-16）

## 2026-03-18 进展补记
1. context 路径继续按层收窄 helper 边界（进行中）
- 动作：已把 `score/context` 从 packet helper 继续下沉到 `per-key` helper，本轮再把 `head group` 的 init/compute/store 收成单独包装块，保持 query 级入口只负责调度。
- 结果：当前代码边界更接近后续 `cache reader / score core / softmax-context core` 的三段式拆分，外部接口未扩大。

2. `catapult_prefill_attention_context` 长时间 compile 观察（进行中）
- 现象：`QWEN_HLS_ENABLE_EXTRACT=0 make catapult_prefill_attention_context` 运行时间明显拉长，短窗口不适合判断成败。
- 当前观测：日志已稳定进入 `compile`，至少推进到 150 秒以上，期间未命中新的 `Error:` / `Failed` 关键字；日志尾部仍处于 `compile` 内联展开阶段，尚未看到终态。
- 结论：当前更像是 compile 长跑，而不是立即回退到新的前端解析错误。

## 成功尝试
1. 输入 token 访问方式修复（成功）
- 现象：早期 compile 触发内部断言，关联 input sequence 的 reinterpret cast 访问。
- 动作：去掉脆弱 2D reinterpret cast，改为扁平偏移访问。
- 结果：相关内部断言不再复现。

2. RMSNorm 路径去递归/定粒度化（部分成功）
- 现象：compile 阶段在 RMSNorm/归约展开处出现明显膨胀与耗时。
- 动作：递归/模板式归约改为固定规模 helper（包括 staged reduction）。
- 结果：compile 前进性提升，旧爆炸点缓解。

3. rope helper 接口冲突修复（成功）
- 现象：`apply_rope_inplace_fp` 相关接口 array mismatch。
- 动作：取消该 helper 的独立 ccore 边界，回归本地 helper。
- 结果：rope 接口连接错误消失。

4. q_token/k_proj 接口错误修复（成功）
- 现象：`q_token.d` / `k_proj.d` CIN-84 + CIN-178 + CIN-236。
- 动作：取消 `attention_max_score_pass_fp` / `attention_value_accum_pass_fp` 的独立 ccore 边界。
- 结果：对应接口尺寸收缩冲突不再作为首发错误出现。

## 失败或无效尝试
1. Tcl 路径级 INLINE off 探测（失败）
- 原因：对象路径在该 flow 中不稳定/不可用，路径匹配失败。

2. 仅靠 pragma unroll 期望“日志无 LOOP-2”（失败）
- 原因：源码仍是 `for` 语句时，Catapult 可能仍报告 LOOP 统计；`unroll` 不等价于“无循环语法”。

3. 以短时窗口判断“compile后不往下跑”（误判）
- 根因：运行窗口/外部观察时长不足，compile 尚未完成，不是 Tcl 缺步骤。

## 关键经验
1. 先区分“接口边界问题”与“性能/展开问题”
- CIN-84/178/236 多是边界推断冲突；LOOP-2 多是循环语法/结构问题。

2. 对切片指针参数 helper，谨慎拆 ccore
- `buffer[idx]` 类切片跨 ccore 容易触发接口尺寸推断问题。

3. 需要“彻底无循环痕迹”时，必须源码级显式展开
- 仅靠 `#pragma hls_unroll yes` 不足以满足该目标。

## 当前状态（简要）
- 已具备 fine/coarse 两种 top 组织方式。
- 已形成可复用 rerun 流程与阶段 marker。
- 当前可作为切换新 Catapult 版本前的 checkpoint 基线。
