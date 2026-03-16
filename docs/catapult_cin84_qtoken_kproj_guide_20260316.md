# Catapult CIN-84/CIN-178（q_token.d / k_proj.d）问题排查与预防指南

## 1. 问题现象
在 `compile` 阶段出现如下典型报错：

- `Unable to reduce array size for variable 'q_token.d' (CIN-84)`
- `Unable to reduce array size for variable 'k_proj.d' (CIN-84)`
- `Interface array size mismatch ... (CIN-178)`
- `Failed to connect 'q_token' ... (CIN-236)`

这类报错常见于“函数间接口推断”而不是算法本身错误。

## 2. 根因机制（本次案例）
本次触发点在 attention context 内部两个 helper：

- `attention_max_score_pass_fp(...)`
- `attention_value_accum_pass_fp(...)`

它们被声明为独立 `ccore` 后，参数中存在切片指针：

- `q_token` 来自 `q_proj[query_index]`
- `k_proj` 来自外部缓存基址指针

Catapult 在跨 `ccore` 边界推断接口尺寸时，会尝试对数组口做“缩减”。
当参数是切片指针、且调用点上下文又有局部二维数组（如 `accum`）时，可能出现接口维度关联失败，最终报 CIN-84/CIN-178/CIN-236。

## 3. 本次修复动作
采用“最小行为改动”原则：

- 去掉上述两个 helper 的 `#pragma hls_design ccore` / `#pragma hls_ccore_type sequential`
- 保持其作为本地 helper（由上层 stage 内联/局部调度），避免形成独立接口边界

这样不会改变功能路径和数学逻辑，只是避免不稳定的接口收缩推断。

## 4. 如何预防这类错误

1. 对“切片指针参数 helper”默认不要单独做 `ccore`
- 典型形式：`const T* token = buffer[idx];` 再传给子函数
- 这类函数更适合作为局部 helper

2. 仅在边界稳定时做 `ccore`
- 参数是固定维度数组（编译期常量）
- 没有复杂切片/偏移复用
- 调用点数量与形态较稳定

3. 对 attention 内核优先按“阶段函数”做 `ccore`
- 如 input_norm / q_proj / k_proj / v_proj / rope / context / output
- 阶段内小 helper 不强制拆 ccore

4. 出现 CIN-84 时先查“新加的 ccore helper”
- 如果报错出现在 `Failed to connect ...`，先怀疑接口边界，而非算法值错误

## 5. 出现后怎么快速处理

1. 先定位调用链
- 看 CIN-236 的 `Failed to connect` 指向哪个函数调用
- 看 CIN-178 的 mismatch 指向哪个参数口

2. 回退最近新增/调整的 helper 级 `ccore`
- 优先回退 pointer-slice 参与的 helper

3. 保留 stage 级 `ccore`，避免一次性大范围回退
- 这样能控制改动面，降低引入新问题的概率

4. 重新 `go compile` 验证
- 若错误消失，再评估是否需要更强约束（比如改成固定数组签名）

## 6. 进阶可选方案（仅在需要时）
若未来必须保留 helper 为独立 `ccore`，可考虑：

- 改成固定维度数组参数（减少指针切片）
- 引入中间缓冲，把切片先拷贝到固定数组再传递
- 减少同一 helper 上下文中多来源指针的混用

这些方案会增加代码复杂度，建议仅在确有收益时采用。
