# Verification

该目录负责 `work_llm_accelerator` 的正确性框架。原则是先锁定真实 Qwen2.5-1.5B 的整网行为，再让 HLS kernel 和 wrapper 去逼近这个行为。

## 当前文件

- `backend_interface.py`
  - 统一 prefill / decode backend 接口
- `torch_reference_backend.py`
  - 基于本地 `module/qwen_model` 的参考后端
- `manual_dispatch_backend.py`
  - 复用同一组单层 decoder layer，显式循环 28 层，验证“单层路径 + layer loop”与原生整网 forward 一致
- `descriptor_dispatch_backend.py`
  - 在 manual dispatch 基础上显式消费 all-layer descriptor，验证 layer reuse 与地址布局方案能驱动同一组单层 layer 执行
- `hls_backend_stub.py`
  - 预留给未来 shared library 或 RTL cosim wrapper 的入口
- `qwen_full_model_validation.py`
  - 整网校验脚本，先做 prefill，再做 decode step，并比较 logits 与 cache
- `run_validation_suite.py`
  - 多 prompt、多步 decode 的整网回归入口
- `validation_prompts.json`
  - 当前默认回归 prompt 集合
- `generate_layer0_prefill_case.py`
  - 导出真实 Qwen2.5-1.5B 的 layer0 prefill 子图参考张量与 cache
- `generate_layer0_decode_case.py`
  - 导出真实 Qwen2.5-1.5B 的 layer0 decode-step 子图参考张量与 cache
- `export_layer0_prefill_params.py`
  - 导出 layer0 prefill reference wrapper 所需的权重、bias 和 norm 参数
- `validate_prefill_layer0_wrapper.py`
  - 通过 ctypes 调用 prefill shared library，检查 ABI 与 layer0 对照
- `validate_layer0_prefill_reference_math.py`
  - 用独立的 layer0 prefill 数学路径重建 Qwen2.5-1.5B layer0，并逐节点对照导出的子图张量
- `layer0_prefill_reference_backend.py`
  - 可复用的 layer0 prefill Python reference backend，供 wrapper 对照和逐节点验证复用
- `validate_prefill_layer0_reference_wrapper_math.py`
  - 通过 ctypes 调用 C++ layer0 prefill reference wrapper，并直接对照导出的 layer0 输出
- `validate_decode_layer0_reference_wrapper_math.py`
  - 通过 ctypes 调用 C++ layer0 decode-step reference wrapper，并对照 layer0 输出与 KV cache
- `validate_decode_attention_smoke.py`
  - 用最小 packed-weight 场景直接调用 decode attention kernel 的 C ABI wrapper，校验 INT4 解包、RMSNorm、KV 追加和 attention 主路径
- `validate_decode_attention_history_regression.py`
  - 用非零历史 KV 和多头 synthetic case 对 decode attention kernel 做更强的 regression，并直接对照 Python 参考实现
- `validate_decode_mlp_smoke.py`
  - 用最小 packed-weight 场景直接调用 decode MLP kernel 的 C ABI wrapper，校验 post-attention RMSNorm、gate/up/down 和 residual add 路径
- `validate_decode_top_wrapper_regression.py`
  - 用同一最小场景对比 decode top wrapper 与直接 kernel wrapper，校验 descriptor 地址解码与端口映射不出错
- `validate_prefill_decode_summary_example.py`
  - 用真实文本总结 prompt 联合执行 prefill + decode，并把 layer0 prefill/decode reference wrapper 直接对照 PyTorch 结果
- `run_host_regression_suite.py`
  - 统一串起 descriptor、decode attention/MLP smoke、top-wrapper regression 和真实 prompt 的 prefill+decode regression
- `validate_layer_dispatch_layout.py`
  - 校验 decode/prefill 的 all-layer descriptor 生成、DDR 地址布局和 1 MB SRAM 分区口径
- `validate_all_layer_manual_dispatch.py`
  - 用手工 28 层 dispatch 路径对照原生 Qwen2.5-1.5B forward，验证层复用方案本身正确
- `validate_all_layer_descriptor_dispatch.py`
  - 用 descriptor 驱动的 28 层 dispatch 路径对照原生 Qwen2.5-1.5B forward，验证 descriptor 方案本身正确

## 当前策略

- 先用 Hugging Face / PyTorch 路径把整网校验框架跑通。
- 再把 prefill-only 和 decode-only 的 HLS wrapper 挂到同一接口上。
- 只有在整网校验能稳定通过后，才扩大 Catapult 设计空间探索。

## 当前补充

- `hls/build_host_wrappers.sh` 可先编译 prefill/decode 的 host-side stub shared library，用于 ABI 冒烟和后续 ctypes 接入。
- `hls/build_host_wrappers.sh` 现在还会编译一个 C++ layer0 prefill reference wrapper，用于把 prefill host wrapper 从 identity stub 推向真实数学路径。
- `hls/build_host_wrappers.sh` 现在也会编译一个 C++ layer0 decode-step reference wrapper，用于把 decode host wrapper 推向真实数学路径。
- 当前 shared library 仍然只是接口骨架，不代表已经实现 Qwen2.5-1.5B 的整网数学路径。
- 当前建议先用 `generate_layer0_prefill_case.py` 固化 layer0 参考，再用 `validate_prefill_layer0_wrapper.py` 逐步把 prefill wrapper 从 ABI 骨架推进到真实数学实现。
- prefill 和 decode 都优先对齐 layer0 子图张量，再逐步扩到更多层和整网闭环。
- 在把 C++/HLS kernel 做实前，先用 `validate_layer0_prefill_reference_math.py` 把 Python 层的数学规格跑通，避免边实现边猜公式。
- `layer0_prefill_reference_backend.py` 是当前 prefill layer0 的单一数学规格来源；后续 wrapper 对照和子图验证都应复用它，而不是复制实现。
- `validate_layer_dispatch_layout.py` 用来锁定 layer 复用和 DDR 地址口径，避免在进入 AXI top-level 之前继续扩 layer-specific 参数接口。
- `manual_dispatch_backend.py` 和 `validate_all_layer_manual_dispatch.py` 用来证明“单层 layer 路径可通过 layer loop 复用到 28 层”，这是后续 descriptor + top wrapper 方案成立的前提。
- `descriptor_dispatch_backend.py` 和 `validate_all_layer_descriptor_dispatch.py` 进一步把这件事落到 descriptor 口径上，确保后续 top wrapper 与 software scheduling 不会在层编号和地址布局上分叉。