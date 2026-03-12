# Verification

该目录负责 `work_llm_accelerator` 的正确性框架。原则是先锁定真实 Qwen2.5-1.5B 的整网行为，再让 HLS kernel 和 wrapper 去逼近这个行为。

## 当前文件

- `backend_interface.py`
  - 统一 prefill / decode backend 接口
- `torch_reference_backend.py`
  - 基于本地 `module/qwen_model` 的参考后端
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
- `validate_prefill_layer0_wrapper.py`
  - 通过 ctypes 调用 prefill shared library，检查 ABI 与 layer0 对照

## 当前策略

- 先用 Hugging Face / PyTorch 路径把整网校验框架跑通。
- 再把 prefill-only 和 decode-only 的 HLS wrapper 挂到同一接口上。
- 只有在整网校验能稳定通过后，才扩大 Catapult 设计空间探索。

## 当前补充

- `hls/build_host_wrappers.sh` 可先编译 prefill/decode 的 host-side stub shared library，用于 ABI 冒烟和后续 ctypes 接入。
- 当前 shared library 仍然只是接口骨架，不代表已经实现 Qwen2.5-1.5B 的整网数学路径。
- 当前建议先用 `generate_layer0_prefill_case.py` 固化 layer0 参考，再用 `validate_prefill_layer0_wrapper.py` 逐步把 prefill wrapper 从 ABI 骨架推进到真实数学实现。
- prefill 和 decode 都优先对齐 layer0 子图张量，再逐步扩到更多层和整网闭环。