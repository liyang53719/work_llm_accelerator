#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="$ROOT_DIR/tmp/host_libs"

mkdir -p "$OUT_DIR"

g++ -std=c++17 -O2 -fPIC -shared \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_attention_kernel.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_mlp_kernel.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_host_wrapper.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_layer0_reference_wrapper.cpp" \
  -o "$OUT_DIR/libqwen_prefill_stub.so"

g++ -std=c++17 -O2 -fPIC -shared \
  "$ROOT_DIR/hls/decode_only/qwen_decode_attention_kernel.cpp" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_mlp_kernel.cpp" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_top_wrapper.cpp" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_host_wrapper.cpp" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_layer0_reference_wrapper.cpp" \
  -o "$OUT_DIR/libqwen_decode_stub.so"

echo "Built host wrapper libraries in $OUT_DIR"