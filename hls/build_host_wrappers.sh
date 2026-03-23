#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="$ROOT_DIR/tmp/host_libs"

if [[ -n "${MGC_HOME:-}" ]]; then
  CATAPULT_SHARED_INCLUDE="$MGC_HOME/shared/include"
else
  CATAPULT_BIN=$(command -v catapult || true)
  if [[ -z "$CATAPULT_BIN" ]]; then
    echo "catapult not found in PATH; cannot resolve ac_channel.h from Catapult shared/include" >&2
    exit 1
  fi
  CATAPULT_SHARED_INCLUDE=$(cd -- "$(dirname -- "$CATAPULT_BIN")/../shared/include" && pwd)
fi

if [[ ! -f "$CATAPULT_SHARED_INCLUDE/ac_channel.h" ]]; then
  echo "ac_channel.h not found under $CATAPULT_SHARED_INCLUDE" >&2
  exit 1
fi

CATAPULT_LIBSTDCPP=$(g++ -print-file-name=libstdc++.so.6)
if [[ ! -f "$CATAPULT_LIBSTDCPP" ]]; then
  echo "libstdc++.so.6 not found for active g++" >&2
  exit 1
fi
CATAPULT_LIBSTDCPP_DIR=$(cd -- "$(dirname -- "$CATAPULT_LIBSTDCPP")" && pwd)

mkdir -p "$OUT_DIR"

g++ -std=c++17 -O2 -fPIC -shared \
  -I"$CATAPULT_SHARED_INCLUDE" \
  -Wl,-rpath,"$CATAPULT_LIBSTDCPP_DIR" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_attention_stream_top.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_attention_kernel.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_glue_top_v1_catapult.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_host_catapult_shims.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_mlp_kernel.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_top_core.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_top_wrapper.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_host_wrapper.cpp" \
  "$ROOT_DIR/hls/prefill_only/qwen_prefill_layer0_reference_wrapper.cpp" \
  -o "$OUT_DIR/libqwen_prefill_stub.so"

g++ -std=c++17 -O2 -fPIC -shared \
  -I"$CATAPULT_SHARED_INCLUDE" \
  -Wl,-rpath,"$CATAPULT_LIBSTDCPP_DIR" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_attention_kernel.cpp" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_mlp_kernel.cpp" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_top_wrapper.cpp" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_host_wrapper.cpp" \
  "$ROOT_DIR/hls/decode_only/qwen_decode_layer0_reference_wrapper.cpp" \
  -o "$OUT_DIR/libqwen_decode_stub.so"

echo "Built host wrapper libraries in $OUT_DIR"