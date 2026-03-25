#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_DIR = ROOT / "verification"
PYTHON_DIR = ROOT / "python"
for path in (VERIFICATION_DIR, PYTHON_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from layer_descriptor_builder import build_layer_parameter_layout, load_qwen_model_spec  # type: ignore


DEFAULT_LIB_PATH = ROOT / "tmp" / "host_libs" / "libqwen_prefill_stub.so"
HIDDEN_SIZE = 1536
KV_WIDTH = 256
INTERMEDIATE_SIZE = 8960
ATTENTION_SEQ_TILE = 128
ATTENTION_QUERY_TILE = 128
ATTENTION_KEY_TILE = 128
ATTENTION_HIDDEN_PROJ_TILE = 256
ATTENTION_KV_PROJ_TILE = 256
ATTENTION_HEAD_DIM_TILE = 128
ATTENTION_QUERY_HEADS_PARALLEL = 2
ATTENTION_KV_HEADS_PARALLEL = 1
MLP_SEQ_TILE = 128
MLP_HIDDEN_TILE = 256
MLP_FF_TILE = 640


def set_packed_weight(packed: np.ndarray, out_dim: int, in_dim: int, out_index: int, in_index: int, value: int) -> None:
    flat_index = out_index * in_dim + in_index
    byte_index = flat_index // 2
    nibble = value & 0xF
    if flat_index % 2 == 0:
        packed[byte_index] = (packed[byte_index] & 0xF0) | nibble
    else:
        packed[byte_index] = (packed[byte_index] & 0x0F) | (nibble << 4)


def build_sequence_case(seq_len: int) -> np.ndarray:
    input_sequence = np.zeros((seq_len, HIDDEN_SIZE), dtype=np.float32)
    for token in range(seq_len):
        input_sequence[token, 0] = float(token + 1)
        input_sequence[token, 1] = -0.5 * float(token + 1)
        input_sequence[token, 2] = 0.25 * float(token + 1)
    return input_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression-test prefill top wrapper address decoding against the direct full-layer smoke wrapper.")
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    spec = load_qwen_model_spec()
    layout = build_layer_parameter_layout(spec)

    input_sequence = build_sequence_case(args.seq_len)
    input_layernorm_weight = np.ones(HIDDEN_SIZE, dtype=np.float32)
    post_attention_layernorm_weight = np.ones(HIDDEN_SIZE, dtype=np.float32)

    weight_ddr = np.zeros(layout.total_parameter_bytes, dtype=np.uint8)
    q_view = weight_ddr[layout.q_weight_offset_bytes : layout.k_weight_offset_bytes]
    k_view = weight_ddr[layout.k_weight_offset_bytes : layout.v_weight_offset_bytes]
    v_view = weight_ddr[layout.v_weight_offset_bytes : layout.o_weight_offset_bytes]
    o_view = weight_ddr[layout.o_weight_offset_bytes : layout.post_attention_layernorm_weight_offset_bytes]
    gate_view = weight_ddr[layout.gate_weight_offset_bytes : layout.up_weight_offset_bytes]
    up_view = weight_ddr[layout.up_weight_offset_bytes : layout.down_weight_offset_bytes]
    down_view = weight_ddr[layout.down_weight_offset_bytes : layout.q_bias_offset_bytes]

    set_packed_weight(q_view, HIDDEN_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(q_view, HIDDEN_SIZE, HIDDEN_SIZE, 128, 1, -1)
    set_packed_weight(k_view, KV_WIDTH, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(k_view, KV_WIDTH, HIDDEN_SIZE, 128, 1, -1)
    set_packed_weight(v_view, KV_WIDTH, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(v_view, KV_WIDTH, HIDDEN_SIZE, 128, 1, 1)
    set_packed_weight(o_view, HIDDEN_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(o_view, HIDDEN_SIZE, HIDDEN_SIZE, 1, 128, 1)
    set_packed_weight(gate_view, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(up_view, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 1, 1)
    set_packed_weight(down_view, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0, 0, 1)

    scale_ddr = np.zeros((layout.total_parameter_bytes + 3) // 4, dtype=np.float32)
    scale_ddr[layout.input_layernorm_weight_offset_bytes // 4 : layout.input_layernorm_weight_offset_bytes // 4 + HIDDEN_SIZE] = input_layernorm_weight
    scale_ddr[
        layout.post_attention_layernorm_weight_offset_bytes // 4 : layout.post_attention_layernorm_weight_offset_bytes // 4 + HIDDEN_SIZE
    ] = post_attention_layernorm_weight
    scale_ddr[layout.q_bias_offset_bytes // 4 : layout.q_bias_offset_bytes // 4 + HIDDEN_SIZE] = 0.0
    scale_ddr[layout.k_bias_offset_bytes // 4 : layout.k_bias_offset_bytes // 4 + KV_WIDTH] = 0.0
    scale_ddr[layout.v_bias_offset_bytes // 4 : layout.v_bias_offset_bytes // 4 + KV_WIDTH] = 0.0
    scale_ddr[layout.q_scale_offset_bytes // 4 : layout.q_scale_offset_bytes // 4 + HIDDEN_SIZE] = 1.0
    scale_ddr[layout.k_scale_offset_bytes // 4 : layout.k_scale_offset_bytes // 4 + KV_WIDTH] = 1.0
    scale_ddr[layout.v_scale_offset_bytes // 4 : layout.v_scale_offset_bytes // 4 + KV_WIDTH] = 1.0
    scale_ddr[layout.o_scale_offset_bytes // 4 : layout.o_scale_offset_bytes // 4 + HIDDEN_SIZE] = 1.0
    scale_ddr[layout.gate_scale_offset_bytes // 4 : layout.gate_scale_offset_bytes // 4 + INTERMEDIATE_SIZE] = 1.0
    scale_ddr[layout.up_scale_offset_bytes // 4 : layout.up_scale_offset_bytes // 4 + INTERMEDIATE_SIZE] = 1.0
    scale_ddr[layout.down_scale_offset_bytes // 4 : layout.down_scale_offset_bytes // 4 + HIDDEN_SIZE] = 1.0

    direct_k_cache = np.zeros(args.seq_len * KV_WIDTH, dtype=np.float32)
    direct_v_cache = np.zeros(args.seq_len * KV_WIDTH, dtype=np.float32)
    direct_output = np.zeros(args.seq_len * HIDDEN_SIZE, dtype=np.float32)

    activation_ddr = np.zeros(args.seq_len * HIDDEN_SIZE * 3, dtype=np.float32)
    activation_ddr[: args.seq_len * HIDDEN_SIZE] = input_sequence.reshape(-1)
    kv_cache_ddr = np.zeros(args.seq_len * KV_WIDTH * 2, dtype=np.float32)
    weight_sram = np.zeros(1, dtype=np.uint8)
    kv_sram = np.zeros(1, dtype=np.float32)
    partial_sum_sram = np.zeros(1, dtype=np.int32)
    softmax_sram = np.zeros(1, dtype=np.float32)
    control_sram = np.zeros(1, dtype=np.float32)

    library = ctypes.CDLL(str(args.lib_path))
    float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    byte_ptr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    int_ptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

    direct_func = library.qwen_prefill_layer_smoke_forward
    direct_func.argtypes = [
        float_ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        float_ptr,
        float_ptr,
        byte_ptr,
        byte_ptr,
        byte_ptr,
        byte_ptr,
        byte_ptr,
        byte_ptr,
        byte_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
    ]
    direct_func.restype = ctypes.c_int

    top_func = library.qwen_prefill_top_smoke_forward
    top_func.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        byte_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        byte_ptr,
        float_ptr,
        int_ptr,
        float_ptr,
        float_ptr,
    ]
    top_func.restype = ctypes.c_int

    status = direct_func(
        np.ascontiguousarray(input_sequence.reshape(-1)),
        args.seq_len,
        ATTENTION_SEQ_TILE,
        ATTENTION_QUERY_TILE,
        ATTENTION_KEY_TILE,
        ATTENTION_HIDDEN_PROJ_TILE,
        ATTENTION_KV_PROJ_TILE,
        ATTENTION_HEAD_DIM_TILE,
        ATTENTION_QUERY_HEADS_PARALLEL,
        ATTENTION_KV_HEADS_PARALLEL,
        MLP_SEQ_TILE,
        MLP_HIDDEN_TILE,
        MLP_FF_TILE,
        input_layernorm_weight,
        post_attention_layernorm_weight,
        q_view,
        k_view,
        v_view,
        o_view,
        gate_view,
        up_view,
        down_view,
        scale_ddr[layout.q_bias_offset_bytes // 4 : layout.q_bias_offset_bytes // 4 + HIDDEN_SIZE],
        scale_ddr[layout.k_bias_offset_bytes // 4 : layout.k_bias_offset_bytes // 4 + KV_WIDTH],
        scale_ddr[layout.v_bias_offset_bytes // 4 : layout.v_bias_offset_bytes // 4 + KV_WIDTH],
        scale_ddr[layout.q_scale_offset_bytes // 4 : layout.q_scale_offset_bytes // 4 + HIDDEN_SIZE],
        scale_ddr[layout.k_scale_offset_bytes // 4 : layout.k_scale_offset_bytes // 4 + KV_WIDTH],
        scale_ddr[layout.v_scale_offset_bytes // 4 : layout.v_scale_offset_bytes // 4 + KV_WIDTH],
        scale_ddr[layout.o_scale_offset_bytes // 4 : layout.o_scale_offset_bytes // 4 + HIDDEN_SIZE],
        scale_ddr[layout.gate_scale_offset_bytes // 4 : layout.gate_scale_offset_bytes // 4 + INTERMEDIATE_SIZE],
        scale_ddr[layout.up_scale_offset_bytes // 4 : layout.up_scale_offset_bytes // 4 + INTERMEDIATE_SIZE],
        scale_ddr[layout.down_scale_offset_bytes // 4 : layout.down_scale_offset_bytes // 4 + HIDDEN_SIZE],
        direct_k_cache,
        direct_v_cache,
        direct_output,
    )
    if status != 0:
        raise RuntimeError(f"Direct prefill layer smoke failed with status {status}")

    status = top_func(
        0,
        args.seq_len,
        ATTENTION_SEQ_TILE,
        ATTENTION_QUERY_TILE,
        ATTENTION_KEY_TILE,
        ATTENTION_HIDDEN_PROJ_TILE,
        ATTENTION_KV_PROJ_TILE,
        ATTENTION_HEAD_DIM_TILE,
        ATTENTION_QUERY_HEADS_PARALLEL,
        ATTENTION_KV_HEADS_PARALLEL,
        MLP_SEQ_TILE,
        MLP_HIDDEN_TILE,
        MLP_FF_TILE,
        0,
        args.seq_len * HIDDEN_SIZE * 4,
        0,
        0,
        0,
        args.seq_len * KV_WIDTH * 4,
        weight_ddr,
        scale_ddr,
        kv_cache_ddr,
        activation_ddr,
        weight_sram,
        kv_sram,
        partial_sum_sram,
        softmax_sram,
        control_sram,
    )
    if status != 0:
        raise RuntimeError(f"Prefill top wrapper smoke failed with status {status}")

    top_output = activation_ddr[args.seq_len * HIDDEN_SIZE : args.seq_len * HIDDEN_SIZE * 2]
    top_k_cache = kv_cache_ddr[: args.seq_len * KV_WIDTH]
    top_v_cache = kv_cache_ddr[args.seq_len * KV_WIDTH : args.seq_len * KV_WIDTH * 2]
    if np.max(np.abs(top_output - direct_output)) > args.atol:
        raise AssertionError("Prefill top-wrapper output diverged from direct layer output")
    if np.max(np.abs(top_k_cache - direct_k_cache)) > args.atol:
        raise AssertionError("Prefill top-wrapper K-cache diverged from direct layer output")
    if np.max(np.abs(top_v_cache - direct_v_cache)) > args.atol:
        raise AssertionError("Prefill top-wrapper V-cache diverged from direct layer output")
    print("Prefill top-wrapper regression PASS")


if __name__ == "__main__":
    main()
