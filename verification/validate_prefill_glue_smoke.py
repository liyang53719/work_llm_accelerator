#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB_PATH = ROOT / "tmp" / "host_libs" / "libqwen_prefill_stub.so"

HIDDEN_SIZE = 1536
NUM_ATTENTION_HEADS = 12
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 128
KV_WIDTH = NUM_KEY_VALUE_HEADS * HEAD_DIM
INTERMEDIATE_SIZE = 8960
ROPE_THETA = 1_000_000.0
RMS_NORM_EPS = 1.0e-6
ATTENTION_SEQ_TILE = 128
ATTENTION_QUERY_TILE = 32
ATTENTION_KEY_TILE = 64
ATTENTION_HIDDEN_PROJ_TILE = 64
ATTENTION_KV_PROJ_TILE = 64
ATTENTION_HEAD_DIM_TILE = 128
ATTENTION_QUERY_HEADS_PARALLEL = 2
ATTENTION_KV_HEADS_PARALLEL = 1
MLP_SEQ_TILE = 128
MLP_HIDDEN_TILE = 256
MLP_FF_TILE = 640


def set_packed_weight(packed: np.ndarray, out_dim: int, in_dim: int, out_index: int, in_index: int, value: int) -> None:
    if value < -8 or value > 7:
        raise ValueError("int4 value out of range")
    flat_index = out_index * in_dim + in_index
    byte_index = flat_index // 2
    nibble = value & 0xF
    if flat_index % 2 == 0:
        packed[byte_index] = (packed[byte_index] & 0xF0) | nibble
    else:
        packed[byte_index] = (packed[byte_index] & 0x0F) | (nibble << 4)


def rotate_head_inplace(head: np.ndarray, token_index: int) -> None:
    half_dim = HEAD_DIM // 2
    for pair in range(half_dim):
        angle = token_index * (ROPE_THETA ** (-2.0 * pair / HEAD_DIM))
        cosv = np.float32(np.cos(angle))
        sinv = np.float32(np.sin(angle))
        even = np.float32(head[pair])
        odd = np.float32(head[pair + half_dim])
        head[pair] = np.float32(even * cosv - odd * sinv)
        head[pair + half_dim] = np.float32(odd * cosv + even * sinv)


def silu(value: np.float32) -> np.float32:
    return np.float32(value / (1.0 + np.exp(-value, dtype=np.float32)))


def rmsnorm(token: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mean_square = np.mean(token.astype(np.float64) ** 2)
    inv_rms = np.float32(1.0 / np.sqrt(mean_square + RMS_NORM_EPS))
    return (token * inv_rms * weight).astype(np.float32)


def build_case(seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    input_sequence = np.zeros((seq_len, HIDDEN_SIZE), dtype=np.float32)
    for token_index in range(seq_len):
        input_sequence[token_index, 0] = np.float32(token_index + 1)
        input_sequence[token_index, 1] = np.float32(-0.5 * (token_index + 1))
    post_attention_layernorm_weight = np.ones(HIDDEN_SIZE, dtype=np.float32)
    return input_sequence, post_attention_layernorm_weight


def prefill_attention_reference(input_sequence: np.ndarray, input_layernorm_weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_len = input_sequence.shape[0]
    q_proj = np.zeros((seq_len, HIDDEN_SIZE), dtype=np.float32)
    k_proj = np.zeros((seq_len, KV_WIDTH), dtype=np.float32)
    v_proj = np.zeros((seq_len, KV_WIDTH), dtype=np.float32)

    normed = np.zeros_like(input_sequence)
    for token_index in range(seq_len):
        normed[token_index] = rmsnorm(input_sequence[token_index], input_layernorm_weight)
        q_proj[token_index, 0] = normed[token_index, 0]
        k_proj[token_index, 0] = normed[token_index, 0]
        v_proj[token_index, 0] = normed[token_index, 0]
        v_proj[token_index, 1] = normed[token_index, 1]
        for head in range(NUM_ATTENTION_HEADS):
            rotate_head_inplace(q_proj[token_index, head * HEAD_DIM : (head + 1) * HEAD_DIM], token_index)
        for head in range(NUM_KEY_VALUE_HEADS):
            rotate_head_inplace(k_proj[token_index, head * HEAD_DIM : (head + 1) * HEAD_DIM], token_index)

    context = np.zeros((seq_len, HIDDEN_SIZE), dtype=np.float32)
    scaling = np.float32(1.0 / np.sqrt(HEAD_DIM))
    num_groups = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS

    for query_index in range(seq_len):
        for head in range(NUM_ATTENTION_HEADS):
            kv_head = head // num_groups
            q_head = q_proj[query_index, head * HEAD_DIM : (head + 1) * HEAD_DIM]
            scores = []
            for key_index in range(query_index + 1):
                k_head = k_proj[key_index, kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
                scores.append(np.float32(np.dot(q_head, k_head) * scaling))
            score_array = np.asarray(scores, dtype=np.float32)
            max_score = np.max(score_array)
            probs = np.exp(score_array - max_score).astype(np.float32)
            probs /= np.sum(probs, dtype=np.float32)
            for key_index in range(query_index + 1):
                v_head = v_proj[key_index, kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM]
                context[query_index, head * HEAD_DIM : (head + 1) * HEAD_DIM] += probs[key_index] * v_head

    output = np.zeros((seq_len, HIDDEN_SIZE), dtype=np.float32)
    output[:, 0] = context[:, 0]
    output[:, 1] = context[:, 1]
    return k_proj.reshape(-1), v_proj.reshape(-1), output.reshape(seq_len, HIDDEN_SIZE)


def prefill_glue_reference(
    input_sequence: np.ndarray,
    input_layernorm_weight: np.ndarray,
    post_attention_layernorm_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_k, expected_v, attention_output = prefill_attention_reference(input_sequence, input_layernorm_weight)
    final_output = attention_output.copy()
    for token_index in range(input_sequence.shape[0]):
        post_norm = rmsnorm(attention_output[token_index], post_attention_layernorm_weight)
        final_output[token_index, 0] = np.float32(final_output[token_index, 0] + silu(post_norm[0]) * post_norm[1])
    return expected_k, expected_v, final_output.reshape(-1)


def max_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the prefill glue stream path against direct software chaining and a Python reference.")
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--check-python-reference", action="store_true")
    args = parser.parse_args()

    input_sequence, post_attention_layernorm_weight = build_case(args.seq_len)
    input_layernorm_weight = np.ones(HIDDEN_SIZE, dtype=np.float32)

    q_weights = np.zeros(HIDDEN_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    k_weights = np.zeros(HIDDEN_SIZE * KV_WIDTH // 2, dtype=np.uint8)
    v_weights = np.zeros(HIDDEN_SIZE * KV_WIDTH // 2, dtype=np.uint8)
    o_weights = np.zeros(HIDDEN_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    gate_weights = np.zeros(INTERMEDIATE_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    up_weights = np.zeros(INTERMEDIATE_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    down_weights = np.zeros(HIDDEN_SIZE * INTERMEDIATE_SIZE // 2, dtype=np.uint8)

    set_packed_weight(q_weights, HIDDEN_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(k_weights, KV_WIDTH, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(v_weights, KV_WIDTH, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(v_weights, KV_WIDTH, HIDDEN_SIZE, 1, 1, 1)
    set_packed_weight(o_weights, HIDDEN_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(o_weights, HIDDEN_SIZE, HIDDEN_SIZE, 1, 1, 1)
    set_packed_weight(gate_weights, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(up_weights, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 1, 1)
    set_packed_weight(down_weights, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0, 0, 1)

    q_bias = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    k_bias = np.zeros(KV_WIDTH, dtype=np.float32)
    v_bias = np.zeros(KV_WIDTH, dtype=np.float32)
    q_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)
    k_scales = np.ones(KV_WIDTH, dtype=np.float32)
    v_scales = np.ones(KV_WIDTH, dtype=np.float32)
    o_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)
    gate_scales = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
    up_scales = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
    down_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)

    direct_k_cache = np.zeros(args.seq_len * KV_WIDTH, dtype=np.float32)
    direct_v_cache = np.zeros(args.seq_len * KV_WIDTH, dtype=np.float32)
    direct_output = np.zeros(args.seq_len * HIDDEN_SIZE, dtype=np.float32)
    glue_k_cache = np.zeros_like(direct_k_cache)
    glue_v_cache = np.zeros_like(direct_v_cache)
    glue_output = np.zeros_like(direct_output)

    library = ctypes.CDLL(str(args.lib_path))
    float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    byte_ptr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    common_argtypes = [
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

    direct_func = library.qwen_prefill_layer_smoke_forward
    direct_func.argtypes = common_argtypes
    direct_func.restype = ctypes.c_int

    glue_func = library.qwen_prefill_glue_smoke_forward
    glue_func.argtypes = common_argtypes
    glue_func.restype = ctypes.c_int

    call_args = [
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
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        gate_weights,
        up_weights,
        down_weights,
        q_bias,
        k_bias,
        v_bias,
        q_scales,
        k_scales,
        v_scales,
        o_scales,
        gate_scales,
        up_scales,
        down_scales,
    ]

    direct_status = direct_func(*call_args, direct_k_cache, direct_v_cache, direct_output)
    if direct_status != 0:
        raise RuntimeError(f"qwen_prefill_layer_smoke_forward failed with status {direct_status}")

    glue_status = glue_func(*call_args, glue_k_cache, glue_v_cache, glue_output)
    if glue_status != 0:
        raise RuntimeError(f"qwen_prefill_glue_smoke_forward failed with status {glue_status}")

    expected_k, expected_v, expected_output = prefill_glue_reference(
        input_sequence,
        input_layernorm_weight,
        post_attention_layernorm_weight,
    )

    print(f"direct vs glue output max diff: {max_diff(direct_output, glue_output):.6g}")
    print(f"direct vs glue k_cache max diff: {max_diff(direct_k_cache, glue_k_cache):.6g}")
    print(f"direct vs glue v_cache max diff: {max_diff(direct_v_cache, glue_v_cache):.6g}")
    print(f"glue vs python output max diff: {max_diff(glue_output, expected_output):.6g}")
    print(f"glue vs python k_cache max diff: {max_diff(glue_k_cache, expected_k):.6g}")
    print(f"glue vs python v_cache max diff: {max_diff(glue_v_cache, expected_v):.6g}")

    if not np.allclose(glue_output, direct_output, atol=args.atol):
        raise AssertionError("Glue output does not match direct software chain")
    if not np.allclose(glue_k_cache, direct_k_cache, atol=args.atol):
        raise AssertionError("Glue K cache does not match direct software chain")
    if not np.allclose(glue_v_cache, direct_v_cache, atol=args.atol):
        raise AssertionError("Glue V cache does not match direct software chain")

    if args.check_python_reference:
        if not np.allclose(glue_output, expected_output, atol=args.atol):
            raise AssertionError("Glue output does not match Python reference")
        if not np.allclose(glue_k_cache, expected_k, atol=args.atol):
            raise AssertionError("Glue K cache does not match Python reference")
        if not np.allclose(glue_v_cache, expected_v, atol=args.atol):
            raise AssertionError("Glue V cache does not match Python reference")

    print("Prefill glue smoke PASS")


if __name__ == "__main__":
    main()