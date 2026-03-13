#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB_PATH = ROOT / "tmp" / "host_libs" / "libqwen_decode_stub.so"

HIDDEN_SIZE = 1536
HEAD_DIM = 128
NUM_HEADS = 12
NUM_KV_HEADS = 2
KV_WIDTH = NUM_KV_HEADS * HEAD_DIM
NUM_GROUPS = NUM_HEADS // NUM_KV_HEADS
RMS_EPS = 1.0e-6
ROPE_THETA = 1000000.0


def set_packed_weight(packed: np.ndarray, out_dim: int, in_dim: int, out_index: int, in_index: int, value: int) -> None:
    flat_index = out_index * in_dim + in_index
    byte_index = flat_index // 2
    nibble = value & 0xF
    if flat_index % 2 == 0:
        packed[byte_index] = (packed[byte_index] & 0xF0) | nibble
    else:
        packed[byte_index] = (packed[byte_index] & 0x0F) | (nibble << 4)


def build_case() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_token = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    input_token[0] = 1.0
    input_token[1] = -0.75
    input_token[2] = 0.5
    input_token[3] = 0.25

    input_layernorm_weight = np.ones(HIDDEN_SIZE, dtype=np.float32)
    q_weights = np.zeros(HIDDEN_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    k_weights = np.zeros(HIDDEN_SIZE * KV_WIDTH // 2, dtype=np.uint8)
    v_weights = np.zeros(HIDDEN_SIZE * KV_WIDTH // 2, dtype=np.uint8)
    o_weights = np.zeros(HIDDEN_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)

    set_packed_weight(q_weights, HIDDEN_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(q_weights, HIDDEN_SIZE, HIDDEN_SIZE, HEAD_DIM, 1, -1)
    set_packed_weight(q_weights, HIDDEN_SIZE, HIDDEN_SIZE, 6 * HEAD_DIM, 2, 1)
    set_packed_weight(q_weights, HIDDEN_SIZE, HIDDEN_SIZE, 7 * HEAD_DIM, 3, 1)

    set_packed_weight(k_weights, KV_WIDTH, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(k_weights, KV_WIDTH, HIDDEN_SIZE, HEAD_DIM, 1, -1)
    set_packed_weight(v_weights, KV_WIDTH, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(v_weights, KV_WIDTH, HIDDEN_SIZE, HEAD_DIM, 2, 1)

    set_packed_weight(o_weights, HIDDEN_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(o_weights, HIDDEN_SIZE, HIDDEN_SIZE, 1, HEAD_DIM, 1)
    set_packed_weight(o_weights, HIDDEN_SIZE, HIDDEN_SIZE, 2, 6 * HEAD_DIM, 1)
    set_packed_weight(o_weights, HIDDEN_SIZE, HIDDEN_SIZE, 3, 7 * HEAD_DIM, 1)

    q_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)
    k_scales = np.ones(KV_WIDTH, dtype=np.float32)
    v_scales = np.ones(KV_WIDTH, dtype=np.float32)
    o_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)
    q_bias = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    k_bias = np.zeros(KV_WIDTH, dtype=np.float32)
    v_bias = np.zeros(KV_WIDTH, dtype=np.float32)

    k_cache = np.zeros(KV_WIDTH * 3, dtype=np.float32)
    v_cache = np.zeros(KV_WIDTH * 3, dtype=np.float32)
    for token in range(2):
        k_cache[token * KV_WIDTH + 0] = 0.25 * (token + 1)
        k_cache[token * KV_WIDTH + HEAD_DIM] = -0.5 * (token + 1)
        v_cache[token * KV_WIDTH + 0] = 1.25 * (token + 1)
        v_cache[token * KV_WIDTH + HEAD_DIM] = -1.5 * (token + 1)
    return input_token, input_layernorm_weight, q_weights, k_weights, v_weights, o_weights, q_bias, k_bias, v_bias, q_scales, k_scales, v_scales, o_scales, k_cache, v_cache


def decode_int4_matrix(packed: np.ndarray, out_dim: int, in_dim: int, scales: np.ndarray) -> np.ndarray:
    matrix = np.zeros((out_dim, in_dim), dtype=np.float32)
    for out_index in range(out_dim):
        for in_index in range(in_dim):
            flat_index = out_index * in_dim + in_index
            packed_value = packed[flat_index // 2]
            nibble = (packed_value >> 4) & 0xF if flat_index % 2 else packed_value & 0xF
            signed_value = nibble - 16 if nibble >= 8 else nibble
            matrix[out_index, in_index] = signed_value * scales[out_index]
    return matrix


def apply_rope(head: np.ndarray, token_index: int) -> np.ndarray:
    rotated = head.copy()
    for pair in range(HEAD_DIM // 2):
        angle = token_index * (ROPE_THETA ** (-2.0 * pair / HEAD_DIM))
        cosv = np.cos(angle).astype(np.float32)
        sinv = np.sin(angle).astype(np.float32)
        even = head[pair]
        odd = head[pair + HEAD_DIM // 2]
        rotated[pair] = even * cosv - odd * sinv
        rotated[pair + HEAD_DIM // 2] = odd * cosv + even * sinv
    return rotated


def reference_decode(
    input_token: np.ndarray,
    input_layernorm_weight: np.ndarray,
    q_weights: np.ndarray,
    k_weights: np.ndarray,
    v_weights: np.ndarray,
    o_weights: np.ndarray,
    q_bias: np.ndarray,
    k_bias: np.ndarray,
    v_bias: np.ndarray,
    q_scales: np.ndarray,
    k_scales: np.ndarray,
    v_scales: np.ndarray,
    o_scales: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    past_seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = input_token * (1.0 / np.sqrt(np.mean(input_token * input_token) + RMS_EPS)) * input_layernorm_weight
    q_matrix = decode_int4_matrix(q_weights, HIDDEN_SIZE, HIDDEN_SIZE, q_scales)
    k_matrix = decode_int4_matrix(k_weights, KV_WIDTH, HIDDEN_SIZE, k_scales)
    v_matrix = decode_int4_matrix(v_weights, KV_WIDTH, HIDDEN_SIZE, v_scales)
    o_matrix = decode_int4_matrix(o_weights, HIDDEN_SIZE, HIDDEN_SIZE, o_scales)

    q_proj = q_matrix @ norm + q_bias
    k_proj = k_matrix @ norm + k_bias
    v_proj = v_matrix @ norm + v_bias
    for head in range(NUM_HEADS):
        q_proj[head * HEAD_DIM : (head + 1) * HEAD_DIM] = apply_rope(q_proj[head * HEAD_DIM : (head + 1) * HEAD_DIM], past_seq_len)
    for kv_head in range(NUM_KV_HEADS):
        k_proj[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM] = apply_rope(k_proj[kv_head * HEAD_DIM : (kv_head + 1) * HEAD_DIM], past_seq_len)

    next_k_cache = k_cache.copy()
    next_v_cache = v_cache.copy()
    next_k_cache[past_seq_len * KV_WIDTH : (past_seq_len + 1) * KV_WIDTH] = k_proj
    next_v_cache[past_seq_len * KV_WIDTH : (past_seq_len + 1) * KV_WIDTH] = v_proj

    context = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    scaling = 1.0 / np.sqrt(HEAD_DIM)
    for head in range(NUM_HEADS):
        kv_head = head // NUM_GROUPS
        q_head = q_proj[head * HEAD_DIM : (head + 1) * HEAD_DIM]
        scores = []
        for token in range(past_seq_len + 1):
            k_head = next_k_cache[token * KV_WIDTH + kv_head * HEAD_DIM : token * KV_WIDTH + (kv_head + 1) * HEAD_DIM]
            scores.append(np.dot(q_head, k_head) * scaling)
        scores = np.asarray(scores, dtype=np.float32)
        probs = np.exp(scores - np.max(scores))
        probs /= np.sum(probs)
        head_context = np.zeros(HEAD_DIM, dtype=np.float32)
        for token, prob in enumerate(probs):
            v_head = next_v_cache[token * KV_WIDTH + kv_head * HEAD_DIM : token * KV_WIDTH + (kv_head + 1) * HEAD_DIM]
            head_context += prob * v_head
        context[head * HEAD_DIM : (head + 1) * HEAD_DIM] = head_context

    output = o_matrix @ context
    return output.astype(np.float32), next_k_cache.astype(np.float32), next_v_cache.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression-test decode attention kernel on a non-zero history KV and multi-head synthetic case.")
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    (
        input_token,
        input_layernorm_weight,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        q_bias,
        k_bias,
        v_bias,
        q_scales,
        k_scales,
        v_scales,
        o_scales,
        k_cache,
        v_cache,
    ) = build_case()

    reference_output, reference_k_cache, reference_v_cache = reference_decode(
        input_token,
        input_layernorm_weight,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        q_bias,
        k_bias,
        v_bias,
        q_scales,
        k_scales,
        v_scales,
        o_scales,
        k_cache,
        v_cache,
        past_seq_len=2,
    )

    library = ctypes.CDLL(str(args.lib_path))
    func = library.qwen_decode_attention_smoke_forward
    float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    byte_ptr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    func.argtypes = [
        float_ptr,
        ctypes.c_int,
        float_ptr,
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
    ]
    func.restype = ctypes.c_int

    kernel_k_cache = k_cache.copy()
    kernel_v_cache = v_cache.copy()
    kernel_output = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    status = func(
        input_token,
        2,
        input_layernorm_weight,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        q_bias,
        k_bias,
        v_bias,
        q_scales,
        k_scales,
        v_scales,
        o_scales,
        kernel_k_cache,
        kernel_v_cache,
        kernel_output,
    )
    if status != 0:
        raise RuntimeError(f"qwen_decode_attention_smoke_forward failed with status {status}")

    if np.max(np.abs(kernel_output - reference_output)) > args.atol:
        raise AssertionError("Kernel output diverged from synthetic reference")
    if np.max(np.abs(kernel_k_cache - reference_k_cache)) > args.atol:
        raise AssertionError("Kernel K-cache diverged from synthetic reference")
    if np.max(np.abs(kernel_v_cache - reference_v_cache)) > args.atol:
        raise AssertionError("Kernel V-cache diverged from synthetic reference")
    print("Decode attention history regression PASS")


if __name__ == "__main__":
    main()