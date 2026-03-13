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
ROPE_THETA = 1_000_000.0
RMS_NORM_EPS = 1.0e-6


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


def rmsnorm(input_token: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mean_square = np.mean(input_token.astype(np.float64) ** 2)
    inv_rms = np.float32(1.0 / np.sqrt(mean_square + RMS_NORM_EPS))
    return (input_token * inv_rms * weight).astype(np.float32)


def prefill_attention_reference(
    input_sequence: np.ndarray,
    input_layernorm_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return k_proj.reshape(-1), v_proj.reshape(-1), output.reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the prefill attention kernel through its C ABI wrapper.")
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--tile-m", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    input_sequence = np.zeros((args.seq_len, HIDDEN_SIZE), dtype=np.float32)
    input_sequence[0, 0] = 1.0
    if args.seq_len > 1:
        input_sequence[1, 0] = 2.0
    for token_index in range(2, args.seq_len):
        input_sequence[token_index, 0] = float(token_index + 1)

    input_layernorm_weight = np.ones(HIDDEN_SIZE, dtype=np.float32)
    q_weights = np.zeros(HIDDEN_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    k_weights = np.zeros(HIDDEN_SIZE * KV_WIDTH // 2, dtype=np.uint8)
    v_weights = np.zeros(HIDDEN_SIZE * KV_WIDTH // 2, dtype=np.uint8)
    o_weights = np.zeros(HIDDEN_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    set_packed_weight(q_weights, HIDDEN_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(k_weights, KV_WIDTH, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(v_weights, KV_WIDTH, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(o_weights, HIDDEN_SIZE, HIDDEN_SIZE, 0, 0, 1)

    q_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)
    k_scales = np.ones(KV_WIDTH, dtype=np.float32)
    v_scales = np.ones(KV_WIDTH, dtype=np.float32)
    o_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)
    k_cache = np.zeros(args.seq_len * KV_WIDTH, dtype=np.float32)
    v_cache = np.zeros(args.seq_len * KV_WIDTH, dtype=np.float32)
    output_sequence = np.zeros(args.seq_len * HIDDEN_SIZE, dtype=np.float32)

    library = ctypes.CDLL(str(args.lib_path))
    func = library.qwen_prefill_attention_smoke_forward
    float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    byte_ptr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    func.argtypes = [
        float_ptr,
        ctypes.c_int,
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
    ]
    func.restype = ctypes.c_int

    status = func(
        np.ascontiguousarray(input_sequence.reshape(-1)),
        args.seq_len,
        args.tile_m,
        input_layernorm_weight,
        q_weights,
        k_weights,
        v_weights,
        o_weights,
        q_scales,
        k_scales,
        v_scales,
        o_scales,
        k_cache,
        v_cache,
        output_sequence,
    )
    if status != 0:
        raise RuntimeError(f"qwen_prefill_attention_smoke_forward failed with status {status}")

    expected_k, expected_v, expected_output = prefill_attention_reference(input_sequence, input_layernorm_weight)
    if not np.allclose(k_cache, expected_k, atol=args.atol):
        raise AssertionError(f"Unexpected K cache values: max diff {np.max(np.abs(k_cache - expected_k))}")
    if not np.allclose(v_cache, expected_v, atol=args.atol):
        raise AssertionError(f"Unexpected V cache values: max diff {np.max(np.abs(v_cache - expected_v))}")
    if not np.allclose(output_sequence, expected_output, atol=args.atol):
        raise AssertionError(f"Unexpected output values: max diff {np.max(np.abs(output_sequence - expected_output))}")
    print("Prefill attention smoke PASS")


if __name__ == "__main__":
    main()
