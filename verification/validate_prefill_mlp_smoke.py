#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB_PATH = ROOT / "tmp" / "host_libs" / "libqwen_prefill_stub.so"
HIDDEN_SIZE = 1536
INTERMEDIATE_SIZE = 8960
RMS_NORM_EPS = 1.0e-6
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


def silu(value: float) -> float:
    return value / (1.0 + np.exp(-value))


def rmsnorm(token: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mean_square = np.mean(token.astype(np.float64) ** 2)
    inv_rms = np.float32(1.0 / np.sqrt(mean_square + RMS_NORM_EPS))
    return (token * inv_rms * weight).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the tiled prefill MLP kernel through its C ABI wrapper.")
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--seq-len", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    attention_residual = np.zeros((args.seq_len, HIDDEN_SIZE), dtype=np.float32)
    attention_residual[0, 0] = 1.0
    attention_residual[0, 1] = -0.5
    if args.seq_len > 1:
        attention_residual[1, 0] = -0.25
        attention_residual[1, 1] = 0.75
    for token_index in range(2, args.seq_len):
        attention_residual[token_index, 0] = 0.1 * (token_index + 1)
        attention_residual[token_index, 1] = -0.2 * (token_index + 1)

    post_attention_layernorm_weight = np.ones(HIDDEN_SIZE, dtype=np.float32)
    gate_weights = np.zeros(INTERMEDIATE_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    up_weights = np.zeros(INTERMEDIATE_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    down_weights = np.zeros(HIDDEN_SIZE * INTERMEDIATE_SIZE // 2, dtype=np.uint8)
    set_packed_weight(gate_weights, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(up_weights, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 1, 1)
    set_packed_weight(down_weights, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0, 0, 1)

    gate_scales = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
    up_scales = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
    down_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)
    output_sequence = np.zeros(args.seq_len * HIDDEN_SIZE, dtype=np.float32)

    library = ctypes.CDLL(str(args.lib_path))
    func = library.qwen_prefill_mlp_smoke_forward
    float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    byte_ptr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    func.argtypes = [
        float_ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        float_ptr,
        byte_ptr,
        byte_ptr,
        byte_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
    ]
    func.restype = ctypes.c_int

    status = func(
        np.ascontiguousarray(attention_residual.reshape(-1)),
        args.seq_len,
        MLP_SEQ_TILE,
        MLP_HIDDEN_TILE,
        MLP_FF_TILE,
        post_attention_layernorm_weight,
        gate_weights,
        up_weights,
        down_weights,
        gate_scales,
        up_scales,
        down_scales,
        output_sequence,
    )
    if status != 0:
        raise RuntimeError(f"qwen_prefill_mlp_smoke_forward failed with status {status}")

    expected = np.zeros_like(attention_residual)
    for token_index in range(args.seq_len):
        norm = rmsnorm(attention_residual[token_index], post_attention_layernorm_weight)
        expected_gate = norm[0]
        expected_up = norm[1]
        expected[token_index] = attention_residual[token_index]
        expected[token_index, 0] += np.float32(silu(expected_gate) * expected_up)

    if not np.allclose(output_sequence.reshape(args.seq_len, HIDDEN_SIZE), expected, atol=args.atol):
        raise AssertionError(
            f"Unexpected prefill MLP output: max diff {np.max(np.abs(output_sequence.reshape(args.seq_len, HIDDEN_SIZE) - expected))}"
        )
    print("Prefill MLP smoke PASS")


if __name__ == "__main__":
    main()
