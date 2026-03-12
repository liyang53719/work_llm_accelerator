#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB_PATH = ROOT / "tmp" / "host_libs" / "libqwen_decode_stub.so"
HIDDEN_SIZE = 1536
INTERMEDIATE_SIZE = 8960


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the tiled decode MLP kernel through its C ABI wrapper.")
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    attention_residual = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    attention_residual[0] = 1.0
    attention_residual[1] = -0.5
    post_attention_layernorm_weight = np.ones(HIDDEN_SIZE, dtype=np.float32)

    gate_weights = np.zeros(INTERMEDIATE_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    up_weights = np.zeros(INTERMEDIATE_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    down_weights = np.zeros(INTERMEDIATE_SIZE * HIDDEN_SIZE // 2, dtype=np.uint8)
    set_packed_weight(gate_weights, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 0, 1)
    set_packed_weight(up_weights, INTERMEDIATE_SIZE, HIDDEN_SIZE, 0, 1, 1)
    set_packed_weight(down_weights, HIDDEN_SIZE, INTERMEDIATE_SIZE, 0, 0, 1)

    gate_scales = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
    up_scales = np.ones(INTERMEDIATE_SIZE, dtype=np.float32)
    down_scales = np.ones(HIDDEN_SIZE, dtype=np.float32)
    output_token = np.zeros(HIDDEN_SIZE, dtype=np.float32)

    library = ctypes.CDLL(str(args.lib_path))
    func = library.qwen_decode_mlp_smoke_forward
    float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    byte_ptr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    func.argtypes = [float_ptr, float_ptr, byte_ptr, byte_ptr, byte_ptr, float_ptr, float_ptr, float_ptr, float_ptr]
    func.restype = ctypes.c_int

    status = func(
        attention_residual,
        post_attention_layernorm_weight,
        gate_weights,
        up_weights,
        down_weights,
        gate_scales,
        up_scales,
        down_scales,
        output_token,
    )
    if status != 0:
        raise RuntimeError(f"qwen_decode_mlp_smoke_forward failed with status {status}")

    expected_norm = attention_residual / np.sqrt(np.mean(attention_residual * attention_residual) + 1.0e-6)
    expected_gate = expected_norm[0]
    expected_up = expected_norm[1]
    expected_output0 = attention_residual[0] + silu(expected_gate) * expected_up
    if not np.isclose(output_token[0], expected_output0, atol=args.atol):
        raise AssertionError(f"Unexpected output value {output_token[0]} vs {expected_output0}")
    if not np.isclose(output_token[1], attention_residual[1], atol=args.atol):
        raise AssertionError("Unexpected residual corruption in untouched dimension")
    print("Decode MLP smoke PASS")


if __name__ == "__main__":
    main()