#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB_PATH = ROOT / "tmp" / "host_libs" / "libqwen_decode_stub.so"

HIDDEN_SIZE = 1536
KV_WIDTH = 256


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the tiled decode attention kernel through its C ABI wrapper.")
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    input_token = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    input_token[0] = 1.0
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
    k_cache = np.zeros(KV_WIDTH, dtype=np.float32)
    v_cache = np.zeros(KV_WIDTH, dtype=np.float32)
    output_token = np.zeros(HIDDEN_SIZE, dtype=np.float32)

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
    ]
    func.restype = ctypes.c_int

    status = func(
        input_token,
        0,
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
        output_token,
    )
    if status != 0:
      raise RuntimeError(f"qwen_decode_attention_smoke_forward failed with status {status}")

    expected_norm = 1.0 / np.sqrt((1.0 / HIDDEN_SIZE) + 1.0e-6)
    if not np.isclose(k_cache[0], expected_norm, atol=args.atol):
        raise AssertionError(f"Unexpected K cache value {k_cache[0]} vs {expected_norm}")
    if not np.isclose(v_cache[0], expected_norm, atol=args.atol):
        raise AssertionError(f"Unexpected V cache value {v_cache[0]} vs {expected_norm}")
    if not np.isclose(output_token[0], expected_norm, atol=args.atol):
        raise AssertionError(f"Unexpected output value {output_token[0]} vs {expected_norm}")
    if np.count_nonzero(output_token[1:]) != 0:
        raise AssertionError("Unexpected non-zero output dimensions in smoke test")
    print("Decode attention smoke PASS")


if __name__ == "__main__":
    main()