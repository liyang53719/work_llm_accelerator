#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


DEFAULT_CASE_PATH = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases" / "layer0_decode_case.npz"
DEFAULT_PARAM_PATH = Path(__file__).resolve().parents[1] / "tmp" / "reference_params" / "layer0_prefill_params.npz"
DEFAULT_LIB_PATH = Path(__file__).resolve().parents[1] / "tmp" / "host_libs" / "libqwen_decode_stub.so"


def diff_report(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    delta = np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))
    return {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the C++ layer0 decode-step reference wrapper against exported tensors.")
    parser.add_argument("--case-path", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--param-path", type=Path, default=DEFAULT_PARAM_PATH)
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--atol", type=float, default=8e-2)
    args = parser.parse_args()

    case = np.load(args.case_path)
    params = np.load(args.param_path)

    input_token = np.ascontiguousarray(case["layer0_input"].reshape(-1).astype(np.float32))
    reference_output = np.ascontiguousarray(case["layer0_output"].reshape(-1).astype(np.float32))
    past_k_cache = np.ascontiguousarray(case["prefill_layer0_k_cache"].reshape(-1).astype(np.float32))
    past_v_cache = np.ascontiguousarray(case["prefill_layer0_v_cache"].reshape(-1).astype(np.float32))
    reference_k_cache = np.ascontiguousarray(case["decode_layer0_k_cache"].reshape(-1).astype(np.float32))
    reference_v_cache = np.ascontiguousarray(case["decode_layer0_v_cache"].reshape(-1).astype(np.float32))
    past_seq_len = int(case["prefill_layer0_k_cache"].shape[2])

    output_token = np.zeros_like(reference_output)
    next_k_cache = np.zeros_like(reference_k_cache)
    next_v_cache = np.zeros_like(reference_v_cache)

    library = ctypes.CDLL(str(args.lib_path))
    func = library.qwen_decode_layer0_reference_forward
    float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    func.argtypes = [
        float_ptr,
        ctypes.c_int,
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
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        float_ptr,
        ctypes.c_float,
        float_ptr,
        float_ptr,
        float_ptr,
    ]
    func.restype = ctypes.c_int

    status = func(
        input_token,
        past_seq_len,
        past_k_cache,
        past_v_cache,
        np.ascontiguousarray(params["input_layernorm_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["q_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["q_bias"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["k_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["k_bias"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["v_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["v_bias"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["o_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["o_bias"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["post_attention_layernorm_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["gate_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["gate_bias"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["up_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["up_bias"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["down_weight"].reshape(-1).astype(np.float32)),
        np.ascontiguousarray(params["down_bias"].reshape(-1).astype(np.float32)),
        float(params["rms_eps"][0]),
        output_token,
        next_k_cache,
        next_v_cache,
    )
    if status != 0:
        raise RuntimeError(f"qwen_decode_layer0_reference_forward failed with status {status}")

    output_report = diff_report(output_token, reference_output)
    k_cache_report = diff_report(next_k_cache, reference_k_cache)
    v_cache_report = diff_report(next_v_cache, reference_v_cache)
    print("Reference decode wrapper output vs layer0 reference:", output_report)
    print("Reference decode wrapper K-cache vs reference:", k_cache_report)
    print("Reference decode wrapper V-cache vs reference:", v_cache_report)
    worst = max(output_report["max_abs_diff"], k_cache_report["max_abs_diff"], v_cache_report["max_abs_diff"])
    if worst > args.atol:
        raise AssertionError(f"Decode reference wrapper exceeded tolerance: {worst} > {args.atol}")
    print("Validation PASS")


if __name__ == "__main__":
    main()