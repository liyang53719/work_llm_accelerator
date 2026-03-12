#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np


DEFAULT_CASE_PATH = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases" / "layer0_prefill_case.npz"
DEFAULT_PARAM_PATH = Path(__file__).resolve().parents[1] / "tmp" / "reference_params" / "layer0_prefill_params.npz"
DEFAULT_LIB_PATH = Path(__file__).resolve().parents[1] / "tmp" / "host_libs" / "libqwen_prefill_stub.so"


def diff_report(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    delta = np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))
    return {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the C++ layer0 prefill reference wrapper against exported tensors.")
    parser.add_argument("--case-path", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--param-path", type=Path, default=DEFAULT_PARAM_PATH)
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--atol", type=float, default=8e-2)
    args = parser.parse_args()

    case = np.load(args.case_path)
    params = np.load(args.param_path)
    input_sequence = np.ascontiguousarray(case["layer0_input"].reshape(-1).astype(np.float32))
    reference_output = np.ascontiguousarray(case["layer0_output"].reshape(-1).astype(np.float32))
    seq_len = int(case["layer0_input"].shape[1])
    output_sequence = np.zeros_like(reference_output)

    library = ctypes.CDLL(str(args.lib_path))
    func = library.qwen_prefill_layer0_reference_forward
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
        ctypes.c_float,
        float_ptr,
    ]
    func.restype = ctypes.c_int

    status = func(
        input_sequence,
        seq_len,
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
        output_sequence,
    )
    if status != 0:
        raise RuntimeError(f"qwen_prefill_layer0_reference_forward failed with status {status}")

    report = diff_report(output_sequence, reference_output)
    print("Reference wrapper output vs layer0 reference:", report)
    if report["max_abs_diff"] > args.atol:
        raise AssertionError(f"Reference wrapper exceeded tolerance: {report['max_abs_diff']} > {args.atol}")
    print("Validation PASS")


if __name__ == "__main__":
    main()