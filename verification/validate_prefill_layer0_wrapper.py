#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys

import numpy as np
import torch

VERIFICATION_DIR = Path(__file__).resolve().parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from layer0_prefill_reference_backend import Layer0PrefillReferenceBackend


DEFAULT_CASE_PATH = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases" / "layer0_prefill_case.npz"
DEFAULT_LIB_PATH = Path(__file__).resolve().parents[1] / "tmp" / "host_libs" / "libqwen_prefill_stub.so"
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


def diff_report(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    delta = np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))
    return {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prefill wrapper ABI against exported layer0 reference case.")
    parser.add_argument("--case-path", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--lib-path", type=Path, default=DEFAULT_LIB_PATH)
    parser.add_argument("--expect-identity", action="store_true")
    parser.add_argument("--compare-reference-backend", action="store_true")
    args = parser.parse_args()

    case = np.load(args.case_path)
    input_sequence = np.ascontiguousarray(case["embedding_output"].reshape(-1).astype(np.float32))
    layer0_output = np.ascontiguousarray(case["layer0_output"].reshape(-1).astype(np.float32))
    seq_len = int(case["embedding_output"].shape[1])
    output_sequence = np.zeros_like(input_sequence)

    library = ctypes.CDLL(str(args.lib_path))
    func = library.qwen_prefill_stub_forward
    float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
    func.argtypes = [
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
    ]
    func.restype = ctypes.c_int

    status = func(
        input_sequence,
        seq_len,
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
        output_sequence,
    )
    if status != 0:
        raise RuntimeError(f"qwen_prefill_stub_forward failed with status {status}")

    identity_diff = diff_report(output_sequence, input_sequence)
    layer0_diff = diff_report(output_sequence, layer0_output)

    print("Wrapper output vs input:", identity_diff)
    print("Wrapper output vs layer0 reference:", layer0_diff)

    if args.compare_reference_backend:
        backend = Layer0PrefillReferenceBackend()
        backend_output = backend.run(torch.from_numpy(case["layer0_input"]).to(torch.float32)).layer0_output.numpy().reshape(-1)
        backend_diff = diff_report(output_sequence, backend_output)
        print("Wrapper output vs layer0 reference backend:", backend_diff)

    if args.expect_identity and identity_diff["max_abs_diff"] != 0.0:
        raise AssertionError("Current prefill wrapper is expected to behave as identity stub, but output drifted.")


if __name__ == "__main__":
    main()