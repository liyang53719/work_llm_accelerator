#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import sys

import torch


VERIFICATION_DIR = Path(__file__).resolve().parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from hls_backend_stub import HlsBackendStub
from descriptor_dispatch_backend import DescriptorDispatchBackend
from manual_dispatch_backend import ManualDispatchBackend
from torch_reference_backend import TorchReferenceBackend, snapshot_cache


def tensor_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, float]:
    delta = (lhs.to(torch.float32) - rhs.to(torch.float32)).abs()
    return {
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
    }


def cache_diff(lhs: Any, rhs: Any) -> dict[str, float]:
    max_abs_diff = 0.0
    mean_abs_accum = 0.0
    tensor_count = 0
    for lhs_layer, rhs_layer in zip(lhs, rhs):
        for lhs_tensor, rhs_tensor in zip(lhs_layer, rhs_layer):
            diff = tensor_diff(lhs_tensor, rhs_tensor)
            max_abs_diff = max(max_abs_diff, diff["max_abs_diff"])
            mean_abs_accum += diff["mean_abs_diff"]
            tensor_count += 1
    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_accum / tensor_count if tensor_count else 0.0,
    }


def build_backend(name: str):
    if name == "torch":
        return TorchReferenceBackend(device="cpu")
    if name == "descriptor-dispatch":
        return DescriptorDispatchBackend()
    if name == "manual-dispatch":
        return ManualDispatchBackend()
    if name == "hls-stub":
        return HlsBackendStub()
    raise ValueError(f"Unsupported backend: {name}")


def run_validation(
    prompt: str,
    decode_steps: int,
    atol: float,
    backend: Any,
    reference_backend: TorchReferenceBackend,
) -> dict[str, Any]:
    tokenizer = reference_backend.tokenizer
    baseline_model = reference_backend.model
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        baseline_prefill = baseline_model(
            input_ids=input_ids.to(reference_backend.device),
            use_cache=True,
            return_dict=True,
        )
    backend_prefill = backend.prefill(input_ids)

    prefill_logits_diff = tensor_diff(
        baseline_prefill.logits.detach().cpu().to(torch.float32),
        backend_prefill.logits,
    )
    prefill_cache_diff = cache_diff(snapshot_cache(baseline_prefill.past_key_values), snapshot_cache(backend_prefill.cache))

    result = {
        "prompt": prompt,
        "prefill_logits_diff": prefill_logits_diff,
        "prefill_cache_diff": prefill_cache_diff,
        "decode_steps": [],
    }

    if prefill_logits_diff["max_abs_diff"] > atol or prefill_cache_diff["max_abs_diff"] > atol:
        raise AssertionError("Prefill validation failed tolerance check.")

    baseline_cache = baseline_prefill.past_key_values
    backend_cache = backend_prefill.cache
    next_token = torch.argmax(backend_prefill.logits[:, -1, :], dim=-1, keepdim=True)

    for step_index in range(decode_steps):
        with torch.no_grad():
            baseline_decode = baseline_model(
                input_ids=next_token.to(reference_backend.device),
                past_key_values=baseline_cache,
                use_cache=True,
                return_dict=True,
            )
        backend_decode = backend.decode_step(next_token, backend_cache)

        decode_logits_diff = tensor_diff(
            baseline_decode.logits.detach().cpu().to(torch.float32),
            backend_decode.logits,
        )
        decode_cache_diff = cache_diff(snapshot_cache(baseline_decode.past_key_values), snapshot_cache(backend_decode.cache))

        result["decode_steps"].append(
            {
                "step_index": step_index,
                "logits_diff": decode_logits_diff,
                "cache_diff": decode_cache_diff,
            }
        )

        if decode_logits_diff["max_abs_diff"] > atol or decode_cache_diff["max_abs_diff"] > atol:
            raise AssertionError(f"Decode validation failed at step {step_index}.")

        baseline_cache = baseline_decode.past_key_values
        backend_cache = backend_decode.cache
        next_token = torch.argmax(backend_decode.logits[:, -1, :], dim=-1, keepdim=True)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full-model prefill + decode against local Qwen2.5-1.5B.")
    parser.add_argument("--backend", choices=["torch", "descriptor-dispatch", "manual-dispatch", "hls-stub"], default="torch")
    parser.add_argument("--prompt", type=str, default="Explain the purpose of blocked attention in one sentence.")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    backend = build_backend(args.backend)
    reference_backend = backend if isinstance(backend, TorchReferenceBackend) else TorchReferenceBackend(device="cpu")
    if isinstance(backend, DescriptorDispatchBackend):
        reference_backend = backend.reference_backend
    if isinstance(backend, ManualDispatchBackend):
        reference_backend = backend.reference_backend
    if isinstance(backend, HlsBackendStub):
        raise NotImplementedError("HLS backend is not wired yet. Use --backend torch to validate the framework.")
    result = run_validation(args.prompt, args.decode_steps, args.atol, backend, reference_backend)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Validation PASS")
        return

    print("Prefill logits diff:", result["prefill_logits_diff"])
    print("Prefill cache diff:", result["prefill_cache_diff"])
    for decode_result in result["decode_steps"]:
        print(f"Decode step {decode_result['step_index']} logits diff:", decode_result["logits_diff"])
        print(f"Decode step {decode_result['step_index']} cache diff:", decode_result["cache_diff"])
    print("Validation PASS")


if __name__ == "__main__":
    main()