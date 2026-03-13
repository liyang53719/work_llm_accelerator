#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


VERIFICATION_DIR = Path(__file__).resolve().parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from qwen_full_model_validation import cache_diff, tensor_diff
from real_host_top_backend import RealHostTopBackend
from torch_reference_backend import snapshot_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the quantized real host top-wrapper backend across all Qwen2.5-1.5B layers.")
    parser.add_argument("--prompt", type=str, default="Summarize why blocked attention helps prefill throughput.")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--logits-atol", type=float, default=32.0)
    parser.add_argument("--cache-atol", type=float, default=320.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    backend = RealHostTopBackend()
    reference_backend = backend.reference_backend
    input_ids = reference_backend.tokenizer(args.prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        baseline_prefill = reference_backend.model(
            input_ids=input_ids.to(reference_backend.device),
            use_cache=True,
            return_dict=True,
        )
    backend_prefill = backend.prefill(input_ids)

    result = {
        "prompt": args.prompt,
        "attention_backend": {
            "reference": reference_backend.attention_backend,
            "backend": backend.attention_backend,
        },
        "backend_metadata": backend.backend_metadata,
        "prefill_logits_diff": tensor_diff(
            baseline_prefill.logits.detach().cpu().to(torch.float32),
            backend_prefill.logits,
        ),
        "prefill_cache_diff": cache_diff(
            snapshot_cache(baseline_prefill.past_key_values),
            snapshot_cache(backend_prefill.cache),
        ),
        "decode_steps": [],
    }

    if result["prefill_logits_diff"]["max_abs_diff"] > args.logits_atol:
        raise AssertionError(f"Prefill logits diff exceeded tolerance: {result['prefill_logits_diff']['max_abs_diff']} > {args.logits_atol}")
    if result["prefill_cache_diff"]["max_abs_diff"] > args.cache_atol:
        raise AssertionError(f"Prefill cache diff exceeded tolerance: {result['prefill_cache_diff']['max_abs_diff']} > {args.cache_atol}")

    baseline_cache = baseline_prefill.past_key_values
    backend_cache = backend_prefill.cache
    next_token = torch.argmax(backend_prefill.logits[:, -1, :], dim=-1, keepdim=True)

    for step_index in range(args.decode_steps):
        with torch.no_grad():
            baseline_decode = reference_backend.model(
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
        decode_cache_diff = cache_diff(
            snapshot_cache(baseline_decode.past_key_values),
            snapshot_cache(backend_decode.cache),
        )
        result["decode_steps"].append(
            {
                "step_index": step_index,
                "logits_diff": decode_logits_diff,
                "cache_diff": decode_cache_diff,
            }
        )

        if decode_logits_diff["max_abs_diff"] > args.logits_atol:
            raise AssertionError(f"Decode logits diff exceeded tolerance at step {step_index}: {decode_logits_diff['max_abs_diff']} > {args.logits_atol}")
        if decode_cache_diff["max_abs_diff"] > args.cache_atol:
            raise AssertionError(f"Decode cache diff exceeded tolerance at step {step_index}: {decode_cache_diff['max_abs_diff']} > {args.cache_atol}")

        baseline_cache = baseline_decode.past_key_values
        backend_cache = backend_decode.cache
        next_token = torch.argmax(backend_decode.logits[:, -1, :], dim=-1, keepdim=True)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Attention backend:", result["attention_backend"])
        print("Backend metadata:", result.get("backend_metadata", {}))
        print("Prefill logits diff:", result["prefill_logits_diff"])
        print("Prefill cache diff:", result["prefill_cache_diff"])
        for decode_result in result["decode_steps"]:
            print(f"Decode step {decode_result['step_index']} logits diff:", decode_result["logits_diff"])
            print(f"Decode step {decode_result['step_index']} cache diff:", decode_result["cache_diff"])
    trace = backend.last_layer_trace
    if trace is not None:
        print("Layer trace:", trace)
    print("Real host top-wrapper validation PASS")


if __name__ == "__main__":
    main()