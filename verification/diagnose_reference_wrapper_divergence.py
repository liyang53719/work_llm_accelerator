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

from reference_wrapper_backend import ReferenceWrapperBackend


DEFAULT_PROMPT = "Explain the purpose of blocked attention in one sentence."


def summarize_first_large_diff(layer_reports: list[dict], threshold: float) -> dict | None:
    for report in layer_reports:
        worst = max(
            report["output_diff"]["max_abs_diff"],
            report["k_cache_diff"]["max_abs_diff"],
            report["v_cache_diff"]["max_abs_diff"],
        )
        if worst > threshold:
            return {
                "layer_id": report["layer_id"],
                "worst_max_abs_diff": worst,
                "output_diff": report["output_diff"],
                "k_cache_diff": report["k_cache_diff"],
                "v_cache_diff": report["v_cache_diff"],
            }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose the first divergent layer in the all-layer reference-wrapper backend.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--threshold", type=float, default=1e-2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    backend = ReferenceWrapperBackend()
    tokenizer = backend.reference_backend.tokenizer
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    reference_prefill = backend.reference_backend.prefill(input_ids)
    prefill_report = backend.diagnose_prefill(input_ids)
    prefill_result = backend.prefill(input_ids)
    next_token = torch.argmax(prefill_result.logits[:, -1, :], dim=-1, keepdim=True)
    decode_report = backend.diagnose_decode(next_token, prefill_result.cache, reference_prefill.cache)

    payload = {
        "prefill_first_large_diff": summarize_first_large_diff(prefill_report["layer_reports"], args.threshold),
        "decode_first_large_diff": summarize_first_large_diff(decode_report["layer_reports"], args.threshold),
        "prefill_last_layer": prefill_report["layer_reports"][-1],
        "decode_last_layer": decode_report["layer_reports"][-1],
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print("Prefill first large diff:", payload["prefill_first_large_diff"])
    print("Decode first large diff:", payload["decode_first_large_diff"])
    print("Prefill last layer:", payload["prefill_last_layer"])
    print("Decode last layer:", payload["decode_last_layer"])


if __name__ == "__main__":
    main()