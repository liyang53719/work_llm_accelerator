#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


VERIFICATION_DIR = Path(__file__).resolve().parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from qwen_full_model_validation import run_validation
from reference_wrapper_backend import ReferenceWrapperBackend


DEFAULT_PROMPT = "Explain the purpose of blocked attention in one sentence."


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 28-layer reference-wrapper backend against local Qwen2.5-1.5B.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    backend = ReferenceWrapperBackend()
    result = run_validation(args.prompt, args.decode_steps, args.atol, backend, backend.reference_backend)
    if backend.last_layer_trace is not None:
        result["wrapper_trace"] = backend.last_layer_trace

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Reference-wrapper validation PASS")
        return

    print("Prefill logits diff:", result["prefill_logits_diff"])
    print("Prefill cache diff:", result["prefill_cache_diff"])
    for decode_result in result["decode_steps"]:
        print(f"Decode step {decode_result['step_index']} logits diff:", decode_result["logits_diff"])
        print(f"Decode step {decode_result['step_index']} cache diff:", decode_result["cache_diff"])
    if backend.last_layer_trace is not None:
        print("Wrapper trace:", backend.last_layer_trace)
    print("Reference-wrapper validation PASS")


if __name__ == "__main__":
    main()