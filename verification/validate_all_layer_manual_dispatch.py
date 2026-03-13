#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_DIR = ROOT / "verification"
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from manual_dispatch_backend import ManualDispatchBackend
from qwen_full_model_validation import run_validation
from torch_reference_backend import TorchReferenceBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate explicit 28-layer manual dispatch against native Qwen2.5-1.5B forward.")
    parser.add_argument("--prompt", type=str, default="Explain the purpose of blocked attention in one sentence.")
    parser.add_argument("--decode-steps", type=int, default=1)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    reference_backend = TorchReferenceBackend(device="cpu")
    backend = ManualDispatchBackend(reference_backend=reference_backend)
    result = run_validation(args.prompt, args.decode_steps, args.atol, backend, reference_backend)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("Manual dispatch validation PASS")
        return

    print("Attention backend:", result["attention_backend"])
    print("Prefill logits diff:", result["prefill_logits_diff"])
    print("Prefill cache diff:", result["prefill_cache_diff"])
    for decode_result in result["decode_steps"]:
        print(f"Decode step {decode_result['step_index']} logits diff:", decode_result["logits_diff"])
        print(f"Decode step {decode_result['step_index']} cache diff:", decode_result["cache_diff"])
    print("Manual dispatch validation PASS")


if __name__ == "__main__":
    main()