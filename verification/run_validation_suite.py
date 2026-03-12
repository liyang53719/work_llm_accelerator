#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


VERIFICATION_DIR = Path(__file__).resolve().parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from hls_backend_stub import HlsBackendStub
from descriptor_dispatch_backend import DescriptorDispatchBackend
from manual_dispatch_backend import ManualDispatchBackend
from qwen_full_model_validation import build_backend, run_validation
from reference_wrapper_backend import ReferenceWrapperBackend
from torch_reference_backend import TorchReferenceBackend


DEFAULT_PROMPTS = Path(__file__).resolve().parent / "validation_prompts.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-prompt Qwen2.5-1.5B validation suite.")
    parser.add_argument("--backend", choices=["torch", "descriptor-dispatch", "manual-dispatch", "reference-wrapper", "hls-stub"], default="torch")
    parser.add_argument("--prompts-file", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--decode-steps", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    with args.prompts_file.open("r", encoding="utf-8") as file_obj:
        prompts = json.load(file_obj)

    backend = build_backend(args.backend)
    reference_backend = backend if isinstance(backend, TorchReferenceBackend) else TorchReferenceBackend(device="cpu")
    if isinstance(backend, DescriptorDispatchBackend):
        reference_backend = backend.reference_backend
    if isinstance(backend, ManualDispatchBackend):
        reference_backend = backend.reference_backend
    if isinstance(backend, ReferenceWrapperBackend):
        reference_backend = backend.reference_backend
    if isinstance(backend, HlsBackendStub):
        raise NotImplementedError("HLS backend is not wired yet. Use --backend torch to validate the framework.")

    summary = []
    for prompt in prompts:
        result = run_validation(prompt, args.decode_steps, args.atol, backend, reference_backend)
        summary.append(result)
        print(f"PASS prompt: {prompt}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()