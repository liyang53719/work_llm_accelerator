#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from layer0_prefill_reference_backend import Layer0PrefillReferenceBackend


DEFAULT_CASE_PATH = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases" / "layer0_prefill_case.npz"


def diff_report(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    delta = (reference.to(torch.float32) - actual.to(torch.float32)).abs()
    return {
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a standalone layer0 prefill math path against exported Qwen2.5-1.5B tensors.")
    parser.add_argument("--case-path", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--atol", type=float, default=5e-2)
    args = parser.parse_args()

    backend = Layer0PrefillReferenceBackend()
    case = np.load(args.case_path)

    layer0_input = torch.from_numpy(case["layer0_input"]).to(torch.float32)
    outputs = backend.run(layer0_input)

    reports = {
        "input_layernorm": diff_report(torch.from_numpy(case["input_layernorm"]), outputs.input_layernorm),
        "q_proj": diff_report(torch.from_numpy(case["q_proj"]), outputs.q_proj),
        "k_proj": diff_report(torch.from_numpy(case["k_proj"]), outputs.k_proj),
        "v_proj": diff_report(torch.from_numpy(case["v_proj"]), outputs.v_proj),
        "self_attn_output": diff_report(torch.from_numpy(case["self_attn_output"]), outputs.self_attn_output),
        "o_proj": diff_report(torch.from_numpy(case["o_proj"]), outputs.o_proj),
        "attention_residual": diff_report(torch.from_numpy(case["attention_residual"]), outputs.attention_residual),
        "post_attention_layernorm": diff_report(torch.from_numpy(case["post_attention_layernorm"]), outputs.post_attention_layernorm),
        "gate_proj": diff_report(torch.from_numpy(case["gate_proj"]), outputs.gate_proj),
        "up_proj": diff_report(torch.from_numpy(case["up_proj"]), outputs.up_proj),
        "silu_mul": diff_report(torch.from_numpy(case["silu_mul"]), outputs.silu_mul),
        "down_proj": diff_report(torch.from_numpy(case["down_proj"]), outputs.down_proj),
        "layer0_output": diff_report(torch.from_numpy(case["layer0_output"]), outputs.layer0_output),
    }

    worst = max(report["max_abs_diff"] for report in reports.values())
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    if worst > args.atol:
        raise AssertionError(f"Standalone layer0 prefill math path exceeded tolerance: {worst} > {args.atol}")
    print("Validation PASS")


if __name__ == "__main__":
    main()