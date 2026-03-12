#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from torch_reference_backend import TorchReferenceBackend


DEFAULT_CASE_PATH = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases" / "layer0_prefill_case.npz"


def module_linear(module: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    dtype = module.weight.dtype
    return module(input_tensor.to(dtype)).to(torch.float32)


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

    backend = TorchReferenceBackend(device="cpu")
    model = backend.model
    layer = model.model.layers[0]
    attn = layer.self_attn
    case = np.load(args.case_path)

    layer0_input = torch.from_numpy(case["layer0_input"]).to(torch.float32)
    seq_len = layer0_input.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        input_layernorm = layer.input_layernorm(layer0_input.to(layer.input_layernorm.weight.dtype)).to(torch.float32)

        q_proj = module_linear(layer.self_attn.q_proj, input_layernorm)
        k_proj = module_linear(layer.self_attn.k_proj, input_layernorm)
        v_proj = module_linear(layer.self_attn.v_proj, input_layernorm)

        batch_size = layer0_input.shape[0]
        q_states = q_proj.view(batch_size, seq_len, model.config.num_attention_heads, attn.head_dim).transpose(1, 2)
        k_states = k_proj.view(batch_size, seq_len, model.config.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v_states = v_proj.view(batch_size, seq_len, model.config.num_key_value_heads, attn.head_dim).transpose(1, 2)

        cos, sin = model.model.rotary_emb(v_states, position_ids)
        q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

        k_states = repeat_kv(k_states, attn.num_key_value_groups)
        v_states = repeat_kv(v_states, attn.num_key_value_groups)

        attn_scores = torch.matmul(q_states, k_states.transpose(-1, -2)) * attn.scaling
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        attn_scores = attn_scores + causal_mask.view(1, 1, seq_len, seq_len)
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_states.dtype)
        attn_context = torch.matmul(attn_probs, v_states)
        attn_context = attn_context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        o_proj = module_linear(layer.self_attn.o_proj, attn_context.to(torch.float32))
        attention_residual = layer0_input + o_proj

        post_attention_layernorm = layer.post_attention_layernorm(attention_residual.to(layer.post_attention_layernorm.weight.dtype)).to(torch.float32)
        gate_proj = module_linear(layer.mlp.gate_proj, post_attention_layernorm)
        up_proj = module_linear(layer.mlp.up_proj, post_attention_layernorm)
        silu_mul = layer.mlp.act_fn(gate_proj.to(layer.mlp.gate_proj.weight.dtype)).to(torch.float32) * up_proj
        down_proj = module_linear(layer.mlp.down_proj, silu_mul)
        layer0_output = attention_residual + down_proj

    reports = {
        "input_layernorm": diff_report(torch.from_numpy(case["input_layernorm"]), input_layernorm),
        "q_proj": diff_report(torch.from_numpy(case["q_proj"]), q_proj),
        "k_proj": diff_report(torch.from_numpy(case["k_proj"]), k_proj),
        "v_proj": diff_report(torch.from_numpy(case["v_proj"]), v_proj),
        "self_attn_output": diff_report(torch.from_numpy(case["self_attn_output"]), o_proj),
        "o_proj": diff_report(torch.from_numpy(case["o_proj"]), o_proj),
        "attention_residual": diff_report(torch.from_numpy(case["attention_residual"]), attention_residual),
        "post_attention_layernorm": diff_report(torch.from_numpy(case["post_attention_layernorm"]), post_attention_layernorm),
        "gate_proj": diff_report(torch.from_numpy(case["gate_proj"]), gate_proj),
        "up_proj": diff_report(torch.from_numpy(case["up_proj"]), up_proj),
        "silu_mul": diff_report(torch.from_numpy(case["silu_mul"]), silu_mul),
        "down_proj": diff_report(torch.from_numpy(case["down_proj"]), down_proj),
        "layer0_output": diff_report(torch.from_numpy(case["layer0_output"]), layer0_output),
    }

    worst = max(report["max_abs_diff"] for report in reports.values())
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    if worst > args.atol:
        raise AssertionError(f"Standalone layer0 prefill math path exceeded tolerance: {worst} > {args.atol}")
    print("Validation PASS")


if __name__ == "__main__":
    main()