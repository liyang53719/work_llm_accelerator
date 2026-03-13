#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

VERIFICATION_DIR = Path(__file__).resolve().parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from diagnose_layer0_prefill_wrapper_stages import (  # noqa: E402
    DEFAULT_CASE_PATH,
    diff_report,
    linear_row_major,
    rmsnorm_token,
    run_wrapper_emulation,
    silu,
)
from layer0_prefill_reference_backend import Layer0PrefillReferenceBackend  # noqa: E402
from real_host_top_backend import quantize_int4_per_channel  # noqa: E402


def unpack_int4_per_channel(packed: np.ndarray, scales: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    flat = packed.astype(np.uint8, copy=False).reshape(-1)
    unpacked = np.empty(out_dim * in_dim, dtype=np.float32)
    for index in range(unpacked.size):
        packed_byte = int(flat[index // 2])
        nibble = ((packed_byte >> 4) & 0xF) if (index & 1) else (packed_byte & 0xF)
        unpacked[index] = float(nibble - 16 if nibble >= 8 else nibble)
    return unpacked.reshape(out_dim, in_dim) * scales[:, None]


def quantized_weight(weight: torch.Tensor, out_dim: int, in_dim: int) -> np.ndarray:
    packed, scales = quantize_int4_per_channel(weight, out_dim, in_dim)
    return unpack_int4_per_channel(packed, scales, out_dim, in_dim)


def optional_bias(module: torch.nn.Module) -> np.ndarray | None:
    if module.bias is None:
        return None
    return module.bias.detach().cpu().to(torch.float32).numpy()


def run_mlp_only_quant_emulation(reference: Layer0PrefillReferenceBackend, float_outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    layer = reference.layer
    hidden_size = reference.model.config.hidden_size
    intermediate_size = reference.model.config.intermediate_size
    rms_eps = float(reference.model.config.rms_norm_eps)

    post_ln_weight = layer.post_attention_layernorm.weight.detach().cpu().to(torch.float32).numpy()
    gate_weight = quantized_weight(layer.mlp.gate_proj.weight, intermediate_size, hidden_size)
    up_weight = quantized_weight(layer.mlp.up_proj.weight, intermediate_size, hidden_size)
    down_weight = quantized_weight(layer.mlp.down_proj.weight, hidden_size, intermediate_size)
    gate_bias = optional_bias(layer.mlp.gate_proj)
    up_bias = optional_bias(layer.mlp.up_proj)
    down_bias = optional_bias(layer.mlp.down_proj)

    seq_len = float_outputs["attention_residual"].shape[0]
    post_attention_layernorm = np.zeros((seq_len, hidden_size), dtype=np.float32)
    gate_proj = np.zeros((seq_len, intermediate_size), dtype=np.float32)
    up_proj = np.zeros((seq_len, intermediate_size), dtype=np.float32)
    silu_mul = np.zeros((seq_len, intermediate_size), dtype=np.float32)
    down_proj = np.zeros((seq_len, hidden_size), dtype=np.float32)
    layer0_output = np.zeros((seq_len, hidden_size), dtype=np.float32)

    for token in range(seq_len):
        post_attention_layernorm[token] = rmsnorm_token(float_outputs["attention_residual"][token], post_ln_weight, rms_eps)
        gate_proj[token] = linear_row_major(post_attention_layernorm[token], gate_weight, gate_bias)
        up_proj[token] = linear_row_major(post_attention_layernorm[token], up_weight, up_bias)
        silu_mul[token] = silu(gate_proj[token]) * up_proj[token]
        down_proj[token] = linear_row_major(silu_mul[token], down_weight, down_bias)
        layer0_output[token] = float_outputs["attention_residual"][token] + down_proj[token]

    outputs = dict(float_outputs)
    outputs.update(
        {
            "post_attention_layernorm": post_attention_layernorm,
            "gate_proj": gate_proj,
            "up_proj": up_proj,
            "silu_mul": silu_mul,
            "down_proj": down_proj,
            "layer0_output": layer0_output,
        }
    )
    return outputs


def run_attention_only_quant_emulation(reference: Layer0PrefillReferenceBackend, layer0_input: torch.Tensor) -> dict[str, np.ndarray]:
    model = reference.model
    layer = reference.layer
    config = model.config

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    kv_width = num_key_value_heads * head_dim
    num_groups = num_attention_heads // num_key_value_heads
    scaling = float(layer.self_attn.scaling)
    rms_eps = float(config.rms_norm_eps)

    input_np = layer0_input.detach().cpu().to(torch.float32).squeeze(0).numpy()
    input_ln_weight = layer.input_layernorm.weight.detach().cpu().to(torch.float32).numpy()
    post_ln_weight = layer.post_attention_layernorm.weight.detach().cpu().to(torch.float32).numpy()

    q_weight = quantized_weight(layer.self_attn.q_proj.weight, hidden_size, hidden_size)
    k_weight = quantized_weight(layer.self_attn.k_proj.weight, kv_width, hidden_size)
    v_weight = quantized_weight(layer.self_attn.v_proj.weight, kv_width, hidden_size)
    o_weight = quantized_weight(layer.self_attn.o_proj.weight, hidden_size, hidden_size)
    q_bias = optional_bias(layer.self_attn.q_proj)
    k_bias = optional_bias(layer.self_attn.k_proj)
    v_bias = optional_bias(layer.self_attn.v_proj)
    o_bias = optional_bias(layer.self_attn.o_proj)
    gate_weight = layer.mlp.gate_proj.weight.detach().cpu().to(torch.float32).numpy()
    gate_bias = optional_bias(layer.mlp.gate_proj)
    up_weight = layer.mlp.up_proj.weight.detach().cpu().to(torch.float32).numpy()
    up_bias = optional_bias(layer.mlp.up_proj)
    down_weight = layer.mlp.down_proj.weight.detach().cpu().to(torch.float32).numpy()
    down_bias = optional_bias(layer.mlp.down_proj)

    seq_len = input_np.shape[0]
    input_layernorm = np.zeros((seq_len, hidden_size), dtype=np.float32)
    q_proj = np.zeros((seq_len, hidden_size), dtype=np.float32)
    k_proj = np.zeros((seq_len, kv_width), dtype=np.float32)
    v_proj = np.zeros((seq_len, kv_width), dtype=np.float32)

    for token in range(seq_len):
        input_layernorm[token] = rmsnorm_token(input_np[token], input_ln_weight, rms_eps)
        q_proj[token] = linear_row_major(input_layernorm[token], q_weight, q_bias)
        k_proj[token] = linear_row_major(input_layernorm[token], k_weight, k_bias)
        v_proj[token] = linear_row_major(input_layernorm[token], v_weight, v_bias)

    q_rot = q_proj.copy().reshape(seq_len, num_attention_heads, head_dim)
    k_rot_kv = k_proj.copy().reshape(seq_len, num_key_value_heads, head_dim)
    inv_freq = np.array([1000000.0 ** (-2.0 * index / head_dim) for index in range(head_dim // 2)], dtype=np.float64)

    for token in range(seq_len):
        for head in range(num_attention_heads):
            for pair in range(head_dim // 2):
                angle = float(token) * float(inv_freq[pair])
                cosv = np.float32(round(float(np.cos(angle)), 7))
                sinv = np.float32(round(float(np.sin(angle)), 7))
                even = q_rot[token, head, pair]
                odd = q_rot[token, head, pair + head_dim // 2]
                q_rot[token, head, pair] = np.float32(even * cosv - odd * sinv)
                q_rot[token, head, pair + head_dim // 2] = np.float32(odd * cosv + even * sinv)
        for kv_head in range(num_key_value_heads):
            for pair in range(head_dim // 2):
                angle = float(token) * float(inv_freq[pair])
                cosv = np.float32(round(float(np.cos(angle)), 7))
                sinv = np.float32(round(float(np.sin(angle)), 7))
                even = k_rot_kv[token, kv_head, pair]
                odd = k_rot_kv[token, kv_head, pair + head_dim // 2]
                k_rot_kv[token, kv_head, pair] = np.float32(even * cosv - odd * sinv)
                k_rot_kv[token, kv_head, pair + head_dim // 2] = np.float32(odd * cosv + even * sinv)

    k_rot = np.repeat(k_rot_kv, num_groups, axis=1)
    attn_probs = np.zeros((num_attention_heads, seq_len, seq_len), dtype=np.float32)
    attn_context = np.zeros((seq_len, hidden_size), dtype=np.float32)

    for token in range(seq_len):
        for head in range(num_attention_heads):
            kv_head = head // num_groups
            scores = np.zeros(token + 1, dtype=np.float64)
            for src in range(token + 1):
                scores[src] = float(np.dot(q_rot[token, head], k_rot[src, head])) * scaling
            max_score = np.max(scores)
            probs = np.exp(scores - max_score)
            probs = probs / np.sum(probs)
            attn_probs[head, token, : token + 1] = probs.astype(np.float32)
            for dim in range(head_dim):
                values = v_proj[: token + 1, kv_head * head_dim + dim]
                attn_context[token, head * head_dim + dim] = np.float32(np.dot(probs, values))

    o_proj = np.zeros((seq_len, hidden_size), dtype=np.float32)
    attention_residual = np.zeros((seq_len, hidden_size), dtype=np.float32)
    post_attention_layernorm = np.zeros((seq_len, hidden_size), dtype=np.float32)
    gate_proj = np.zeros((seq_len, intermediate_size), dtype=np.float32)
    up_proj = np.zeros((seq_len, intermediate_size), dtype=np.float32)
    silu_mul = np.zeros((seq_len, intermediate_size), dtype=np.float32)
    down_proj = np.zeros((seq_len, hidden_size), dtype=np.float32)
    layer0_output = np.zeros((seq_len, hidden_size), dtype=np.float32)

    for token in range(seq_len):
        o_proj[token] = linear_row_major(attn_context[token], o_weight, o_bias)
        attention_residual[token] = input_np[token] + o_proj[token]
        post_attention_layernorm[token] = rmsnorm_token(attention_residual[token], post_ln_weight, rms_eps)
        gate_proj[token] = linear_row_major(post_attention_layernorm[token], gate_weight, gate_bias)
        up_proj[token] = linear_row_major(post_attention_layernorm[token], up_weight, up_bias)
        silu_mul[token] = silu(gate_proj[token]) * up_proj[token]
        down_proj[token] = linear_row_major(silu_mul[token], down_weight, down_bias)
        layer0_output[token] = attention_residual[token] + down_proj[token]

    return {
        "input_layernorm": input_layernorm,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "q_rot": np.transpose(q_rot, (1, 0, 2)),
        "k_rot": np.transpose(k_rot, (1, 0, 2)),
        "attn_probs": attn_probs,
        "attn_context": attn_context,
        "self_attn_output": o_proj,
        "o_proj": o_proj,
        "attention_residual": attention_residual,
        "post_attention_layernorm": post_attention_layernorm,
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "silu_mul": silu_mul,
        "down_proj": down_proj,
        "layer0_output": layer0_output,
    }


def summarize_against_reference(reference_outputs: dict[str, torch.Tensor], candidate_outputs: dict[str, np.ndarray]) -> dict[str, dict[str, float | tuple[int, ...]]]:
    report: dict[str, dict[str, float | tuple[int, ...]]] = {}
    for name, reference_value in reference_outputs.items():
        if name not in candidate_outputs:
            continue
        reference_array = reference_value.detach().cpu().to(torch.float32).squeeze(0).numpy()
        if reference_array.size == 0:
            continue
        report[name] = diff_report(candidate_outputs[name], reference_array)
    return report


def extract_focus(report: dict[str, dict[str, float | tuple[int, ...]]]) -> dict[str, dict[str, float | tuple[int, ...]]]:
    focus_names = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "self_attn_output",
        "attention_residual",
        "post_attention_layernorm",
        "down_proj",
        "layer0_output",
    ]
    return {name: report[name] for name in focus_names if name in report}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose layer0 selective quantization attribution for prefill attention and MLP.")
    parser.add_argument("--case-path", type=Path, default=DEFAULT_CASE_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    case = np.load(args.case_path)
    layer0_input = torch.from_numpy(case["layer0_input"]).to(torch.float32)

    reference = Layer0PrefillReferenceBackend()
    reference_outputs = reference.run(layer0_input).as_dict()
    float_outputs = run_wrapper_emulation(reference, layer0_input)
    attention_only_outputs = run_attention_only_quant_emulation(reference, layer0_input)
    mlp_only_outputs = run_mlp_only_quant_emulation(reference, float_outputs)
    full_quant_outputs = run_mlp_only_quant_emulation(reference, attention_only_outputs)

    result = {
        "case_path": str(args.case_path),
        "float_wrapper": extract_focus(summarize_against_reference(reference_outputs, float_outputs)),
        "attention_only_quant": extract_focus(summarize_against_reference(reference_outputs, attention_only_outputs)),
        "mlp_only_quant": extract_focus(summarize_against_reference(reference_outputs, mlp_only_outputs)),
        "full_quant": extract_focus(summarize_against_reference(reference_outputs, full_quant_outputs)),
    }

    result["headline"] = {
        "float_layer0_output_max_abs_diff": result["float_wrapper"]["layer0_output"]["max_abs_diff"],
        "attention_only_layer0_output_max_abs_diff": result["attention_only_quant"]["layer0_output"]["max_abs_diff"],
        "mlp_only_layer0_output_max_abs_diff": result["mlp_only_quant"]["layer0_output"]["max_abs_diff"],
        "full_quant_layer0_output_max_abs_diff": result["full_quant"]["layer0_output"]["max_abs_diff"],
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("Case:", result["case_path"])
    print("Headline:", result["headline"])
    for section in ("float_wrapper", "attention_only_quant", "mlp_only_quant", "full_quant"):
        print(section + ":")
        for name, report in result[section].items():
            print(f"  {name}: {report}")


if __name__ == "__main__":
    main()
