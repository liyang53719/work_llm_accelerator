#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

VERIFICATION_DIR = Path(__file__).resolve().parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from layer0_prefill_reference_backend import Layer0PrefillReferenceBackend


DEFAULT_CASE_PATH = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases" / "layer0_prefill_case.npz"


def round_bfloat16_scalar(value: float) -> float:
    bits = np.array([value], dtype=np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    bits += np.uint32(0x7FFF) + lsb
    bits &= np.uint32(0xFFFF0000)
    return bits.view(np.float32)[0].item()


def diff_report(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float | tuple[int, ...]]:
    delta = np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))
    max_flat_index = int(delta.argmax())
    max_index = tuple(int(index) for index in np.unravel_index(max_flat_index, delta.shape))
    return {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "max_index": max_index,
        "lhs_at_max": float(lhs[max_index]),
        "rhs_at_max": float(rhs[max_index]),
    }


def rmsnorm_token(input_token: np.ndarray, weight: np.ndarray, rms_eps: float) -> np.ndarray:
    mean_square = np.mean(input_token.astype(np.float32) * input_token.astype(np.float32), dtype=np.float32)
    inv_rms = np.float32(1.0 / np.sqrt(np.float32(mean_square + np.float32(rms_eps))))
    output = np.zeros_like(input_token, dtype=np.float32)
    for dim in range(input_token.shape[0]):
        normalized = np.float32(input_token[dim] * inv_rms)
        normalized_bf16 = np.float32(round_bfloat16_scalar(float(normalized)))
        output[dim] = np.float32(round_bfloat16_scalar(float(normalized_bf16 * weight[dim])))
    return output


def linear_row_major(input_token: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    out_dim, in_dim = weight.shape
    output = np.zeros(out_dim, dtype=np.float32)
    for out_index in range(out_dim):
        total = float(bias[out_index]) if bias is not None else 0.0
        row = weight[out_index]
        for in_index in range(in_dim):
            total += float(input_token[in_index]) * float(row[in_index])
        output[out_index] = np.float32(round_bfloat16_scalar(total))
    return output


def silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def optional_bias(module: torch.nn.Module) -> np.ndarray | None:
    if module.bias is None:
        return None
    return module.bias.detach().cpu().to(torch.float32).numpy()


def run_wrapper_emulation(reference: Layer0PrefillReferenceBackend, layer0_input: torch.Tensor) -> dict[str, np.ndarray]:
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

    q_weight = layer.self_attn.q_proj.weight.detach().cpu().to(torch.float32).numpy()
    q_bias = optional_bias(layer.self_attn.q_proj)
    k_weight = layer.self_attn.k_proj.weight.detach().cpu().to(torch.float32).numpy()
    k_bias = optional_bias(layer.self_attn.k_proj)
    v_weight = layer.self_attn.v_proj.weight.detach().cpu().to(torch.float32).numpy()
    v_bias = optional_bias(layer.self_attn.v_proj)
    o_weight = layer.self_attn.o_proj.weight.detach().cpu().to(torch.float32).numpy()
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
    inv_freq = np.array(
        [1000000.0 ** (-2.0 * index / head_dim) for index in range(head_dim // 2)],
        dtype=np.float64,
    )

    for token in range(seq_len):
        for head in range(num_attention_heads):
            for pair in range(head_dim // 2):
                angle = float(token) * float(inv_freq[pair])
                cosv = np.float32(round_bfloat16_scalar(float(np.cos(angle))))
                sinv = np.float32(round_bfloat16_scalar(float(np.sin(angle))))
                even = q_rot[token, head, pair]
                odd = q_rot[token, head, pair + head_dim // 2]
                even_cos = np.float32(round_bfloat16_scalar(float(even * cosv)))
                odd_sin = np.float32(round_bfloat16_scalar(float(odd * sinv)))
                odd_cos = np.float32(round_bfloat16_scalar(float(odd * cosv)))
                even_sin = np.float32(round_bfloat16_scalar(float(even * sinv)))
                q_rot[token, head, pair] = np.float32(round_bfloat16_scalar(float(even_cos - odd_sin)))
                q_rot[token, head, pair + head_dim // 2] = np.float32(round_bfloat16_scalar(float(odd_cos + even_sin)))
        for kv_head in range(num_key_value_heads):
            for pair in range(head_dim // 2):
                angle = float(token) * float(inv_freq[pair])
                cosv = np.float32(round_bfloat16_scalar(float(np.cos(angle))))
                sinv = np.float32(round_bfloat16_scalar(float(np.sin(angle))))
                even = k_rot_kv[token, kv_head, pair]
                odd = k_rot_kv[token, kv_head, pair + head_dim // 2]
                even_cos = np.float32(round_bfloat16_scalar(float(even * cosv)))
                odd_sin = np.float32(round_bfloat16_scalar(float(odd * sinv)))
                odd_cos = np.float32(round_bfloat16_scalar(float(odd * cosv)))
                even_sin = np.float32(round_bfloat16_scalar(float(even * sinv)))
                k_rot_kv[token, kv_head, pair] = np.float32(round_bfloat16_scalar(float(even_cos - odd_sin)))
                k_rot_kv[token, kv_head, pair + head_dim // 2] = np.float32(round_bfloat16_scalar(float(odd_cos + even_sin)))

    k_rot = np.repeat(k_rot_kv, num_groups, axis=1)
    attn_probs = np.zeros((num_attention_heads, seq_len, seq_len), dtype=np.float32)

    attn_context = np.zeros((seq_len, hidden_size), dtype=np.float32)
    for token in range(seq_len):
        for head in range(num_attention_heads):
            kv_head = head // num_groups
            scores = np.zeros(token + 1, dtype=np.float64)
            for src in range(token + 1):
                dot = 0.0
                for dim in range(head_dim):
                    dot += float(q_rot[token, head, dim]) * float(k_rot[src, head, dim])
                scores[src] = dot * scaling
            max_score = np.max(scores)
            probs = np.exp(scores - max_score)
            probs = probs / np.sum(probs)
            attn_probs[head, token, : token + 1] = probs.astype(np.float32)
            for dim in range(head_dim):
                accum = 0.0
                for src in range(token + 1):
                    accum += float(probs[src]) * float(v_proj[src, kv_head * head_dim + dim])
                attn_context[token, head * head_dim + dim] = np.float32(accum)

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose stage-level diffs between the layer0 prefill wrapper math and the torch reference path.")
    parser.add_argument("--case-path", type=Path, default=DEFAULT_CASE_PATH)
    args = parser.parse_args()

    case = np.load(args.case_path)
    layer0_input = torch.from_numpy(case["layer0_input"]).to(torch.float32)

    reference = Layer0PrefillReferenceBackend()
    reference_outputs = reference.run(layer0_input).as_dict()
    wrapper_outputs = run_wrapper_emulation(reference, layer0_input)

    for name, reference_value in reference_outputs.items():
        wrapper_value = wrapper_outputs[name]
        reference_array = reference_value.detach().cpu().to(torch.float32).squeeze(0).numpy()
        if reference_array.size == 0:
            print(name, {"status": "unavailable_for_current_attention_backend"})
            continue
        report = diff_report(wrapper_value, reference_array)
        print(name, report)


if __name__ == "__main__":
    main()