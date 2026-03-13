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


def round_bfloat16_array(values: np.ndarray) -> np.ndarray:
    rounded = np.asarray(values, dtype=np.float32).copy()
    bits = rounded.view(np.uint32)
    lsb = (bits >> 16) & np.uint32(1)
    bits += np.uint32(0x7FFF) + lsb
    bits &= np.uint32(0xFFFF0000)
    return bits.view(np.float32)


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
    normalized = np.asarray(input_token, dtype=np.float32) * inv_rms
    normalized_bf16 = round_bfloat16_array(normalized)
    return round_bfloat16_array(normalized_bf16 * np.asarray(weight, dtype=np.float32))


def rmsnorm_rows(input_rows: np.ndarray, weight: np.ndarray, rms_eps: float) -> np.ndarray:
    output = np.zeros_like(input_rows, dtype=np.float32)
    for row_index in range(input_rows.shape[0]):
        output[row_index] = rmsnorm_token(input_rows[row_index], weight, rms_eps)
    return output


def linear_row_major(input_token: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    return linear_row_major_rows(np.asarray(input_token, dtype=np.float32)[None, :], weight, bias)[0]


def linear_row_major_rows(input_rows: np.ndarray, weight: np.ndarray, bias: np.ndarray | None) -> np.ndarray:
    totals = np.matmul(
        np.asarray(input_rows, dtype=np.float32).astype(np.float64, copy=False),
        np.asarray(weight, dtype=np.float32).T.astype(np.float64, copy=False),
    )
    if bias is not None:
        totals = totals + np.asarray(bias, dtype=np.float32).astype(np.float64, copy=False)[None, :]
    return round_bfloat16_array(totals.astype(np.float32, copy=False))


def silu(value: np.ndarray) -> np.ndarray:
    return value / (1.0 + np.exp(-value))


def apply_rotary_bf16(states: np.ndarray, cos_values: np.ndarray, sin_values: np.ndarray) -> np.ndarray:
    half_dim = states.shape[-1] // 2
    even = states[..., :half_dim]
    odd = states[..., half_dim:]
    cos_term = cos_values[:, None, :]
    sin_term = sin_values[:, None, :]

    even_cos = round_bfloat16_array(even * cos_term)
    odd_sin = round_bfloat16_array(odd * sin_term)
    odd_cos = round_bfloat16_array(odd * cos_term)
    even_sin = round_bfloat16_array(even * sin_term)

    rotated_even = round_bfloat16_array(even_cos - odd_sin)
    rotated_odd = round_bfloat16_array(odd_cos + even_sin)
    return np.concatenate([rotated_even, rotated_odd], axis=-1)


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
    input_layernorm = rmsnorm_rows(input_np, input_ln_weight, rms_eps)
    q_proj = linear_row_major_rows(input_layernorm, q_weight, q_bias)
    k_proj = linear_row_major_rows(input_layernorm, k_weight, k_bias)
    v_proj = linear_row_major_rows(input_layernorm, v_weight, v_bias)

    q_rot = q_proj.copy().reshape(seq_len, num_attention_heads, head_dim)
    k_rot_kv = k_proj.copy().reshape(seq_len, num_key_value_heads, head_dim)
    inv_freq = np.array([1000000.0 ** (-2.0 * index / head_dim) for index in range(head_dim // 2)], dtype=np.float64)
    positions = np.arange(seq_len, dtype=np.float64)[:, None]
    angles = positions * inv_freq[None, :]
    cos_values = round_bfloat16_array(np.cos(angles).astype(np.float32))
    sin_values = round_bfloat16_array(np.sin(angles).astype(np.float32))

    q_rot = apply_rotary_bf16(q_rot, cos_values, sin_values)
    k_rot_kv = apply_rotary_bf16(k_rot_kv, cos_values, sin_values)

    k_rot = np.repeat(k_rot_kv, num_groups, axis=1)
    attn_probs = np.zeros((num_attention_heads, seq_len, seq_len), dtype=np.float32)
    v_heads = v_proj.reshape(seq_len, num_key_value_heads, head_dim)
    v_repeated = np.repeat(v_heads, num_groups, axis=1)

    attn_context = np.zeros((seq_len, hidden_size), dtype=np.float32)
    for token in range(seq_len):
        token_scores = np.einsum(
            "hd,shd->hs",
            q_rot[token].astype(np.float64, copy=False),
            k_rot[: token + 1].astype(np.float64, copy=False),
            optimize=True,
        )
        token_scores = token_scores * scaling
        max_scores = np.max(token_scores, axis=1, keepdims=True)
        probs = np.exp(token_scores - max_scores)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        attn_probs[:, token, : token + 1] = probs.astype(np.float32)
        token_context = np.einsum(
            "hs,shd->hd",
            probs.astype(np.float64, copy=False),
            v_repeated[: token + 1].astype(np.float64, copy=False),
            optimize=True,
        )
        attn_context[token] = token_context.astype(np.float32, copy=False).reshape(hidden_size)

    o_proj = linear_row_major_rows(attn_context, o_weight, o_bias)
    attention_residual = input_np + o_proj
    post_attention_layernorm = rmsnorm_rows(attention_residual, post_ln_weight, rms_eps)
    gate_proj = linear_row_major_rows(post_attention_layernorm, gate_weight, gate_bias)
    up_proj = linear_row_major_rows(post_attention_layernorm, up_weight, up_bias)
    silu_mul = silu(gate_proj) * up_proj
    down_proj = linear_row_major_rows(silu_mul, down_weight, down_bias)
    layer0_output = attention_residual + down_proj

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