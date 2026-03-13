#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
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


DEFAULT_CASE_DIR = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases"
DEFAULT_CASE_GLOB = "*_prefill_case.npz"


@dataclass
class InputCase:
    name: str
    source: str
    layer0_input: torch.Tensor


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


def range_report(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float | tuple[int, ...]]:
    report = diff_report(lhs, rhs)
    signed_delta = lhs.astype(np.float32) - rhs.astype(np.float32)

    min_flat_index = int(signed_delta.argmin())
    min_index = tuple(int(index) for index in np.unravel_index(min_flat_index, signed_delta.shape))
    max_flat_index = int(signed_delta.argmax())
    max_index = tuple(int(index) for index in np.unravel_index(max_flat_index, signed_delta.shape))

    report.update(
        {
            "min_diff": float(signed_delta.min()),
            "min_diff_index": min_index,
            "lhs_at_min_diff": float(lhs[min_index]),
            "rhs_at_min_diff": float(rhs[min_index]),
            "max_diff": float(signed_delta.max()),
            "max_diff_index": max_index,
            "lhs_at_max_diff": float(lhs[max_index]),
            "rhs_at_max_diff": float(rhs[max_index]),
        }
    )
    return report


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
        report[name] = range_report(candidate_outputs[name], reference_array)
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


def parse_seq_lens(seq_lens_text: str) -> list[int]:
    if not seq_lens_text.strip():
        return []
    seq_lens = [int(item.strip()) for item in seq_lens_text.split(",") if item.strip()]
    if any(seq_len <= 0 for seq_len in seq_lens):
        raise ValueError("Sequence lengths must be positive integers.")
    return seq_lens


def discover_case_paths(case_dir: Path, case_glob: str, explicit_case_paths: list[Path]) -> list[Path]:
    if explicit_case_paths:
        return explicit_case_paths
    return sorted(path for path in case_dir.glob(case_glob) if path.is_file())


def load_cases(case_paths: list[Path]) -> list[InputCase]:
    cases: list[InputCase] = []
    for case_path in case_paths:
        with np.load(case_path) as case_data:
            if "layer0_input" not in case_data:
                continue
            layer0_input = torch.from_numpy(case_data["layer0_input"]).to(torch.float32)
        cases.append(InputCase(name=case_path.stem, source="real", layer0_input=layer0_input))
    return cases


def build_random_cases(real_cases: list[InputCase], random_cases: int, random_seed: int, random_seq_lens: list[int]) -> list[InputCase]:
    if random_cases <= 0:
        return []
    if not real_cases:
        raise ValueError("Random case generation requires at least one real case to estimate input distribution.")

    hidden_size = int(real_cases[0].layer0_input.shape[-1])
    observed_seq_lens = [int(case.layer0_input.shape[1]) for case in real_cases]
    seq_lens = random_seq_lens if random_seq_lens else observed_seq_lens

    flat_inputs = np.concatenate([case.layer0_input.numpy().reshape(-1) for case in real_cases])
    mean = float(flat_inputs.mean())
    std = float(flat_inputs.std())
    if std == 0.0:
        std = 1.0

    rng = np.random.default_rng(random_seed)
    generated_cases: list[InputCase] = []
    for index in range(random_cases):
        seq_len = int(rng.choice(seq_lens))
        layer0_input = rng.normal(loc=mean, scale=std, size=(1, seq_len, hidden_size)).astype(np.float32)
        generated_cases.append(
            InputCase(
                name=f"random_prefill_case_{index:03d}_seq{seq_len}",
                source="random",
                layer0_input=torch.from_numpy(layer0_input),
            )
        )
    return generated_cases


def aggregate_metric_reports(case_results: list[dict[str, object]], mode_name: str, metric_name: str) -> dict[str, object]:
    reports = [(case_result["case_name"], case_result[mode_name][metric_name]) for case_result in case_results]
    max_abs_values = [float(report["max_abs_diff"]) for _, report in reports]
    mean_abs_values = [float(report["mean_abs_diff"]) for _, report in reports]
    min_diff_values = [float(report["min_diff"]) for _, report in reports]
    max_diff_values = [float(report["max_diff"]) for _, report in reports]

    worst_max_abs_index = int(np.argmax(max_abs_values))
    worst_min_index = int(np.argmin(min_diff_values))
    worst_max_index = int(np.argmax(max_diff_values))

    return {
        "case_count": len(reports),
        "mean_max_abs_diff": float(np.mean(max_abs_values)),
        "worst_max_abs_diff": float(max_abs_values[worst_max_abs_index]),
        "worst_max_abs_case": str(reports[worst_max_abs_index][0]),
        "mean_mean_abs_diff": float(np.mean(mean_abs_values)),
        "mean_min_diff": float(np.mean(min_diff_values)),
        "worst_min_diff": float(min_diff_values[worst_min_index]),
        "worst_min_diff_case": str(reports[worst_min_index][0]),
        "mean_max_diff": float(np.mean(max_diff_values)),
        "worst_max_diff": float(max_diff_values[worst_max_index]),
        "worst_max_diff_case": str(reports[worst_max_index][0]),
    }


def aggregate_results(case_results: list[dict[str, object]]) -> dict[str, dict[str, dict[str, object]]]:
    mode_names = ["attention_only_quant", "mlp_only_quant", "full_quant"]
    metric_names = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "layer0_output"]
    grouped_results = {
        "all_cases": case_results,
        "real_cases": [case_result for case_result in case_results if case_result["source"] == "real"],
        "random_cases": [case_result for case_result in case_results if case_result["source"] == "random"],
    }

    aggregates: dict[str, dict[str, dict[str, object]]] = {}
    for group_name, group_results in grouped_results.items():
        if not group_results:
            continue
        aggregates[group_name] = {}
        for mode_name in mode_names:
            aggregates[group_name][mode_name] = {}
            for metric_name in metric_names:
                aggregates[group_name][mode_name][metric_name] = aggregate_metric_reports(group_results, mode_name, metric_name)
    return aggregates


def run_case(reference: Layer0PrefillReferenceBackend, input_case: InputCase) -> dict[str, object]:
    reference_outputs = reference.run(input_case.layer0_input).as_dict()
    float_outputs = run_wrapper_emulation(reference, input_case.layer0_input)
    attention_only_outputs = run_attention_only_quant_emulation(reference, input_case.layer0_input)
    mlp_only_outputs = run_mlp_only_quant_emulation(reference, float_outputs)
    full_quant_outputs = run_mlp_only_quant_emulation(reference, attention_only_outputs)

    result = {
        "case_name": input_case.name,
        "source": input_case.source,
        "seq_len": int(input_case.layer0_input.shape[1]),
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
        "attention_only_layer0_output_min_diff": result["attention_only_quant"]["layer0_output"]["min_diff"],
        "attention_only_layer0_output_max_diff": result["attention_only_quant"]["layer0_output"]["max_diff"],
        "mlp_only_layer0_output_min_diff": result["mlp_only_quant"]["layer0_output"]["min_diff"],
        "mlp_only_layer0_output_max_diff": result["mlp_only_quant"]["layer0_output"]["max_diff"],
    }
    return result


def print_text_summary(result: dict[str, object]) -> None:
    print("Cases:")
    for case_result in result["cases"]:
        headline = case_result["headline"]
        print(
            f"  {case_result['case_name']} ({case_result['source']}, seq_len={case_result['seq_len']}): "
            f"attn={headline['attention_only_layer0_output_max_abs_diff']:.6f}, "
            f"mlp={headline['mlp_only_layer0_output_max_abs_diff']:.6f}, "
            f"full={headline['full_quant_layer0_output_max_abs_diff']:.6f}, "
            f"attn_range=[{headline['attention_only_layer0_output_min_diff']:.6f}, {headline['attention_only_layer0_output_max_diff']:.6f}], "
            f"mlp_range=[{headline['mlp_only_layer0_output_min_diff']:.6f}, {headline['mlp_only_layer0_output_max_diff']:.6f}]"
        )

    print("Aggregates:")
    for group_name, group_summary in result["aggregates"].items():
        print(group_name + ":")
        for mode_name in ("attention_only_quant", "mlp_only_quant", "full_quant"):
            layer0_summary = group_summary[mode_name]["layer0_output"]
            print(
                f"  {mode_name}: mean_max_abs={layer0_summary['mean_max_abs_diff']:.6f}, "
                f"worst_max_abs={layer0_summary['worst_max_abs_diff']:.6f} ({layer0_summary['worst_max_abs_case']}), "
                f"mean_range=[{layer0_summary['mean_min_diff']:.6f}, {layer0_summary['mean_max_diff']:.6f}], "
                f"worst_range=[{layer0_summary['worst_min_diff']:.6f}, {layer0_summary['worst_max_diff']:.6f}]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose layer0 selective quantization attribution for prefill attention and MLP.")
    parser.add_argument("--case-path", type=Path, action="append", default=[])
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--case-glob", type=str, default=DEFAULT_CASE_GLOB)
    parser.add_argument("--random-cases", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--random-seq-lens", type=str, default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    reference = Layer0PrefillReferenceBackend()
    case_paths = discover_case_paths(args.case_dir, args.case_glob, args.case_path)
    if not case_paths and args.random_cases <= 0:
        raise FileNotFoundError(f"No cases matched glob '{args.case_glob}' under {args.case_dir}.")

    real_cases = load_cases(case_paths)
    random_seq_lens = parse_seq_lens(args.random_seq_lens)
    random_cases = build_random_cases(real_cases, args.random_cases, args.random_seed, random_seq_lens)
    input_cases = [*real_cases, *random_cases]
    if not input_cases:
        raise RuntimeError("No valid input cases available for diagnosis.")

    result = {
        "case_dir": str(args.case_dir),
        "case_glob": args.case_glob,
        "default_single_case_path": str(DEFAULT_CASE_PATH),
        "explicit_case_paths": [str(path) for path in args.case_path],
        "random_cases": args.random_cases,
        "random_seed": args.random_seed,
        "random_seq_lens": random_seq_lens,
        "cases": [run_case(reference, input_case) for input_case in input_cases],
    }
    result["aggregates"] = aggregate_results(result["cases"])

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print_text_summary(result)


if __name__ == "__main__":
    main()
