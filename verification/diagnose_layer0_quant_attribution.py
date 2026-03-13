#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
import faulthandler
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

VERIFICATION_DIR = Path(__file__).resolve().parent
if str(VERIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(VERIFICATION_DIR))

from diagnose_layer0_prefill_wrapper_stages import (  # noqa: E402
    DEFAULT_CASE_PATH,
    linear_row_major_rows,
    diff_report,
    linear_row_major,
    rmsnorm_rows,
    rmsnorm_token,
    run_wrapper_emulation,
    silu,
)
from layer0_prefill_reference_backend import Layer0PrefillReferenceBackend  # noqa: E402
from real_host_top_backend import quantize_int4_per_channel  # noqa: E402


DEFAULT_CASE_DIR = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases"
DEFAULT_CASE_GLOB = "*_prefill_case.npz"
DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "analysis"
DEFAULT_JSON_REPORT_PATH = DEFAULT_REPORT_DIR / "layer0_quant_batch_baseline.json"
DEFAULT_MARKDOWN_REPORT_PATH = DEFAULT_REPORT_DIR / "layer0_quant_batch_baseline.md"


@dataclass
class InputCase:
    name: str
    source: str
    layer0_input: torch.Tensor


@dataclass
class ProgressLogger:
    enabled: bool = True

    def log(self, message: str) -> None:
        if self.enabled:
            print(message, flush=True)


def elapsed_seconds(start_time: float) -> float:
    return time.monotonic() - start_time


def check_timeout(start_time: float, timeout_seconds: float, logger: ProgressLogger, stage: str) -> None:
    if timeout_seconds > 0.0 and elapsed_seconds(start_time) > timeout_seconds:
        logger.log(f"[timeout] exceeded {timeout_seconds:.2f}s after {stage}")
        raise TimeoutError(f"Timed out after {timeout_seconds:.2f}s during {stage}.")


def arm_hard_timeout(timeout_seconds: float, logger: ProgressLogger) -> None:
    if timeout_seconds <= 0.0:
        return
    logger.log(f"[run] arm_hard_timeout={timeout_seconds:.2f}s")
    faulthandler.dump_traceback_later(timeout_seconds, repeat=False, exit=True)


def cancel_hard_timeout() -> None:
    faulthandler.cancel_dump_traceback_later()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def unpack_int4_per_channel(packed: np.ndarray, scales: np.ndarray, out_dim: int, in_dim: int) -> np.ndarray:
    flat = packed.astype(np.uint8, copy=False).reshape(-1)
    low = flat & np.uint8(0x0F)
    high = (flat >> np.uint8(4)) & np.uint8(0x0F)
    interleaved = np.empty(flat.size * 2, dtype=np.int8)
    interleaved[0::2] = low.astype(np.int8)
    interleaved[1::2] = high.astype(np.int8)
    signed = np.where(interleaved >= 8, interleaved - 16, interleaved).astype(np.float32, copy=False)
    return signed[: out_dim * in_dim].reshape(out_dim, in_dim) * scales[:, None]


def quantized_weight(weight: torch.Tensor, out_dim: int, in_dim: int) -> np.ndarray:
    packed, scales = quantize_int4_per_channel(weight, out_dim, in_dim)
    return unpack_int4_per_channel(packed, scales, out_dim, in_dim)


def optional_bias(module: torch.nn.Module) -> np.ndarray | None:
    if module.bias is None:
        return None
    return module.bias.detach().cpu().to(torch.float32).numpy()


def get_cached_quant_params(reference: Layer0PrefillReferenceBackend) -> dict[str, np.ndarray | None]:
    cache = getattr(reference, "_quant_attr_cache", None)
    if cache is not None:
        return cache

    layer = reference.layer
    hidden_size = reference.model.config.hidden_size
    intermediate_size = reference.model.config.intermediate_size
    head_dim = hidden_size // reference.model.config.num_attention_heads
    kv_width = reference.model.config.num_key_value_heads * head_dim

    cache = {
        "q_weight_quant": quantized_weight(layer.self_attn.q_proj.weight, hidden_size, hidden_size),
        "k_weight_quant": quantized_weight(layer.self_attn.k_proj.weight, kv_width, hidden_size),
        "v_weight_quant": quantized_weight(layer.self_attn.v_proj.weight, kv_width, hidden_size),
        "o_weight_quant": quantized_weight(layer.self_attn.o_proj.weight, hidden_size, hidden_size),
        "gate_weight_quant": quantized_weight(layer.mlp.gate_proj.weight, intermediate_size, hidden_size),
        "up_weight_quant": quantized_weight(layer.mlp.up_proj.weight, intermediate_size, hidden_size),
        "down_weight_quant": quantized_weight(layer.mlp.down_proj.weight, hidden_size, intermediate_size),
        "q_bias": optional_bias(layer.self_attn.q_proj),
        "k_bias": optional_bias(layer.self_attn.k_proj),
        "v_bias": optional_bias(layer.self_attn.v_proj),
        "o_bias": optional_bias(layer.self_attn.o_proj),
        "gate_bias": optional_bias(layer.mlp.gate_proj),
        "up_bias": optional_bias(layer.mlp.up_proj),
        "down_bias": optional_bias(layer.mlp.down_proj),
    }
    reference._quant_attr_cache = cache
    return cache


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


def apply_rotary_decimal(states: np.ndarray, cos_values: np.ndarray, sin_values: np.ndarray) -> np.ndarray:
    half_dim = states.shape[-1] // 2
    even = states[..., :half_dim]
    odd = states[..., half_dim:]
    cos_term = cos_values[:, None, :]
    sin_term = sin_values[:, None, :]
    rotated_even = np.float32(even * cos_term - odd * sin_term)
    rotated_odd = np.float32(odd * cos_term + even * sin_term)
    return np.concatenate([rotated_even, rotated_odd], axis=-1)


def run_mlp_only_quant_emulation(reference: Layer0PrefillReferenceBackend, float_outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    layer = reference.layer
    hidden_size = reference.model.config.hidden_size
    rms_eps = float(reference.model.config.rms_norm_eps)
    quant_params = get_cached_quant_params(reference)

    post_ln_weight = layer.post_attention_layernorm.weight.detach().cpu().to(torch.float32).numpy()
    gate_weight = quant_params["gate_weight_quant"]
    up_weight = quant_params["up_weight_quant"]
    down_weight = quant_params["down_weight_quant"]
    gate_bias = quant_params["gate_bias"]
    up_bias = quant_params["up_bias"]
    down_bias = quant_params["down_bias"]

    attention_residual = float_outputs["attention_residual"]
    post_attention_layernorm = rmsnorm_rows(attention_residual, post_ln_weight, rms_eps)
    gate_proj = linear_row_major_rows(post_attention_layernorm, gate_weight, gate_bias)
    up_proj = linear_row_major_rows(post_attention_layernorm, up_weight, up_bias)
    silu_mul = silu(gate_proj) * up_proj
    down_proj = linear_row_major_rows(silu_mul, down_weight, down_bias)
    layer0_output = attention_residual + down_proj

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
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads
    head_dim = hidden_size // num_attention_heads
    kv_width = num_key_value_heads * head_dim
    num_groups = num_attention_heads // num_key_value_heads
    scaling = float(layer.self_attn.scaling)
    rms_eps = float(config.rms_norm_eps)
    quant_params = get_cached_quant_params(reference)

    input_np = layer0_input.detach().cpu().to(torch.float32).squeeze(0).numpy()
    input_ln_weight = layer.input_layernorm.weight.detach().cpu().to(torch.float32).numpy()
    post_ln_weight = layer.post_attention_layernorm.weight.detach().cpu().to(torch.float32).numpy()

    q_weight = quant_params["q_weight_quant"]
    k_weight = quant_params["k_weight_quant"]
    v_weight = quant_params["v_weight_quant"]
    o_weight = quant_params["o_weight_quant"]
    q_bias = quant_params["q_bias"]
    k_bias = quant_params["k_bias"]
    v_bias = quant_params["v_bias"]
    o_bias = quant_params["o_bias"]
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
    cos_values = np.round(np.cos(angles), 7).astype(np.float32)
    sin_values = np.round(np.sin(angles), 7).astype(np.float32)

    q_rot = apply_rotary_decimal(q_rot, cos_values, sin_values)
    k_rot_kv = apply_rotary_decimal(k_rot_kv, cos_values, sin_values)

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


def build_result_metadata(
    reference: Layer0PrefillReferenceBackend,
    case_paths: list[Path],
    random_seq_lens: list[int],
    random_cases: list[InputCase],
    case_results: list[dict[str, object]],
) -> dict[str, object]:
    config = reference.model.config
    return {
        "model_scope": {
            "model_type": str(getattr(config, "model_type", "unknown")),
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "analyzed_layer": 0,
            "parameter_scope": "Qwen2.5 layer0 weights taken from the full pretrained model",
            "activation_scope": "Layer0 prefill input activations only; not a whole-model end-to-end full-quant rollout",
        },
        "statistical_basis": {
            "case_count": len(case_results),
            "real_case_count": sum(1 for case_result in case_results if case_result["source"] == "real"),
            "random_case_count": sum(1 for case_result in case_results if case_result["source"] == "random"),
            "real_case_paths": [str(path) for path in case_paths],
            "random_case_generation": {
                "enabled": bool(random_cases),
                "count": len(random_cases),
                "seq_lens": random_seq_lens,
                "distribution": "Gaussian estimated from concatenated real layer0_input values",
            },
            "metric_definition": "mean_max_abs_diff is the arithmetic mean of per-case max_abs_diff for the target tensor.",
            "layer0_output_definition": "Per-case comparison uses the layer0_output tensor between selective-quant emulation and Torch layer0 reference.",
        },
    }


def build_markdown_report(result: dict[str, object]) -> str:
    metadata = result["metadata"]
    model_scope = metadata["model_scope"]
    basis = metadata["statistical_basis"]
    lines: list[str] = []
    lines.append("# Layer0 Selective-Quant 基线报告")
    lines.append("")
    lines.append("## 统计口径")
    lines.append("")
    lines.append(f"- 模型参数范围：{model_scope['parameter_scope']}")
    lines.append(f"- 激活统计范围：{model_scope['activation_scope']}")
    lines.append(f"- 当前分析层：layer {model_scope['analyzed_layer']}")
    lines.append(
        f"- 模型维度：hidden_size={model_scope['hidden_size']}，intermediate_size={model_scope['intermediate_size']}，"
        f"num_hidden_layers={model_scope['num_hidden_layers']}，num_attention_heads={model_scope['num_attention_heads']}，"
        f"num_key_value_heads={model_scope['num_key_value_heads']}"
    )
    lines.append(f"- 总样本数：{basis['case_count']} = 真实 case {basis['real_case_count']} + 随机 case {basis['random_case_count']}")
    lines.append(f"- 均值定义：{basis['metric_definition']}")
    lines.append(f"- layer0_output 定义：{basis['layer0_output_definition']}")
    if basis["real_case_paths"]:
        lines.append("- 真实 case 来源：")
        for case_path in basis["real_case_paths"]:
            lines.append(f"  - {case_path}")
    random_generation = basis["random_case_generation"]
    if random_generation["enabled"]:
        lines.append(
            f"- 随机 case：count={random_generation['count']}，seq_lens={random_generation['seq_lens']}，"
            f"distribution={random_generation['distribution']}"
        )
    lines.append("")
    lines.append("## 关键结论")
    lines.append("")
    if "all_cases" in result["aggregates"]:
        all_full = result["aggregates"]["all_cases"]["full_quant"]["layer0_output"]
        all_attn = result["aggregates"]["all_cases"]["attention_only_quant"]["layer0_output"]
        all_mlp = result["aggregates"]["all_cases"]["mlp_only_quant"]["layer0_output"]
        lines.append(
            f"- all_cases 下，full-quant 的 layer0_output mean_max_abs_diff = {all_full['mean_max_abs_diff']:.6f}，"
            f"这是对 {all_full['case_count']} 个 case 的逐 case max_abs_diff 取算术平均得到。"
        )
        lines.append(
            f"- all_cases 下，attention-only 的 layer0_output mean_max_abs_diff = {all_attn['mean_max_abs_diff']:.6f}，"
            f"最坏值 = {all_attn['worst_max_abs_diff']:.6f}。"
        )
        lines.append(
            f"- all_cases 下，mlp-only 的 layer0_output mean_max_abs_diff = {all_mlp['mean_max_abs_diff']:.6f}，"
            f"最坏值 = {all_mlp['worst_max_abs_diff']:.6f}。"
        )
    if "real_cases" in result["aggregates"]:
        real_attn = result["aggregates"]["real_cases"]["attention_only_quant"]["layer0_output"]
        real_mlp = result["aggregates"]["real_cases"]["mlp_only_quant"]["layer0_output"]
        real_full = result["aggregates"]["real_cases"]["full_quant"]["layer0_output"]
        lines.append(
            f"- 只看真实 case 时，attention-only / mlp-only / full-quant 的均值分别为 "
            f"{real_attn['mean_max_abs_diff']:.6f} / {real_mlp['mean_max_abs_diff']:.6f} / {real_full['mean_max_abs_diff']:.6f}。"
        )
    lines.append("")
    lines.append("## 聚合结果")
    lines.append("")
    lines.append("| 样本组 | 模式 | case_count | mean_max_abs_diff | worst_max_abs_diff | mean_min_diff | mean_max_diff |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for group_name, group_summary in result["aggregates"].items():
        for mode_name in ("attention_only_quant", "mlp_only_quant", "full_quant"):
            layer0_summary = group_summary[mode_name]["layer0_output"]
            lines.append(
                f"| {group_name} | {mode_name} | {layer0_summary['case_count']} | "
                f"{layer0_summary['mean_max_abs_diff']:.6f} | {layer0_summary['worst_max_abs_diff']:.6f} | "
                f"{layer0_summary['mean_min_diff']:.6f} | {layer0_summary['mean_max_diff']:.6f} |"
            )
    lines.append("")
    lines.append("## 分 case 摘要")
    lines.append("")
    lines.append("| case_name | source | seq_len | attn_max_abs | mlp_max_abs | full_max_abs | attn_range | mlp_range |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for case_result in result["cases"]:
        headline = case_result["headline"]
        lines.append(
            f"| {case_result['case_name']} | {case_result['source']} | {case_result['seq_len']} | "
            f"{headline['attention_only_layer0_output_max_abs_diff']:.6f} | "
            f"{headline['mlp_only_layer0_output_max_abs_diff']:.6f} | "
            f"{headline['full_quant_layer0_output_max_abs_diff']:.6f} | "
            f"[{headline['attention_only_layer0_output_min_diff']:.6f}, {headline['attention_only_layer0_output_max_diff']:.6f}] | "
            f"[{headline['mlp_only_layer0_output_min_diff']:.6f}, {headline['mlp_only_layer0_output_max_diff']:.6f}] |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("")
    lines.append("- 这里的 full-quant 均值不是整网 end-to-end logits 误差均值，而是 layer0 selective-quant 诊断里 `layer0_output` 的逐 case `max_abs_diff` 平均值。")
    lines.append("- 如果后续要作为 RTL tile 切分的 signoff 基线，建议优先参考 real_cases 与 all_cases 两组的 layer0_output 包络，而不是只看均值。")
    lines.append("")
    return "\n".join(lines)


def write_reports(result: dict[str, object], json_report_path: Path, markdown_report_path: Path) -> None:
    ensure_parent_dir(json_report_path)
    ensure_parent_dir(markdown_report_path)
    json_report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_report_path.write_text(build_markdown_report(result), encoding="utf-8")


def run_case(
    reference: Layer0PrefillReferenceBackend,
    input_case: InputCase,
    global_start_time: float,
    timeout_seconds: float,
    logger: ProgressLogger,
) -> dict[str, object]:
    case_start_time = time.monotonic()
    logger.log(f"[case] start name={input_case.name} source={input_case.source} seq_len={int(input_case.layer0_input.shape[1])}")

    phase_start = time.monotonic()
    reference_outputs = reference.run(input_case.layer0_input).as_dict()
    logger.log(f"[case] {input_case.name} reference_run={elapsed_seconds(phase_start):.3f}s")
    check_timeout(global_start_time, timeout_seconds, logger, f"reference.run for {input_case.name}")

    phase_start = time.monotonic()
    float_outputs = run_wrapper_emulation(reference, input_case.layer0_input)
    logger.log(f"[case] {input_case.name} float_wrapper={elapsed_seconds(phase_start):.3f}s")
    check_timeout(global_start_time, timeout_seconds, logger, f"float wrapper for {input_case.name}")

    phase_start = time.monotonic()
    attention_only_outputs = run_attention_only_quant_emulation(reference, input_case.layer0_input)
    logger.log(f"[case] {input_case.name} attention_only={elapsed_seconds(phase_start):.3f}s")
    check_timeout(global_start_time, timeout_seconds, logger, f"attention-only emulation for {input_case.name}")

    phase_start = time.monotonic()
    mlp_only_outputs = run_mlp_only_quant_emulation(reference, float_outputs)
    logger.log(f"[case] {input_case.name} mlp_only={elapsed_seconds(phase_start):.3f}s")
    check_timeout(global_start_time, timeout_seconds, logger, f"mlp-only emulation for {input_case.name}")

    phase_start = time.monotonic()
    full_quant_outputs = run_mlp_only_quant_emulation(reference, attention_only_outputs)
    logger.log(f"[case] {input_case.name} full_quant={elapsed_seconds(phase_start):.3f}s")
    check_timeout(global_start_time, timeout_seconds, logger, f"full-quant emulation for {input_case.name}")

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
    logger.log(f"[case] done name={input_case.name} total={elapsed_seconds(case_start_time):.3f}s")
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
    parser.add_argument("--json-report-path", type=Path, default=DEFAULT_JSON_REPORT_PATH)
    parser.add_argument("--markdown-report-path", type=Path, default=DEFAULT_MARKDOWN_REPORT_PATH)
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    run_start_time = time.monotonic()
    logger = ProgressLogger(enabled=not args.quiet)
    logger.log(f"[run] start timeout_seconds={args.timeout_seconds:.2f}")
    arm_hard_timeout(args.timeout_seconds, logger)

    try:
        phase_start = time.monotonic()
        reference = Layer0PrefillReferenceBackend()
        logger.log(f"[run] init_reference_backend={elapsed_seconds(phase_start):.3f}s")
        check_timeout(run_start_time, args.timeout_seconds, logger, "reference backend initialization")

        phase_start = time.monotonic()
        case_paths = discover_case_paths(args.case_dir, args.case_glob, args.case_path)
        if not case_paths and args.random_cases <= 0:
            raise FileNotFoundError(f"No cases matched glob '{args.case_glob}' under {args.case_dir}.")
        logger.log(f"[run] discover_case_paths={elapsed_seconds(phase_start):.3f}s count={len(case_paths)}")

        phase_start = time.monotonic()
        real_cases = load_cases(case_paths)
        random_seq_lens = parse_seq_lens(args.random_seq_lens)
        random_cases = build_random_cases(real_cases, args.random_cases, args.random_seed, random_seq_lens)
        input_cases = [*real_cases, *random_cases]
        if not input_cases:
            raise RuntimeError("No valid input cases available for diagnosis.")
        logger.log(
            f"[run] prepare_cases={elapsed_seconds(phase_start):.3f}s real={len(real_cases)} random={len(random_cases)} total={len(input_cases)}"
        )
        check_timeout(run_start_time, args.timeout_seconds, logger, "case preparation")

        case_results: list[dict[str, object]] = []
        timed_out = False
        timeout_message: str | None = None
        for input_case in input_cases:
            try:
                case_results.append(run_case(reference, input_case, run_start_time, args.timeout_seconds, logger))
            except TimeoutError as error:
                timed_out = True
                timeout_message = str(error)
                logger.log(f"[run] stop_after_case name={input_case.name} reason={timeout_message}")
                break

        result = {
            "case_dir": str(args.case_dir),
            "case_glob": args.case_glob,
            "default_single_case_path": str(DEFAULT_CASE_PATH),
            "explicit_case_paths": [str(path) for path in args.case_path],
            "random_cases": args.random_cases,
            "random_seed": args.random_seed,
            "random_seq_lens": random_seq_lens,
            "timeout_seconds": args.timeout_seconds,
            "timed_out": timed_out,
            "timeout_message": timeout_message,
            "cases_completed": len(case_results),
            "cases_requested": len(input_cases),
            "cases": case_results,
        }
        result["aggregates"] = aggregate_results(result["cases"])
        result["metadata"] = build_result_metadata(reference, case_paths, random_seq_lens, random_cases, result["cases"])
        result["report_paths"] = {
            "json": str(args.json_report_path),
            "markdown": str(args.markdown_report_path),
        }

        phase_start = time.monotonic()
        write_reports(result, args.json_report_path, args.markdown_report_path)
        logger.log(f"[run] write_reports={elapsed_seconds(phase_start):.3f}s")
        logger.log(f"[run] total_elapsed={elapsed_seconds(run_start_time):.3f}s completed_cases={len(case_results)}/{len(input_cases)}")

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return

        print_text_summary(result)
        print(f"JSON report: {args.json_report_path}")
        print(f"Markdown report: {args.markdown_report_path}")
    finally:
        cancel_hard_timeout()


if __name__ == "__main__":
    main()
