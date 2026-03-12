#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass


@dataclass
class ModelSpec:
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    vocab_size: int = 151936

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def bytes_for_bits(bits: int) -> float:
    return bits / 8.0


def param_breakdown(spec: ModelSpec) -> dict[str, int]:
    kv_width = spec.num_key_value_heads * spec.head_dim
    attention = (
        spec.hidden_size * spec.hidden_size
        + spec.hidden_size * kv_width
        + spec.hidden_size * kv_width
        + spec.hidden_size * spec.hidden_size
    )
    mlp = (
        spec.hidden_size * spec.intermediate_size
        + spec.hidden_size * spec.intermediate_size
        + spec.intermediate_size * spec.hidden_size
    )
    norms = 2 * spec.hidden_size
    per_layer = attention + mlp + norms
    embed = spec.vocab_size * spec.hidden_size
    return {
        "attention_per_layer": attention,
        "mlp_per_layer": mlp,
        "norms_per_layer": norms,
        "per_layer": per_layer,
        "layers_total": per_layer * spec.num_hidden_layers,
        "embed_or_lm_head": embed,
        "model_total": per_layer * spec.num_hidden_layers + embed,
    }


def kv_cache_bytes_per_token(spec: ModelSpec, kv_bits: int) -> float:
    elements_per_layer = 2 * spec.num_key_value_heads * spec.head_dim
    elements_all_layers = elements_per_layer * spec.num_hidden_layers
    return elements_all_layers * bytes_for_bits(kv_bits)


def decode_macs_per_token(spec: ModelSpec, context_len: int) -> dict[str, int]:
    dims = param_breakdown(spec)
    qk_scores = spec.num_hidden_layers * spec.num_attention_heads * spec.head_dim * context_len
    av_mix = spec.num_hidden_layers * spec.num_attention_heads * spec.head_dim * context_len
    attention_context = qk_scores + av_mix
    return {
        "dense_stack": dims["layers_total"],
        "lm_head": dims["embed_or_lm_head"],
        "attention_context": attention_context,
        "total": dims["layers_total"] + dims["embed_or_lm_head"] + attention_context,
    }


def prefill_macs(spec: ModelSpec, prompt_len: int, include_lm_head: bool) -> dict[str, int]:
    dims = param_breakdown(spec)
    dense_stack = dims["layers_total"] * prompt_len
    attention_qk = (
        spec.num_hidden_layers
        * spec.num_attention_heads
        * spec.head_dim
        * prompt_len
        * prompt_len
    )
    attention_av = attention_qk
    lm_head = dims["embed_or_lm_head"] * prompt_len if include_lm_head else 0
    total = dense_stack + attention_qk + attention_av + lm_head
    return {
        "dense_stack": dense_stack,
        "attention_qk": attention_qk,
        "attention_av": attention_av,
        "attention_context": attention_qk + attention_av,
        "lm_head": lm_head,
        "total": total,
        "per_token_average": math.ceil(total / prompt_len),
    }


def sram_budget_report(
    spec: ModelSpec,
    sram_bytes: int,
    act_bits: int,
    weight_bits: int,
    kv_bits: int,
    prefill_tile_m: int,
    tile_n: int,
    softmax_scratch_factor: float,
) -> dict[str, float]:
    act_bytes = bytes_for_bits(act_bits)
    weight_bytes = bytes_for_bits(weight_bits)
    kv_bytes = bytes_for_bits(kv_bits)

    decode_input = spec.hidden_size * act_bytes
    decode_weight = spec.hidden_size * tile_n * weight_bytes
    decode_output = tile_n * act_bytes
    decode_kv_slice = 2 * spec.num_key_value_heads * spec.head_dim * kv_bytes
    decode_total = decode_input + decode_weight + decode_output + decode_kv_slice

    prefill_input = prefill_tile_m * spec.hidden_size * act_bytes
    prefill_weight = spec.hidden_size * tile_n * weight_bytes
    prefill_q_tile = prefill_tile_m * spec.hidden_size * act_bytes
    prefill_scores = prefill_tile_m * prefill_tile_m * act_bytes * softmax_scratch_factor
    prefill_output = prefill_tile_m * tile_n * act_bytes
    prefill_kv_stage = prefill_tile_m * 2 * spec.num_key_value_heads * spec.head_dim * kv_bytes
    prefill_total = (
        prefill_input
        + prefill_weight
        + prefill_q_tile
        + prefill_scores
        + prefill_output
        + prefill_kv_stage
    )

    return {
        "sram_bytes": sram_bytes,
        "decode_working_set_bytes": decode_total,
        "decode_fits": decode_total <= sram_bytes,
        "prefill_working_set_bytes": prefill_total,
        "prefill_fits": prefill_total <= sram_bytes,
        "prefill_tile_m": prefill_tile_m,
        "prefill_softmax_scratch_bytes": prefill_scores,
        "prefill_utilization": prefill_total / sram_bytes,
    }


def decode_bandwidth_report(
    spec: ModelSpec,
    params: dict[str, int],
    context_len: int,
    weight_bits: int,
    kv_bits: int,
    target_decode_tok_s: float,
) -> dict[str, float]:
    weight_stream_bytes = params["model_total"] * bytes_for_bits(weight_bits)
    kv_bytes_per_tok = kv_cache_bytes_per_token(spec, kv_bits)
    kv_read_bytes = context_len * kv_bytes_per_tok
    kv_write_bytes = kv_bytes_per_tok
    return {
        "weight_stream_bytes_per_token": weight_stream_bytes,
        "weight_stream_gb_per_s_for_target": weight_stream_bytes * target_decode_tok_s / 1e9,
        "kv_read_bytes_per_token": kv_read_bytes,
        "kv_write_bytes_per_token": kv_write_bytes,
        "kv_total_gb_per_s_for_target": (kv_read_bytes + kv_write_bytes) * target_decode_tok_s / 1e9,
    }


def prefill_bandwidth_report(
    spec: ModelSpec,
    params: dict[str, int],
    prompt_len: int,
    prefill_tile_m: int,
    kv_bits: int,
    weight_bits: int,
    target_prefill_tok_s: float,
    prefill_kv_block: int,
) -> dict[str, float]:
    model_weight_bytes = params["model_total"] * bytes_for_bits(weight_bits)
    kv_bytes_per_tok = kv_cache_bytes_per_token(spec, kv_bits)
    effective_weight_bytes_per_token = model_weight_bytes / max(prefill_tile_m, 1)
    num_query_blocks = math.ceil(prompt_len / prefill_tile_m)
    num_kv_blocks = math.ceil(prompt_len / prefill_kv_block)
    blocked_kv_reads = (
        num_query_blocks
        * num_kv_blocks
        * prefill_tile_m
        * kv_bytes_per_tok
    )
    kv_write_bytes = prompt_len * kv_bytes_per_tok
    return {
        "effective_weight_bytes_per_token": effective_weight_bytes_per_token,
        "effective_weight_gb_per_s_for_target": effective_weight_bytes_per_token * target_prefill_tok_s / 1e9,
        "kv_write_bytes_total": kv_write_bytes,
        "kv_write_gb_per_s_for_target": kv_bytes_per_tok * target_prefill_tok_s / 1e9,
        "blocked_attention_kv_read_bytes_total": blocked_kv_reads,
        "blocked_attention_kv_read_bytes_per_token": blocked_kv_reads / max(prompt_len, 1),
        "blocked_attention_kv_read_gb_per_s_for_target": (blocked_kv_reads / max(prompt_len, 1)) * target_prefill_tok_s / 1e9,
    }


def mixed_workload_report(
    prefill_total_macs: int,
    decode_total_macs: int,
    mac_units: int,
    freq_ghz: float,
    prompt_len: int,
    decode_tokens: int,
) -> dict[str, float]:
    peak_macs_per_s = mac_units * freq_ghz * 1e9
    total_tokens = prompt_len + decode_tokens
    total_macs = prefill_total_macs + decode_total_macs * decode_tokens
    seconds = total_macs / peak_macs_per_s
    return {
        "total_tokens": total_tokens,
        "total_macs": total_macs,
        "ideal_seconds": seconds,
        "ideal_tokens_per_s": total_tokens / seconds if seconds else 0.0,
        "ideal_decode_only_tokens_per_s": decode_tokens / seconds if seconds else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Qwen2.5-1.5B prefill + decode feasibility.")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-context-len", type=int, default=2048)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prefill-tile-m", type=int, default=64)
    parser.add_argument("--prefill-kv-block", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--target-decode-tok-s", type=float, default=10.0)
    parser.add_argument("--target-prefill-tok-s", type=float, default=200.0)
    parser.add_argument("--mac-units", type=int, default=2048)
    parser.add_argument("--freq-ghz", type=float, default=1.0)
    parser.add_argument("--sram-mb", type=float, default=1.0)
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--kv-bits", type=int, default=8)
    parser.add_argument("--softmax-scratch-factor", type=float, default=2.0)
    parser.add_argument("--include-prefill-lm-head", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = ModelSpec()
    params = param_breakdown(spec)
    decode_macs = decode_macs_per_token(spec, args.decode_context_len)
    prefill = prefill_macs(spec, args.prompt_len, args.include_prefill_lm_head)
    sram_bytes = int(args.sram_mb * 1024 * 1024)
    peak_macs_per_s = args.mac_units * args.freq_ghz * 1e9

    decode_perf = {
        "peak_tokens_per_s": peak_macs_per_s / decode_macs["total"],
        "target_utilization": args.target_decode_tok_s / (peak_macs_per_s / decode_macs["total"]),
    }
    prefill_perf = {
        "ideal_tokens_per_s": peak_macs_per_s / prefill["per_token_average"],
        "target_utilization": args.target_prefill_tok_s / (peak_macs_per_s / prefill["per_token_average"]),
    }

    report = {
        "model": asdict(spec),
        "assumptions": {
            "prompt_len": args.prompt_len,
            "decode_context_len": args.decode_context_len,
            "decode_tokens": args.decode_tokens,
            "prefill_tile_m": args.prefill_tile_m,
            "prefill_kv_block": args.prefill_kv_block,
            "tile_n": args.tile_n,
            "target_decode_tok_s": args.target_decode_tok_s,
            "target_prefill_tok_s": args.target_prefill_tok_s,
            "mac_units": args.mac_units,
            "freq_ghz": args.freq_ghz,
            "sram_mb": args.sram_mb,
            "weight_bits": args.weight_bits,
            "act_bits": args.act_bits,
            "kv_bits": args.kv_bits,
            "softmax_scratch_factor": args.softmax_scratch_factor,
            "include_prefill_lm_head": args.include_prefill_lm_head,
        },
        "params": params,
        "decode": {
            "macs_per_token": decode_macs,
            "performance": decode_perf,
            "bandwidth": decode_bandwidth_report(
                spec,
                params,
                args.decode_context_len,
                args.weight_bits,
                args.kv_bits,
                args.target_decode_tok_s,
            ),
        },
        "prefill": {
            "macs": prefill,
            "performance": prefill_perf,
            "bandwidth": prefill_bandwidth_report(
                spec,
                params,
                args.prompt_len,
                args.prefill_tile_m,
                args.kv_bits,
                args.weight_bits,
                args.target_prefill_tok_s,
                args.prefill_kv_block,
            ),
        },
        "sram": sram_budget_report(
            spec,
            sram_bytes,
            args.act_bits,
            args.weight_bits,
            args.kv_bits,
            args.prefill_tile_m,
            args.tile_n,
            args.softmax_scratch_factor,
        ),
        "mixed_workload": mixed_workload_report(
            prefill["total"],
            decode_macs["total"],
            args.mac_units,
            args.freq_ghz,
            args.prompt_len,
            args.decode_tokens,
        ),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("Model:")
    print(json.dumps(report["model"], indent=2))
    print("\nAssumptions:")
    print(json.dumps(report["assumptions"], indent=2))
    print("\nParameter breakdown:")
    print(json.dumps(report["params"], indent=2))
    print("\nDecode:")
    print(json.dumps(report["decode"], indent=2))
    print("\nPrefill:")
    print(json.dumps(report["prefill"], indent=2))
    print("\nSRAM:")
    print(json.dumps(report["sram"], indent=2))
    print("\nMixed workload:")
    print(json.dumps(report["mixed_workload"], indent=2))


if __name__ == "__main__":
    main()