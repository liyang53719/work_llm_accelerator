#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from layer_descriptor_builder import (  # noqa: E402
    CONTROL_SCRATCH_BYTES,
    KV_WORKING_SET_BYTES,
    PARTIAL_SUM_BYTES,
    SOFTMAX_SCRATCH_BYTES,
    SRAM_BUDGET_BYTES,
    WEIGHT_BUFFER_BYTES,
    build_decode_descriptors,
    build_prefill_descriptors,
    summary_dict,
)
from qwen_model_spec import load_qwen_model_spec  # noqa: E402


def assert_unique(values: list[int], name: str) -> None:
    if len(values) != len(set(values)):
        raise AssertionError(f"{name} contains overlapping addresses")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate descriptor generation and DDR/SRAM layout invariants.")
    parser.add_argument("--past-seq-len", type=int, default=128)
    parser.add_argument("--prefill-seq-len", type=int, default=256)
    parser.add_argument("--tile-m", type=int, default=64)
    parser.add_argument("--activation-base", type=int, default=0)
    parser.add_argument("--weight-base", type=int, default=1 << 20)
    parser.add_argument("--scale-base", type=int, default=1 << 30)
    parser.add_argument("--kv-base", type=int, default=2 << 30)
    parser.add_argument("--scratch-base", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    spec = load_qwen_model_spec()
    summary = summary_dict(spec)
    if WEIGHT_BUFFER_BYTES + KV_WORKING_SET_BYTES + PARTIAL_SUM_BYTES + SOFTMAX_SCRATCH_BYTES + CONTROL_SCRATCH_BYTES != SRAM_BUDGET_BYTES:
        raise AssertionError("SRAM partition does not sum to 1 MB")

    decode_descriptors = build_decode_descriptors(
        past_seq_len=args.past_seq_len,
        activation_base_addr=args.activation_base,
        weight_base_addr=args.weight_base,
        scale_base_addr=args.scale_base,
        kv_cache_base_addr=args.kv_base,
        scratch_base_addr=args.scratch_base,
        spec=spec,
    )
    prefill_descriptors = build_prefill_descriptors(
        seq_len=args.prefill_seq_len,
        tile_m=args.tile_m,
        activation_base_addr=args.activation_base,
        weight_base_addr=args.weight_base,
        scale_base_addr=args.scale_base,
        kv_cache_base_addr=args.kv_base,
        scratch_base_addr=args.scratch_base,
        spec=spec,
    )

    assert len(decode_descriptors) == spec.num_hidden_layers
    assert len(prefill_descriptors) == spec.num_hidden_layers
    assert all(descriptor.layer_id == index for index, descriptor in enumerate(decode_descriptors))
    assert all(descriptor.layer_id == index for index, descriptor in enumerate(prefill_descriptors))

    assert_unique([descriptor.layer_weights_base_addr for descriptor in decode_descriptors], "decode layer weight bases")
    assert_unique([descriptor.k_cache_base_addr for descriptor in decode_descriptors], "decode K cache bases")
    assert_unique([descriptor.v_cache_base_addr for descriptor in decode_descriptors], "decode V cache bases")

    if decode_descriptors[-1].k_cache_base_addr >= decode_descriptors[0].v_cache_base_addr:
        raise AssertionError("K/V cache regions overlap")

    payload = {
        "summary": summary,
        "decode_descriptor_example": asdict(decode_descriptors[0]),
        "decode_last_layer_example": asdict(decode_descriptors[-1]),
        "prefill_descriptor_example": asdict(prefill_descriptors[0]),
        "prefill_last_layer_example": asdict(prefill_descriptors[-1]),
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("SRAM partition bytes:", summary["sram"])
        print("Decode layer0 descriptor:", payload["decode_descriptor_example"])
        print("Decode last-layer descriptor:", payload["decode_last_layer_example"])
        print("Prefill layer0 descriptor:", payload["prefill_descriptor_example"])
        print("Prefill last-layer descriptor:", payload["prefill_last_layer_example"])
    print("Layout validation PASS")


if __name__ == "__main__":
    main()