#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from torch_reference_backend import TorchReferenceBackend


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "reference_params"


def as_numpy(parameter: torch.Tensor | None, shape: tuple[int, ...]) -> np.ndarray:
    if parameter is None:
        return np.zeros(shape, dtype=np.float32)
    return parameter.detach().cpu().to(torch.float32).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen2.5-1.5B layer0 prefill parameters.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--file-name", type=str, default="layer0_prefill_params")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = TorchReferenceBackend(device="cpu")
    layer = backend.model.model.layers[0]

    npz_path = args.output_dir / f"{args.file_name}.npz"
    json_path = args.output_dir / f"{args.file_name}.json"

    np.savez_compressed(
        npz_path,
        input_layernorm_weight=as_numpy(layer.input_layernorm.weight, (1536,)),
        q_weight=as_numpy(layer.self_attn.q_proj.weight, (1536, 1536)),
        q_bias=as_numpy(layer.self_attn.q_proj.bias, (1536,)),
        k_weight=as_numpy(layer.self_attn.k_proj.weight, (256, 1536)),
        k_bias=as_numpy(layer.self_attn.k_proj.bias, (256,)),
        v_weight=as_numpy(layer.self_attn.v_proj.weight, (256, 1536)),
        v_bias=as_numpy(layer.self_attn.v_proj.bias, (256,)),
        o_weight=as_numpy(layer.self_attn.o_proj.weight, (1536, 1536)),
        o_bias=as_numpy(layer.self_attn.o_proj.bias, (1536,)),
        post_attention_layernorm_weight=as_numpy(layer.post_attention_layernorm.weight, (1536,)),
        gate_weight=as_numpy(layer.mlp.gate_proj.weight, (8960, 1536)),
        gate_bias=as_numpy(layer.mlp.gate_proj.bias, (8960,)),
        up_weight=as_numpy(layer.mlp.up_proj.weight, (8960, 1536)),
        up_bias=as_numpy(layer.mlp.up_proj.bias, (8960,)),
        down_weight=as_numpy(layer.mlp.down_proj.weight, (1536, 8960)),
        down_bias=as_numpy(layer.mlp.down_proj.bias, (1536,)),
        rms_eps=np.asarray([backend.model.config.rms_norm_eps], dtype=np.float32),
    )

    metadata = {
        "npz_path": str(npz_path),
        "rms_eps": float(backend.model.config.rms_norm_eps),
        "layer_index": 0,
    }
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote parameter set to {npz_path}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()