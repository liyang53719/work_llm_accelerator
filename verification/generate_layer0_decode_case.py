#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch_reference_backend import TorchReferenceBackend, snapshot_cache


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen2.5-1.5B layer0 decode-step reference tensors.")
    parser.add_argument("--prompt", type=str, default="Explain the purpose of blocked attention in one sentence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-name", type=str, default="layer0_decode_case")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = TorchReferenceBackend(device="cpu")
    tokenizer = backend.tokenizer
    model = backend.model
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        prefill_outputs = model(
            input_ids=input_ids.to(backend.device),
            use_cache=True,
            return_dict=True,
        )

    decode_input_ids = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
    prefill_cache_snapshot = snapshot_cache(prefill_outputs.past_key_values)

    captures: dict[str, torch.Tensor] = {}

    def layer0_pre_hook(_module, inputs):
        captures["layer0_input"] = inputs[0].detach().cpu().to(torch.float32)

    def layer0_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captures["layer0_output"] = hidden.detach().cpu().to(torch.float32)

    def capture_tensor(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captures[name] = tensor.detach().cpu().to(torch.float32)

        return hook

    layer0_pre_handle = model.model.layers[0].register_forward_pre_hook(layer0_pre_hook)
    layer0_handle = model.model.layers[0].register_forward_hook(layer0_hook)
    input_norm_handle = model.model.layers[0].input_layernorm.register_forward_hook(capture_tensor("input_layernorm"))
    self_attn_handle = model.model.layers[0].self_attn.register_forward_hook(capture_tensor("self_attn_output"))
    q_proj_handle = model.model.layers[0].self_attn.q_proj.register_forward_hook(capture_tensor("q_proj"))
    k_proj_handle = model.model.layers[0].self_attn.k_proj.register_forward_hook(capture_tensor("k_proj"))
    v_proj_handle = model.model.layers[0].self_attn.v_proj.register_forward_hook(capture_tensor("v_proj"))
    o_proj_handle = model.model.layers[0].self_attn.o_proj.register_forward_hook(capture_tensor("o_proj"))
    post_norm_handle = model.model.layers[0].post_attention_layernorm.register_forward_hook(capture_tensor("post_attention_layernorm"))
    gate_proj_handle = model.model.layers[0].mlp.gate_proj.register_forward_hook(capture_tensor("gate_proj"))
    up_proj_handle = model.model.layers[0].mlp.up_proj.register_forward_hook(capture_tensor("up_proj"))
    down_proj_handle = model.model.layers[0].mlp.down_proj.register_forward_hook(capture_tensor("down_proj"))

    try:
        with torch.no_grad():
            decode_outputs = model(
                input_ids=decode_input_ids.to(backend.device),
                past_key_values=prefill_outputs.past_key_values,
                use_cache=True,
                return_dict=True,
            )
    finally:
        layer0_pre_handle.remove()
        layer0_handle.remove()
        input_norm_handle.remove()
        self_attn_handle.remove()
        q_proj_handle.remove()
        k_proj_handle.remove()
        v_proj_handle.remove()
        o_proj_handle.remove()
        post_norm_handle.remove()
        gate_proj_handle.remove()
        up_proj_handle.remove()
        down_proj_handle.remove()

    captures["attention_residual"] = captures["layer0_input"] + captures["self_attn_output"]
    captures["silu_mul"] = F.silu(captures["gate_proj"]) * captures["up_proj"]

    decode_cache_snapshot = snapshot_cache(decode_outputs.past_key_values)
    prefill_layer0_k = prefill_cache_snapshot[0][0].numpy()
    prefill_layer0_v = prefill_cache_snapshot[0][1].numpy()
    decode_layer0_k = decode_cache_snapshot[0][0].numpy()
    decode_layer0_v = decode_cache_snapshot[0][1].numpy()

    npz_path = args.output_dir / f"{args.case_name}.npz"
    json_path = args.output_dir / f"{args.case_name}.json"

    np.savez_compressed(
        npz_path,
        prompt_input_ids=input_ids.detach().cpu().numpy(),
        decode_input_ids=decode_input_ids.detach().cpu().numpy(),
        layer0_input=captures["layer0_input"].numpy(),
        input_layernorm=captures["input_layernorm"].numpy(),
        q_proj=captures["q_proj"].numpy(),
        k_proj=captures["k_proj"].numpy(),
        v_proj=captures["v_proj"].numpy(),
        self_attn_output=captures["self_attn_output"].numpy(),
        o_proj=captures["o_proj"].numpy(),
        attention_residual=captures["attention_residual"].numpy(),
        post_attention_layernorm=captures["post_attention_layernorm"].numpy(),
        gate_proj=captures["gate_proj"].numpy(),
        up_proj=captures["up_proj"].numpy(),
        silu_mul=captures["silu_mul"].numpy(),
        down_proj=captures["down_proj"].numpy(),
        layer0_output=captures["layer0_output"].numpy(),
        decode_logits=decode_outputs.logits.detach().cpu().to(torch.float32).numpy(),
        prefill_layer0_k_cache=prefill_layer0_k,
        prefill_layer0_v_cache=prefill_layer0_v,
        decode_layer0_k_cache=decode_layer0_k,
        decode_layer0_v_cache=decode_layer0_v,
    )

    metadata = {
        "prompt": args.prompt,
        "prefill_seq_len": int(input_ids.shape[1]),
        "decode_seq_len": int(decode_input_ids.shape[1]),
        "decode_token_id": int(decode_input_ids.item()),
        "tensors": sorted(captures.keys()),
        "npz_path": str(npz_path),
    }
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote decode reference case to {npz_path}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()