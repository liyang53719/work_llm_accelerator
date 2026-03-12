#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from torch_reference_backend import TorchReferenceBackend, snapshot_cache


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp" / "reference_cases"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Qwen2.5-1.5B layer0 prefill reference tensors.")
    parser.add_argument("--prompt", type=str, default="Explain the purpose of blocked attention in one sentence.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-name", type=str, default="layer0_prefill_case")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    backend = TorchReferenceBackend(device="cpu")
    tokenizer = backend.tokenizer
    model = backend.model
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids

    captures: dict[str, torch.Tensor] = {}

    def embedding_hook(_module, _inputs, output):
        captures["embedding_output"] = output.detach().cpu().to(torch.float32)

    def layer0_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captures["layer0_output"] = hidden.detach().cpu().to(torch.float32)

    embed_handle = model.model.embed_tokens.register_forward_hook(embedding_hook)
    layer0_handle = model.model.layers[0].register_forward_hook(layer0_hook)

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(backend.device),
                use_cache=True,
                return_dict=True,
            )
    finally:
        embed_handle.remove()
        layer0_handle.remove()

    cache_snapshot = snapshot_cache(outputs.past_key_values)
    layer0_k = cache_snapshot[0][0].numpy()
    layer0_v = cache_snapshot[0][1].numpy()

    npz_path = args.output_dir / f"{args.case_name}.npz"
    json_path = args.output_dir / f"{args.case_name}.json"

    np.savez_compressed(
        npz_path,
        input_ids=input_ids.detach().cpu().numpy(),
        embedding_output=captures["embedding_output"].numpy(),
        layer0_output=captures["layer0_output"].numpy(),
        final_logits=outputs.logits.detach().cpu().to(torch.float32).numpy(),
        layer0_k_cache=layer0_k,
        layer0_v_cache=layer0_v,
    )

    metadata = {
        "prompt": args.prompt,
        "seq_len": int(input_ids.shape[1]),
        "hidden_size": int(captures["embedding_output"].shape[-1]),
        "npz_path": str(npz_path),
    }
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote reference case to {npz_path}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()