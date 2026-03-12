from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend_interface import BackendInterface, DecodeResult, PrefillResult


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from qwen_model_spec import MODEL_PATH  # noqa: E402


def snapshot_cache(cache: Any) -> Any:
    if cache is None:
        return None
    detached_layers = []
    for layer_cache in cache:
        detached_layers.append(tuple(tensor.detach().cpu().to(torch.float32) for tensor in layer_cache))
    return tuple(detached_layers)


def move_cache_to_device(cache: Any, device: torch.device) -> Any:
    if cache is None:
        return None
    if hasattr(cache, "get_seq_length"):
        return cache
    if hasattr(cache, "to"):
        return cache.to(device)
    moved_layers = []
    for layer_cache in cache:
        moved_layers.append(tuple(tensor.to(device=device) for tensor in layer_cache))
    return tuple(moved_layers)


class TorchReferenceBackend(BackendInterface):
    def __init__(self, model_path: Path = MODEL_PATH, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.to(self.device)

    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                use_cache=True,
                return_dict=True,
            )
        return PrefillResult(
            logits=outputs.logits.detach().cpu().to(torch.float32),
            cache=outputs.past_key_values,
        )

    def decode_step(self, input_ids: torch.Tensor, cache: Any) -> DecodeResult:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                past_key_values=move_cache_to_device(cache, self.device),
                use_cache=True,
                return_dict=True,
            )
        return DecodeResult(
            logits=outputs.logits.detach().cpu().to(torch.float32),
            cache=outputs.past_key_values,
        )