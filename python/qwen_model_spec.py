from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT.parent / "module" / "qwen_model"
CONFIG_PATH = MODEL_PATH / "config.json"


@dataclass
class QwenModelSpec:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    rms_norm_eps: float
    max_position_embeddings: int

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def load_qwen_model_spec(config_path: Path = CONFIG_PATH) -> QwenModelSpec:
    with config_path.open("r", encoding="utf-8") as file_obj:
        config = json.load(file_obj)
    return QwenModelSpec(
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config["num_key_value_heads"],
        vocab_size=config["vocab_size"],
        rms_norm_eps=config["rms_norm_eps"],
        max_position_embeddings=config["max_position_embeddings"],
    )