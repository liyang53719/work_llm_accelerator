from __future__ import annotations

from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask

from backend_interface import BackendInterface, DecodeResult, PrefillResult
from torch_reference_backend import TorchReferenceBackend, move_cache_to_device


class ManualDispatchBackend(BackendInterface):
    def __init__(self, reference_backend: TorchReferenceBackend | None = None) -> None:
        self.reference_backend = TorchReferenceBackend(device="cpu") if reference_backend is None else reference_backend
        self.device = self.reference_backend.device
        self.tokenizer = self.reference_backend.tokenizer
        self.model = self.reference_backend.model

    def _run_manual_dispatch(self, input_ids: torch.Tensor, cache: Any) -> tuple[torch.Tensor, Any]:
        qwen_model = self.model.model
        inputs_embeds = qwen_model.embed_tokens(input_ids.to(self.device))

        past_key_values = move_cache_to_device(cache, self.device)
        if past_key_values is None:
            past_key_values = DynamicCache(config=self.model.config)

        past_seen_tokens = past_key_values.get_seq_length()
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
        position_ids = cache_position.unsqueeze(0)

        mask_kwargs = {
            "config": self.model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": None,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if qwen_model.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = qwen_model.rotary_emb(hidden_states, position_ids)

        with torch.no_grad():
            for decoder_layer in qwen_model.layers[: self.model.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = qwen_model.norm(hidden_states)
            logits = self.model.lm_head(hidden_states)

        return logits.detach().cpu().to(torch.float32), past_key_values

    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        logits, cache = self._run_manual_dispatch(input_ids, cache=None)
        return PrefillResult(logits=logits, cache=cache)

    def decode_step(self, input_ids: torch.Tensor, cache: Any) -> DecodeResult:
        logits, next_cache = self._run_manual_dispatch(input_ids, cache=cache)
        return DecodeResult(logits=logits, cache=next_cache)