from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask

from backend_interface import BackendInterface, DecodeResult, PrefillResult
from manual_dispatch_backend import ManualDispatchBackend
from torch_reference_backend import SDPA_ATTENTION_BACKEND, get_attention_backend, move_cache_to_device


class DescriptorDispatchBackend(BackendInterface):
    def __init__(
        self,
        manual_backend: ManualDispatchBackend | None = None,
        activation_base_addr: int = 0,
        weight_base_addr: int = 1 << 20,
        scale_base_addr: int = 1 << 30,
        kv_cache_base_addr: int = 2 << 30,
        scratch_base_addr: int = 0,
    ) -> None:
        self.manual_backend = ManualDispatchBackend() if manual_backend is None else manual_backend
        self.reference_backend = self.manual_backend.reference_backend
        self.device = self.manual_backend.device
        self.model = self.manual_backend.model
        self.tokenizer = self.manual_backend.tokenizer
        self.attention_backend = get_attention_backend(self.model)
        if self.attention_backend != SDPA_ATTENTION_BACKEND:
            raise AssertionError(
                f"DescriptorDispatchBackend requires attention backend '{SDPA_ATTENTION_BACKEND}', got '{self.attention_backend}'."
            )
        self.activation_base_addr = activation_base_addr
        self.weight_base_addr = weight_base_addr
        self.scale_base_addr = scale_base_addr
        self.kv_cache_base_addr = kv_cache_base_addr
        self.scratch_base_addr = scratch_base_addr
        self.last_descriptor_trace: dict[str, Any] | None = None

    def _run_with_descriptors(self, input_ids: torch.Tensor, cache: Any, is_prefill: bool) -> tuple[torch.Tensor, Any]:
        from layer_descriptor_builder import build_decode_descriptors, build_prefill_descriptors, default_prefill_tile_config

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

        if is_prefill:
            descriptors = build_prefill_descriptors(
                seq_len=input_ids.shape[1],
                activation_base_addr=self.activation_base_addr,
                weight_base_addr=self.weight_base_addr,
                scale_base_addr=self.scale_base_addr,
                kv_cache_base_addr=self.kv_cache_base_addr,
                scratch_base_addr=self.scratch_base_addr,
                tile_config=default_prefill_tile_config(),
            )
        else:
            descriptors = build_decode_descriptors(
                past_seq_len=past_seen_tokens,
                activation_base_addr=self.activation_base_addr,
                weight_base_addr=self.weight_base_addr,
                scale_base_addr=self.scale_base_addr,
                kv_cache_base_addr=self.kv_cache_base_addr,
                scratch_base_addr=self.scratch_base_addr,
            )

        hidden_states = inputs_embeds
        position_embeddings = qwen_model.rotary_emb(hidden_states, position_ids)

        with torch.no_grad():
            for descriptor in descriptors:
                decoder_layer = qwen_model.layers[descriptor.layer_id]
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

        self.last_descriptor_trace = {
            "mode": "prefill" if is_prefill else "decode",
            "layer_count": len(descriptors),
            "first_descriptor": asdict(descriptors[0]),
            "last_descriptor": asdict(descriptors[-1]),
        }
        return logits.detach().cpu().to(torch.float32), past_key_values

    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        logits, cache = self._run_with_descriptors(input_ids, cache=None, is_prefill=True)
        return PrefillResult(logits=logits, cache=cache)

    def decode_step(self, input_ids: torch.Tensor, cache: Any) -> DecodeResult:
        logits, next_cache = self._run_with_descriptors(input_ids, cache=cache, is_prefill=False)
        return DecodeResult(logits=logits, cache=next_cache)