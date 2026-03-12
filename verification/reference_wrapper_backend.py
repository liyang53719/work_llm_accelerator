from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import create_causal_mask, create_sliding_window_causal_mask

from backend_interface import BackendInterface, DecodeResult, PrefillResult
from torch_reference_backend import TorchReferenceBackend, move_cache_to_device


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFILL_LIB_PATH = PROJECT_ROOT / "tmp" / "host_libs" / "libqwen_prefill_stub.so"
DEFAULT_DECODE_LIB_PATH = PROJECT_ROOT / "tmp" / "host_libs" / "libqwen_decode_stub.so"


def as_numpy(parameter: torch.Tensor | None, shape: tuple[int, ...]) -> np.ndarray:
    if parameter is None:
        return np.zeros(shape, dtype=np.float32)
    return np.ascontiguousarray(parameter.detach().cpu().to(torch.float32).numpy())


def numpy_diff(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    delta = np.abs(lhs.astype(np.float32) - rhs.astype(np.float32))
    max_flat_index = int(delta.argmax())
    max_index = tuple(int(index) for index in np.unravel_index(max_flat_index, delta.shape))
    return {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "max_index": max_index,
        "lhs_at_max": float(lhs[max_index]),
        "rhs_at_max": float(rhs[max_index]),
    }


def round_bfloat16_array(array: np.ndarray) -> np.ndarray:
    rounded = np.ascontiguousarray(array.astype(np.float32)).copy()
    bits = rounded.view(np.uint32)
    lsb = (bits >> 16) & 1
    bits += np.uint32(0x7FFF) + lsb
    bits &= np.uint32(0xFFFF0000)
    return rounded


class ReferenceWrapperBackend(BackendInterface):
    def __init__(
        self,
        reference_backend: TorchReferenceBackend | None = None,
        prefill_lib_path: Path = DEFAULT_PREFILL_LIB_PATH,
        decode_lib_path: Path = DEFAULT_DECODE_LIB_PATH,
    ) -> None:
        self.reference_backend = reference_backend or TorchReferenceBackend(device="cpu")
        self.device = self.reference_backend.device
        self.tokenizer = self.reference_backend.tokenizer
        self.model = self.reference_backend.model
        self.prefill_lib = ctypes.CDLL(str(prefill_lib_path))
        self.decode_lib = ctypes.CDLL(str(decode_lib_path))
        self.hidden_size = self.model.config.hidden_size
        self.intermediate_size = self.model.config.intermediate_size
        self.num_hidden_layers = self.model.config.num_hidden_layers
        self.num_key_value_heads = self.model.config.num_key_value_heads
        self.head_dim = self.hidden_size // self.model.config.num_attention_heads
        self.kv_width = self.num_key_value_heads * self.head_dim
        self.rms_eps = float(self.model.config.rms_norm_eps)
        self.last_layer_trace: dict[str, Any] | None = None
        self._configure_abis()

    def _configure_abis(self) -> None:
        float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")

        self.prefill_func = self.prefill_lib.qwen_prefill_layer0_reference_forward_with_cache
        self.prefill_func.argtypes = [
            float_ptr,
            ctypes.c_int,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            ctypes.c_float,
            float_ptr,
            float_ptr,
            float_ptr,
        ]
        self.prefill_func.restype = ctypes.c_int

        self.decode_func = self.decode_lib.qwen_decode_layer0_reference_forward
        self.decode_func.argtypes = [
            float_ptr,
            ctypes.c_int,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            ctypes.c_float,
            float_ptr,
            float_ptr,
            float_ptr,
        ]
        self.decode_func.restype = ctypes.c_int

    def _layer_parameters(self, layer_id: int) -> dict[str, np.ndarray]:
        layer = self.model.model.layers[layer_id]
        return {
            "input_layernorm_weight": as_numpy(layer.input_layernorm.weight, (self.hidden_size,)),
            "q_weight": as_numpy(layer.self_attn.q_proj.weight, (self.hidden_size, self.hidden_size)),
            "q_bias": as_numpy(layer.self_attn.q_proj.bias, (self.hidden_size,)),
            "k_weight": as_numpy(layer.self_attn.k_proj.weight, (self.kv_width, self.hidden_size)),
            "k_bias": as_numpy(layer.self_attn.k_proj.bias, (self.kv_width,)),
            "v_weight": as_numpy(layer.self_attn.v_proj.weight, (self.kv_width, self.hidden_size)),
            "v_bias": as_numpy(layer.self_attn.v_proj.bias, (self.kv_width,)),
            "o_weight": as_numpy(layer.self_attn.o_proj.weight, (self.hidden_size, self.hidden_size)),
            "o_bias": as_numpy(layer.self_attn.o_proj.bias, (self.hidden_size,)),
            "post_attention_layernorm_weight": as_numpy(layer.post_attention_layernorm.weight, (self.hidden_size,)),
            "gate_weight": as_numpy(layer.mlp.gate_proj.weight, (self.intermediate_size, self.hidden_size)),
            "gate_bias": as_numpy(layer.mlp.gate_proj.bias, (self.intermediate_size,)),
            "up_weight": as_numpy(layer.mlp.up_proj.weight, (self.intermediate_size, self.hidden_size)),
            "up_bias": as_numpy(layer.mlp.up_proj.bias, (self.intermediate_size,)),
            "down_weight": as_numpy(layer.mlp.down_proj.weight, (self.hidden_size, self.intermediate_size)),
            "down_bias": as_numpy(layer.mlp.down_proj.bias, (self.hidden_size,)),
        }

    def _finalize_logits(self, hidden_states: np.ndarray) -> torch.Tensor:
        hidden_tensor = torch.from_numpy(hidden_states).unsqueeze(0).to(
            device=self.device,
            dtype=self.model.model.norm.weight.dtype,
        )
        with torch.no_grad():
            normalized = self.model.model.norm(hidden_tensor)
            logits = self.model.lm_head(normalized)
        return logits.detach().cpu().to(torch.float32)

    def _prepare_reference_layer_state(
        self,
        input_ids: torch.Tensor,
        cache: Any,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Any, dict[str, torch.Tensor]]:
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
        return qwen_model, hidden_states, position_embeddings, cache_position, past_key_values, causal_mask_mapping

    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(input_ids.to(self.device)).detach().cpu().to(torch.float32).squeeze(0).numpy()

        seq_len = int(input_ids.shape[1])
        cache_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        trace_layers: list[dict[str, int]] = []

        for layer_id in range(self.num_hidden_layers):
            params = self._layer_parameters(layer_id)
            output_sequence = np.zeros((seq_len, self.hidden_size), dtype=np.float32)
            k_cache = np.zeros((self.num_key_value_heads, seq_len, self.head_dim), dtype=np.float32)
            v_cache = np.zeros((self.num_key_value_heads, seq_len, self.head_dim), dtype=np.float32)

            status = self.prefill_func(
                np.ascontiguousarray(hidden_states.reshape(-1)),
                seq_len,
                params["input_layernorm_weight"],
                params["q_weight"].reshape(-1),
                params["q_bias"],
                params["k_weight"].reshape(-1),
                params["k_bias"],
                params["v_weight"].reshape(-1),
                params["v_bias"],
                params["o_weight"].reshape(-1),
                params["o_bias"],
                params["post_attention_layernorm_weight"],
                params["gate_weight"].reshape(-1),
                params["gate_bias"],
                params["up_weight"].reshape(-1),
                params["up_bias"],
                params["down_weight"].reshape(-1),
                params["down_bias"],
                self.rms_eps,
                output_sequence.reshape(-1),
                k_cache.reshape(-1),
                v_cache.reshape(-1),
            )
            if status != 0:
                raise RuntimeError(f"Prefill wrapper failed at layer {layer_id} with status {status}")

            hidden_states = round_bfloat16_array(output_sequence)
            cache_layers.append((torch.from_numpy(k_cache.copy()).unsqueeze(0), torch.from_numpy(v_cache.copy()).unsqueeze(0)))
            trace_layers.append({"layer_id": layer_id, "seq_len": seq_len})

        self.last_layer_trace = {
            "mode": "prefill",
            "layer_count": self.num_hidden_layers,
            "first_layer": trace_layers[0],
            "last_layer": trace_layers[-1],
        }
        return PrefillResult(logits=self._finalize_logits(hidden_states), cache=tuple(cache_layers))

    def decode_step(self, input_ids: torch.Tensor, cache: Any) -> DecodeResult:
        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(input_ids.to(self.device)).detach().cpu().to(torch.float32).squeeze(0).squeeze(0).numpy()

        if cache is None:
            raise ValueError("ReferenceWrapperBackend.decode_step requires a prefill cache.")

        past_seq_len = int(cache[0][0].shape[2])
        next_seq_len = past_seq_len + 1
        next_cache_layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        trace_layers: list[dict[str, int]] = []

        for layer_id in range(self.num_hidden_layers):
            params = self._layer_parameters(layer_id)
            past_k_cache = np.ascontiguousarray(cache[layer_id][0].detach().cpu().to(torch.float32).numpy().reshape(-1))
            past_v_cache = np.ascontiguousarray(cache[layer_id][1].detach().cpu().to(torch.float32).numpy().reshape(-1))
            output_token = np.zeros(self.hidden_size, dtype=np.float32)
            next_k_cache = np.zeros((self.num_key_value_heads, next_seq_len, self.head_dim), dtype=np.float32)
            next_v_cache = np.zeros((self.num_key_value_heads, next_seq_len, self.head_dim), dtype=np.float32)

            status = self.decode_func(
                np.ascontiguousarray(hidden_states.reshape(-1)),
                past_seq_len,
                past_k_cache,
                past_v_cache,
                params["input_layernorm_weight"],
                params["q_weight"].reshape(-1),
                params["q_bias"],
                params["k_weight"].reshape(-1),
                params["k_bias"],
                params["v_weight"].reshape(-1),
                params["v_bias"],
                params["o_weight"].reshape(-1),
                params["o_bias"],
                params["post_attention_layernorm_weight"],
                params["gate_weight"].reshape(-1),
                params["gate_bias"],
                params["up_weight"].reshape(-1),
                params["up_bias"],
                params["down_weight"].reshape(-1),
                params["down_bias"],
                self.rms_eps,
                output_token,
                next_k_cache.reshape(-1),
                next_v_cache.reshape(-1),
            )
            if status != 0:
                raise RuntimeError(f"Decode wrapper failed at layer {layer_id} with status {status}")

            hidden_states = round_bfloat16_array(output_token)
            next_cache_layers.append((torch.from_numpy(next_k_cache.copy()).unsqueeze(0), torch.from_numpy(next_v_cache.copy()).unsqueeze(0)))
            trace_layers.append({"layer_id": layer_id, "past_seq_len": past_seq_len})

        self.last_layer_trace = {
            "mode": "decode",
            "layer_count": self.num_hidden_layers,
            "first_layer": trace_layers[0],
            "last_layer": trace_layers[-1],
        }
        return DecodeResult(logits=self._finalize_logits(hidden_states.reshape(1, self.hidden_size)), cache=tuple(next_cache_layers))

    def diagnose_prefill(self, input_ids: torch.Tensor) -> dict[str, Any]:
        qwen_model, hidden_states_ref, position_embeddings, cache_position, past_key_values_ref, causal_mask_mapping = self._prepare_reference_layer_state(
            input_ids,
            cache=None,
        )
        hidden_states_wrapper = hidden_states_ref.detach().cpu().to(torch.float32).squeeze(0).numpy()
        seq_len = int(input_ids.shape[1])
        layer_reports: list[dict[str, Any]] = []

        with torch.no_grad():
            for layer_id in range(self.num_hidden_layers):
                decoder_layer = qwen_model.layers[layer_id]
                hidden_states_ref = decoder_layer(
                    hidden_states_ref,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=cache_position.unsqueeze(0),
                    past_key_values=past_key_values_ref,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                params = self._layer_parameters(layer_id)
                output_sequence = np.zeros((seq_len, self.hidden_size), dtype=np.float32)
                k_cache = np.zeros((self.num_key_value_heads, seq_len, self.head_dim), dtype=np.float32)
                v_cache = np.zeros((self.num_key_value_heads, seq_len, self.head_dim), dtype=np.float32)
                status = self.prefill_func(
                    np.ascontiguousarray(hidden_states_wrapper.reshape(-1)),
                    seq_len,
                    params["input_layernorm_weight"],
                    params["q_weight"].reshape(-1),
                    params["q_bias"],
                    params["k_weight"].reshape(-1),
                    params["k_bias"],
                    params["v_weight"].reshape(-1),
                    params["v_bias"],
                    params["o_weight"].reshape(-1),
                    params["o_bias"],
                    params["post_attention_layernorm_weight"],
                    params["gate_weight"].reshape(-1),
                    params["gate_bias"],
                    params["up_weight"].reshape(-1),
                    params["up_bias"],
                    params["down_weight"].reshape(-1),
                    params["down_bias"],
                    self.rms_eps,
                    output_sequence.reshape(-1),
                    k_cache.reshape(-1),
                    v_cache.reshape(-1),
                )
                if status != 0:
                    raise RuntimeError(f"Prefill wrapper failed at layer {layer_id} with status {status}")

                rounded_output_sequence = round_bfloat16_array(output_sequence)
                reference_hidden = hidden_states_ref.detach().cpu().to(torch.float32).squeeze(0).numpy()
                reference_k = past_key_values_ref[layer_id][0].detach().cpu().to(torch.float32).squeeze(0).numpy()
                reference_v = past_key_values_ref[layer_id][1].detach().cpu().to(torch.float32).squeeze(0).numpy()
                layer_reports.append(
                    {
                        "layer_id": layer_id,
                        "output_diff": numpy_diff(rounded_output_sequence, reference_hidden),
                        "k_cache_diff": numpy_diff(k_cache, reference_k),
                        "v_cache_diff": numpy_diff(v_cache, reference_v),
                    }
                )
                hidden_states_wrapper = rounded_output_sequence

        return {"mode": "prefill", "layer_reports": layer_reports}

    def diagnose_decode(self, input_ids: torch.Tensor, wrapper_cache: Any, reference_cache: Any) -> dict[str, Any]:
        qwen_model, hidden_states_ref, position_embeddings, cache_position, past_key_values_ref, causal_mask_mapping = self._prepare_reference_layer_state(
            input_ids,
            cache=reference_cache,
        )
        hidden_states_wrapper = hidden_states_ref.detach().cpu().to(torch.float32).squeeze(0).squeeze(0).numpy()
        past_seq_len = int(wrapper_cache[0][0].shape[2])
        next_seq_len = past_seq_len + 1
        layer_reports: list[dict[str, Any]] = []

        with torch.no_grad():
            for layer_id in range(self.num_hidden_layers):
                decoder_layer = qwen_model.layers[layer_id]
                hidden_states_ref = decoder_layer(
                    hidden_states_ref,
                    attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                    position_ids=cache_position.unsqueeze(0),
                    past_key_values=past_key_values_ref,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

                params = self._layer_parameters(layer_id)
                past_k_cache = np.ascontiguousarray(wrapper_cache[layer_id][0].detach().cpu().to(torch.float32).numpy().reshape(-1))
                past_v_cache = np.ascontiguousarray(wrapper_cache[layer_id][1].detach().cpu().to(torch.float32).numpy().reshape(-1))
                output_token = np.zeros(self.hidden_size, dtype=np.float32)
                next_k_cache = np.zeros((self.num_key_value_heads, next_seq_len, self.head_dim), dtype=np.float32)
                next_v_cache = np.zeros((self.num_key_value_heads, next_seq_len, self.head_dim), dtype=np.float32)
                status = self.decode_func(
                    np.ascontiguousarray(hidden_states_wrapper.reshape(-1)),
                    past_seq_len,
                    past_k_cache,
                    past_v_cache,
                    params["input_layernorm_weight"],
                    params["q_weight"].reshape(-1),
                    params["q_bias"],
                    params["k_weight"].reshape(-1),
                    params["k_bias"],
                    params["v_weight"].reshape(-1),
                    params["v_bias"],
                    params["o_weight"].reshape(-1),
                    params["o_bias"],
                    params["post_attention_layernorm_weight"],
                    params["gate_weight"].reshape(-1),
                    params["gate_bias"],
                    params["up_weight"].reshape(-1),
                    params["up_bias"],
                    params["down_weight"].reshape(-1),
                    params["down_bias"],
                    self.rms_eps,
                    output_token,
                    next_k_cache.reshape(-1),
                    next_v_cache.reshape(-1),
                )
                if status != 0:
                    raise RuntimeError(f"Decode wrapper failed at layer {layer_id} with status {status}")

                rounded_output_token = round_bfloat16_array(output_token)
                reference_hidden = hidden_states_ref.detach().cpu().to(torch.float32).squeeze(0).squeeze(0).numpy()
                reference_k = past_key_values_ref[layer_id][0].detach().cpu().to(torch.float32).squeeze(0).numpy()
                reference_v = past_key_values_ref[layer_id][1].detach().cpu().to(torch.float32).squeeze(0).numpy()
                layer_reports.append(
                    {
                        "layer_id": layer_id,
                        "output_diff": numpy_diff(rounded_output_token, reference_hidden),
                        "k_cache_diff": numpy_diff(next_k_cache, reference_k),
                        "v_cache_diff": numpy_diff(next_v_cache, reference_v),
                    }
                )
                hidden_states_wrapper = rounded_output_token

        return {"mode": "decode", "layer_reports": layer_reports}
