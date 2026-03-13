from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backend_interface import BackendInterface, DecodeResult, PrefillResult
from reference_wrapper_backend import DEFAULT_DECODE_LIB_PATH, DEFAULT_PREFILL_LIB_PATH, round_bfloat16_array
from torch_reference_backend import TorchReferenceBackend


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
if str(PYTHON_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PYTHON_DIR))

from layer_descriptor_builder import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    build_layer_parameter_layout,
    default_prefill_tile_config,
    load_qwen_model_spec,
)


def as_numpy(parameter: torch.Tensor, shape: tuple[int, ...]) -> np.ndarray:
    return np.ascontiguousarray(parameter.detach().cpu().to(torch.float32).numpy().reshape(shape))


def quantize_int4_per_channel(weight: torch.Tensor, out_dim: int, in_dim: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = as_numpy(weight, (out_dim, in_dim))
    max_abs = np.max(np.abs(matrix), axis=1)
    scales = np.where(max_abs > 0.0, max_abs / 7.0, 1.0).astype(np.float32)
    quantized = np.rint(matrix / scales[:, None]).astype(np.int32)
    quantized = np.clip(quantized, -8, 7)
    nibbles = np.where(quantized < 0, quantized + 16, quantized).astype(np.uint8).reshape(-1)
    if nibbles.size % 2 != 0:
        nibbles = np.concatenate([nibbles, np.zeros(1, dtype=np.uint8)])
    packed = np.bitwise_or(nibbles[0::2], np.left_shift(nibbles[1::2], 4)).astype(np.uint8)
    return np.ascontiguousarray(packed), np.ascontiguousarray(scales)


class RealHostTopBackend(BackendInterface):
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
        self.attention_backend = self.reference_backend.attention_backend
        self.prefill_lib = ctypes.CDLL(str(prefill_lib_path))
        self.decode_lib = ctypes.CDLL(str(decode_lib_path))
        self.spec = load_qwen_model_spec()
        self.param_layout = build_layer_parameter_layout(self.spec)
        self.hidden_size = self.spec.hidden_size
        self.intermediate_size = self.spec.intermediate_size
        self.num_hidden_layers = self.spec.num_hidden_layers
        self.num_key_value_heads = self.spec.num_key_value_heads
        self.head_dim = self.hidden_size // self.spec.num_attention_heads
        self.kv_width = self.num_key_value_heads * self.head_dim
        self.prefill_tile_config = default_prefill_tile_config()
        self.backend_metadata = {
            "path": "top-wrapper",
            "quantization": "int4-per-output-channel",
            "prefill_tile_config": {
                "attention": self.prefill_tile_config.attention.__dict__,
                "mlp": self.prefill_tile_config.mlp.__dict__,
            },
        }
        self.last_layer_trace: dict[str, Any] | None = None
        self._configure_abis()

    def _configure_abis(self) -> None:
        float_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        byte_ptr = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
        int_ptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")

        self.prefill_func = self.prefill_lib.qwen_prefill_top_smoke_forward
        self.prefill_func.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            byte_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            byte_ptr,
            float_ptr,
            int_ptr,
            float_ptr,
            float_ptr,
        ]
        self.prefill_func.restype = ctypes.c_int

        self.decode_func = self.decode_lib.qwen_decode_top_smoke_forward
        self.decode_func.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            byte_ptr,
            float_ptr,
            float_ptr,
            float_ptr,
            byte_ptr,
            float_ptr,
            int_ptr,
            float_ptr,
            float_ptr,
        ]
        self.decode_func.restype = ctypes.c_int

    def _finalize_logits(self, hidden_states: np.ndarray) -> torch.Tensor:
        hidden_tensor = torch.from_numpy(hidden_states).unsqueeze(0).to(
            device=self.device,
            dtype=self.model.model.norm.weight.dtype,
        )
        with torch.no_grad():
            normalized = self.model.model.norm(hidden_tensor)
            logits = self.model.lm_head(normalized)
        return logits.detach().cpu().to(torch.float32)

    def _pack_layer_parameters(self, layer_id: int) -> tuple[np.ndarray, np.ndarray]:
        layer = self.model.model.layers[layer_id]
        layout = self.param_layout
        weight_ddr = np.zeros(layout.total_parameter_bytes, dtype=np.uint8)
        scale_ddr = np.zeros((layout.total_parameter_bytes + 3) // 4, dtype=np.float32)

        def copy_scale(offset_bytes: int, values: np.ndarray) -> None:
            start = offset_bytes // 4
            scale_ddr[start : start + values.size] = values.reshape(-1)

        def copy_weight(start_offset: int, end_offset: int, packed: np.ndarray) -> None:
            weight_ddr[start_offset:end_offset] = packed.reshape(-1)

        q_packed, q_scales = quantize_int4_per_channel(layer.self_attn.q_proj.weight, self.hidden_size, self.hidden_size)
        k_packed, k_scales = quantize_int4_per_channel(layer.self_attn.k_proj.weight, self.kv_width, self.hidden_size)
        v_packed, v_scales = quantize_int4_per_channel(layer.self_attn.v_proj.weight, self.kv_width, self.hidden_size)
        o_packed, o_scales = quantize_int4_per_channel(layer.self_attn.o_proj.weight, self.hidden_size, self.hidden_size)
        gate_packed, gate_scales = quantize_int4_per_channel(layer.mlp.gate_proj.weight, self.intermediate_size, self.hidden_size)
        up_packed, up_scales = quantize_int4_per_channel(layer.mlp.up_proj.weight, self.intermediate_size, self.hidden_size)
        down_packed, down_scales = quantize_int4_per_channel(layer.mlp.down_proj.weight, self.hidden_size, self.intermediate_size)

        copy_scale(layout.input_layernorm_weight_offset_bytes, as_numpy(layer.input_layernorm.weight, (self.hidden_size,)))
        copy_scale(layout.post_attention_layernorm_weight_offset_bytes, as_numpy(layer.post_attention_layernorm.weight, (self.hidden_size,)))
        copy_scale(layout.q_bias_offset_bytes, as_numpy(layer.self_attn.q_proj.bias, (self.hidden_size,)))
        copy_scale(layout.k_bias_offset_bytes, as_numpy(layer.self_attn.k_proj.bias, (self.kv_width,)))
        copy_scale(layout.v_bias_offset_bytes, as_numpy(layer.self_attn.v_proj.bias, (self.kv_width,)))
        copy_weight(layout.q_weight_offset_bytes, layout.k_weight_offset_bytes, q_packed)
        copy_weight(layout.k_weight_offset_bytes, layout.v_weight_offset_bytes, k_packed)
        copy_weight(layout.v_weight_offset_bytes, layout.o_weight_offset_bytes, v_packed)
        copy_weight(layout.o_weight_offset_bytes, layout.post_attention_layernorm_weight_offset_bytes, o_packed)
        copy_weight(layout.gate_weight_offset_bytes, layout.up_weight_offset_bytes, gate_packed)
        copy_weight(layout.up_weight_offset_bytes, layout.down_weight_offset_bytes, up_packed)
        copy_weight(layout.down_weight_offset_bytes, layout.q_bias_offset_bytes, down_packed)
        copy_scale(layout.q_scale_offset_bytes, q_scales)
        copy_scale(layout.k_scale_offset_bytes, k_scales)
        copy_scale(layout.v_scale_offset_bytes, v_scales)
        copy_scale(layout.o_scale_offset_bytes, o_scales)
        copy_scale(layout.gate_scale_offset_bytes, gate_scales)
        copy_scale(layout.up_scale_offset_bytes, up_scales)
        copy_scale(layout.down_scale_offset_bytes, down_scales)
        return np.ascontiguousarray(weight_ddr), np.ascontiguousarray(scale_ddr)

    def _scratch_buffers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros(1, dtype=np.uint8),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.int32),
            np.zeros(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
        )

    def _kv_ddr_to_cache_tensor(self, flat_cache: np.ndarray, seq_len: int) -> np.ndarray:
        token_major = flat_cache.reshape(seq_len, self.num_key_value_heads, self.head_dim)
        return np.transpose(token_major, (1, 0, 2)).copy()

    def _cache_tensor_to_kv_ddr(self, cache_tensor: torch.Tensor, seq_len: int) -> np.ndarray:
        head_major = cache_tensor.detach().cpu().to(torch.float32).numpy().reshape(self.num_key_value_heads, seq_len, self.head_dim)
        token_major = np.transpose(head_major, (1, 0, 2))
        return np.ascontiguousarray(token_major.reshape(-1))

    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(input_ids.to(self.device)).detach().cpu().to(torch.float32).squeeze(0).numpy()

        seq_len = int(input_ids.shape[1])
        attention_tiles = self.prefill_tile_config.attention
        mlp_tiles = self.prefill_tile_config.mlp
        trace_layers: list[dict[str, int]] = []
        cache_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_id in range(self.num_hidden_layers):
            weight_ddr, scale_ddr = self._pack_layer_parameters(layer_id)
            activation_stride = seq_len * self.hidden_size
            kv_stride = seq_len * self.kv_width
            activation_ddr = np.zeros(activation_stride * 2, dtype=np.float32)
            activation_ddr[:activation_stride] = hidden_states.reshape(-1)
            kv_cache_ddr = np.zeros(kv_stride * 2, dtype=np.float32)
            weight_sram, kv_sram, partial_sum_sram, softmax_sram, control_sram = self._scratch_buffers()

            status = self.prefill_func(
                layer_id,
                seq_len,
                attention_tiles.seq,
                attention_tiles.query,
                attention_tiles.key,
                attention_tiles.hidden_proj,
                attention_tiles.kv_proj,
                attention_tiles.head_dim,
                attention_tiles.query_heads_parallel,
                attention_tiles.kv_heads_parallel,
                mlp_tiles.seq,
                mlp_tiles.hidden,
                mlp_tiles.ff,
                0,
                activation_stride * 4,
                0,
                0,
                0,
                kv_stride * 4,
                weight_ddr,
                scale_ddr,
                kv_cache_ddr,
                activation_ddr,
                weight_sram,
                kv_sram,
                partial_sum_sram,
                softmax_sram,
                control_sram,
            )
            if status != 0:
                raise RuntimeError(f"Prefill top wrapper failed at layer {layer_id} with status {status}")

            hidden_states = round_bfloat16_array(activation_ddr[activation_stride:].reshape(seq_len, self.hidden_size))
            layer_k_cache = self._kv_ddr_to_cache_tensor(kv_cache_ddr[:kv_stride], seq_len)
            layer_v_cache = self._kv_ddr_to_cache_tensor(kv_cache_ddr[kv_stride:], seq_len)
            cache_layers.append(
                (
                    torch.from_numpy(layer_k_cache.copy()).unsqueeze(0),
                    torch.from_numpy(layer_v_cache.copy()).unsqueeze(0),
                )
            )
            trace_layers.append(
                {
                    "layer_id": layer_id,
                    "seq_len": seq_len,
                    "attention_seq_tile": attention_tiles.seq,
                    "attention_query_tile": attention_tiles.query,
                    "attention_key_tile": attention_tiles.key,
                    "mlp_seq_tile": mlp_tiles.seq,
                    "mlp_hidden_tile": mlp_tiles.hidden,
                    "mlp_ff_tile": mlp_tiles.ff,
                }
            )

        self.last_layer_trace = {
            "mode": "prefill",
            "layer_count": self.num_hidden_layers,
            "first_layer": trace_layers[0],
            "last_layer": trace_layers[-1],
        }
        return PrefillResult(logits=self._finalize_logits(hidden_states), cache=tuple(cache_layers))

    def decode_step(self, input_ids: torch.Tensor, cache: Any) -> DecodeResult:
        if cache is None:
            raise ValueError("RealHostTopBackend.decode_step requires a prefill cache.")

        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(input_ids.to(self.device)).detach().cpu().to(torch.float32).squeeze(0).squeeze(0).numpy()

        past_seq_len = int(cache[0][0].shape[2])
        next_seq_len = past_seq_len + 1
        trace_layers: list[dict[str, int]] = []
        next_cache_layers: list[tuple[torch.Tensor, torch.Tensor]] = []

        for layer_id in range(self.num_hidden_layers):
            weight_ddr, scale_ddr = self._pack_layer_parameters(layer_id)
            kv_stride = next_seq_len * self.kv_width
            activation_ddr = np.zeros(self.hidden_size * 2, dtype=np.float32)
            activation_ddr[: self.hidden_size] = hidden_states.reshape(-1)
            kv_cache_ddr = np.zeros(kv_stride * 2, dtype=np.float32)
            kv_cache_ddr[: past_seq_len * self.kv_width] = self._cache_tensor_to_kv_ddr(cache[layer_id][0], past_seq_len)
            kv_cache_ddr[kv_stride : kv_stride + past_seq_len * self.kv_width] = self._cache_tensor_to_kv_ddr(cache[layer_id][1], past_seq_len)
            weight_sram, kv_sram, partial_sum_sram, softmax_sram, control_sram = self._scratch_buffers()

            status = self.decode_func(
                layer_id,
                past_seq_len,
                0,
                self.hidden_size * 4,
                0,
                0,
                0,
                kv_stride * 4,
                weight_ddr,
                scale_ddr,
                kv_cache_ddr,
                activation_ddr,
                weight_sram,
                kv_sram,
                partial_sum_sram,
                softmax_sram,
                control_sram,
            )
            if status != 0:
                raise RuntimeError(f"Decode top wrapper failed at layer {layer_id} with status {status}")

            hidden_states = round_bfloat16_array(activation_ddr[self.hidden_size :].reshape(self.hidden_size))
            layer_k_cache = self._kv_ddr_to_cache_tensor(kv_cache_ddr[:kv_stride], next_seq_len)
            layer_v_cache = self._kv_ddr_to_cache_tensor(kv_cache_ddr[kv_stride:], next_seq_len)
            next_cache_layers.append(
                (
                    torch.from_numpy(layer_k_cache.copy()).unsqueeze(0),
                    torch.from_numpy(layer_v_cache.copy()).unsqueeze(0),
                )
            )
            trace_layers.append({"layer_id": layer_id, "past_seq_len": past_seq_len})

        self.last_layer_trace = {
            "mode": "decode",
            "layer_count": self.num_hidden_layers,
            "first_layer": trace_layers[0],
            "last_layer": trace_layers[-1],
        }
        return DecodeResult(logits=self._finalize_logits(hidden_states.reshape(1, self.hidden_size)), cache=tuple(next_cache_layers))