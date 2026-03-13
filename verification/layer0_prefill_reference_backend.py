from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from transformers.models.qwen2.modeling_qwen2 import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, repeat_kv

from torch_reference_backend import SDPA_ATTENTION_BACKEND, TorchReferenceBackend, get_attention_backend


@dataclass
class Layer0PrefillOutputs:
    input_layernorm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
    q_rot: torch.Tensor
    k_rot: torch.Tensor
    attn_probs: torch.Tensor
    attn_context: torch.Tensor
    self_attn_output: torch.Tensor
    o_proj: torch.Tensor
    attention_residual: torch.Tensor
    post_attention_layernorm: torch.Tensor
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    silu_mul: torch.Tensor
    down_proj: torch.Tensor
    layer0_output: torch.Tensor

    def as_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "input_layernorm": self.input_layernorm,
            "q_proj": self.q_proj,
            "k_proj": self.k_proj,
            "v_proj": self.v_proj,
            "q_rot": self.q_rot,
            "k_rot": self.k_rot,
            "attn_probs": self.attn_probs,
            "attn_context": self.attn_context,
            "self_attn_output": self.self_attn_output,
            "o_proj": self.o_proj,
            "attention_residual": self.attention_residual,
            "post_attention_layernorm": self.post_attention_layernorm,
            "gate_proj": self.gate_proj,
            "up_proj": self.up_proj,
            "silu_mul": self.silu_mul,
            "down_proj": self.down_proj,
            "layer0_output": self.layer0_output,
        }
class Layer0PrefillReferenceBackend:
    def __init__(self, torch_backend: TorchReferenceBackend | None = None) -> None:
        self.torch_backend = TorchReferenceBackend(device="cpu") if torch_backend is None else torch_backend
        self.model = self.torch_backend.model
        self.layer = self.model.model.layers[0]
        self.attn = self.layer.self_attn

    def run(self, layer0_input: torch.Tensor) -> Layer0PrefillOutputs:
        attention_backend = get_attention_backend(self.model)
        if attention_backend != SDPA_ATTENTION_BACKEND:
            raise AssertionError(
                f"Layer0PrefillReferenceBackend requires attention backend '{SDPA_ATTENTION_BACKEND}', got '{attention_backend}'."
            )
        model_dtype = self.layer.input_layernorm.weight.dtype
        layer0_input = layer0_input.to(model_dtype)
        seq_len = layer0_input.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32), diagonal=1
        ).view(1, 1, seq_len, seq_len)
        position_embeddings = self.model.model.rotary_emb(layer0_input, position_ids)

        with torch.no_grad():
            input_layernorm = self.layer.input_layernorm(layer0_input)

            input_shape = input_layernorm.shape[:-1]
            hidden_shape = (*input_shape, -1, self.attn.head_dim)
            q_proj = self.attn.q_proj(input_layernorm)
            k_proj = self.attn.k_proj(input_layernorm)
            v_proj = self.attn.v_proj(input_layernorm)

            q_states = q_proj.view(hidden_shape).transpose(1, 2)
            k_states = k_proj.view(hidden_shape).transpose(1, 2)
            v_states = v_proj.view(hidden_shape).transpose(1, 2)

            cos, sin = position_embeddings
            q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

            attention_interface = ALL_ATTENTION_FUNCTIONS.get(attention_backend)
            if attention_interface is None:
                raise AssertionError(f"Attention backend '{attention_backend}' is unavailable.")
            attn_output, attn_probs = attention_interface(
                self.attn,
                q_states,
                k_states,
                v_states,
                attention_mask,
                scaling=self.attn.scaling,
                dropout=0.0,
            )
            attn_context = attn_output.reshape(*input_shape, -1).contiguous()

            o_proj = self.attn.o_proj(attn_context)
            attention_residual = layer0_input + o_proj

            post_attention_layernorm = self.layer.post_attention_layernorm(attention_residual)
            gate_proj = self.layer.mlp.gate_proj(post_attention_layernorm)
            up_proj = self.layer.mlp.up_proj(post_attention_layernorm)
            silu_mul = self.layer.mlp.act_fn(gate_proj) * up_proj
            down_proj = self.layer.mlp.down_proj(silu_mul)
            layer0_output = attention_residual + down_proj
            repeated_k_states = repeat_kv(k_states, self.attn.num_key_value_groups)

        return Layer0PrefillOutputs(
            input_layernorm=input_layernorm.to(torch.float32),
            q_proj=q_proj.to(torch.float32),
            k_proj=k_proj.to(torch.float32),
            v_proj=v_proj.to(torch.float32),
            q_rot=q_states.to(torch.float32),
            k_rot=repeated_k_states.to(torch.float32),
            attn_probs=torch.empty(0, dtype=torch.float32) if attn_probs is None else attn_probs.to(torch.float32),
            attn_context=attn_context.to(torch.float32),
            self_attn_output=o_proj.to(torch.float32),
            o_proj=o_proj.to(torch.float32),
            attention_residual=attention_residual.to(torch.float32),
            post_attention_layernorm=post_attention_layernorm.to(torch.float32),
            gate_proj=gate_proj.to(torch.float32),
            up_proj=up_proj.to(torch.float32),
            silu_mul=silu_mul.to(torch.float32),
            down_proj=down_proj.to(torch.float32),
            layer0_output=layer0_output.to(torch.float32),
        )