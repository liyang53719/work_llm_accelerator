from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from torch_reference_backend import TorchReferenceBackend


@dataclass
class Layer0PrefillOutputs:
    input_layernorm: torch.Tensor
    q_proj: torch.Tensor
    k_proj: torch.Tensor
    v_proj: torch.Tensor
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


def module_linear(module: torch.nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    dtype = module.weight.dtype
    return module(input_tensor.to(dtype)).to(torch.float32)


class Layer0PrefillReferenceBackend:
    def __init__(self, torch_backend: TorchReferenceBackend | None = None) -> None:
        self.torch_backend = TorchReferenceBackend(device="cpu") if torch_backend is None else torch_backend
        self.model = self.torch_backend.model
        self.layer = self.model.model.layers[0]
        self.attn = self.layer.self_attn

    def run(self, layer0_input: torch.Tensor) -> Layer0PrefillOutputs:
        layer0_input = layer0_input.to(torch.float32)
        seq_len = layer0_input.shape[1]
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            input_layernorm = self.layer.input_layernorm(
                layer0_input.to(self.layer.input_layernorm.weight.dtype)
            ).to(torch.float32)

            q_proj = module_linear(self.layer.self_attn.q_proj, input_layernorm)
            k_proj = module_linear(self.layer.self_attn.k_proj, input_layernorm)
            v_proj = module_linear(self.layer.self_attn.v_proj, input_layernorm)

            batch_size = layer0_input.shape[0]
            q_states = q_proj.view(batch_size, seq_len, self.model.config.num_attention_heads, self.attn.head_dim).transpose(1, 2)
            k_states = k_proj.view(batch_size, seq_len, self.model.config.num_key_value_heads, self.attn.head_dim).transpose(1, 2)
            v_states = v_proj.view(batch_size, seq_len, self.model.config.num_key_value_heads, self.attn.head_dim).transpose(1, 2)

            cos, sin = self.model.model.rotary_emb(v_states, position_ids)
            q_states, k_states = apply_rotary_pos_emb(q_states, k_states, cos, sin)

            k_states = repeat_kv(k_states, self.attn.num_key_value_groups)
            v_states = repeat_kv(v_states, self.attn.num_key_value_groups)

            attn_scores = torch.matmul(q_states, k_states.transpose(-1, -2)) * self.attn.scaling
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
            attn_scores = attn_scores + causal_mask.view(1, 1, seq_len, seq_len)
            attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q_states.dtype)
            attn_context = torch.matmul(attn_probs, v_states)
            attn_context = attn_context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

            o_proj = module_linear(self.layer.self_attn.o_proj, attn_context.to(torch.float32))
            attention_residual = layer0_input + o_proj

            post_attention_layernorm = self.layer.post_attention_layernorm(
                attention_residual.to(self.layer.post_attention_layernorm.weight.dtype)
            ).to(torch.float32)
            gate_proj = module_linear(self.layer.mlp.gate_proj, post_attention_layernorm)
            up_proj = module_linear(self.layer.mlp.up_proj, post_attention_layernorm)
            silu_mul = self.layer.mlp.act_fn(gate_proj.to(self.layer.mlp.gate_proj.weight.dtype)).to(torch.float32) * up_proj
            down_proj = module_linear(self.layer.mlp.down_proj, silu_mul)
            layer0_output = attention_residual + down_proj

        return Layer0PrefillOutputs(
            input_layernorm=input_layernorm,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            self_attn_output=o_proj,
            o_proj=o_proj,
            attention_residual=attention_residual,
            post_attention_layernorm=post_attention_layernorm,
            gate_proj=gate_proj,
            up_proj=up_proj,
            silu_mul=silu_mul,
            down_proj=down_proj,
            layer0_output=layer0_output,
        )