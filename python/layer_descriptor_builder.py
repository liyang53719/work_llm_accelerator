from __future__ import annotations

from dataclasses import asdict, dataclass

from qwen_model_spec import QwenModelSpec, load_qwen_model_spec


SRAM_BUDGET_BYTES = 1 << 20
WEIGHT_BUFFER_BYTES = 256 << 10
KV_WORKING_SET_BYTES = 256 << 10
PARTIAL_SUM_BYTES = 128 << 10
SOFTMAX_SCRATCH_BYTES = 128 << 10
CONTROL_SCRATCH_BYTES = SRAM_BUDGET_BYTES - WEIGHT_BUFFER_BYTES - KV_WORKING_SET_BYTES - PARTIAL_SUM_BYTES - SOFTMAX_SCRATCH_BYTES


@dataclass(frozen=True)
class LayerParameterLayout:
    input_layernorm_weight_offset_bytes: int
    q_weight_offset_bytes: int
    k_weight_offset_bytes: int
    v_weight_offset_bytes: int
    o_weight_offset_bytes: int
    post_attention_layernorm_weight_offset_bytes: int
    gate_weight_offset_bytes: int
    up_weight_offset_bytes: int
    down_weight_offset_bytes: int
    q_bias_offset_bytes: int
    k_bias_offset_bytes: int
    v_bias_offset_bytes: int
    q_scale_offset_bytes: int
    k_scale_offset_bytes: int
    v_scale_offset_bytes: int
    o_scale_offset_bytes: int
    gate_scale_offset_bytes: int
    up_scale_offset_bytes: int
    down_scale_offset_bytes: int
    total_parameter_bytes: int


@dataclass(frozen=True)
class KvCacheLayout:
    k_base_offset_bytes: int
    v_base_offset_bytes: int
    layer_stride_bytes: int
    token_stride_bytes: int


@dataclass(frozen=True)
class DecodeLayerDescriptor:
    layer_id: int
    past_seq_len: int
    input_token_addr: int
    output_token_addr: int
    layer_weights_base_addr: int
    layer_scales_base_addr: int
    k_cache_base_addr: int
    v_cache_base_addr: int
    scratch_base_addr: int


@dataclass(frozen=True)
class PrefillLayerDescriptor:
    layer_id: int
    seq_len: int
    tile_config: dict[str, object]
    input_sequence_addr: int
    output_sequence_addr: int
    layer_weights_base_addr: int
    layer_scales_base_addr: int
    k_cache_base_addr: int
    v_cache_base_addr: int
    scratch_base_addr: int


@dataclass(frozen=True)
class PrefillAttentionTileConfig:
    seq: int
    query: int
    key: int
    hidden_proj: int
    kv_proj: int
    head_dim: int
    query_heads_parallel: int
    kv_heads_parallel: int


@dataclass(frozen=True)
class PrefillMLPTileConfig:
    seq: int
    hidden: int
    ff: int


@dataclass(frozen=True)
class PrefillTileConfig:
    attention: PrefillAttentionTileConfig
    mlp: PrefillMLPTileConfig


def default_prefill_tile_config() -> PrefillTileConfig:
    return PrefillTileConfig(
        attention=PrefillAttentionTileConfig(
            seq=128,
            query=128,
            key=128,
            hidden_proj=256,
            kv_proj=256,
            head_dim=128,
            query_heads_parallel=2,
            kv_heads_parallel=1,
        ),
        mlp=PrefillMLPTileConfig(
            seq=128,
            hidden=256,
            ff=640,
        ),
    )


def build_layer_parameter_layout(spec: QwenModelSpec) -> LayerParameterLayout:
    q_weight_bytes = spec.hidden_size * spec.hidden_size // 2
    k_width = spec.num_key_value_heads * spec.head_dim
    k_weight_bytes = spec.hidden_size * k_width // 2
    v_weight_bytes = k_weight_bytes
    o_weight_bytes = q_weight_bytes
    gate_weight_bytes = spec.hidden_size * spec.intermediate_size // 2
    up_weight_bytes = gate_weight_bytes
    down_weight_bytes = spec.intermediate_size * spec.hidden_size // 2

    offset = 0
    input_layernorm_weight_offset_bytes = offset
    offset += spec.hidden_size * 4
    q_weight_offset_bytes = offset
    offset += q_weight_bytes
    k_weight_offset_bytes = offset
    offset += k_weight_bytes
    v_weight_offset_bytes = offset
    offset += v_weight_bytes
    o_weight_offset_bytes = offset
    offset += o_weight_bytes
    post_attention_layernorm_weight_offset_bytes = offset
    offset += spec.hidden_size * 4
    gate_weight_offset_bytes = offset
    offset += gate_weight_bytes
    up_weight_offset_bytes = offset
    offset += up_weight_bytes
    down_weight_offset_bytes = offset
    offset += down_weight_bytes
    q_bias_offset_bytes = offset
    offset += spec.hidden_size * 4
    k_bias_offset_bytes = offset
    offset += k_width * 4
    v_bias_offset_bytes = offset
    offset += k_width * 4
    q_scale_offset_bytes = offset
    offset += spec.hidden_size * 4
    k_scale_offset_bytes = offset
    offset += k_width * 4
    v_scale_offset_bytes = offset
    offset += k_width * 4
    o_scale_offset_bytes = offset
    offset += spec.hidden_size * 4
    gate_scale_offset_bytes = offset
    offset += spec.intermediate_size * 4
    up_scale_offset_bytes = offset
    offset += spec.intermediate_size * 4
    down_scale_offset_bytes = offset
    offset += spec.hidden_size * 4

    return LayerParameterLayout(
        input_layernorm_weight_offset_bytes=input_layernorm_weight_offset_bytes,
        q_weight_offset_bytes=q_weight_offset_bytes,
        k_weight_offset_bytes=k_weight_offset_bytes,
        v_weight_offset_bytes=v_weight_offset_bytes,
        o_weight_offset_bytes=o_weight_offset_bytes,
        post_attention_layernorm_weight_offset_bytes=post_attention_layernorm_weight_offset_bytes,
        gate_weight_offset_bytes=gate_weight_offset_bytes,
        up_weight_offset_bytes=up_weight_offset_bytes,
        down_weight_offset_bytes=down_weight_offset_bytes,
        q_bias_offset_bytes=q_bias_offset_bytes,
        k_bias_offset_bytes=k_bias_offset_bytes,
        v_bias_offset_bytes=v_bias_offset_bytes,
        q_scale_offset_bytes=q_scale_offset_bytes,
        k_scale_offset_bytes=k_scale_offset_bytes,
        v_scale_offset_bytes=v_scale_offset_bytes,
        o_scale_offset_bytes=o_scale_offset_bytes,
        gate_scale_offset_bytes=gate_scale_offset_bytes,
        up_scale_offset_bytes=up_scale_offset_bytes,
        down_scale_offset_bytes=down_scale_offset_bytes,
        total_parameter_bytes=offset,
    )


def build_kv_cache_layout(spec: QwenModelSpec) -> KvCacheLayout:
    token_stride_bytes = spec.num_key_value_heads * spec.head_dim * 4
    layer_stride_bytes = spec.max_position_embeddings * token_stride_bytes
    return KvCacheLayout(
        k_base_offset_bytes=0,
        v_base_offset_bytes=spec.num_hidden_layers * layer_stride_bytes,
        layer_stride_bytes=layer_stride_bytes,
        token_stride_bytes=token_stride_bytes,
    )


def build_decode_descriptors(
    past_seq_len: int,
    activation_base_addr: int,
    weight_base_addr: int,
    scale_base_addr: int,
    kv_cache_base_addr: int,
    scratch_base_addr: int,
    spec: QwenModelSpec | None = None,
) -> list[DecodeLayerDescriptor]:
    spec = load_qwen_model_spec() if spec is None else spec
    param_layout = build_layer_parameter_layout(spec)
    kv_layout = build_kv_cache_layout(spec)
    activation_stride = spec.hidden_size * 4

    descriptors: list[DecodeLayerDescriptor] = []
    for layer_id in range(spec.num_hidden_layers):
      layer_weight_base = weight_base_addr + layer_id * param_layout.total_parameter_bytes
      layer_scale_base = scale_base_addr + layer_id * param_layout.total_parameter_bytes
      layer_k_base = kv_cache_base_addr + kv_layout.k_base_offset_bytes + layer_id * kv_layout.layer_stride_bytes
      layer_v_base = kv_cache_base_addr + kv_layout.v_base_offset_bytes + layer_id * kv_layout.layer_stride_bytes
      descriptors.append(
          DecodeLayerDescriptor(
              layer_id=layer_id,
              past_seq_len=past_seq_len,
              input_token_addr=activation_base_addr + layer_id * activation_stride,
              output_token_addr=activation_base_addr + (layer_id + 1) * activation_stride,
              layer_weights_base_addr=layer_weight_base,
              layer_scales_base_addr=layer_scale_base,
              k_cache_base_addr=layer_k_base,
              v_cache_base_addr=layer_v_base,
              scratch_base_addr=scratch_base_addr,
          )
      )
    return descriptors


def build_prefill_descriptors(
    seq_len: int,
    activation_base_addr: int,
    weight_base_addr: int,
    scale_base_addr: int,
    kv_cache_base_addr: int,
    scratch_base_addr: int,
    tile_config: PrefillTileConfig | None = None,
    spec: QwenModelSpec | None = None,
) -> list[PrefillLayerDescriptor]:
    spec = load_qwen_model_spec() if spec is None else spec
    tile_config = default_prefill_tile_config() if tile_config is None else tile_config
    param_layout = build_layer_parameter_layout(spec)
    kv_layout = build_kv_cache_layout(spec)
    activation_stride = seq_len * spec.hidden_size * 4

    descriptors: list[PrefillLayerDescriptor] = []
    for layer_id in range(spec.num_hidden_layers):
        layer_weight_base = weight_base_addr + layer_id * param_layout.total_parameter_bytes
        layer_scale_base = scale_base_addr + layer_id * param_layout.total_parameter_bytes
        layer_k_base = kv_cache_base_addr + kv_layout.k_base_offset_bytes + layer_id * kv_layout.layer_stride_bytes
        layer_v_base = kv_cache_base_addr + kv_layout.v_base_offset_bytes + layer_id * kv_layout.layer_stride_bytes
        descriptors.append(
            PrefillLayerDescriptor(
                layer_id=layer_id,
                seq_len=seq_len,
                tile_config=asdict(tile_config),
                input_sequence_addr=activation_base_addr + layer_id * activation_stride,
                output_sequence_addr=activation_base_addr + (layer_id + 1) * activation_stride,
                layer_weights_base_addr=layer_weight_base,
                layer_scales_base_addr=layer_scale_base,
                k_cache_base_addr=layer_k_base,
                v_cache_base_addr=layer_v_base,
                scratch_base_addr=scratch_base_addr,
            )
        )
    return descriptors


def summary_dict(spec: QwenModelSpec | None = None) -> dict[str, object]:
    spec = load_qwen_model_spec() if spec is None else spec
    return {
        "model_spec": asdict(spec),
        "sram": {
            "budget_bytes": SRAM_BUDGET_BYTES,
            "weight_buffer_bytes": WEIGHT_BUFFER_BYTES,
            "kv_working_set_bytes": KV_WORKING_SET_BYTES,
            "partial_sum_bytes": PARTIAL_SUM_BYTES,
            "softmax_scratch_bytes": SOFTMAX_SCRATCH_BYTES,
            "control_scratch_bytes": CONTROL_SCRATCH_BYTES,
        },
        "parameter_layout": asdict(build_layer_parameter_layout(spec)),
        "kv_layout": asdict(build_kv_cache_layout(spec)),
    }