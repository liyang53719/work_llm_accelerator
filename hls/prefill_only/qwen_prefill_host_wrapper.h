#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int qwen_prefill_stub_forward(
    const float* input_sequence,
    int seq_len,
    int tile_m,
    float* output_sequence);

#ifdef __cplusplus
}
#endif