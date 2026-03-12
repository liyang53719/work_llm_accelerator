#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int qwen_decode_stub_forward(
    const float* input_token,
    int past_seq_len,
    float* output_token);

#ifdef __cplusplus
}
#endif