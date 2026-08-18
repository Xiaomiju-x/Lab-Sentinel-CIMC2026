#ifndef FORGE200_RUNTIME_V8_H
#define FORGE200_RUNTIME_V8_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    F2RT_OK = 0,
    F2RT_ERR_ARGUMENT = -1,
    F2RT_ERR_SCHEMA = -2,
    F2RT_ERR_BOUNDS = -3,
    F2RT_ERR_ENGINE = -4,
    F2RT_ERR_WORKSPACE = -5,
    F2RT_ERR_TENSOR = -6,
    F2RT_ERR_OUTPUT = -7
} f2rt_status_t;

typedef struct {
    const uint8_t *payload;
    uint32_t payload_bytes;
    uint32_t kind;
    uint32_t activation;
    uint32_t layer_count;
    uint32_t tensor_count;
    uint32_t input_elems;
    uint32_t output_elems;
    uint32_t workspace_elems;
    uint32_t postprocess;
    uint32_t aux[4];
} f2rt_model_t;

f2rt_status_t f2rt_bind(const void *payload, uint32_t payload_bytes,
                        f2rt_model_t *model);

f2rt_status_t f2rt_infer_f32(const f2rt_model_t *model,
                             const float *input, uint32_t input_elems,
                             float *output, uint32_t output_capacity,
                             float *workspace, uint32_t workspace_elems);

f2rt_status_t f2rt_generate_u16(const f2rt_model_t *model,
                                const uint16_t *prompt, uint32_t prompt_length,
                                uint16_t *generated, uint32_t generated_capacity,
                                float *workspace, uint32_t workspace_elems);

int f2rt_golden_layout_ok(uint64_t golden_bytes, uint32_t input_count,
                          uint32_t output_count, uint32_t element_bytes);

#ifdef __cplusplus
}
#endif

#endif
