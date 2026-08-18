#include "forge200_runtime_v8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define HEADER_BYTES 64U
#define TENSOR_BYTES 48U

static void wr32(uint8_t *target, uint32_t value)
{
    target[0] = (uint8_t)value;
    target[1] = (uint8_t)(value >> 8);
    target[2] = (uint8_t)(value >> 16);
    target[3] = (uint8_t)(value >> 24);
}

static void wrf32(uint8_t *target, float value)
{
    memcpy(target, &value, sizeof(value));
}

static void tensor_entry(uint8_t *payload, uint32_t index, uint32_t dtype,
                         uint32_t ndim, uint32_t dim0, uint32_t dim1,
                         uint32_t data_offset, uint32_t data_bytes,
                         uint32_t scale_offset, uint32_t scale_count)
{
    uint8_t *entry = payload + HEADER_BYTES + index * TENSOR_BYTES;
    wr32(entry + 0U, index + 1U);
    wr32(entry + 4U, dtype);
    wr32(entry + 8U, ndim);
    wr32(entry + 16U, dim0);
    wr32(entry + 20U, dim1);
    wr32(entry + 24U, 1U);
    wr32(entry + 28U, 1U);
    wr32(entry + 32U, data_offset);
    wr32(entry + 36U, data_bytes);
    wr32(entry + 40U, scale_offset);
    wr32(entry + 44U, scale_count);
}

static void valid_sequence(uint8_t payload[176])
{
    memset(payload, 0, 176U);
    memcpy(payload, "F2RT", 4U);
    wr32(payload + 4U, 1U);
    wr32(payload + 8U, HEADER_BYTES);
    wr32(payload + 12U, 1U); /* sequence */
    wr32(payload + 16U, 0U); /* linear */
    wr32(payload + 20U, 1U); /* one layer */
    wr32(payload + 24U, 2U); /* weight + bias */
    wr32(payload + 28U, 1U);
    wr32(payload + 32U, 1U);
    wr32(payload + 36U, 2U);
    wr32(payload + 40U, 0U);
    wr32(payload + 60U, 176U);
    tensor_entry(payload, 0U, 2U, 2U, 1U, 1U, 160U, 4U, 164U, 1U);
    tensor_entry(payload, 1U, 2U, 1U, 1U, 1U, 168U, 4U, 172U, 1U);
    wrf32(payload + 160U, 2.0f);
    wrf32(payload + 164U, 1.0f);
    wrf32(payload + 168U, 1.0f);
    wrf32(payload + 172U, 1.0f);
}

static void valid_grouped_sequence(uint8_t payload[256])
{
    uint32_t i;
    memset(payload, 0, 256U);
    memcpy(payload, "F2RT", 4U);
    wr32(payload + 4U, 1U);
    wr32(payload + 8U, HEADER_BYTES);
    wr32(payload + 12U, 1U);
    wr32(payload + 16U, 0U);
    wr32(payload + 20U, 1U);
    wr32(payload + 24U, 2U);
    wr32(payload + 28U, 33U);
    wr32(payload + 32U, 2U);
    wr32(payload + 36U, 66U);
    wr32(payload + 40U, 0U);
    wr32(payload + 60U, 256U);
    tensor_entry(payload, 0U, 1U, 2U, 2U, 33U, 160U, 66U, 228U, 4U);
    tensor_entry(payload, 1U, 2U, 1U, 2U, 1U, 244U, 8U, 252U, 1U);
    for (i = 0U; i < 66U; ++i) payload[160U + i] = 1U;
    wrf32(payload + 228U, 1.0f);
    wrf32(payload + 232U, 2.0f);
    wrf32(payload + 236U, 3.0f);
    wrf32(payload + 240U, 4.0f);
    wrf32(payload + 244U, 0.0f);
    wrf32(payload + 248U, 0.0f);
    wrf32(payload + 252U, 1.0f);
}

static int require(int condition, const char *message)
{
    if (condition) return 1;
    fprintf(stderr, "FAIL:%s\n", message);
    return 0;
}

int main(void)
{
    uint8_t payload[176];
    uint8_t unaligned_storage[177];
    uint8_t grouped[256];
    f2rt_model_t model;
    float input[1] = {3.0f}, output[1] = {0.0f}, workspace[2] = {0.0f, 0.0f};
    uint16_t prompt[1] = {1U}, generated[1] = {0U};
    float grouped_input[33], grouped_output[2] = {0.0f, 0.0f}, grouped_workspace[66];
    uint32_t i;

    valid_sequence(payload);
    if (!require(f2rt_bind(payload, sizeof(payload), &model) == F2RT_OK, "valid bind")) return 1;
    if (!require(f2rt_infer_f32(&model, input, 1U, output, 1U, workspace, 2U) == F2RT_OK,
                 "valid inference")) return 1;
    if (!require(fabsf(output[0] - 7.0f) < 1e-6f, "valid numerical output")) return 1;

    valid_sequence(unaligned_storage + 1U);
    output[0] = 0.0f;
    if (!require(f2rt_bind(unaligned_storage + 1U, sizeof(unaligned_storage) - 1U, &model) == F2RT_OK,
                 "unaligned payload bind")) return 1;
    if (!require(f2rt_infer_f32(&model, input, 1U, output, 1U, workspace, 2U) == F2RT_OK,
                 "unaligned payload inference")) return 1;
    if (!require(fabsf(output[0] - 7.0f) < 1e-6f, "unaligned numerical output")) return 1;

    for (i = 0U; i < 33U; ++i) grouped_input[i] = 1.0f;
    valid_grouped_sequence(grouped);
    if (!require(f2rt_bind(grouped, sizeof(grouped), &model) == F2RT_OK,
                 "grouped-scale bind")) return 1;
    if (!require(f2rt_infer_f32(&model, grouped_input, 33U, grouped_output, 2U,
                                grouped_workspace, 66U) == F2RT_OK,
                 "grouped-scale inference")) return 1;
    if (!require(fabsf(grouped_output[0] - 34.0f) < 1e-6f &&
                 fabsf(grouped_output[1] - 100.0f) < 1e-6f,
                 "last grouped scale index")) return 1;

    valid_grouped_sequence(grouped);
    wr32(grouped + HEADER_BYTES + 44U, 3U);
    if (!require(f2rt_bind(grouped, sizeof(grouped), &model) == F2RT_ERR_TENSOR,
                 "wrong grouped scale count")) return 1;

    valid_grouped_sequence(grouped);
    wr32(grouped + HEADER_BYTES + 36U, 65U);
    if (!require(f2rt_bind(grouped, sizeof(grouped), &model) == F2RT_ERR_BOUNDS,
                 "wrong tensor data bytes")) return 1;

    valid_grouped_sequence(grouped);
    wr32(grouped + HEADER_BYTES + 32U, HEADER_BYTES);
    if (!require(f2rt_bind(grouped, sizeof(grouped), &model) == F2RT_ERR_BOUNDS,
                 "tensor data overlaps table")) return 1;

    valid_grouped_sequence(grouped);
    wr32(grouped + HEADER_BYTES + 40U, HEADER_BYTES);
    if (!require(f2rt_bind(grouped, sizeof(grouped), &model) == F2RT_ERR_BOUNDS,
                 "tensor scale overlaps table")) return 1;

    valid_grouped_sequence(grouped);
    wr32(grouped + HEADER_BYTES + 24U, 2U);
    if (!require(f2rt_bind(grouped, sizeof(grouped), &model) == F2RT_ERR_TENSOR,
                 "inactive dimension must be one")) return 1;

    valid_sequence(payload);
    wr32(payload + 24U, UINT32_MAX);
    if (!require(f2rt_bind(payload, sizeof(payload), &model) == F2RT_ERR_BOUNDS,
                 "tensor table multiplication overflow")) return 1;

    valid_sequence(payload);
    wr32(payload + HEADER_BYTES + 44U, UINT32_MAX);
    if (!require(f2rt_bind(payload, sizeof(payload), &model) == F2RT_ERR_BOUNDS,
                 "scale byte multiplication overflow")) return 1;

    valid_sequence(payload);
    wr32(payload + HEADER_BYTES + 16U, UINT32_MAX);
    wr32(payload + HEADER_BYTES + 20U, UINT32_MAX);
    if (!require(f2rt_bind(payload, sizeof(payload), &model) != F2RT_OK,
                 "tensor dimension multiplication overflow")) return 1;

    valid_sequence(payload);
    if (!require(f2rt_bind(payload, sizeof(payload), &model) == F2RT_OK, "rebind")) return 1;
    model.layer_count = UINT32_MAX;
    if (!require(f2rt_infer_f32(&model, input, 1U, output, 1U, workspace, 2U) == F2RT_ERR_TENSOR,
                 "sequence tensor-index overflow")) return 1;

    model.kind = 9U; /* convolution */
    model.layer_count = 1U;
    model.aux[0] = 24U;
    model.aux[1] = UINT32_MAX;
    model.aux[2] = UINT32_MAX;
    if (!require(f2rt_infer_f32(&model, input, 1U, output, 1U, workspace, 2U) == F2RT_ERR_TENSOR,
                 "convolution geometry overflow")) return 1;

    model.kind = 10U; /* NanoLM */
    if (!require(f2rt_generate_u16(&model, prompt, UINT32_MAX, generated, 1U, workspace, 2U) ==
                 F2RT_ERR_ARGUMENT, "prompt length addition overflow")) return 1;

    valid_sequence(payload);
    wr32(payload + 12U, 8U); /* multihead */
    wr32(payload + 20U, 0U);
    if (!require(f2rt_bind(payload, sizeof(payload), &model) == F2RT_OK,
                 "multihead parser path")) return 1;
    if (!require(f2rt_infer_f32(&model, input, 1U, output, 1U, workspace, 2U) ==
                 F2RT_ERR_TENSOR, "multihead zero-layer rejection")) return 1;

    valid_sequence(payload);
    wr32(payload + 12U, 10U); /* NanoLM */
    wr32(payload + 44U, 1U);
    wr32(payload + 48U, 1U);
    wr32(payload + 52U, 1U);
    wr32(payload + 56U, 4U);
    if (!require(f2rt_bind(payload, sizeof(payload), &model) == F2RT_OK,
                 "NanoLM parser path")) return 1;
    if (!require(f2rt_generate_u16(&model, prompt, 1U, generated, 1U, workspace, 2U) ==
                 F2RT_ERR_TENSOR, "NanoLM tensor-count rejection")) return 1;

    if (!require(f2rt_golden_layout_ok(72U, 2U, 2U, 2U),
                 "valid golden layout")) return 1;
    if (!require(!f2rt_golden_layout_ok(64U, UINT32_MAX, 1U, 4U),
                 "golden element addition overflow")) return 1;
    if (!require(!f2rt_golden_layout_ok(UINT64_MAX, UINT32_MAX, UINT32_MAX, 4U),
                 "golden byte multiplication mismatch")) return 1;

    puts("PASS");
    return 0;
}
