#include "forge200_runtime_v8.h"

#include <math.h>
#include <string.h>

#define F2RT_HEADER_BYTES 64U
#define F2RT_TENSOR_BYTES 48U
#define F2RT_DTYPE_INT8 1U
#define F2RT_DTYPE_FLOAT32 2U
#define F2RT_DTYPE_UINT8 3U
#define F2RT_DTYPE_UINT16 4U

#define F2RT_KIND_SEQUENCE 1U
#define F2RT_KIND_RESIDUAL 2U
#define F2RT_KIND_RIDGE_PRIOR 3U
#define F2RT_KIND_POLYNOMIAL 4U
#define F2RT_KIND_CIE_RESIDUAL 5U
#define F2RT_KIND_INPUT_PRIOR_RESIDUAL 6U
#define F2RT_KIND_SKIP 7U
#define F2RT_KIND_MULTIHEAD 8U
#define F2RT_KIND_CONV_SEQUENCE 9U
#define F2RT_KIND_NANOLM 10U

#define F2RT_ACT_LINEAR 0U
#define F2RT_ACT_RELU 1U
#define F2RT_ACT_GELU 2U

#define F2RT_POST_RAW 0U
#define F2RT_POST_LAST_SIGMOID 1U
#define F2RT_POST_FIRST_SIGMOID_REST_SOFTMAX 2U
#define F2RT_POST_FIRST_RAW_LAST_SIGMOID 3U
#define F2RT_POST_SOFTPLUS 4U

typedef struct {
    uint32_t dtype;
    uint32_t ndim;
    uint32_t dims[4];
    const uint8_t *data;
    uint32_t data_bytes;
    const uint8_t *scale;
    uint32_t scale_count;
} f2rt_tensor_t;

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static float rdf32(const uint8_t *p)
{
    uint32_t bits = rd32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int span_ok(uint32_t offset, uint32_t bytes, uint32_t total)
{
    return offset <= total && bytes <= total - offset;
}

static int mul_u32(uint32_t left, uint32_t right, uint32_t *product)
{
    uint64_t value;
    if (product == NULL) return 0;
    value = (uint64_t)left * (uint64_t)right;
    if (value > UINT32_MAX) return 0;
    *product = (uint32_t)value;
    return 1;
}

static int count_fits_size_t(uintmax_t elements, size_t element_bytes)
{
    if (element_bytes == 0U) return 0;
    return elements <= (uintmax_t)SIZE_MAX / (uintmax_t)element_bytes;
}

int f2rt_golden_layout_ok(uint64_t golden_bytes, uint32_t input_count,
                          uint32_t output_count, uint32_t element_bytes)
{
    uint64_t total_elems;
    if (golden_bytes < 64U || output_count == 0U ||
        (element_bytes != 2U && element_bytes != 4U)) return 0;
    total_elems = (uint64_t)input_count + (uint64_t)output_count;
    return total_elems <= (golden_bytes - 64U) / element_bytes &&
           golden_bytes == 64U + total_elems * element_bytes;
}

static int tensor_shape_elems(const f2rt_tensor_t *tensor, uint32_t *elements)
{
    uint64_t count = 1U;
    uint32_t dimension;
    if (tensor == NULL || elements == NULL || tensor->ndim == 0U || tensor->ndim > 4U) return 0;
    for (dimension = 0U; dimension < tensor->ndim; ++dimension) {
        if (tensor->dims[dimension] == 0U) return 0;
        count *= tensor->dims[dimension];
        if (count > UINT32_MAX) return 0;
    }
    *elements = (uint32_t)count;
    return 1;
}

static int tensor_is_vector(const f2rt_tensor_t *tensor, uint32_t elements)
{
    return tensor != NULL && tensor->ndim == 1U && tensor->dims[0] == elements;
}

static int tensor_is_matrix(const f2rt_tensor_t *tensor, uint32_t rows, uint32_t columns)
{
    return tensor != NULL && tensor->ndim == 2U &&
           tensor->dims[0] == rows && tensor->dims[1] == columns;
}

static f2rt_status_t tensor_at(const f2rt_model_t *model, uint32_t index,
                               f2rt_tensor_t *tensor)
{
    const uint8_t *entry;
    uint32_t data_offset, scale_offset, table_bytes, table_end, entry_offset;
    uint32_t elements, element_bytes, scale_bytes, columns, groups, expected_scales, i;
    if (model == NULL || tensor == NULL || index >= model->tensor_count) {
        return F2RT_ERR_TENSOR;
    }
    if (model->payload == NULL || model->payload_bytes < F2RT_HEADER_BYTES ||
        model->tensor_count > (model->payload_bytes - F2RT_HEADER_BYTES) / F2RT_TENSOR_BYTES ||
        !mul_u32(model->tensor_count, F2RT_TENSOR_BYTES, &table_bytes)) {
        return F2RT_ERR_BOUNDS;
    }
    table_end = F2RT_HEADER_BYTES + table_bytes;
    if (!mul_u32(index, F2RT_TENSOR_BYTES, &entry_offset) ||
        entry_offset > table_bytes - F2RT_TENSOR_BYTES) {
        return F2RT_ERR_BOUNDS;
    }
    entry = model->payload + F2RT_HEADER_BYTES + entry_offset;
    tensor->dtype = rd32(entry + 4U);
    tensor->ndim = rd32(entry + 8U);
    if (tensor->ndim == 0U || tensor->ndim > 4U) {
        return F2RT_ERR_TENSOR;
    }
    for (i = 0U; i < 4U; ++i) {
        tensor->dims[i] = rd32(entry + 16U + i * 4U);
    }
    for (i = tensor->ndim; i < 4U; ++i) {
        if (tensor->dims[i] != 1U) return F2RT_ERR_TENSOR;
    }
    data_offset = rd32(entry + 32U);
    tensor->data_bytes = rd32(entry + 36U);
    scale_offset = rd32(entry + 40U);
    tensor->scale_count = rd32(entry + 44U);
    if (!tensor_shape_elems(tensor, &elements)) return F2RT_ERR_TENSOR;
    if (tensor->dtype == F2RT_DTYPE_INT8 || tensor->dtype == F2RT_DTYPE_UINT8) {
        element_bytes = 1U;
    } else if (tensor->dtype == F2RT_DTYPE_UINT16) {
        element_bytes = 2U;
    } else if (tensor->dtype == F2RT_DTYPE_FLOAT32) {
        element_bytes = 4U;
    } else {
        return F2RT_ERR_TENSOR;
    }
    if (!mul_u32(elements, element_bytes, &element_bytes) ||
        tensor->data_bytes != element_bytes ||
        tensor->scale_count == 0U ||
        !mul_u32(tensor->scale_count, 4U, &scale_bytes) ||
        data_offset < table_end || scale_offset < table_end ||
        !span_ok(data_offset, tensor->data_bytes, model->payload_bytes) ||
        !span_ok(scale_offset, scale_bytes, model->payload_bytes) ||
        (scale_offset & 3U) != 0U ||
        (tensor->dtype == F2RT_DTYPE_FLOAT32 && (data_offset & 3U) != 0U) ||
        (tensor->dtype == F2RT_DTYPE_UINT16 && (data_offset & 1U) != 0U)) {
        return F2RT_ERR_BOUNDS;
    }
    if (tensor->dtype != F2RT_DTYPE_FLOAT32 && tensor->scale_count != 1U) {
        if (tensor->ndim == 1U) {
            if (tensor->scale_count != elements) return F2RT_ERR_TENSOR;
        } else if (tensor->scale_count != tensor->dims[0]) {
            columns = elements / tensor->dims[0];
            groups = columns / 32U + (columns % 32U != 0U ? 1U : 0U);
            if (!mul_u32(tensor->dims[0], groups, &expected_scales) ||
                tensor->scale_count != expected_scales) return F2RT_ERR_TENSOR;
        }
    }
    tensor->data = model->payload + data_offset;
    tensor->scale = model->payload + scale_offset;
    return F2RT_OK;
}

static float qvalue(const f2rt_tensor_t *tensor, uint32_t row, uint32_t column)
{
    uint32_t dimension, offset;
    uint32_t columns = tensor->ndim >= 2U ? 1U : tensor->dims[0];
    if (tensor->ndim >= 2U) {
        for (dimension = 1U; dimension < tensor->ndim; ++dimension) columns *= tensor->dims[dimension];
    }
    offset = tensor->ndim >= 2U ? row * columns + column : column;
    float scale;
    if (tensor->dtype == F2RT_DTYPE_FLOAT32) {
        return rdf32(tensor->data + offset * 4U);
    }
    if (tensor->scale_count <= 1U) {
        scale = rdf32(tensor->scale);
    } else if (tensor->ndim >= 2U && tensor->scale_count == tensor->dims[0]) {
        scale = rdf32(tensor->scale + row * 4U);
    } else if (tensor->ndim >= 2U) {
        uint32_t groups = columns / 32U + (columns % 32U != 0U ? 1U : 0U);
        scale = rdf32(tensor->scale + (row * groups + column / 32U) * 4U);
    } else {
        scale = rdf32(tensor->scale + (offset < tensor->scale_count ? offset : 0U) * 4U);
    }
    if (tensor->dtype == F2RT_DTYPE_INT8) {
        return (float)((const int8_t *)(const void *)tensor->data)[offset] * scale;
    }
    if (tensor->dtype == F2RT_DTYPE_UINT8) {
        return (float)((const uint8_t *)(const void *)tensor->data)[offset] * scale;
    }
    return (float)rd16(tensor->data + offset * 2U) * scale;
}

static f2rt_status_t dense(const f2rt_tensor_t *weight,
                           const f2rt_tensor_t *bias,
                           const float *input, uint32_t input_elems,
                           float *output, uint32_t output_capacity)
{
    uint32_t out, in, outputs;
    if (weight->ndim != 2U || bias->ndim != 1U ||
        weight->dims[1] != input_elems || bias->dims[0] != weight->dims[0] ||
        weight->dims[0] > output_capacity) {
        return F2RT_ERR_TENSOR;
    }
    outputs = weight->dims[0];
    for (out = 0U; out < outputs; ++out) {
        float sum = qvalue(bias, 0U, out);
        for (in = 0U; in < input_elems; ++in) {
            sum += input[in] * qvalue(weight, out, in);
        }
        output[out] = sum;
    }
    return F2RT_OK;
}

static float gelu(float x)
{
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
}

static void activate(float *values, uint32_t count, uint32_t activation)
{
    uint32_t i;
    if (activation == F2RT_ACT_LINEAR) {
        return;
    }
    for (i = 0U; i < count; ++i) {
        values[i] = activation == F2RT_ACT_GELU ? gelu(values[i]) :
                    (values[i] > 0.0f ? values[i] : 0.0f);
    }
}

static void vector_softmax(float *values, uint32_t count)
{
    uint32_t i;
    float maximum, sum = 0.0f;
    if (count == 0U) return;
    maximum = values[0];
    for (i = 1U; i < count; ++i) if (values[i] > maximum) maximum = values[i];
    for (i = 0U; i < count; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    if (sum <= 0.0f) sum = 1.0f;
    for (i = 0U; i < count; ++i) values[i] /= sum;
}

static float logistic(float value)
{
    if (value >= 0.0f) {
        float inverse = expf(-value);
        return 1.0f / (1.0f + inverse);
    }
    {
        float exponential = expf(value);
        return exponential / (1.0f + exponential);
    }
}

static void apply_postprocess(float *output, uint32_t count, uint32_t postprocess)
{
    uint32_t i;
    if (postprocess == F2RT_POST_LAST_SIGMOID && count > 0U) {
        output[count - 1U] = logistic(output[count - 1U]);
    } else if (postprocess == F2RT_POST_FIRST_SIGMOID_REST_SOFTMAX && count > 1U) {
        output[0] = logistic(output[0]);
        vector_softmax(output + 1U, count - 1U);
    } else if (postprocess == F2RT_POST_FIRST_RAW_LAST_SIGMOID && count > 1U) {
        output[count - 1U] = logistic(output[count - 1U]);
    } else if (postprocess == F2RT_POST_SOFTPLUS) {
        for (i = 0U; i < count; ++i) {
            float x = output[i];
            output[i] = log1pf(expf(-fabsf(x))) + (x > 0.0f ? x : 0.0f);
        }
    }
}

static f2rt_status_t run_sequence(const f2rt_model_t *model, uint32_t tensor_start,
                                  uint32_t layers, uint32_t activation,
                                  const float *input, uint32_t input_elems,
                                  float *output, uint32_t output_capacity,
                                  float *workspace, uint32_t workspace_elems)
{
    float *a = workspace;
    float *b;
    const float *current = input;
    uint32_t current_count = input_elems, layer, maximum = input_elems;
    f2rt_tensor_t weight, bias;
    if (model == NULL || input == NULL || output == NULL || workspace == NULL ||
        layers == 0U || tensor_start > model->tensor_count ||
        layers > (model->tensor_count - tensor_start) / 2U) return F2RT_ERR_TENSOR;
    for (layer = 0U; layer < layers; ++layer) {
        uint32_t tensor_index = tensor_start + layer * 2U;
        if (tensor_at(model, tensor_index, &weight) != F2RT_OK ||
            tensor_at(model, tensor_index + 1U, &bias) != F2RT_OK) {
            return F2RT_ERR_TENSOR;
        }
        if (weight.dims[0] > maximum) maximum = weight.dims[0];
    }
    if (maximum > workspace_elems / 2U ||
        !count_fits_size_t((uintmax_t)maximum * 2U, sizeof(float))) return F2RT_ERR_WORKSPACE;
    b = workspace + maximum;
    for (layer = 0U; layer < layers; ++layer) {
        uint32_t tensor_index = tensor_start + layer * 2U;
        float *next = (layer & 1U) == 0U ? a : b;
        f2rt_status_t status;
        if (tensor_at(model, tensor_index, &weight) != F2RT_OK ||
            tensor_at(model, tensor_index + 1U, &bias) != F2RT_OK) return F2RT_ERR_TENSOR;
        status = dense(&weight, &bias, current, current_count, next, maximum);
        if (status != F2RT_OK) return status;
        current_count = weight.dims[0];
        if (layer + 1U != layers) activate(next, current_count, activation);
        current = next;
    }
    if (current_count > output_capacity ||
        !count_fits_size_t(current_count, sizeof(float))) return F2RT_ERR_OUTPUT;
    memcpy(output, current, (size_t)current_count * sizeof(float));
    return F2RT_OK;
}

static f2rt_status_t sequence_output_count(const f2rt_model_t *model, uint32_t tensor_start,
                                           uint32_t layers, uint32_t *output_count)
{
    f2rt_tensor_t weight;
    uint32_t last_index;
    if (model == NULL || output_count == NULL || layers == 0U ||
        tensor_start > model->tensor_count || layers > (model->tensor_count - tensor_start) / 2U) {
        return F2RT_ERR_TENSOR;
    }
    last_index = tensor_start + (layers - 1U) * 2U;
    if (tensor_at(model, last_index, &weight) != F2RT_OK || weight.ndim != 2U) return F2RT_ERR_TENSOR;
    *output_count = weight.dims[0];
    return F2RT_OK;
}

f2rt_status_t f2rt_bind(const void *payload, uint32_t payload_bytes,
                        f2rt_model_t *model)
{
    const uint8_t *raw = (const uint8_t *)payload;
    f2rt_model_t candidate;
    uint32_t table_bytes, total, i;
    f2rt_status_t status;
    if (payload == NULL || model == NULL || payload_bytes < F2RT_HEADER_BYTES) {
        return F2RT_ERR_ARGUMENT;
    }
    if (memcmp(raw, "F2RT", 4U) != 0 || rd32(raw + 4U) != 1U ||
        rd32(raw + 8U) != F2RT_HEADER_BYTES) {
        return F2RT_ERR_SCHEMA;
    }
    memset(model, 0, sizeof(*model));
    memset(&candidate, 0, sizeof(candidate));
    candidate.payload = raw;
    candidate.payload_bytes = payload_bytes;
    candidate.kind = rd32(raw + 12U);
    candidate.activation = rd32(raw + 16U);
    candidate.layer_count = rd32(raw + 20U);
    candidate.tensor_count = rd32(raw + 24U);
    candidate.input_elems = rd32(raw + 28U);
    candidate.output_elems = rd32(raw + 32U);
    candidate.workspace_elems = rd32(raw + 36U);
    candidate.postprocess = rd32(raw + 40U);
    for (i = 0U; i < 4U; ++i) candidate.aux[i] = rd32(raw + 44U + i * 4U);
    total = rd32(raw + 60U);
    if (candidate.tensor_count == 0U ||
        candidate.tensor_count > (payload_bytes - F2RT_HEADER_BYTES) / F2RT_TENSOR_BYTES ||
        !mul_u32(candidate.tensor_count, F2RT_TENSOR_BYTES, &table_bytes) ||
        !span_ok(F2RT_HEADER_BYTES, table_bytes, payload_bytes) ||
        total != payload_bytes || candidate.kind < F2RT_KIND_SEQUENCE || candidate.kind > F2RT_KIND_NANOLM ||
        candidate.activation > F2RT_ACT_GELU || candidate.postprocess > F2RT_POST_SOFTPLUS ||
        candidate.input_elems == 0U || candidate.output_elems == 0U ||
        !count_fits_size_t(candidate.input_elems, sizeof(float)) ||
        !count_fits_size_t(candidate.output_elems, sizeof(float)) ||
        !count_fits_size_t(candidate.workspace_elems, sizeof(float))) {
        return F2RT_ERR_BOUNDS;
    }
    for (i = 0U; i < candidate.tensor_count; ++i) {
        f2rt_tensor_t tensor;
        status = tensor_at(&candidate, i, &tensor);
        if (status != F2RT_OK) return status;
    }
    *model = candidate;
    return F2RT_OK;
}

static f2rt_status_t infer_residual(const f2rt_model_t *model, const float *input,
                                    float *output, uint32_t output_capacity,
                                    float *workspace, uint32_t workspace_elems)
{
    f2rt_tensor_t weight, bias, residual_scale;
    float baseline[8], residual[8];
    uint32_t residual_count;
    f2rt_status_t status;
    if (tensor_at(model, 0U, &weight) != F2RT_OK || tensor_at(model, 1U, &bias) != F2RT_OK ||
        tensor_at(model, 2U, &residual_scale) != F2RT_OK || model->output_elems > 8U ||
        model->output_elems == 0U || weight.ndim != 2U || weight.dims[0] != model->output_elems ||
        !tensor_is_vector(&residual_scale, 1U) ||
        sequence_output_count(model, 3U, model->layer_count, &residual_count) != F2RT_OK ||
        residual_count != model->output_elems) return F2RT_ERR_TENSOR;
    status = dense(&weight, &bias, input, model->input_elems, baseline, 8U);
    if (status != F2RT_OK) return status;
    status = run_sequence(model, 3U, model->layer_count, model->activation, input, model->input_elems,
                          residual, 8U, workspace, workspace_elems);
    if (status != F2RT_OK || model->output_elems > output_capacity) return status != F2RT_OK ? status : F2RT_ERR_OUTPUT;
    {
        uint32_t i;
        float scale = qvalue(&residual_scale, 0U, 0U);
        for (i = 0U; i < model->output_elems; ++i) output[i] = baseline[i] + residual[i] * scale;
    }
    return F2RT_OK;
}

static f2rt_status_t infer_ridge(const f2rt_model_t *model, const float *input, float *output)
{
    f2rt_tensor_t weight;
    uint32_t i;
    float sum;
    if (tensor_at(model, 0U, &weight) != F2RT_OK || weight.ndim != 1U ||
        model->input_elems == 0U || weight.dims[0] != model->input_elems - 1U ||
        model->output_elems != 1U) return F2RT_ERR_TENSOR;
    sum = input[model->input_elems - 1U];
    for (i = 0U; i < weight.dims[0]; ++i) sum += input[i] * qvalue(&weight, 0U, i);
    output[0] = sum;
    return F2RT_OK;
}

static f2rt_status_t infer_polynomial(const f2rt_model_t *model, const float *input, float *output)
{
    f2rt_tensor_t coefficient;
    uint32_t i;
    float result = 0.0f;
    if (tensor_at(model, 0U, &coefficient) != F2RT_OK || coefficient.ndim != 1U ||
        model->input_elems != 1U || model->output_elems != 1U) return F2RT_ERR_TENSOR;
    for (i = 0U; i < coefficient.dims[0]; ++i) result = result * input[0] + qvalue(&coefficient, 0U, i);
    output[0] = result * 100.0f;
    return F2RT_OK;
}

static f2rt_status_t infer_skip(const f2rt_model_t *model, const float *input, float *output,
                                float *workspace, uint32_t workspace_elems)
{
    f2rt_tensor_t weight, bias;
    float direct[8], hidden[8];
    uint32_t i, hidden_count;
    f2rt_status_t status;
    if (model->output_elems == 0U || model->output_elems > 8U ||
        tensor_at(model, 0U, &weight) != F2RT_OK || tensor_at(model, 1U, &bias) != F2RT_OK ||
        weight.ndim != 2U || weight.dims[0] != model->output_elems ||
        sequence_output_count(model, 2U, model->layer_count, &hidden_count) != F2RT_OK ||
        hidden_count != model->output_elems) return F2RT_ERR_TENSOR;
    status = dense(&weight, &bias, input, model->input_elems, direct, 8U);
    if (status != F2RT_OK) return status;
    status = run_sequence(model, 2U, model->layer_count, model->activation, input, model->input_elems,
                          hidden, 8U, workspace, workspace_elems);
    if (status != F2RT_OK) return status;
    for (i = 0U; i < model->output_elems; ++i) output[i] = direct[i] + hidden[i];
    return F2RT_OK;
}

static f2rt_status_t infer_multihead(const f2rt_model_t *model, const float *input, float *output,
                                     float *workspace, uint32_t workspace_elems)
{
    f2rt_tensor_t last_body_weight, alpha_w, alpha_b, cls_w, cls_b;
    uint32_t body_tensors;
    uint32_t hidden_count;
    float *hidden = workspace;
    f2rt_status_t status;
    if (model->layer_count == 0U || model->output_elems < 2U || model->tensor_count < 4U ||
        model->layer_count > (model->tensor_count - 4U) / 2U ||
        !mul_u32(model->layer_count, 2U, &body_tensors) || body_tensors > model->tensor_count - 4U) {
        return F2RT_ERR_TENSOR;
    }
    if (tensor_at(model, body_tensors - 2U, &last_body_weight) != F2RT_OK ||
        tensor_at(model, body_tensors, &alpha_w) != F2RT_OK || tensor_at(model, body_tensors + 1U, &alpha_b) != F2RT_OK ||
        tensor_at(model, body_tensors + 2U, &cls_w) != F2RT_OK || tensor_at(model, body_tensors + 3U, &cls_b) != F2RT_OK) return F2RT_ERR_TENSOR;
    hidden_count = last_body_weight.dims[0];
    if (hidden_count == 0U || hidden_count > workspace_elems ||
        !count_fits_size_t(hidden_count, sizeof(float)) || alpha_w.ndim != 2U || alpha_w.dims[0] != 1U ||
        alpha_w.dims[1] != hidden_count || !tensor_is_vector(&alpha_b, 1U) ||
        cls_w.ndim != 2U || cls_w.dims[0] != model->output_elems - 1U ||
        cls_w.dims[1] != hidden_count || !tensor_is_vector(&cls_b, model->output_elems - 1U)) {
        return F2RT_ERR_TENSOR;
    }
    status = run_sequence(model, 0U, model->layer_count, model->activation, input, model->input_elems,
                          hidden, hidden_count, workspace + hidden_count, workspace_elems - hidden_count);
    if (status != F2RT_OK) return status;
    status = dense(&alpha_w, &alpha_b, hidden, hidden_count, output, model->output_elems);
    if (status != F2RT_OK) return status;
    status = dense(&cls_w, &cls_b, hidden, hidden_count, output + 1U, model->output_elems - 1U);
    if (status != F2RT_OK) return status;
    apply_postprocess(output, model->output_elems, F2RT_POST_FIRST_SIGMOID_REST_SOFTMAX);
    return F2RT_OK;
}

static f2rt_status_t infer_conv(const f2rt_model_t *model, const float *input,
                                float *output, uint32_t output_capacity,
                                float *workspace, uint32_t workspace_elems)
{
    uint32_t channels = model->aux[0], height = model->aux[1], width = model->aux[2];
    uint64_t plane64;
    uint64_t output_elems;
    uint32_t plane, maximum;
    float *a = workspace, *b;
    const float *current = input;
    uint32_t current_channels = channels, layer;
    if (height == 0U || width == 0U || height > INT32_MAX || width > INT32_MAX ||
        channels == 0U || channels > 24U || model->layer_count == 0U ||
        model->layer_count > model->tensor_count / 2U) return F2RT_ERR_TENSOR;
    plane64 = (uint64_t)height * (uint64_t)width;
    if (plane64 > UINT32_MAX / 24U) return F2RT_ERR_WORKSPACE;
    plane = (uint32_t)plane64;
    if (channels > UINT32_MAX / plane || channels * plane != model->input_elems) return F2RT_ERR_TENSOR;
    maximum = 24U * plane;
    if (maximum > workspace_elems / 2U ||
        !count_fits_size_t((uintmax_t)maximum * 2U, sizeof(float))) return F2RT_ERR_WORKSPACE;
    b = workspace + maximum;
    for (layer = 0U; layer < model->layer_count; ++layer) {
        f2rt_tensor_t weight, bias;
        uint32_t out_channel, y, x, in_channel, ky, kx, kernel, padding, outputs;
        float *next = (layer & 1U) == 0U ? a : b;
        if (tensor_at(model, layer * 2U, &weight) != F2RT_OK || tensor_at(model, layer * 2U + 1U, &bias) != F2RT_OK ||
            weight.ndim != 4U || weight.dims[1] != current_channels || weight.dims[0] == 0U ||
            weight.dims[0] > 24U || !tensor_is_vector(&bias, weight.dims[0])) return F2RT_ERR_TENSOR;
        kernel = weight.dims[2];
        if (kernel == 0U || kernel > 15U || weight.dims[3] != kernel) return F2RT_ERR_TENSOR;
        padding = kernel / 2U; outputs = weight.dims[0];
        for (out_channel = 0U; out_channel < outputs; ++out_channel) {
            for (y = 0U; y < height; ++y) for (x = 0U; x < width; ++x) {
                float sum = qvalue(&bias, 0U, out_channel);
                for (in_channel = 0U; in_channel < current_channels; ++in_channel) {
                    for (ky = 0U; ky < kernel; ++ky) for (kx = 0U; kx < kernel; ++kx) {
                        int32_t iy = (int32_t)y + (int32_t)ky - (int32_t)padding;
                        int32_t ix = (int32_t)x + (int32_t)kx - (int32_t)padding;
                        if (iy >= 0 && ix >= 0 && iy < (int32_t)height && ix < (int32_t)width) {
                            uint32_t input_offset = (in_channel * height + (uint32_t)iy) * width + (uint32_t)ix;
                            uint32_t weight_column = ((in_channel * kernel + ky) * kernel + kx);
                            sum += current[input_offset] * qvalue(&weight, out_channel, weight_column);
                        }
                    }
                }
                next[(out_channel * height + y) * width + x] = sum;
            }
        }
        if (layer + 1U != model->layer_count) activate(next, outputs * height * width, model->activation);
        current = next; current_channels = outputs;
    }
    output_elems = (uint64_t)current_channels * plane64;
    if (output_elems != model->output_elems || output_elems > output_capacity ||
        !count_fits_size_t(output_elems, sizeof(float))) return F2RT_ERR_OUTPUT;
    memcpy(output, current, (size_t)output_elems * sizeof(float));
    return F2RT_OK;
}

f2rt_status_t f2rt_infer_f32(const f2rt_model_t *model,
                             const float *input, uint32_t input_elems,
                             float *output, uint32_t output_capacity,
                             float *workspace, uint32_t workspace_elems)
{
    f2rt_status_t status;
    if (model == NULL || input == NULL || output == NULL || workspace == NULL ||
        input_elems != model->input_elems || output_capacity < model->output_elems ||
        workspace_elems < model->workspace_elems || model->kind == F2RT_KIND_NANOLM ||
        !count_fits_size_t(model->input_elems, sizeof(float)) ||
        !count_fits_size_t(model->output_elems, sizeof(float)) ||
        !count_fits_size_t(model->workspace_elems, sizeof(float))) {
        return F2RT_ERR_ARGUMENT;
    }
    if (model->kind == F2RT_KIND_SEQUENCE) {
        uint32_t sequence_count;
        if (sequence_output_count(model, 0U, model->layer_count, &sequence_count) != F2RT_OK ||
            sequence_count != model->output_elems) return F2RT_ERR_TENSOR;
        status = run_sequence(model, 0U, model->layer_count, model->activation, input, input_elems,
                              output, output_capacity, workspace, workspace_elems);
        if (status == F2RT_OK) apply_postprocess(output, model->output_elems, model->postprocess);
        return status;
    }
    if (model->kind == F2RT_KIND_RESIDUAL) return infer_residual(model, input, output, output_capacity, workspace, workspace_elems);
    if (model->kind == F2RT_KIND_RIDGE_PRIOR) return infer_ridge(model, input, output);
    if (model->kind == F2RT_KIND_POLYNOMIAL) return infer_polynomial(model, input, output);
    if (model->kind == F2RT_KIND_CIE_RESIDUAL) {
        float residual[8]; uint32_t i, residual_count;
        if (input_elems < 2U || model->output_elems != 2U ||
            sequence_output_count(model, 0U, model->layer_count, &residual_count) != F2RT_OK ||
            residual_count != 2U) return F2RT_ERR_TENSOR;
        status = run_sequence(model, 0U, model->layer_count, model->activation, input, input_elems - 2U,
                              residual, 8U, workspace, workspace_elems);
        if (status != F2RT_OK) return status;
        for (i = 0U; i < 2U; ++i) output[i] = input[input_elems - 2U + i] + residual[i] * 0.25f;
        return F2RT_OK;
    }
    if (model->kind == F2RT_KIND_INPUT_PRIOR_RESIDUAL) {
        float residual[8]; uint32_t residual_count;
        if (input_elems < 1U || model->output_elems != 1U ||
            sequence_output_count(model, 0U, model->layer_count, &residual_count) != F2RT_OK ||
            residual_count != 1U) return F2RT_ERR_TENSOR;
        status = run_sequence(model, 0U, model->layer_count, model->activation, input, input_elems - 1U,
                              residual, 8U, workspace, workspace_elems);
        if (status == F2RT_OK) output[0] = input[input_elems - 1U] + residual[0];
        return status;
    }
    if (model->kind == F2RT_KIND_SKIP) return infer_skip(model, input, output, workspace, workspace_elems);
    if (model->kind == F2RT_KIND_MULTIHEAD) return infer_multihead(model, input, output, workspace, workspace_elems);
    if (model->kind == F2RT_KIND_CONV_SEQUENCE) return infer_conv(model, input, output, output_capacity, workspace, workspace_elems);
    return F2RT_ERR_ENGINE;
}

static f2rt_status_t layernorm(const float *input, const f2rt_tensor_t *gain,
                               const f2rt_tensor_t *bias, float *output, uint32_t count)
{
    uint32_t i;
    float mean = 0.0f, variance = 0.0f;
    if (input == NULL || output == NULL || count == 0U ||
        !tensor_is_vector(gain, count) || !tensor_is_vector(bias, count)) return F2RT_ERR_TENSOR;
    for (i = 0U; i < count; ++i) mean += input[i];
    mean /= (float)count;
    for (i = 0U; i < count; ++i) {
        float difference = input[i] - mean;
        variance += difference * difference;
    }
    variance = 1.0f / sqrtf(variance / (float)count + 1e-5f);
    for (i = 0U; i < count; ++i) output[i] = (input[i] - mean) * variance * qvalue(gain, 0U, i) + qvalue(bias, 0U, i);
    return F2RT_OK;
}

static f2rt_status_t nanolm_forward(const f2rt_model_t *model, uint16_t token, uint32_t position,
                                    float *workspace, uint16_t *argmax_out)
{
    uint32_t d = model->aux[0], heads = model->aux[1], ff = model->aux[2], vocab = model->aux[3];
    uint32_t layers = model->layer_count, context = 192U, head_width, layer, i, tensor_index, triple_d;
    uint32_t kv_each;
    uint64_t kv_each64, required64, expected_tensors64;
    float *key_cache = workspace, *value_cache;
    float *scratch;
    float *x, *normalized, *qkv, *attention;
    f2rt_tensor_t token_embedding, position_embedding;
    if (model == NULL || workspace == NULL || argmax_out == NULL || position >= context ||
        d == 0U || heads == 0U || layers == 0U || ff == 0U || vocab == 0U || vocab > 65536U ||
        token >= vocab || d % heads != 0U || !mul_u32(d, 3U, &triple_d) || ff > triple_d) {
        return F2RT_ERR_ARGUMENT;
    }
    kv_each64 = (uint64_t)layers * context * d;
    required64 = kv_each64 * 2U + (uint64_t)d * 5U + context;
    expected_tensors64 = (uint64_t)layers * 12U + 4U;
    if (expected_tensors64 != model->tensor_count) return F2RT_ERR_TENSOR;
    if (kv_each64 > UINT32_MAX || required64 > model->workspace_elems ||
        !count_fits_size_t(required64, sizeof(float))) {
        return F2RT_ERR_WORKSPACE;
    }
    kv_each = (uint32_t)kv_each64;
    value_cache = workspace + kv_each;
    scratch = workspace + kv_each * 2U;
    x = scratch;
    normalized = x + d;
    qkv = normalized + d;
    attention = qkv + triple_d;
    head_width = d / heads;
    if (tensor_at(model, 0U, &token_embedding) != F2RT_OK ||
        tensor_at(model, 1U, &position_embedding) != F2RT_OK ||
        !tensor_is_matrix(&token_embedding, vocab, d) ||
        position_embedding.ndim != 2U || position_embedding.dims[0] < context ||
        position_embedding.dims[1] != d) return F2RT_ERR_TENSOR;
    for (i = 0U; i < d; ++i) x[i] = qvalue(&token_embedding, token, i) + qvalue(&position_embedding, position, i);
    tensor_index = 2U;
    for (layer = 0U; layer < layers; ++layer, tensor_index += 12U) {
        f2rt_tensor_t n1w, n1b, qkvw, qkvb, projw, projb, n2w, n2b, f1w, f1b, f2w, f2b;
        uint32_t head, past, j;
        float *q = qkv, *k = qkv + d, *v = qkv + 2U * d;
        if (tensor_at(model, tensor_index + 0U, &n1w) != F2RT_OK || tensor_at(model, tensor_index + 1U, &n1b) != F2RT_OK ||
            tensor_at(model, tensor_index + 2U, &qkvw) != F2RT_OK || tensor_at(model, tensor_index + 3U, &qkvb) != F2RT_OK ||
            tensor_at(model, tensor_index + 4U, &projw) != F2RT_OK || tensor_at(model, tensor_index + 5U, &projb) != F2RT_OK ||
            tensor_at(model, tensor_index + 6U, &n2w) != F2RT_OK || tensor_at(model, tensor_index + 7U, &n2b) != F2RT_OK ||
            tensor_at(model, tensor_index + 8U, &f1w) != F2RT_OK || tensor_at(model, tensor_index + 9U, &f1b) != F2RT_OK ||
            tensor_at(model, tensor_index + 10U, &f2w) != F2RT_OK || tensor_at(model, tensor_index + 11U, &f2b) != F2RT_OK) return F2RT_ERR_TENSOR;
        if (!tensor_is_vector(&n1w, d) || !tensor_is_vector(&n1b, d) ||
            !tensor_is_matrix(&qkvw, triple_d, d) || !tensor_is_vector(&qkvb, triple_d) ||
            !tensor_is_matrix(&projw, d, d) || !tensor_is_vector(&projb, d) ||
            !tensor_is_vector(&n2w, d) || !tensor_is_vector(&n2b, d) ||
            !tensor_is_matrix(&f1w, ff, d) || !tensor_is_vector(&f1b, ff) ||
            !tensor_is_matrix(&f2w, d, ff) || !tensor_is_vector(&f2b, d)) return F2RT_ERR_TENSOR;
        if (layernorm(x, &n1w, &n1b, normalized, d) != F2RT_OK) return F2RT_ERR_TENSOR;
        if (dense(&qkvw, &qkvb, normalized, d, qkv, triple_d) != F2RT_OK) return F2RT_ERR_TENSOR;
        memcpy(key_cache + (layer * context + position) * d, k, (size_t)d * sizeof(float));
        memcpy(value_cache + (layer * context + position) * d, v, (size_t)d * sizeof(float));
        for (head = 0U; head < heads; ++head) {
            uint32_t offset = head * head_width;
            float maximum, sum = 0.0f, scale = 1.0f / sqrtf((float)head_width);
            for (past = 0U; past <= position; ++past) {
                const float *cached = key_cache + (layer * context + past) * d + offset;
                float dot = 0.0f;
                for (j = 0U; j < head_width; ++j) dot += q[offset + j] * cached[j];
                attention[past] = dot * scale;
            }
            maximum = attention[0];
            for (past = 1U; past <= position; ++past) if (attention[past] > maximum) maximum = attention[past];
            for (past = 0U; past <= position; ++past) { attention[past] = expf(attention[past] - maximum); sum += attention[past]; }
            for (past = 0U; past <= position; ++past) attention[past] /= sum;
            for (j = 0U; j < head_width; ++j) {
                float value = 0.0f;
                for (past = 0U; past <= position; ++past) value += attention[past] * value_cache[(layer * context + past) * d + offset + j];
                normalized[offset + j] = value;
            }
        }
        if (dense(&projw, &projb, normalized, d, qkv, d) != F2RT_OK) return F2RT_ERR_TENSOR;
        for (i = 0U; i < d; ++i) x[i] += qkv[i];
        if (layernorm(x, &n2w, &n2b, normalized, d) != F2RT_OK) return F2RT_ERR_TENSOR;
        if (dense(&f1w, &f1b, normalized, d, qkv, triple_d) != F2RT_OK) return F2RT_ERR_TENSOR;
        activate(qkv, ff, F2RT_ACT_RELU);
        if (dense(&f2w, &f2b, qkv, ff, normalized, d) != F2RT_OK) return F2RT_ERR_TENSOR;
        for (i = 0U; i < d; ++i) x[i] += normalized[i];
    }
    {
        f2rt_tensor_t final_w, final_b;
        uint32_t best = 0U, word;
        float best_logit = -3.402823466e38F;
        if (tensor_at(model, tensor_index, &final_w) != F2RT_OK || tensor_at(model, tensor_index + 1U, &final_b) != F2RT_OK) return F2RT_ERR_TENSOR;
        if (layernorm(x, &final_w, &final_b, normalized, d) != F2RT_OK) return F2RT_ERR_TENSOR;
        for (word = 0U; word < vocab; ++word) {
            float logit = 0.0f;
            for (i = 0U; i < d; ++i) logit += normalized[i] * qvalue(&token_embedding, word, i);
            if (logit > best_logit) { best_logit = logit; best = word; }
        }
        *argmax_out = (uint16_t)best;
    }
    return F2RT_OK;
}

f2rt_status_t f2rt_generate_u16(const f2rt_model_t *model,
                                const uint16_t *prompt, uint32_t prompt_length,
                                uint16_t *generated, uint32_t generated_capacity,
                                float *workspace, uint32_t workspace_elems)
{
    uint32_t position, step;
    uint16_t next = 0U;
    f2rt_status_t status;
    if (model == NULL || prompt == NULL || generated == NULL || workspace == NULL ||
        model->kind != F2RT_KIND_NANOLM || prompt_length == 0U ||
        prompt_length > 192U || generated_capacity > 192U - prompt_length ||
        workspace_elems < model->workspace_elems ||
        !count_fits_size_t(model->workspace_elems, sizeof(float)) ||
        !count_fits_size_t(generated_capacity, sizeof(uint16_t))) return F2RT_ERR_ARGUMENT;
    for (position = 0U; position < prompt_length; ++position) {
        status = nanolm_forward(model, prompt[position], position, workspace, &next);
        if (status != F2RT_OK) return status;
    }
    memset(generated, 0, generated_capacity * sizeof(uint16_t));
    for (step = 0U; step < generated_capacity; ++step) {
        generated[step] = next;
        if (next == 2U) break;
        position = prompt_length + step;
        status = nanolm_forward(model, next, position, workspace, &next);
        if (status != F2RT_OK) return status;
    }
    return F2RT_OK;
}
