#include "forge200_rag_v9.h"

#include "forge200_runtime_v8.h"
#include "sha256.h"

#include <math.h>
#include <string.h>

#define F2RAG_SUPPORT_HEADER_BYTES 128U
#define F2RAG_SUPPORT_ENTRY_BYTES 160U
#define F2RAG_WORKLOAD_HEADER_BYTES 128U
#define F2RAG_WORKLOAD_RECORD_BYTES 13376U
#define F2RAG_PACKAGE_HEADER_BYTES 256U
#define F2RAG_GOLDEN_HEADER_BYTES 64U
#define F2RAG_LM_PACKAGE_BYTES_MAX 2097152U
#define F2RAG_SUPPORT_BUNDLE_BYTES_MAX 1048576U
#define F2RAG_WORKLOAD_BYTES_MAX 1048576U
#define F2RAG_WORKSPACE_RESERVE 4096U

#define F2RAG_OFF_PROMPT 128U
#define F2RAG_OFF_TARGET 464U
#define F2RAG_OFF_ROUTER 512U
#define F2RAG_OFF_SUFF 1316U
#define F2RAG_OFF_ARBITRATION 2120U
#define F2RAG_OFF_REFUSAL 2924U
#define F2RAG_OFF_SPAN 3728U
#define F2RAG_OFF_PROVENANCE 4532U
#define F2RAG_OFF_QUALITY 5336U
#define F2RAG_OFF_TASK_ROUTER 6140U
#define F2RAG_OFF_OOD 6944U
#define F2RAG_OFF_NLI 7748U
#define F2RAG_OFF_RERANK0 8552U
#define F2RAG_OFF_RERANK1 9356U
#define F2RAG_OFF_RERANK2 10160U
#define F2RAG_OFF_TEMPORAL 10964U
#define F2RAG_OFF_ENCODER_Q 11224U
#define F2RAG_OFF_ENCODER_E 11996U
#define F2RAG_OFF_ENCODER_EMBED 12768U
#define F2RAG_OFF_Q_SPARSE 12836U
#define F2RAG_OFF_E_SPARSE 13096U

typedef struct {
    const uint8_t *raw;
    const uint8_t *package;
    const uint8_t *golden;
    uint32_t logical_model;
    uint32_t package_bytes;
    uint32_t golden_bytes;
} f2rag_support_entry_t;

static uint16_t read_u16_le(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t read_u64_le(const uint8_t *p)
{
    return (uint64_t)read_u32_le(p) |
           ((uint64_t)read_u32_le(p + 4U) << 32);
}

static int bytes_all_zero(const uint8_t *p, uint32_t bytes)
{
    uint32_t i;
    for (i = 0U; i < bytes; ++i) {
        if (p[i] != 0U) {
            return 0;
        }
    }
    return 1;
}

static int bounds_ok(uint32_t offset, uint32_t bytes, uint32_t total)
{
    return offset <= total && bytes <= total - offset;
}

static void secure_zero(void *value, uint32_t bytes)
{
    volatile uint8_t *p = (volatile uint8_t *)value;
    while (bytes != 0U) {
        *p++ = 0U;
        --bytes;
    }
}

static uint32_t stage_begin(f2rag_clock_fn clock_fn, void *context)
{
    return clock_fn != NULL ? clock_fn(context) : 0U;
}

static void stage_end(f2rag_metrics_t *metrics, f2rag_stage_t stage,
                      uint32_t start, f2rag_clock_fn clock_fn, void *context)
{
    uint32_t end = clock_fn != NULL ? clock_fn(context) : start;
    metrics->stage_ticks[(uint32_t)stage] = end - start;
    metrics->state_mask |= 1UL << (uint32_t)stage;
}

static int package_header_valid(const uint8_t *package, uint32_t package_bytes,
                                uint16_t expected_engine,
                                const uint8_t expected_candidate[32])
{
    uint8_t digest[32];
    uint64_t payload_bytes;
    if (package == NULL || package_bytes < F2RAG_PACKAGE_HEADER_BYTES ||
        memcmp(package, "ICMF", 4U) != 0 ||
        read_u16_le(package + 4U) != 1U ||
        read_u16_le(package + 6U) != F2RAG_PACKAGE_HEADER_BYTES ||
        read_u16_le(package + 8U) != expected_engine ||
        package[12] != 0U || read_u64_le(package + 16U) == 0U ||
        !bytes_all_zero(package + 204U, 52U)) {
        return 0;
    }
    payload_bytes = read_u64_le(package + 24U);
    if (payload_bytes > 0xFFFFFFFFULL ||
        payload_bytes + F2RAG_PACKAGE_HEADER_BYTES != package_bytes) {
        return 0;
    }
    if (expected_candidate != NULL && memcmp(package + 44U, expected_candidate, 32U) != 0) {
        return 0;
    }
    sha256(package + F2RAG_PACKAGE_HEADER_BYTES, (size_t)payload_bytes, digest);
    return memcmp(digest, package + 76U, 32U) == 0;
}

static int golden_header_valid(const uint8_t *golden, uint32_t golden_bytes,
                               uint16_t expected_engine)
{
    if (golden == NULL || golden_bytes < F2RAG_GOLDEN_HEADER_BYTES ||
        memcmp(golden, "F2GV", 4U) != 0 ||
        read_u32_le(golden + 4U) != 1U ||
        read_u32_le(golden + 8U) != F2RAG_GOLDEN_HEADER_BYTES ||
        read_u32_le(golden + 12U) != (uint32_t)expected_engine) {
        return 0;
    }
    return bytes_all_zero(golden + 44U, 20U);
}

static f2rag_status_t run_golden(const uint8_t *package, uint32_t package_bytes,
                                 const uint8_t *golden, uint32_t golden_bytes,
                                 float *workspace, uint32_t workspace_elems)
{
    f2rt_model_t model;
    f2rt_status_t status;
    uint32_t input_dtype;
    uint32_t output_dtype;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t prompt_length;
    uint32_t tolerance_bits;
    uint32_t i;
    if (!package_header_valid(package, package_bytes, read_u16_le(package + 8U), NULL) ||
        !golden_header_valid(golden, golden_bytes, read_u16_le(package + 8U)) ||
        f2rt_bind(package + F2RAG_PACKAGE_HEADER_BYTES,
                  package_bytes - F2RAG_PACKAGE_HEADER_BYTES, &model) != F2RT_OK ||
        model.workspace_elems > workspace_elems) {
        return F2RAG_ERR_GOLDEN;
    }
    input_dtype = read_u32_le(golden + 20U);
    output_dtype = read_u32_le(golden + 24U);
    input_count = read_u32_le(golden + 28U);
    output_count = read_u32_le(golden + 32U);
    prompt_length = read_u32_le(golden + 36U);
    tolerance_bits = read_u32_le(golden + 40U);
    if (input_dtype == 2U && output_dtype == 2U) {
        float tolerance;
        const float *input;
        const float *expected;
        float actual[64];
        if (output_count == 0U || output_count > 64U ||
            golden_bytes != F2RAG_GOLDEN_HEADER_BYTES +
                (input_count + output_count) * (uint32_t)sizeof(float)) {
            return F2RAG_ERR_GOLDEN;
        }
        memcpy(&tolerance, &tolerance_bits, sizeof(tolerance));
        input = (const float *)(const void *)(golden + F2RAG_GOLDEN_HEADER_BYTES);
        expected = input + input_count;
        status = f2rt_infer_f32(&model, input, input_count, actual, output_count,
                                workspace, workspace_elems);
        if (status != F2RT_OK) {
            return F2RAG_ERR_MODEL;
        }
        for (i = 0U; i < output_count; ++i) {
            if (!isfinite(actual[i]) || fabsf(actual[i] - expected[i]) > tolerance) {
                return F2RAG_ERR_GOLDEN;
            }
        }
        return F2RAG_OK;
    }
    if (input_dtype == 4U && output_dtype == 4U) {
        const uint16_t *input;
        const uint16_t *expected;
        uint16_t actual[F2RAG_GENERATION_TOKENS_MAX];
        if (output_count == 0U || output_count > F2RAG_GENERATION_TOKENS_MAX ||
            prompt_length == 0U || prompt_length > input_count ||
            golden_bytes != F2RAG_GOLDEN_HEADER_BYTES +
                (input_count + output_count) * (uint32_t)sizeof(uint16_t)) {
            return F2RAG_ERR_GOLDEN;
        }
        input = (const uint16_t *)(const void *)(golden + F2RAG_GOLDEN_HEADER_BYTES);
        expected = input + input_count;
        status = f2rt_generate_u16(&model, input, prompt_length, actual, output_count,
                                   workspace, workspace_elems);
        if (status != F2RT_OK) {
            return F2RAG_ERR_MODEL;
        }
        for (i = 0U; i < output_count; ++i) {
            if (actual[i] != expected[i]) {
                return F2RAG_ERR_GOLDEN;
            }
        }
        return F2RAG_OK;
    }
    return F2RAG_ERR_GOLDEN;
}

static f2rag_status_t support_entry_at(const uint8_t *bundle, uint32_t bundle_bytes,
                                       uint32_t index, f2rag_support_entry_t *entry)
{
    const uint8_t *raw;
    uint32_t table_offset;
    uint32_t package_offset;
    uint32_t golden_offset;
    if (bundle == NULL || entry == NULL || index >= F2RAG_SUPPORT_MODEL_COUNT) {
        return F2RAG_ERR_ARGUMENT;
    }
    table_offset = read_u32_le(bundle + 28U);
    if (!bounds_ok(table_offset, F2RAG_SUPPORT_MODEL_COUNT * F2RAG_SUPPORT_ENTRY_BYTES,
                   bundle_bytes)) {
        return F2RAG_ERR_BOUNDS;
    }
    raw = bundle + table_offset + index * F2RAG_SUPPORT_ENTRY_BYTES;
    package_offset = read_u32_le(raw + 4U);
    golden_offset = read_u32_le(raw + 12U);
    entry->raw = raw;
    entry->logical_model = read_u16_le(raw);
    entry->package_bytes = read_u32_le(raw + 8U);
    entry->golden_bytes = read_u32_le(raw + 16U);
    if (!bounds_ok(package_offset, entry->package_bytes, bundle_bytes) ||
        !bounds_ok(golden_offset, entry->golden_bytes, bundle_bytes)) {
        return F2RAG_ERR_BOUNDS;
    }
    entry->package = bundle + package_offset;
    entry->golden = bundle + golden_offset;
    return F2RAG_OK;
}

static f2rag_status_t find_support_entry(const uint8_t *bundle, uint32_t bundle_bytes,
                                         uint32_t logical_model,
                                         f2rag_support_entry_t *entry)
{
    uint32_t i;
    for (i = 0U; i < F2RAG_SUPPORT_MODEL_COUNT; ++i) {
        f2rag_status_t status = support_entry_at(bundle, bundle_bytes, i, entry);
        if (status != F2RAG_OK) {
            return status;
        }
        if (entry->logical_model == logical_model) {
            return F2RAG_OK;
        }
    }
    return F2RAG_ERR_MODEL;
}

f2rag_status_t f2rag_validate_support_bundle(
    const uint8_t *bundle, uint32_t bundle_bytes,
    float *workspace, uint32_t workspace_elems)
{
    uint8_t digest[32];
    uint32_t i;
    if (bundle == NULL || workspace == NULL ||
        bundle_bytes < F2RAG_SUPPORT_HEADER_BYTES ||
        bundle_bytes > F2RAG_SUPPORT_BUNDLE_BYTES_MAX ||
        memcmp(bundle, "F2SB", 4U) != 0 || read_u16_le(bundle + 4U) != 1U ||
        read_u16_le(bundle + 6U) != F2RAG_SUPPORT_HEADER_BYTES ||
        read_u32_le(bundle + 8U) >= F2RAG_DOMAIN_COUNT ||
        read_u32_le(bundle + 12U) != F2RAG_SUPPORT_MODEL_COUNT ||
        read_u64_le(bundle + 16U) == 0U ||
        read_u32_le(bundle + 24U) != F2RAG_SUPPORT_ENTRY_BYTES ||
        read_u32_le(bundle + 28U) != F2RAG_SUPPORT_HEADER_BYTES ||
        read_u32_le(bundle + 32U) != bundle_bytes ||
        read_u32_le(bundle + 36U) != 0U ||
        !bytes_all_zero(bundle + 104U, 24U)) {
        return F2RAG_ERR_SCHEMA;
    }
    sha256(bundle + F2RAG_SUPPORT_HEADER_BYTES,
           bundle_bytes - F2RAG_SUPPORT_HEADER_BYTES, digest);
    if (memcmp(digest, bundle + 40U, 32U) != 0) {
        return F2RAG_ERR_HASH;
    }
    for (i = 0U; i < F2RAG_SUPPORT_MODEL_COUNT; ++i) {
        f2rag_support_entry_t entry;
        f2rag_status_t status = support_entry_at(bundle, bundle_bytes, i, &entry);
        if (status != F2RAG_OK) {
            return status;
        }
        sha256(entry.package, entry.package_bytes, digest);
        if (memcmp(digest, entry.raw + 60U, 32U) != 0) {
            return F2RAG_ERR_HASH;
        }
        sha256(entry.golden, entry.golden_bytes, digest);
        if (memcmp(digest, entry.raw + 92U, 32U) != 0 ||
            !package_header_valid(entry.package, entry.package_bytes,
                                  read_u16_le(entry.raw + 2U), entry.raw + 28U) ||
            memcmp(entry.package + 108U, entry.raw + 92U, 32U) != 0 ||
            memcmp(entry.package + 140U, entry.raw + 124U, 32U) != 0 ||
            entry.package[12] != 0U) {
            return F2RAG_ERR_HASH;
        }
        status = run_golden(entry.package, entry.package_bytes,
                            entry.golden, entry.golden_bytes,
                            workspace, workspace_elems);
        if (status != F2RAG_OK) {
            return status;
        }
    }
    return F2RAG_OK;
}

f2rag_status_t f2rag_validate_workload(const uint8_t *workload,
                                       uint32_t workload_bytes)
{
    uint8_t digest[32];
    if (workload == NULL || workload_bytes < F2RAG_WORKLOAD_HEADER_BYTES ||
        workload_bytes > F2RAG_WORKLOAD_BYTES_MAX ||
        memcmp(workload, "F2RW", 4U) != 0 || read_u16_le(workload + 4U) != 1U ||
        read_u16_le(workload + 6U) != F2RAG_WORKLOAD_HEADER_BYTES ||
        read_u32_le(workload + 8U) != F2RAG_WORKLOAD_PER_DOMAIN ||
        read_u32_le(workload + 12U) != F2RAG_WORKLOAD_RECORD_BYTES ||
        read_u32_le(workload + 16U) >= F2RAG_DOMAIN_COUNT ||
        read_u32_le(workload + 20U) != F2RAG_WORKLOAD_HEADER_BYTES ||
        read_u32_le(workload + 24U) != workload_bytes ||
        read_u32_le(workload + 28U) != 0U) {
        return F2RAG_ERR_SCHEMA;
    }
    sha256(workload + F2RAG_WORKLOAD_HEADER_BYTES,
           workload_bytes - F2RAG_WORKLOAD_HEADER_BYTES, digest);
    if (memcmp(digest, workload + 32U, 32U) != 0) {
        return F2RAG_ERR_HASH;
    }
    return F2RAG_OK;
}

static void dequantize(const uint8_t *raw, uint32_t count, float *output)
{
    float scale;
    uint32_t i;
    memcpy(&scale, raw, sizeof(scale));
    for (i = 0U; i < count; ++i) {
        output[i] = scale * (float)(int8_t)raw[4U + i];
    }
}

static uint16_t output_label(const float *output, uint32_t count)
{
    uint32_t best = 0U;
    uint32_t i;
    for (i = 1U; i < count; ++i) {
        if (output[i] > output[best]) {
            best = i;
        }
    }
    return (uint16_t)best;
}

static f2rag_status_t run_support_model(
    const uint8_t *bundle, uint32_t bundle_bytes, uint32_t logical_model,
    const uint8_t *quantized_input, uint32_t input_count,
    float *output, uint32_t output_capacity,
    float *workspace, uint32_t model_workspace_elems,
    float *input_buffer, f2rt_model_t *bound_model)
{
    f2rag_support_entry_t entry;
    f2rag_status_t status = find_support_entry(bundle, bundle_bytes, logical_model, &entry);
    if (status != F2RAG_OK || quantized_input == NULL || input_count > 800U) {
        return status != F2RAG_OK ? status : F2RAG_ERR_ARGUMENT;
    }
    if (f2rt_bind(entry.package + F2RAG_PACKAGE_HEADER_BYTES,
                  entry.package_bytes - F2RAG_PACKAGE_HEADER_BYTES,
                  bound_model) != F2RT_OK ||
        bound_model->input_elems != input_count ||
        bound_model->output_elems > output_capacity ||
        bound_model->workspace_elems > model_workspace_elems) {
        return F2RAG_ERR_MODEL;
    }
    dequantize(quantized_input, input_count, input_buffer);
    if (f2rt_infer_f32(bound_model, input_buffer, input_count,
                       output, output_capacity, workspace,
                       model_workspace_elems) != F2RT_OK) {
        return F2RAG_ERR_MODEL;
    }
    return F2RAG_OK;
}

static float dot_quantized(const uint8_t *a, const uint8_t *b, uint32_t count)
{
    float scale_a;
    float scale_b;
    int32_t accumulator = 0;
    uint32_t i;
    memcpy(&scale_a, a, sizeof(scale_a));
    memcpy(&scale_b, b, sizeof(scale_b));
    for (i = 0U; i < count; ++i) {
        accumulator += (int32_t)(int8_t)a[4U + i] * (int32_t)(int8_t)b[4U + i];
    }
    return scale_a * scale_b * (float)accumulator;
}

static uint32_t rank_of(const float scores[F2RAG_WORKLOAD_PER_DOMAIN], uint32_t index)
{
    uint32_t rank = 0U;
    uint32_t i;
    if (index >= F2RAG_WORKLOAD_PER_DOMAIN) {
        return F2RAG_WORKLOAD_PER_DOMAIN;
    }
    for (i = 0U; i < F2RAG_WORKLOAD_PER_DOMAIN; ++i) {
        if (scores[i] > scores[index] ||
            (scores[i] == scores[index] && i < index)) {
            ++rank;
        }
    }
    return rank;
}

static f2rag_status_t validate_lm(const uint8_t *package, uint32_t package_bytes,
                                  const uint8_t *golden, uint32_t golden_bytes,
                                  const uint8_t expected_candidate[32],
                                  float *workspace, uint32_t workspace_elems)
{
    uint8_t digest[32];
    if (package_bytes > F2RAG_LM_PACKAGE_BYTES_MAX ||
        !package_header_valid(package, package_bytes, 5U, expected_candidate) ||
        read_u16_le(package + 10U) != 1U || golden == NULL) {
        return F2RAG_ERR_SCHEMA;
    }
    sha256(golden, golden_bytes, digest);
    if (memcmp(digest, package + 108U, 32U) != 0) {
        return F2RAG_ERR_HASH;
    }
    return run_golden(package, package_bytes, golden, golden_bytes,
                      workspace, workspace_elems);
}

f2rag_status_t f2rag_run_query(
    const uint8_t *support_bundle, uint32_t support_bundle_bytes,
    const uint8_t *workload, uint32_t workload_bytes,
    uint32_t local_query_index,
    uint8_t *lm_package, uint32_t lm_package_bytes,
    const uint8_t *lm_golden, uint32_t lm_golden_bytes,
    float *workspace, uint32_t workspace_elems,
    f2rag_clock_fn clock_fn, void *clock_context,
    f2rag_metrics_t *metrics, f2rag_result_t *result)
{
    const uint8_t *record;
    const uint8_t *records;
    float *scratch;
    float *input;
    float *output;
    float *query_embedding;
    float sparse_scores[F2RAG_WORKLOAD_PER_DOMAIN];
    float dense_scores[F2RAG_WORKLOAD_PER_DOMAIN];
    float fused_scores[F2RAG_WORKLOAD_PER_DOMAIN];
    float rerank_scores[3];
    uint16_t top3[3];
    uint32_t model_workspace_elems;
    uint32_t domain;
    uint32_t domain_encoder;
    uint32_t domain_reranker;
    uint32_t domain_nli;
    uint32_t i;
    uint32_t start;
    uint32_t best;
    uint32_t target_length;
    uint32_t prompt_length;
    uint32_t retrieved_self;
    uint32_t reranked_self;
    uint32_t generation_exact = 1U;
    uint32_t gate_refuse;
    f2rt_model_t model;
    f2rag_status_t status = F2RAG_OK;

    if (support_bundle == NULL || workload == NULL || lm_package == NULL ||
        lm_golden == NULL || workspace == NULL || metrics == NULL || result == NULL ||
        local_query_index >= F2RAG_WORKLOAD_PER_DOMAIN ||
        workspace_elems <= F2RAG_WORKSPACE_RESERVE) {
        return F2RAG_ERR_ARGUMENT;
    }
    memset(metrics, 0, sizeof(*metrics));
    memset(result, 0, sizeof(*result));
    model_workspace_elems = workspace_elems - F2RAG_WORKSPACE_RESERVE;
    scratch = workspace + model_workspace_elems;
    input = scratch;
    output = input + 800U;
    query_embedding = output + 128U;
    records = workload + F2RAG_WORKLOAD_HEADER_BYTES;
    record = records + local_query_index * F2RAG_WORKLOAD_RECORD_BYTES;

    start = stage_begin(clock_fn, clock_context);
    status = f2rag_validate_support_bundle(support_bundle, support_bundle_bytes,
                                           workspace, model_workspace_elems);
    if (status != F2RAG_OK) {
        goto fail;
    }
    if (f2rag_validate_workload(workload, workload_bytes) != F2RAG_OK ||
        read_u32_le(support_bundle + 8U) != read_u32_le(workload + 16U)) {
        status = F2RAG_ERR_SCHEMA;
        goto fail;
    }
    domain = read_u32_le(workload + 16U);
    if (read_u16_le(record + 4U) != domain || (read_u16_le(record + 6U) & 2U) == 0U) {
        status = F2RAG_ERR_SCHEMA;
        goto fail;
    }
    result->query_id = read_u32_le(record);
    result->domain_id = (uint8_t)domain;
    result->expected_refusal = (uint8_t)(read_u16_le(record + 6U) & 1U);
    metrics->cold_sd_read_bytes = support_bundle_bytes + workload_bytes +
                                  lm_package_bytes + lm_golden_bytes;
    stage_end(metrics, F2RAG_LOAD_SUPPORT_A, start, clock_fn, clock_context);

    start = stage_begin(clock_fn, clock_context);
    domain_encoder = 181U + domain;
    domain_reranker = 187U + domain;
    domain_nli = 193U + domain;

#define RUN_SUPPORT(LOGICAL, OFFSET, COUNT) \
    do { \
        status = run_support_model(support_bundle, support_bundle_bytes, (LOGICAL), \
                                   record + (OFFSET), (COUNT), output, 128U, \
                                   workspace, model_workspace_elems, input, &model); \
        if (status != F2RAG_OK) { goto fail; } \
    } while (0)

    RUN_SUPPORT(173U, F2RAG_OFF_ROUTER, 800U);
    metrics->router_label = output_label(output, model.output_elems);
    RUN_SUPPORT(174U, F2RAG_OFF_SUFF, 800U);
    metrics->sufficiency_label = output_label(output, model.output_elems);
    RUN_SUPPORT(175U, F2RAG_OFF_ARBITRATION, 800U);
    RUN_SUPPORT(176U, F2RAG_OFF_REFUSAL, 800U);
    metrics->refusal_label = output_label(output, model.output_elems);
    RUN_SUPPORT(177U, F2RAG_OFF_SPAN, 800U);
    RUN_SUPPORT(178U, F2RAG_OFF_PROVENANCE, 800U);
    metrics->provenance_label = output_label(output, model.output_elems);
    RUN_SUPPORT(179U, F2RAG_OFF_QUALITY, 800U);
    metrics->quality_score = output[0];
    RUN_SUPPORT(180U, F2RAG_OFF_TEMPORAL, 256U);
    RUN_SUPPORT(199U, F2RAG_OFF_TASK_ROUTER, 800U);
    RUN_SUPPORT(200U, F2RAG_OFF_OOD, 800U);
    metrics->ood_label = output_label(output, model.output_elems);
    RUN_SUPPORT(domain_encoder, F2RAG_OFF_ENCODER_Q, 768U);
    if (model.output_elems != 64U) {
        status = F2RAG_ERR_MODEL;
        goto fail;
    }
    memcpy(query_embedding, output, 64U * sizeof(float));
    {
        float norm = 0.0f;
        for (i = 0U; i < 64U; ++i) {
            norm += query_embedding[i] * query_embedding[i];
        }
        norm = sqrtf(norm);
        if (!(norm > 0.0f)) {
            status = F2RAG_ERR_MODEL;
            goto fail;
        }
        for (i = 0U; i < 64U; ++i) {
            query_embedding[i] /= norm;
        }
    }
    for (i = 0U; i < F2RAG_WORKLOAD_PER_DOMAIN; ++i) {
        const uint8_t *peer = records + i * F2RAG_WORKLOAD_RECORD_BYTES;
        float evidence[64];
        uint32_t j;
        sparse_scores[i] = dot_quantized(record + F2RAG_OFF_Q_SPARSE,
                                         peer + F2RAG_OFF_E_SPARSE, 256U);
        dequantize(peer + F2RAG_OFF_ENCODER_EMBED, 64U, evidence);
        dense_scores[i] = 0.0f;
        for (j = 0U; j < 64U; ++j) {
            dense_scores[i] += query_embedding[j] * evidence[j];
        }
    }
    best = 0U;
    for (i = 0U; i < F2RAG_WORKLOAD_PER_DOMAIN; ++i) {
        fused_scores[i] = 1.0f / (60.0f + (float)rank_of(sparse_scores, i)) +
                          1.0f / (60.0f + (float)rank_of(dense_scores, i));
        if (fused_scores[i] > fused_scores[best]) {
            best = i;
        }
    }
    metrics->retrieved_local_index = best;
    top3[0] = read_u16_le(record + 112U);
    top3[1] = read_u16_le(record + 114U);
    top3[2] = read_u16_le(record + 116U);
    best = 0U;
    for (i = 0U; i < 3U; ++i) {
        static const uint32_t rerank_offsets[3] = {
            F2RAG_OFF_RERANK0, F2RAG_OFF_RERANK1, F2RAG_OFF_RERANK2
        };
        if (top3[i] >= F2RAG_WORKLOAD_PER_DOMAIN) {
            status = F2RAG_ERR_SCHEMA;
            goto fail;
        }
        status = run_support_model(
            support_bundle, support_bundle_bytes, domain_reranker,
            record + rerank_offsets[i], domain == 4U ? 2U : 800U,
            output, 128U, workspace, model_workspace_elems, input, &model);
        if (status != F2RAG_OK) {
            goto fail;
        }
        rerank_scores[i] = model.output_elems > 1U ? output[1] : output[0];
        if (fused_scores[top3[i]] + 0.02f * rerank_scores[i] >
            fused_scores[top3[best]] + 0.02f * rerank_scores[best]) {
            best = i;
        }
    }
    metrics->reranked_local_index = top3[best];
    metrics->support_models_executed = F2RAG_SUPPORT_MODEL_COUNT;
    stage_end(metrics, F2RAG_ROUTE_ENCODE_RETRIEVE_RERANK, start,
              clock_fn, clock_context);

    start = stage_begin(clock_fn, clock_context);
    status = validate_lm(lm_package, lm_package_bytes, lm_golden, lm_golden_bytes,
                         record + 80U, workspace, model_workspace_elems);
    if (status != F2RAG_OK) {
        goto fail;
    }
    stage_end(metrics, F2RAG_LOAD_LM_B, start, clock_fn, clock_context);

    start = stage_begin(clock_fn, clock_context);
    prompt_length = read_u16_le(record + 10U);
    target_length = read_u16_le(record + 12U);
    if (prompt_length == 0U || prompt_length > 168U || target_length == 0U ||
        target_length > F2RAG_GENERATION_TOKENS_MAX ||
        f2rt_bind(lm_package + F2RAG_PACKAGE_HEADER_BYTES,
                  lm_package_bytes - F2RAG_PACKAGE_HEADER_BYTES, &model) != F2RT_OK ||
        model.workspace_elems > model_workspace_elems ||
        f2rt_generate_u16(&model,
            (const uint16_t *)(const void *)(record + F2RAG_OFF_PROMPT),
            prompt_length, result->generated, target_length,
            workspace, model_workspace_elems) != F2RT_OK) {
        status = F2RAG_ERR_GENERATION;
        goto fail;
    }
    metrics->generation_tokens = target_length;
    for (i = 0U; i < target_length; ++i) {
        if (result->generated[i] != read_u16_le(record + F2RAG_OFF_TARGET + i * 2U)) {
            generation_exact = 0U;
        }
    }
    result->generation_exact = (uint8_t)generation_exact;
    stage_end(metrics, F2RAG_GENERATE, start, clock_fn, clock_context);

    start = stage_begin(clock_fn, clock_context);
    secure_zero(lm_package, lm_package_bytes);
    result->lm_slot_zeroized = (uint8_t)bytes_all_zero(lm_package, lm_package_bytes);
    if (result->lm_slot_zeroized == 0U) {
        status = F2RAG_ERR_STATE;
        goto fail;
    }
    stage_end(metrics, F2RAG_UNLOAD_LM_B, start, clock_fn, clock_context);

    start = stage_begin(clock_fn, clock_context);
    RUN_SUPPORT(domain_nli, F2RAG_OFF_NLI, 800U);
    metrics->nli_label = output_label(output, model.output_elems);
    stage_end(metrics, F2RAG_NLI_QUALITY_A, start, clock_fn, clock_context);

    start = stage_begin(clock_fn, clock_context);
    retrieved_self = metrics->retrieved_local_index == local_query_index;
    reranked_self = metrics->reranked_local_index == local_query_index;
    gate_refuse = !retrieved_self || !reranked_self || metrics->router_label != domain ||
                  metrics->ood_label != 0U || metrics->sufficiency_label != 1U ||
                  metrics->refusal_label != 0U || metrics->provenance_label != 0U ||
                  metrics->nli_label != 0U || metrics->quality_score < 0.5f ||
                  generation_exact == 0U;
    result->refused = (uint8_t)(gate_refuse != 0U);
    result->published = (uint8_t)(gate_refuse == 0U);
    result->source_bound = (uint8_t)(result->published != 0U && retrieved_self != 0U &&
                                     reranked_self != 0U && generation_exact != 0U &&
                                     metrics->nli_label == 0U &&
                                     metrics->provenance_label == 0U);
    result->safe_outcome = (uint8_t)(result->refused != 0U || result->source_bound != 0U);
    if (result->expected_refusal != 0U && result->refused == 0U) {
        result->safe_outcome = 0U;
    }
    stage_end(metrics, F2RAG_COMMIT_OR_REFUSE, start, clock_fn, clock_context);

    start = stage_begin(clock_fn, clock_context);
    secure_zero(workspace, workspace_elems * (uint32_t)sizeof(float));
    result->workspace_zeroized = 1U;
    stage_end(metrics, F2RAG_ZEROIZE, start, clock_fn, clock_context);
    if (metrics->state_mask != 0xFFU || result->safe_outcome == 0U) {
        return F2RAG_ERR_STATE;
    }
    return F2RAG_OK;

fail:
    if (lm_package != NULL) {
        secure_zero(lm_package, lm_package_bytes);
        result->lm_slot_zeroized = 1U;
    }
    if (workspace != NULL) {
        secure_zero(workspace, workspace_elems * (uint32_t)sizeof(float));
        result->workspace_zeroized = 1U;
    }
    return status;

#undef RUN_SUPPORT
}
