#ifndef FORGE200_RAG_V9_H
#define FORGE200_RAG_V9_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define F2RAG_DOMAIN_COUNT 6U
#define F2RAG_SUPPORT_MODEL_COUNT 13U
#define F2RAG_WORKLOAD_PER_DOMAIN 20U
#define F2RAG_STAGE_COUNT 8U
#define F2RAG_GENERATION_TOKENS_MAX 24U

typedef enum {
    F2RAG_OK = 0,
    F2RAG_ERR_ARGUMENT = -1,
    F2RAG_ERR_SCHEMA = -2,
    F2RAG_ERR_HASH = -3,
    F2RAG_ERR_BOUNDS = -4,
    F2RAG_ERR_AUTHORITY = -5,
    F2RAG_ERR_GENERATION = -6,
    F2RAG_ERR_MODEL = -7,
    F2RAG_ERR_GOLDEN = -8,
    F2RAG_ERR_STATE = -9,
    F2RAG_ERR_WORKSPACE = -10
} f2rag_status_t;

typedef enum {
    F2RAG_LOAD_SUPPORT_A = 0,
    F2RAG_ROUTE_ENCODE_RETRIEVE_RERANK = 1,
    F2RAG_LOAD_LM_B = 2,
    F2RAG_GENERATE = 3,
    F2RAG_UNLOAD_LM_B = 4,
    F2RAG_NLI_QUALITY_A = 5,
    F2RAG_COMMIT_OR_REFUSE = 6,
    F2RAG_ZEROIZE = 7
} f2rag_stage_t;

typedef uint32_t (*f2rag_clock_fn)(void *context);

typedef struct {
    uint32_t state_mask;
    uint32_t stage_ticks[F2RAG_STAGE_COUNT];
    uint32_t cold_sd_read_bytes;
    uint32_t support_models_executed;
    uint32_t generation_tokens;
    uint32_t retrieved_local_index;
    uint32_t reranked_local_index;
    uint16_t router_label;
    uint16_t ood_label;
    uint16_t sufficiency_label;
    uint16_t refusal_label;
    uint16_t provenance_label;
    uint16_t nli_label;
    float quality_score;
} f2rag_metrics_t;

typedef struct {
    uint32_t query_id;
    uint8_t domain_id;
    uint8_t expected_refusal;
    uint8_t refused;
    uint8_t published;
    uint8_t source_bound;
    uint8_t generation_exact;
    uint8_t lm_slot_zeroized;
    uint8_t workspace_zeroized;
    uint8_t safe_outcome;
    uint16_t generated[F2RAG_GENERATION_TOKENS_MAX];
} f2rag_result_t;

f2rag_status_t f2rag_validate_support_bundle(
    const uint8_t *bundle,
    uint32_t bundle_bytes,
    float *workspace,
    uint32_t workspace_elems);

f2rag_status_t f2rag_validate_workload(
    const uint8_t *workload,
    uint32_t workload_bytes);

f2rag_status_t f2rag_run_query(
    const uint8_t *support_bundle,
    uint32_t support_bundle_bytes,
    const uint8_t *workload,
    uint32_t workload_bytes,
    uint32_t local_query_index,
    uint8_t *lm_package,
    uint32_t lm_package_bytes,
    const uint8_t *lm_golden,
    uint32_t lm_golden_bytes,
    float *workspace,
    uint32_t workspace_elems,
    f2rag_clock_fn clock_fn,
    void *clock_context,
    f2rag_metrics_t *metrics,
    f2rag_result_t *result);

#ifdef __cplusplus
}
#endif

#endif
