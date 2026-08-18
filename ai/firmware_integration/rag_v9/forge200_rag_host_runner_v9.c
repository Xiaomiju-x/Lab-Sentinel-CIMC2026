#include "forge200_rag_v9.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HOST_WORKSPACE_ELEMS 655360U

typedef struct {
    uint8_t *data;
    uint32_t bytes;
} host_file_t;

static int read_file(const char *path, host_file_t *file)
{
    FILE *handle;
    long bytes;
    if (path == NULL || file == NULL) {
        return -1;
    }
    memset(file, 0, sizeof(*file));
    handle = fopen(path, "rb");
    if (handle == NULL || fseek(handle, 0L, SEEK_END) != 0) {
        if (handle != NULL) {
            fclose(handle);
        }
        return -1;
    }
    bytes = ftell(handle);
    if (bytes <= 0L || (unsigned long)bytes > 0xFFFFFFFFUL ||
        fseek(handle, 0L, SEEK_SET) != 0) {
        fclose(handle);
        return -1;
    }
    file->data = (uint8_t *)malloc((size_t)bytes);
    if (file->data == NULL || fread(file->data, 1U, (size_t)bytes, handle) != (size_t)bytes) {
        free(file->data);
        memset(file, 0, sizeof(*file));
        fclose(handle);
        return -1;
    }
    fclose(handle);
    file->bytes = (uint32_t)bytes;
    return 0;
}

static void release_file(host_file_t *file)
{
    if (file != NULL) {
        free(file->data);
        memset(file, 0, sizeof(*file));
    }
}

static uint32_t host_clock(void *context)
{
    (void)context;
    return (uint32_t)clock();
}

int main(int argc, char **argv)
{
    host_file_t support;
    host_file_t workload;
    host_file_t lm_original;
    host_file_t lm_golden;
    uint8_t *lm_mutable = NULL;
    float *workspace = NULL;
    uint32_t safe = 0U;
    uint32_t published = 0U;
    uint32_t refused = 0U;
    uint32_t source_bound = 0U;
    uint32_t positive_published = 0U;
    uint32_t negative_refused = 0U;
    uint32_t max_cold_read = 0U;
    uint32_t max_stage_ticks[F2RAG_STAGE_COUNT] = {0U};
    uint32_t i;
    int exit_code = 0;

    memset(&support, 0, sizeof(support));
    memset(&workload, 0, sizeof(workload));
    memset(&lm_original, 0, sizeof(lm_original));
    memset(&lm_golden, 0, sizeof(lm_golden));
    if (argc != 5 || read_file(argv[1], &support) != 0 ||
        read_file(argv[2], &workload) != 0 ||
        read_file(argv[3], &lm_original) != 0 ||
        read_file(argv[4], &lm_golden) != 0) {
        fprintf(stderr, "usage_or_read_error\n");
        exit_code = 2;
        goto done;
    }
    lm_mutable = (uint8_t *)malloc(lm_original.bytes);
    workspace = (float *)calloc(HOST_WORKSPACE_ELEMS, sizeof(float));
    if (lm_mutable == NULL || workspace == NULL) {
        fprintf(stderr, "allocation_error\n");
        exit_code = 3;
        goto done;
    }

    printf("{\"queries\":[");
    for (i = 0U; i < F2RAG_WORKLOAD_PER_DOMAIN; ++i) {
        f2rag_metrics_t metrics;
        f2rag_result_t result;
        f2rag_status_t status;
        uint32_t stage;
        memcpy(lm_mutable, lm_original.data, lm_original.bytes);
        status = f2rag_run_query(
            support.data, support.bytes, workload.data, workload.bytes, i,
            lm_mutable, lm_original.bytes, lm_golden.data, lm_golden.bytes,
            workspace, HOST_WORKSPACE_ELEMS, host_clock, NULL, &metrics, &result);
        if (i != 0U) {
            printf(",");
        }
        printf("{\"id\":%u,\"status\":%d,\"expected_refusal\":%u,"
               "\"refused\":%u,\"published\":%u,\"source_bound\":%u,"
               "\"generation_exact\":%u,\"safe\":%u,\"state_mask\":%u,"
               "\"support_models\":%u,\"cold_read_bytes\":%u,"
               "\"generation_tokens\":%u,"
               "\"lm_zeroized\":%u,\"workspace_zeroized\":%u,"
               "\"retrieved\":%u,\"reranked\":%u,\"router\":%u,"
               "\"ood\":%u,\"sufficiency\":%u,\"refusal_gate\":%u,"
               "\"provenance\":%u,\"nli\":%u,\"quality\":%.9g,"
               "\"stage_ticks\":[",
               result.query_id, (int)status, result.expected_refusal,
               result.refused, result.published, result.source_bound,
               result.generation_exact, result.safe_outcome, metrics.state_mask,
               metrics.support_models_executed, metrics.cold_sd_read_bytes,
               metrics.generation_tokens,
               result.lm_slot_zeroized, result.workspace_zeroized,
               metrics.retrieved_local_index, metrics.reranked_local_index,
               metrics.router_label, metrics.ood_label,
               metrics.sufficiency_label, metrics.refusal_label,
               metrics.provenance_label, metrics.nli_label,
               (double)metrics.quality_score);
        for (stage = 0U; stage < F2RAG_STAGE_COUNT; ++stage) {
            if (stage != 0U) {
                printf(",");
            }
            printf("%u", metrics.stage_ticks[stage]);
            if (metrics.stage_ticks[stage] > max_stage_ticks[stage]) {
                max_stage_ticks[stage] = metrics.stage_ticks[stage];
            }
        }
        printf("]}");
        if (status != F2RAG_OK) {
            exit_code = 10 + (int)i;
            break;
        }
        safe += result.safe_outcome;
        published += result.published;
        refused += result.refused;
        source_bound += result.source_bound;
        if (result.expected_refusal == 0U) {
            positive_published += result.published;
        } else {
            negative_refused += result.refused;
        }
        if (metrics.cold_sd_read_bytes > max_cold_read) {
            max_cold_read = metrics.cold_sd_read_bytes;
        }
    }
    printf("],\"summary\":{\"safe\":%u,\"published\":%u,\"refused\":%u,"
           "\"source_bound\":%u,\"positive_published\":%u,"
           "\"negative_refused\":%u,\"max_cold_read_bytes\":%u,"
           "\"max_stage_ticks\":[",
           safe, published, refused, source_bound, positive_published,
           negative_refused, max_cold_read);
    for (i = 0U; i < F2RAG_STAGE_COUNT; ++i) {
        if (i != 0U) {
            printf(",");
        }
        printf("%u", max_stage_ticks[i]);
    }
    printf("]}}\n");

done:
    free(lm_mutable);
    free(workspace);
    release_file(&support);
    release_file(&workload);
    release_file(&lm_original);
    release_file(&lm_golden);
    return exit_code;
}
