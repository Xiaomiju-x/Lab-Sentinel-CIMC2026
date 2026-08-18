#include "forge200_rag_board_v9.h"

#include "HeaderFiles.h"
#include "FreeRTOS.h"
#include "task.h"
#include "FatFs/ff.h"
#include "forge200_bus_guard.h"

#include <string.h>

#define F2RAG_SLOT_A ((uint8_t *)0xC0F80000UL)
#define F2RAG_SLOT_B ((uint8_t *)0xC16C0000UL)
#define F2RAG_SLOT_BYTES 0x740000UL
#define F2RAG_WORKLOAD_OFFSET 0x00100000UL
#define F2RAG_WORKSPACE ((float *)0xC0C00000UL)
#define F2RAG_WORKSPACE_ELEMS (0x280000UL / 4UL)
#define F2RAG_GOLDEN_BUFFER ((uint8_t *)0xC0F00000UL)
#define F2RAG_GOLDEN_BUFFER_BYTES 0x00020000UL

static const char *const s_support_paths[F2RAG_DOMAIN_COUNT] = {
    "0:/F200/RAG/D0.F2S", "0:/F200/RAG/D1.F2S", "0:/F200/RAG/D2.F2S",
    "0:/F200/RAG/D3.F2S", "0:/F200/RAG/D4.F2S", "0:/F200/RAG/D5.F2S"
};

static const char *const s_workload_paths[F2RAG_DOMAIN_COUNT] = {
    "0:/F200/RAG/D0.RIX", "0:/F200/RAG/D1.RIX", "0:/F200/RAG/D2.RIX",
    "0:/F200/RAG/D3.RIX", "0:/F200/RAG/D4.RIX", "0:/F200/RAG/D5.RIX"
};

static const char *const s_lm_paths[F2RAG_DOMAIN_COUNT] = {
    "0:/F200/RAG/G001.ICM", "0:/F200/RAG/G012.ICM", "0:/F200/RAG/G003.ICM",
    "0:/F200/RAG/G004.ICM", "0:/F200/RAG/G005.ICM", "0:/F200/RAG/G006.ICM"
};

static const char *const s_golden_paths[F2RAG_DOMAIN_COUNT] = {
    "0:/F200/RAG/G001.GLD", "0:/F200/RAG/G012.GLD", "0:/F200/RAG/G003.GLD",
    "0:/F200/RAG/G004.GLD", "0:/F200/RAG/G005.GLD", "0:/F200/RAG/G006.GLD"
};

static uint32_t s_support_bytes;
static uint32_t s_workload_bytes;
static uint8_t s_cached_domain = 0xFFU;

static uint32_t rag_clock(void *context)
{
    (void)context;
    return DWT->CYCCNT;
}

static int load_file(const char *path, uint8_t *destination,
                     uint32_t capacity, uint32_t *loaded)
{
    FIL file;
    uint32_t total;
    uint32_t offset = 0U;
    UINT got;
    if (path == NULL || destination == NULL || loaded == NULL ||
        f_open(&file, path, FA_READ) != FR_OK) {
        return -1;
    }
    total = (uint32_t)f_size(&file);
    if ((FSIZE_t)total != f_size(&file) || total == 0U || total > capacity) {
        (void)f_close(&file);
        return -2;
    }
    while (offset < total) {
        uint32_t chunk = total - offset;
        if (chunk > 32768U) {
            chunk = 32768U;
        }
        got = 0U;
        if (f_read(&file, destination + offset, chunk, &got) != FR_OK || got != chunk) {
            (void)f_close(&file);
            return -3;
        }
        offset += chunk;
    }
    if (f_close(&file) != FR_OK) {
        return -4;
    }
    *loaded = total;
    return 0;
}

void forge200_rag_board_reset_cache(void)
{
    s_cached_domain = 0xFFU;
    s_support_bytes = 0U;
    s_workload_bytes = 0U;
    memset(F2RAG_SLOT_A, 0, F2RAG_WORKLOAD_OFFSET + 0x00080000UL);
}

int forge200_rag_board_run(uint32_t domain_id, uint32_t local_query_index,
                           uint8_t force_cold,
                           f2rag_metrics_t *metrics,
                           f2rag_result_t *result)
{
    uint32_t lm_bytes = 0U;
    uint32_t golden_bytes = 0U;
    uint32_t sd_read_bytes = 0U;
    f2rag_status_t status;
    int locked = 0;
    int rc = -1;
    if (domain_id >= F2RAG_DOMAIN_COUNT ||
        local_query_index >= F2RAG_WORKLOAD_PER_DOMAIN ||
        metrics == NULL || result == NULL || force_cold > 1U) {
        return -1;
    }
    if (forge200_inference_guard_acquire(30000U) != 0) {
        return -2;
    }
    locked = 1;
    if (force_cold != 0U || s_cached_domain != domain_id) {
        forge200_rag_board_reset_cache();
        if (load_file(s_support_paths[domain_id], F2RAG_SLOT_A,
                      F2RAG_WORKLOAD_OFFSET, &s_support_bytes) != 0 ||
            load_file(s_workload_paths[domain_id],
                      F2RAG_SLOT_A + F2RAG_WORKLOAD_OFFSET,
                      F2RAG_SLOT_BYTES - F2RAG_WORKLOAD_OFFSET,
                      &s_workload_bytes) != 0) {
            rc = -3;
            goto done;
        }
        sd_read_bytes += s_support_bytes + s_workload_bytes;
        s_cached_domain = (uint8_t)domain_id;
    }
    if (load_file(s_lm_paths[domain_id], F2RAG_SLOT_B,
                  F2RAG_SLOT_BYTES, &lm_bytes) != 0 ||
        load_file(s_golden_paths[domain_id], F2RAG_GOLDEN_BUFFER,
                  F2RAG_GOLDEN_BUFFER_BYTES, &golden_bytes) != 0) {
        rc = -4;
        goto done;
    }
    sd_read_bytes += lm_bytes + golden_bytes;
    status = f2rag_run_query(
        F2RAG_SLOT_A, s_support_bytes,
        F2RAG_SLOT_A + F2RAG_WORKLOAD_OFFSET, s_workload_bytes,
        local_query_index, F2RAG_SLOT_B, lm_bytes,
        F2RAG_GOLDEN_BUFFER, golden_bytes,
        F2RAG_WORKSPACE, F2RAG_WORKSPACE_ELEMS,
        rag_clock, NULL, metrics, result);
    metrics->cold_sd_read_bytes = sd_read_bytes;
    rc = status == F2RAG_OK ? 0 : -100 + (int)status;

done:
    if (rc != 0) {
        memset(F2RAG_SLOT_B, 0, lm_bytes);
        memset(F2RAG_WORKSPACE, 0, F2RAG_WORKSPACE_ELEMS * sizeof(float));
    }
    if (locked != 0) {
        (void)forge200_inference_guard_release();
    }
    return rc;
}
