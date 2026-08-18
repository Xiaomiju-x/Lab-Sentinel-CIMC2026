#ifndef FORGE200_RAG_BOARD_V9_H
#define FORGE200_RAG_BOARD_V9_H

#include <stdint.h>

#include "forge200_rag_v9.h"

#ifdef __cplusplus
extern "C" {
#endif

int forge200_rag_board_run(uint32_t domain_id, uint32_t local_query_index,
                           uint8_t force_cold,
                           f2rag_metrics_t *metrics,
                           f2rag_result_t *result);
void forge200_rag_board_reset_cache(void);

#ifdef __cplusplus
}
#endif

#endif
