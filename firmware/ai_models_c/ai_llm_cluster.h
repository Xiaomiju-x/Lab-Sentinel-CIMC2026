/******************************************************************************
 * ai_llm_cluster.h  —  GD32 EDGE LLM CLUSTER: 5 role-specialized nano-LM experts
 * swap-loaded one-at-a-time from external 8MB SPI flash into SDRAM (mirrors a
 * server-class BPU swap-load cluster, brought down to a no-NPU Cortex-M7).
 *
 *   experts (shared vocab, identical arch, distilled by ROLE not architecture):
 *     0 e1 diag   故障诊断       3 e4 qc     质量批次判读
 *     1 e2 recipe 工艺配方建议    4 e5 brief  操作简报
 *     2 e3 energy 能耗碳排
 *
 * Storage: provisioned cluster_image.bin lives in SPI flash; expert k at
 * NLM_CL_SPI_BASE + k*NLM_CL_STRIDE. cluster_load_expert(k) streams that blob
 * into an SDRAM working buffer and binds the forward engine to it (~0.3s @18MHz).
 ******************************************************************************/
#ifndef AI_LLM_CLUSTER_H
#define AI_LLM_CLUSTER_H

#include "nlm_cluster_vocab.h"

/* Load expert `id` (0..NLM_CL_NEXPERT-1) from SPI flash into SDRAM and bind it.
 * Returns 0 on success, <0 on error (bad id / flash id mismatch). Idempotent:
 * re-loading the already-bound expert is a fast no-op. */
int cluster_load_expert(int id);

/* Currently bound expert id, or -1 if none. */
int cluster_current(void);

/* Generate one role sentence for the 12 control tokens with the CURRENTLY bound
 * expert. out is NUL-terminated UTF-8; conf = mean top-1 prob. Returns token count.
 * (Call cluster_load_expert first.) */
int cluster_generate(const short *ctx12, char *out, int cap, float *conf);

/* Boot self-test: load each expert, reproduce its golden greedy ids + check the
 * demo0 logits. per_ok[NLM_CL_NEXPERT] (may be NULL) gets 1/0 per expert; returns
 * 1 if ALL pass. *max_err (may be NULL) = worst logit |err| across experts. */
int cluster_selftest(int *per_ok, float *max_err);

#ifdef NLM_HOST_TEST
/* host only: bind the in-memory provisioning image (5 blobs) so the engine can
 * run without SPI flash. */
void cluster_set_host_image(const unsigned char *image);
#endif

#endif
