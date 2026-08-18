/******************************************************************************
 * ai_nanolm.h  —  GD32 on-chip edge nano-LM (generative diagnosis).
 *
 * A real decoder-only GPT (3 layers, d=128, 4 heads, ~0.6M params, INT8) that
 * autoregressively GENERATES a one-sentence Chinese furnace diagnosis from the
 * live sentinel state. Distilled from DeepSeek (a large teacher model
 * family) -> the edge speaks without the cloud. NOT a template lookup: it runs
 * token-by-token transformer decoding with a KV cache on the M7 FPU.
 *
 * This is the "big brain distills nano brain" piece of the two-tier story; it
 * fires on an event (~0.5-1.5 s / sentence), not continuously.
 ******************************************************************************/
#ifndef AI_NANOLM_H
#define AI_NANOLM_H

#include "nanolm_vocab.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Generate a Chinese diagnosis from a 12-slot control-token context.
 * ctx12 holds NLM_NCTX control-token ids (use the NLM_CTX_* macros), in
 * SLOT_ORDER: [stage,temp,risk,ramp,drift,tc,gas,ae,vib,energy,host,elem].
 * Writes a UTF-8 NUL-terminated sentence into out (at most cap-1 bytes).
 * If conf != 0, sets *conf to the mean top-1 probability over generated
 * tokens (0..1) — used as the edge-cloud cascade confidence.
 * Returns the number of generated character tokens (0 on bad args). */
int nanolm_generate(const short *ctx12, char *out, int cap, float *conf);

/* On-chip golden self-test: reproduce the deployed INT8 model on the 3 baked
 * demo contexts. Pass = every demo's greedy token-id sequence matches exactly
 * AND demo0 last-position logits are within tol. Sets *logit_err = max|err|.
 * Returns 1 on pass, 0 on fail. */
int nanolm_selftest(float *logit_err);

#ifdef __cplusplus
}
#endif

#endif /* AI_NANOLM_H */
