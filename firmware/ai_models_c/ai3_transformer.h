/******************************************************************************
 * ai3_transformer.h  —  AI-3 sintering-curve TinyTransformer (on-chip)
 *
 * Input  : 64 consecutive minutes x 8 temperature features
 * Output : 5-class softmax  [normal, fast_ramp, undertemp, temp_drift, slow_ramp]
 * Arch   : Linear(8->64)+sinusoidal PE, 2x pre-norm blocks (4-head attn + FFN
 *          64->128->64), final LayerNorm, mean-pool over time, Linear(64->5).
 *
 * "Cortex-M7 上的 Transformer" — the SBC-BPU 24-layer Qwen2 story (XRD Round 8)
 * pushed down to a bare MCU. Activation buffers sit in SDRAM (SDRAM_AI_SCRATCH).
 * Full forward ~ few ms at 600 MHz; runs at 1 Hz in env_task.
 ******************************************************************************/
#ifndef AI3_TRANSFORMER_H
#define AI3_TRANSFORMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AI3_SEQ_LEN   64
#define AI3_N_FEAT    8
#define AI3_N_CLASS   5

/* Pure forward on a NORMALISED sequence (64*8 row-major). Used by self-test. */
void ai3_forward_norm(const float *seqn, float *logits5);

/* Classify a RAW sequence (64*8 row-major, time-major). Normalises each
 * timestep with mu/std, runs the transformer, softmaxes into probs5.
 * Returns argmax class index. */
int ai3_classify(const float *seq_raw, float *probs5);

/* Attention saliency over the 64-step temperature window from the most recent
 * forward (last-block attention received per timestep, averaged over heads &
 * queries; sums to ~1). Explainability readout for the HMI / report.
 * Call AFTER ai3_forward_norm()/ai3_classify(). out_seq must hold 64 floats. */
void ai3_get_attention(float *out_seq);

#ifdef __cplusplus
}
#endif

#endif /* AI3_TRANSFORMER_H */
