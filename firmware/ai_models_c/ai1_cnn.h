/******************************************************************************
 * ai1_cnn.h  —  AI-1 vision TinyCNN inference (on-chip)
 *
 * Conv4-MaxPool-Conv8-MaxPool-FC32-FC10, 13,242 params, ~51.7 KB FP32.
 * Weights are the trained MNIST model (97.3%); the CNN *engine* is fully
 * deployed and self-tested on-chip (ai_golden.h). The crucible 4-class head
 * is retrained once camera + PI-lab crucible images are available — only the
 * final FC weights change, the engine below is unchanged.
 *
 * The FC32 ReLU activations double as a 32-D image embedding consumed by the
 * AI-1b few-shot NCM classifier (ai1b_ncm.c).
 ******************************************************************************/
#ifndef AI1_CNN_H
#define AI1_CNN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AI1_INPUT_LEN   784   /* 1 x 28 x 28 */
#define AI1_EMB_LEN     32
#define AI1_LOGITS_LEN  10

/* Full forward. `in` = 784 floats (1x28x28, row-major, normalised externally).
 * Writes 10 raw logits and the 32-D embedding. emb may be NULL if unused. */
void ai1_cnn_forward(const float *in, float *logits, float *emb);

/* argmax over the 10 logits. */
int ai1_cnn_argmax(const float *logits);

#ifdef __cplusplus
}
#endif

#endif /* AI1_CNN_H */
