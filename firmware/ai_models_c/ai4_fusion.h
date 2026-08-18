/******************************************************************************
 * ai4_fusion.h  —  AI-4 multi-modal decision fusion (on-chip)
 *
 * Input  : 16-D vector assembled from AI-1/AI-2/AI-3 outputs + DOA + progress
 *          (layout in ai4_data_info.json / feature_build.c)
 * Output : 4 risk classes  good(0) / suspected(1) / bad(2) / critical(3)
 * Arch   : Linear(16->32)-ReLU-Linear(32->16)-ReLU-Linear(16->4)
 *          (BatchNorm folded into the Linears at export). Test acc 97.25%.
 ******************************************************************************/
#ifndef AI4_FUSION_H
#define AI4_FUSION_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AI4_IN_DIM   16
#define AI4_N_CLASS  4

/* Pure forward → raw logits (used by self-test). */
void ai4_forward(const float *feat16, float *logits4);

/* Forward + softmax into probs4, returns argmax risk level (0..3). */
int ai4_fuse(const float *feat16, float *probs4);

/* Calibrated fuse + selective abstention (ai4_calib.h):
 *   probs4      <- softmax(logits / AI4_TEMP)   (temperature-calibrated)
 *   out_conf    <- top calibrated probability   (NULL ok)
 *   out_abstain <- 1 if conf < AI4_ABSTAIN_CONF  => defer to operator (NULL ok)
 * Returns argmax risk level (0..3). The caller decides the abstention policy;
 * the deployed policy (lab_sentinel fusion_task) is CONSERVATIVE: it floors an
 * uncertain "good" up to "warning/review" and never downgrades a bad/critical. */
int ai4_fuse_calibrated(const float *feat16, float *probs4,
                        float *out_conf, uint8_t *out_abstain);

#ifdef __cplusplus
}
#endif

#endif /* AI4_FUSION_H */
