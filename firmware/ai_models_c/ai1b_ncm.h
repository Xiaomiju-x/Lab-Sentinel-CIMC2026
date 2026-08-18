/******************************************************************************
 * ai1b_ncm.h  —  AI-1b On-device Few-Shot learning (Nearest Class Mean)
 *
 * The innovation (ADR-8): add a brand-new crucible/defect class on the factory
 * floor without retraining or going back to the cloud. AI-1's frozen 32-D
 * embedding (ai1_cnn.c) is the feature; a new class is just its running mean
 * vector (128 bytes). Classify = nearest class mean (Euclidean).
 *
 * No back-prop, no optimiser — works with 1..10 samples per class on the MCU.
 ******************************************************************************/
#ifndef AI1B_NCM_H
#define AI1B_NCM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AI1B_EMB_DIM     32
#define AI1B_MAX_CLASS   8

/* Forget all enrolled classes. */
void ai1b_reset(void);

/* Number of classes currently enrolled. */
int ai1b_num_classes(void);

/* Enrol one sample into class `cid` (0..AI1B_MAX_CLASS-1), updating its
 * running mean. Auto-registers the class on first sample.
 * Returns 0 on success, -1 if cid out of range. */
int ai1b_add_sample(int cid, const float *emb);

/* Classify an embedding by nearest class mean.
 * Returns class id (0..) or -1 if no classes enrolled.
 * out_dist (NULL ok) = Euclidean distance to the nearest mean. */
int ai1b_classify(const float *emb, float *out_dist);

#ifdef __cplusplus
}
#endif

#endif /* AI1B_NCM_H */
