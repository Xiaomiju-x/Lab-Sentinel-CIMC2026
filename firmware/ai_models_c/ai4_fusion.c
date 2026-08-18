/******************************************************************************
 * ai4_fusion.c  —  AI-4 Fusion MLP forward. Weights (BN folded) from
 * ai4_fusion_weights.h.
 ******************************************************************************/
#include "ai4_fusion.h"
#include "nn_ops.h"
#include "ai4_fusion_weights.h"
#include "ai4_calib.h"   /* temperature + abstention threshold + golden */
#include "nn_opt.h"    /* force -O3 -Otime on this TU (project default is -O0) */

void ai4_forward(const float *feat16, float *logits4)
{
    float h0[AI4_H0];   /* 32 */
    float h1[AI4_H1];   /* 16 */
    nn_linear(feat16, ai4_fc0_w, ai4_fc0_b, h0, AI4_IN, AI4_H0);
    nn_relu(h0, AI4_H0);
    nn_linear(h0, ai4_fc1_w, ai4_fc1_b, h1, AI4_H0, AI4_H1);
    nn_relu(h1, AI4_H1);
    nn_linear(h1, ai4_fc2_w, ai4_fc2_b, logits4, AI4_H1, AI4_NCLS);
}

int ai4_fuse(const float *feat16, float *probs4)
{
    ai4_forward(feat16, probs4);
    nn_softmax(probs4, AI4_N_CLASS);
    return nn_argmax(probs4, AI4_N_CLASS);
}

int ai4_fuse_calibrated(const float *feat16, float *probs4,
                        float *out_conf, uint8_t *out_abstain)
{
    int   cls, i;
    float conf;
    ai4_forward(feat16, probs4);
    /* temperature scaling: softmax(logits / T). AI4_TEMP may be 1.0 (no-op)
     * when the model is already well-calibrated (see ai4_calibration_report). */
    if (AI4_TEMP != 1.0f) {
        for (i = 0; i < AI4_N_CLASS; i++) probs4[i] /= AI4_TEMP;
    }
    nn_softmax(probs4, AI4_N_CLASS);
    cls  = nn_argmax(probs4, AI4_N_CLASS);
    conf = probs4[cls];
    if (out_conf)    *out_conf = conf;
    if (out_abstain) *out_abstain = (conf < AI4_ABSTAIN_CONF) ? 1u : 0u;
    return cls;
}
