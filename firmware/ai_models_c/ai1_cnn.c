/******************************************************************************
 * ai1_cnn.c  —  AI-1 vision TinyCNN forward pass.
 * Weights from ai1_cnn_weights.h (trained, MNIST). See ai1_cnn.h.
 ******************************************************************************/
#include "ai1_cnn.h"
#include "nn_ops.h"
#include "ai1_cnn_weights.h"
#include "nn_opt.h"    /* force -O3 -Otime on this TU (project default is -O0) */

/* Static activation buffers (SRAM .bss, ~23 KB). Single-threaded use:
 * AI-1 only runs inside vision_task / boot self-test, never re-entrant. */
static float s_c0[AI1_C0 * AI1_IN_H * AI1_IN_W];   /* 4 x 28 x 28 = 3136 */
static float s_p0[AI1_C0 * 14 * 14];               /* 4 x 14 x 14 =  784 */
static float s_c1[AI1_C1 * 14 * 14];               /* 8 x 14 x 14 = 1568 */
static float s_p1[AI1_C1 * 7 * 7];                 /* 8 x  7 x  7 =  392 */

void ai1_cnn_forward(const float *in, float *logits, float *emb)
{
    float emb_local[AI1_EMB];

    /* conv0: 1x28x28 -> 4x28x28, pad 1 */
    nn_conv2d(in, AI1_IN_CH, AI1_IN_H, AI1_IN_W,
              ai1_conv0_w, ai1_conv0_b, AI1_C0, 3, 3, 1, s_c0);
    nn_relu(s_c0, AI1_C0 * AI1_IN_H * AI1_IN_W);

    /* maxpool 2x2 -> 4x14x14 */
    nn_maxpool2d(s_c0, AI1_C0, AI1_IN_H, AI1_IN_W, 2, 2, s_p0);

    /* conv1: 4x14x14 -> 8x14x14, pad 1 */
    nn_conv2d(s_p0, AI1_C0, 14, 14,
              ai1_conv1_w, ai1_conv1_b, AI1_C1, 3, 3, 1, s_c1);
    nn_relu(s_c1, AI1_C1 * 14 * 14);

    /* maxpool 2x2 -> 8x7x7 = 392 (already in NCHW = flatten order) */
    nn_maxpool2d(s_c1, AI1_C1, 14, 14, 2, 2, s_p1);

    /* fc0: 392 -> 32, ReLU  (= embedding) */
    nn_linear(s_p1, ai1_fc0_w, ai1_fc0_b, emb_local, AI1_FLAT, AI1_EMB);
    nn_relu(emb_local, AI1_EMB);
    if (emb != 0) {
        int i;
        for (i = 0; i < AI1_EMB; i++) emb[i] = emb_local[i];
    }

    /* fc1: 32 -> 10 logits */
    nn_linear(emb_local, ai1_fc1_w, ai1_fc1_b, logits, AI1_EMB, AI1_NCLS);
}

int ai1_cnn_argmax(const float *logits)
{
    return nn_argmax(logits, AI1_NCLS);
}
