/* ai1_crucible.c — AI-1 crucible 3-class CNN inference (CIMC Lab-Sentinel)
 *
 * Mirrors crucible_cnn.py / export_crucible_to_c.py exactly (host golden verified
 * folded-numpy vs torch = 2.4e-7). Pipeline:
 *   in 3x64x64
 *   conv0 3x3 s2 p1 +relu -> 8 x32x32   (s_bufB)
 *   conv1 3x3 s2 p1 +relu -> 16x16x16   (s_bufA)
 *   conv2 3x3 s2 p1 +relu -> 24x 8x 8   (s_bufB)
 *   conv3 3x3 s2 p1 +relu -> 32x 4x 4   (s_bufA)
 *   GAP(4x4) -> 32  ->  FC 32xNCLS -> logits
 *
 * Stride-2 conv (not conv+maxpool) keeps the largest activation at 8x32x32 (32 KB),
 * so the two ping-pong scratch buffers total only ~49 KB SRAM (static .bss).
 */
#include "ai1_crucible.h"
#include "ai1_crucible_weights.h"

/* ping-pong activation scratch (static .bss -> AXI-SRAM, never on the stack).
 * bufB holds conv0(8*32*32=8192) and conv2(24*8*8=1536) -> 8192 max.
 * bufA holds conv1(16*16*16=4096) and conv3(32*4*4=512) -> 4096 max. */
static float s_bufA[4096];
static float s_bufB[8192];

/* 3x3 stride-2 pad-1 conv + ReLU. in:[IC,IH,IW] w:[OC,IC,3,3] b:[OC] -> out:[OC,OH,OW]
 * OH=(IH-1)/2+1, OW=(IW-1)/2+1. Folded-BN weights, so a plain conv + ReLU. */
static void conv3x3s2_relu(const float *in, int IC, int IH, int IW,
                           const float *w, const float *b, int OC, float *out)
{
    int OH = (IH - 1) / 2 + 1;
    int OW = (IW - 1) / 2 + 1;
    int oc, oy, ox, ic, ky, kx;

    for (oc = 0; oc < OC; oc++) {
        const float *wo = w + (long)oc * IC * 9;
        float        bo = b[oc];
        for (oy = 0; oy < OH; oy++) {
            for (ox = 0; ox < OW; ox++) {
                float acc = bo;
                for (ic = 0; ic < IC; ic++) {
                    const float *wi  = wo + ic * 9;
                    const float *inc = in + (long)ic * IH * IW;
                    for (ky = 0; ky < 3; ky++) {
                        int iy = oy * 2 + ky - 1;
                        if (iy < 0 || iy >= IH) continue;
                        {
                            const float *inr = inc + (long)iy * IW;
                            const float *wky = wi + ky * 3;
                            for (kx = 0; kx < 3; kx++) {
                                int ix = ox * 2 + kx - 1;
                                if (ix < 0 || ix >= IW) continue;
                                acc += wky[kx] * inr[ix];
                            }
                        }
                    }
                }
                out[(long)oc * OH * OW + (long)oy * OW + ox] = (acc > 0.0f) ? acc : 0.0f;
            }
        }
    }
}

int ai1_crucible_forward(const float *img_chw, float *logits_out, float *emb_out32)
{
    float g[CRU_C3];
    int   i, c, o, best;
    float bestv;

    conv3x3s2_relu(img_chw, CRU_IN_CH, 64, 64, cru_conv0_w, cru_conv0_b, CRU_C0, s_bufB);
    conv3x3s2_relu(s_bufB,  CRU_C0,    32, 32, cru_conv1_w, cru_conv1_b, CRU_C1, s_bufA);
    conv3x3s2_relu(s_bufA,  CRU_C1,    16, 16, cru_conv2_w, cru_conv2_b, CRU_C2, s_bufB);
    conv3x3s2_relu(s_bufB,  CRU_C2,     8,  8, cru_conv3_w, cru_conv3_b, CRU_C3, s_bufA);

    /* GAP over the final 4x4 map -> g[32] (s_bufA holds 32 x 4x4).
     * g[] doubles as the 32-D image embedding for the AI-1b few-shot NCM. */
    for (c = 0; c < CRU_C3; c++) {
        const float *p = s_bufA + (long)c * 16;   /* 4*4 = 16 */
        float s = 0.0f;
        for (i = 0; i < 16; i++) s += p[i];
        g[c] = s * (1.0f / 16.0f);
        if (emb_out32) emb_out32[c] = g[c];
    }

    /* FC 32->NCLS + argmax. */
    best = 0; bestv = 0.0f;
    for (o = 0; o < CRU_NCLS; o++) {
        const float *wr = cru_fc_w + (long)o * CRU_C3;
        float acc = cru_fc_b[o];
        for (i = 0; i < CRU_C3; i++) acc += wr[i] * g[i];
        logits_out[o] = acc;
        if (o == 0 || acc > bestv) { bestv = acc; best = o; }
    }
    return best;
}

/* ai1_crucible_cam — B3 explainability backend. Because AI-1 ends in GAP->FC, the
 * Class Activation Map is FORWARD-ONLY (no backward pass): for the predicted class,
 * CAM[h,w] = sum_k fc_w[cls,k] * conv3_featmap[k,h,w], then ReLU + max-normalise to
 * [0,1]. cam16 = 4x4 heatmap (row-major h*4+w) over the crucible crop. Returns the
 * predicted class. ~512 extra MACs on top of the forward pass. */
int ai1_crucible_cam(const float *img_chw, float *cam16)
{
    float g[CRU_C3];
    int   i, c, o, cls;
    float bestv, cmax;

    conv3x3s2_relu(img_chw, CRU_IN_CH, 64, 64, cru_conv0_w, cru_conv0_b, CRU_C0, s_bufB);
    conv3x3s2_relu(s_bufB,  CRU_C0,    32, 32, cru_conv1_w, cru_conv1_b, CRU_C1, s_bufA);
    conv3x3s2_relu(s_bufA,  CRU_C1,    16, 16, cru_conv2_w, cru_conv2_b, CRU_C2, s_bufB);
    conv3x3s2_relu(s_bufB,  CRU_C2,     8,  8, cru_conv3_w, cru_conv3_b, CRU_C3, s_bufA);

    /* GAP -> logits -> argmax (s_bufA holds the 32 x 4x4 conv3 map) */
    for (c = 0; c < CRU_C3; c++) {
        const float *p = s_bufA + (long)c * 16;
        float s = 0.0f;
        for (i = 0; i < 16; i++) s += p[i];
        g[c] = s * (1.0f / 16.0f);
    }
    cls = 0; bestv = 0.0f;
    for (o = 0; o < CRU_NCLS; o++) {
        const float *wr = cru_fc_w + (long)o * CRU_C3;
        float acc = cru_fc_b[o];
        for (i = 0; i < CRU_C3; i++) acc += wr[i] * g[i];
        if (o == 0 || acc > bestv) { bestv = acc; cls = o; }
    }

    /* CAM for the predicted class over the 4x4 spatial map */
    {
        const float *wr = cru_fc_w + (long)cls * CRU_C3;
        cmax = 0.0f;
        for (i = 0; i < 16; i++) {
            float v = 0.0f;
            int k;
            for (k = 0; k < CRU_C3; k++) v += wr[k] * s_bufA[(long)k * 16 + i];
            if (v < 0.0f) v = 0.0f;
            cam16[i] = v;
            if (v > cmax) cmax = v;
        }
        if (cmax > 0.0f) {
            float inv = 1.0f / cmax;
            for (i = 0; i < 16; i++) cam16[i] *= inv;
        }
    }
    return cls;
}
