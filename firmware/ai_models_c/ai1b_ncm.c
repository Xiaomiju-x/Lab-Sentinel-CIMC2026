/******************************************************************************
 * ai1b_ncm.c  —  Nearest-Class-Mean few-shot classifier over AI-1 embeddings.
 ******************************************************************************/
#include "ai1b_ncm.h"
#include "nn_opt.h"    /* force -O3 -Otime on this TU (project default is -O0) */

static float   s_mean[AI1B_MAX_CLASS][AI1B_EMB_DIM];
static uint32_t s_count[AI1B_MAX_CLASS];
static int     s_nclass;

void ai1b_reset(void)
{
    int c, d;
    for (c = 0; c < AI1B_MAX_CLASS; c++) {
        s_count[c] = 0u;
        for (d = 0; d < AI1B_EMB_DIM; d++) s_mean[c][d] = 0.0f;
    }
    s_nclass = 0;
}

int ai1b_num_classes(void)
{
    return s_nclass;
}

int ai1b_add_sample(int cid, const float *emb)
{
    int d;
    float n1;
    if (cid < 0 || cid >= AI1B_MAX_CLASS) return -1;

    if (s_count[cid] == 0u && (cid + 1) > s_nclass) {
        s_nclass = cid + 1;
    }
    /* incremental mean: mean += (x - mean) / (n+1) */
    s_count[cid]++;
    n1 = 1.0f / (float)s_count[cid];
    for (d = 0; d < AI1B_EMB_DIM; d++) {
        s_mean[cid][d] += (emb[d] - s_mean[cid][d]) * n1;
    }
    return 0;
}

int ai1b_classify(const float *emb, float *out_dist)
{
    int c, d, best = -1;
    float best_d2 = 0.0f;

    for (c = 0; c < s_nclass; c++) {
        if (s_count[c] == 0u) continue;
        float d2 = 0.0f;
        for (d = 0; d < AI1B_EMB_DIM; d++) {
            float diff = emb[d] - s_mean[c][d];
            d2 += diff * diff;
        }
        if (best < 0 || d2 < best_d2) { best_d2 = d2; best = c; }
    }
    if (out_dist != 0) {
        /* sqrt without pulling math.h here: best_d2 is small, use Newton */
        float r = best_d2, x = (best_d2 > 1.0f) ? best_d2 : 1.0f;
        int k;
        for (k = 0; k < 12 && r > 0.0f; k++) x = 0.5f * (x + r / x);
        *out_dist = (best >= 0) ? x : 0.0f;
    }
    return best;
}
