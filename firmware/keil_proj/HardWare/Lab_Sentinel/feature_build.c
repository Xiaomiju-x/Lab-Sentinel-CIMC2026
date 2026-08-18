/******************************************************************************
 * feature_build.c  —  runtime assembly of AI-2 / AI-3 / AI-4 feature vectors.
 * Layout constants mirror CIMC/model/.../synth_data*.py exactly.
 ******************************************************************************/
#include "feature_build.h"
#include <math.h>

#define AI3_SEQ 64
#define AI3_NF  8

/* ---- AI-3 sliding window ring (oldest..newest) ---- */
static float s_ring[AI3_SEQ][AI3_NF];
static int   s_head;        /* index of oldest slot */
static int   s_filled;
static float s_lin[AI3_SEQ * AI3_NF];

/* ---- per-call history for room/gas derivative + MQ-135 air baseline ---- */
static float s_prev_room_t;
static float s_prev_room_h;
static float s_mq_ring[5];
static int   s_mq_n;
static float s_mq_baseline;   /* auto-calibrated clean-air ADC baseline */
static int   s_mq_cal_n;
static float s_cum_dev;
static int   s_inited;
static int   s_first_room;   /* seed room derivative baseline on first sample */

void fb_reset(void)
{
    int i, j;
    for (i = 0; i < AI3_SEQ; i++)
        for (j = 0; j < AI3_NF; j++) s_ring[i][j] = 0.0f;
    s_head = 0; s_filled = 0;
    s_prev_room_t = 25.0f; s_prev_room_h = 50.0f;
    for (i = 0; i < 5; i++) s_mq_ring[i] = 0.55f;
    s_mq_n = 0; s_mq_baseline = 0.0f; s_mq_cal_n = 0;
    s_cum_dev = 0.0f;
    s_first_room = 1;
    s_inited = 1;
}

void fb_push_ai3(const float temp_feat8[8])
{
    int j;
    int slot = (s_head + s_filled) % AI3_SEQ;
    if (s_filled < AI3_SEQ) {
        for (j = 0; j < AI3_NF; j++) s_ring[slot][j] = temp_feat8[j];
        s_filled++;
    } else {
        /* overwrite oldest, advance head */
        for (j = 0; j < AI3_NF; j++) s_ring[s_head][j] = temp_feat8[j];
        s_head = (s_head + 1) % AI3_SEQ;
    }
}

const float *fb_ai3_window(void)
{
    int i, j, idx;
    /* If not yet full, pad the front with the oldest available row so the
     * 64-length window is always valid (early-run behaviour). */
    for (i = 0; i < AI3_SEQ; i++) {
        if (s_filled == 0) {
            for (j = 0; j < AI3_NF; j++) s_lin[i * AI3_NF + j] = 0.0f;
            continue;
        }
        if (i < AI3_SEQ - s_filled) {
            idx = s_head;                                   /* repeat oldest */
        } else {
            idx = (s_head + (i - (AI3_SEQ - s_filled))) % AI3_SEQ;
        }
        for (j = 0; j < AI3_NF; j++) s_lin[i * AI3_NF + j] = s_ring[idx][j];
    }
    return s_lin;
}

void fb_build_ai2(const furnace_out_t *f, const fb_sensors_t *s, float feat32[32])
{
    int i;
    float room_t, room_h, mq_raw, mq_avg, mq_std, mq_trend;
    float vib_g, cen, energy;

    if (!s_inited) fb_reset();

    /* ---- Group A: temperature (from furnace sim) [0:8] ---- */
    for (i = 0; i < 8; i++) feat32[i] = f->temp_feat[i];

    /* ---- Group B: gas / atmosphere [8:14] ---- */
    {
        float adc = (float)s->mq135_adc / 4095.0f;          /* 0..1 */
        /* auto-calibrate clean-air baseline over the first 8 samples */
        if (s_mq_cal_n < 8) { s_mq_baseline += adc; s_mq_cal_n++; }
        float base = (s_mq_cal_n > 0) ? (s_mq_baseline / (float)s_mq_cal_n) : adc;
        /* map real sensor onto the training air baseline (0.55), real rises detected */
        mq_raw = 0.55f + (adc - base) * 2.0f;
        if (mq_raw < 0.0f) mq_raw = 0.0f; if (mq_raw > 1.0f) mq_raw = 1.0f;
    }
    /* 5-sample ring */
    for (i = 0; i < 4; i++) s_mq_ring[i] = s_mq_ring[i + 1];
    s_mq_ring[4] = mq_raw;
    if (s_mq_n < 5) s_mq_n++;
    mq_avg = 0.0f; for (i = 0; i < 5; i++) mq_avg += s_mq_ring[i]; mq_avg /= 5.0f;
    mq_std = 0.0f; for (i = 0; i < 5; i++) { float d = s_mq_ring[i] - mq_avg; mq_std += d * d; }
    mq_std = sqrtf(mq_std / 5.0f) / 0.05f;
    mq_trend = (s_mq_ring[4] - s_mq_ring[0]) / 4.0f / 0.01f;   /* slope proxy */
    feat32[8]  = mq_raw;
    feat32[9]  = mq_avg;
    feat32[10] = (float)f->atm_code / 3.0f;
    feat32[11] = mq_std;
    feat32[12] = mq_trend;
    feat32[13] = 1.0f;                                          /* atm_match (always 1 in training) */

    /* ---- Group C: room environment (real SHT30) [14:18] ---- */
    room_t = (float)s->temp_c_q8 / 256.0f;
    room_h = (float)s->hum_q8    / 256.0f;
    if (s_first_room) { s_prev_room_t = room_t; s_prev_room_h = room_h; s_first_room = 0; }
    feat32[14] = (room_t - 25.0f) / 5.0f;
    feat32[15] = (room_h - 50.0f) / 10.0f;
    /* Room temp/humidity gradients: training saw smooth ~0.05 C/min changes
     * (std ~0.25). Real SHT30 at 1 Hz can jitter; clamp to the training 3-4 sigma
     * band so sensor noise on this minor channel can't drive a false anomaly
     * (the furnace-temp + atmosphere + vibration groups carry the real signal). */
    {
        float gt = (room_t - s_prev_room_t) / 0.2f;   /* 0 on first sample */
        float gh = (room_h - s_prev_room_h) / 0.5f;
        if (gt >  1.0f) gt =  1.0f; else if (gt < -1.0f) gt = -1.0f;
        if (gh >  1.0f) gh =  1.0f; else if (gh < -1.0f) gh = -1.0f;
        feat32[16] = gt;
        feat32[17] = gh;
    }
    s_prev_room_t = room_t; s_prev_room_h = room_h;

    /* ---- Group D: vibration (real ADXL345, single-axis magnitude) [18:26] ---- */
    vib_g  = (float)s->vib_rms_mg / 1000.0f;                    /* g */
    cen    = 0.1f + 0.3f * f->grinding_flag;                    /* centroid proxy (norm) */
    energy = 3.0f * (vib_g * vib_g) / 0.3f;                     /* sum of 3 axes^2 / 0.3 */
    feat32[18] = vib_g / 0.1f;
    feat32[19] = vib_g / 0.1f;
    feat32[20] = vib_g / 0.1f;
    feat32[21] = cen;
    feat32[22] = cen;
    feat32[23] = cen;
    feat32[24] = energy;
    feat32[25] = f->grinding_flag;

    /* ---- Group E: vision proxy (stage-conditioned) [26:30] ---- */
    feat32[26] = f->vis_proxy[0];
    feat32[27] = f->vis_proxy[1];
    feat32[28] = f->vis_proxy[2];
    feat32[29] = f->vis_proxy[3];

    /* ---- Group F: progress [30:32] ---- */
    s_cum_dev += fabsf(f->temp_feat[2]);                        /* |temp_dev|/100 per tick */
    feat32[30] = f->progress;
    feat32[31] = (s_cum_dev / 1000.0f > 1.0f) ? 1.0f : s_cum_dev / 1000.0f;
}

void fb_build_ai4(const float ai1_probs4[4], float ai2_ratio, const float ai2_resid3[3],
                  const float ai3_probs5[5], float progress, float feat16[16])
{
    int i;
    for (i = 0; i < 4; i++) feat16[i] = ai1_probs4[i];          /* [0:4]  AI-1 */
    feat16[4] = ai2_ratio;                                      /* [4]    AI-2 ratio */
    feat16[5] = ai2_resid3[0];                                  /* [5]    temp resid */
    feat16[6] = ai2_resid3[1];                                  /* [6]    vib  resid */
    feat16[7] = ai2_resid3[2];                                  /* [7]    gas  resid */
    for (i = 0; i < 5; i++) feat16[8 + i] = ai3_probs5[i];      /* [8:13] AI-3 */
    feat16[13] = 0.0f;                                          /* [13]   DOA flag (mic disabled) */
    feat16[14] = 0.0f;                                          /* [14]   DOA intensity */
    feat16[15] = progress;                                      /* [15]   progress */
}
