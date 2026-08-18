/******************************************************************************
 * furnace_sim.c  —  sintering furnace temperature playback (see furnace_sim.h)
 *
 * Profile: garnet / YAG:Cr family  (predict_engine/sintering_profiles.json "garnet")
 *   calcine 900 C, 4 h, ramp 5 C/min, air
 *   sinter  1500 C, 6 h, ramp 5 C/min, air
 *   cool    ~3 C/min
 *   Source: solid-state ceramic route, lit. consensus for Al-garnet hosts
 *           (sintering_profiles.json "garnet" sources block).
 ******************************************************************************/
#include "furnace_sim.h"
#include <math.h>

#define T_ROOM       25.0f
#define CALCINE_C    900.0f
#define SINTER_C     1500.0f
#define BASE_RAMP    5.0f      /* C/min, calcine & sinter ramps */
#define COOL_RATE    3.0f      /* C/min */
#define CALCINE_MIN  240       /* 4 h hold */
#define SINTER_MIN   360       /* 6 h hold */
#define GRIND_MIN    30
#define HISTN        60

/* stage-conditioned vision proxy logits (synth_data.py stage_to_logit) */
static const float k_vis[6][4] = {
    {0.10f, 0.70f, 0.10f, 0.10f},  /* 0 ramp1   : loaded   */
    {0.10f, 0.60f, 0.20f, 0.10f},  /* 1 calcine : loaded   */
    {0.60f, 0.30f, 0.05f, 0.05f},  /* 2 grind   : empty    */
    {0.10f, 0.20f, 0.60f, 0.10f},  /* 3 ramp2   : sintering*/
    {0.05f, 0.10f, 0.80f, 0.05f},  /* 4 sinter  : sintering*/
    {0.10f, 0.10f, 0.10f, 0.70f},  /* 5 cool    : done     */
};

static furnace_state_t   s_state;
static furnace_anomaly_t s_anom;
static int   s_min;            /* absolute simulated minute */
static int   s_total;          /* total minutes for current run */
static float s_hist[HISTN];    /* last 60 temps */
static int   s_hist_n;
static float s_hold_cum;
static float s_prev_t;
static uint32_t s_rng = 0x1234abcdu;

/* stage boundaries (cumulative minute at which each stage ends) */
static int s_b[6];

static float frand_unit(void)   /* uniform [-0.5,0.5] via LCG */
{
    s_rng = s_rng * 1664525u + 1013904223u;
    return ((float)((s_rng >> 8) & 0xFFFFFFu) / (float)0x1000000u) - 0.5f;
}
static float noise3(void)       /* ~N(0,3): sum of 4 uniforms (CLT-ish) * scale */
{
    float u = frand_unit() + frand_unit() + frand_unit() + frand_unit(); /* ~[-2,2], var~1/3 */
    return u * 3.0f;
}

static float ramp_rate(void)
{
    if (s_anom == FURN_ANOM_FAST_RAMP) return BASE_RAMP * 4.0f;
    if (s_anom == FURN_ANOM_SLOW_RAMP) {
        float r = BASE_RAMP * 0.25f;
        return (r < 0.5f) ? 0.5f : r;
    }
    return BASE_RAMP;
}

static void recompute_stages(void)
{
    float r = ramp_rate();
    int ramp1 = (int)((CALCINE_C - T_ROOM) / (r > 1.0f ? r : 1.0f));
    int ramp2 = (int)((SINTER_C  - T_ROOM) / (r > 1.0f ? r : 1.0f));
    int cool  = (int)((SINTER_C  - T_ROOM) / COOL_RATE);
    s_b[0] = ramp1;
    s_b[1] = s_b[0] + CALCINE_MIN;
    s_b[2] = s_b[1] + GRIND_MIN;
    s_b[3] = s_b[2] + ramp2;
    s_b[4] = s_b[3] + SINTER_MIN;
    s_b[5] = s_b[4] + cool;
    s_total = s_b[5];
}

static int stage_of(int minute, int *step_in_stage, int *stage_len)
{
    int prev = 0, i;
    for (i = 0; i < 6; i++) {
        if (minute < s_b[i]) {
            *step_in_stage = minute - prev;
            *stage_len     = s_b[i] - prev;
            return i;
        }
        prev = s_b[i];
    }
    *step_in_stage = 0; *stage_len = 1;
    return 5;   /* past end -> cooling tail */
}

void furnace_sim_init(void)
{
    int i;
    s_state = FURN_IDLE;
    s_anom  = FURN_ANOM_NONE;
    s_min = 0; s_hold_cum = 0.0f; s_prev_t = T_ROOM; s_hist_n = HISTN;
    for (i = 0; i < HISTN; i++) s_hist[i] = T_ROOM;
    recompute_stages();
}

void furnace_sim_start(void)
{
    int i;
    s_min = 0; s_hold_cum = 0.0f; s_prev_t = T_ROOM; s_hist_n = HISTN;
    for (i = 0; i < HISTN; i++) s_hist[i] = T_ROOM;
    recompute_stages();
    s_state = FURN_RUNNING;
}

void furnace_sim_stop(void)            { s_state = FURN_IDLE; }
void furnace_sim_set_anomaly(furnace_anomaly_t a) { s_anom = a; recompute_stages(); }
furnace_anomaly_t furnace_sim_get_anomaly(void)   { return s_anom; }
furnace_state_t   furnace_sim_state(void)         { return s_state; }

/* compute target furnace temp at a given (stage, step) */
static float target_temp(int stage, int step, int stage_len)
{
    float frac;
    switch (stage) {
        case 0: frac = (stage_len > 1) ? (float)step / (float)(stage_len - 1) : 1.0f;
                return T_ROOM + (CALCINE_C - T_ROOM) * frac;
        case 1: return CALCINE_C;
        case 2: return T_ROOM;
        case 3: frac = (stage_len > 1) ? (float)step / (float)(stage_len - 1) : 1.0f;
                return T_ROOM + (SINTER_C - T_ROOM) * frac;
        case 4: return SINTER_C;
        default: {  /* cooling */
            float t = SINTER_C - COOL_RATE * (float)step;
            return (t < T_ROOM) ? T_ROOM : t;
        }
    }
}

/* advance exactly one simulated minute, updating state */
static void step_one(furnace_out_t *out)
{
    int step, slen, stage;
    float t_target, t_current, temp_dev, temp_grad, rms, var, mean;
    int i, k;

    if (s_state != FURN_RUNNING) {
        /* idle: sit at room temp, stage 0 baseline */
        stage = 0; t_target = T_ROOM;
    } else {
        stage = stage_of(s_min, &step, &slen);
        t_target = target_temp(stage, step, slen);
    }

    t_current = t_target + noise3();

    /* anomaly injections (mirror synth_data_ai3.py) */
    if (s_state == FURN_RUNNING) {
        if (s_anom == FURN_ANOM_TEMP_DRIFT) {
            t_current += 64.0f;                /* +~60 C drift everywhere */
        } else if (s_anom == FURN_ANOM_UNDERTEMP && stage == 4) {
            t_current -= 100.0f;               /* -100 C during sinter hold */
        }
    }

    /* push into history ring (keep last 60) */
    for (i = 0; i < HISTN - 1; i++) s_hist[i] = s_hist[i + 1];
    s_hist[HISTN - 1] = t_current;

    temp_dev  = t_current - t_target;
    if ((stage == 1 || stage == 4) && fabsf(temp_dev) <= 10.0f) s_hold_cum += 1.0f;
    temp_grad = s_hist[HISTN - 1] - s_hist[HISTN - 2];

    mean = 0.0f; for (k = 0; k < HISTN; k++) mean += s_hist[k]; mean /= (float)HISTN;
    var = 0.0f; rms = 0.0f;
    for (k = 0; k < HISTN; k++) { float v = s_hist[k]; rms += v * v; var += (v - mean) * (v - mean); }
    rms = sqrtf(rms / (float)HISTN);
    var = sqrtf(var / (float)HISTN);    /* std */

    out->temp_feat[0] = t_current / 1600.0f;
    out->temp_feat[1] = t_target  / 1600.0f;
    out->temp_feat[2] = temp_dev  / 100.0f;
    out->temp_feat[3] = temp_grad / 20.0f;
    out->temp_feat[4] = rms / 1600.0f;
    out->temp_feat[5] = var / 50.0f;
    out->temp_feat[6] = s_hold_cum / 600.0f;
    out->temp_feat[7] = (float)stage / 5.0f;

    for (k = 0; k < 4; k++) out->vis_proxy[k] = k_vis[stage][k];
    out->stage_id     = stage;
    out->atm_code     = 0;                     /* garnet = air */
    out->grinding_flag = (stage == 2) ? 1.0f : 0.0f;
    out->t_current_C  = t_current;
    out->t_target_C   = t_target;
    out->progress     = (s_total > 1) ? (float)s_min / (float)(s_total - 1) : 0.0f;
    if (out->progress > 1.0f) out->progress = 1.0f;
    out->state        = s_state;

    if (s_state == FURN_RUNNING) {
        s_min++;
        if (s_min >= s_total) s_state = FURN_DONE;
    }
}

void furnace_sim_advance(int n_minutes, furnace_out_t *out)
{
    int i;
    if (n_minutes < 1) n_minutes = 1;
    for (i = 0; i < n_minutes; i++) step_one(out);
}
