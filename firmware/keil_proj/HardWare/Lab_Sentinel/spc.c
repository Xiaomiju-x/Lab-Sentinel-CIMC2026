/******************************************************************************
 * spc.c — online SPC for the sintering soak. See spc.h.
 * O(1) memory; host-validated by host_test/spc_test.c before hardware.
 ******************************************************************************/
#include "spc.h"
#include <math.h>

void spc_init(spc_t *s, float lsl, float usl, float target,
              float ewma_lambda, int warmup_n)
{
    int k;
    s->lsl = lsl; s->usl = usl;
    s->ewma_lambda = (ewma_lambda > 0.0f && ewma_lambda <= 1.0f) ? ewma_lambda : 0.2f;
    s->warmup_n = (warmup_n < 5) ? 5 : warmup_n;
    s->mean = 0.0; s->m2 = 0.0; s->n = 0;
    s->cl = (double)target; s->sigma_cl = 0.0; s->frozen = 0;
    s->ewma = 0.0f; s->ewma_started = 0;
    for (k = 0; k < SPC_HIST; k++) { s->r_side[k] = 0; s->r_level[k] = 0; }
    s->ring_i = 0; s->ring_fill = 0;
    s->n_alarms = 0; s->first_rule = SPC_OK; s->first_alarm_n = 0;
}

/* classify a sample into a Western Electric zone given the frozen baseline */
static void classify(const spc_t *s, float x, int8_t *side, int8_t *level)
{
    double z, az;
    if (s->sigma_cl < 1e-6) { *side = (x >= (float)s->cl) ? 1 : -1; *level = 0; return; }
    z = ((double)x - s->cl) / s->sigma_cl;
    *side = (z >= 0.0) ? 1 : -1;
    az = fabs(z);
    if (az > 3.0)      *level = 3;
    else if (az > 2.0) *level = 2;
    else if (az > 1.0) *level = 1;
    else               *level = 0;
}

/* read the k-th most recent ring entry (0 = current). returns 0 if not present */
static int recent(const spc_t *s, int k, int8_t *side, int8_t *level)
{
    int idx;
    if (k >= s->ring_fill) return 0;
    idx = (s->ring_i - 1 - k + 2 * SPC_HIST) % SPC_HIST;
    *side = s->r_side[idx]; *level = s->r_level[idx];
    return 1;
}

/* Western Electric / Nelson runtime rules on the current ring state */
static spc_rule_t check_we(const spc_t *s)
{
    int8_t cs, cl, sd, lv;
    int j, cnt;

    if (!recent(s, 0, &cs, &cl)) return SPC_OK;

    /* Rule 1: current point beyond 3 sigma */
    if (cl >= 3) return SPC_WE1_BEYOND_3SIGMA;

    /* Rule 2: 2 of the last 3 in zone A+ (level>=2) on the same side, incl. current */
    if (cl >= 2 && s->ring_fill >= 2) {
        cnt = 0;
        for (j = 0; j < 3; j++)
            if (recent(s, j, &sd, &lv) && sd == cs && lv >= 2) cnt++;
        if (cnt >= 2) return SPC_WE2_2OF3_2SIGMA;
    }

    /* Rule 3: 4 of the last 5 in zone B+ (level>=1) on the same side, incl. current */
    if (cl >= 1 && s->ring_fill >= 4) {
        cnt = 0;
        for (j = 0; j < 5; j++)
            if (recent(s, j, &sd, &lv) && sd == cs && lv >= 1) cnt++;
        if (cnt >= 4) return SPC_WE3_4OF5_1SIGMA;
    }

    /* Rule 4: 8 consecutive on the same side of center */
    if (s->ring_fill >= 8) {
        cnt = 0;
        for (j = 0; j < 8; j++)
            if (recent(s, j, &sd, &lv) && sd == cs) cnt++;
        if (cnt == 8) return SPC_WE4_8_RUN;
    }
    return SPC_OK;
}

spc_rule_t spc_update(spc_t *s, float x)
{
    double d, d2;
    int8_t side, level;
    spc_rule_t rule = SPC_OK;

    /* full-run Welford (all samples feed Cp/Cpk) */
    s->n++;
    d = (double)x - s->mean;
    s->mean += d / (double)s->n;
    d2 = (double)x - s->mean;
    s->m2 += d * d2;

    /* EWMA */
    if (!s->ewma_started) { s->ewma = x; s->ewma_started = 1; }
    else s->ewma = s->ewma_lambda * x + (1.0f - s->ewma_lambda) * s->ewma;

    /* Phase-I: freeze the spread once warmup is complete (CL stays at target) */
    if (!s->frozen) {
        if (s->n >= s->warmup_n) {
            s->sigma_cl = (s->n > 1) ? sqrt(s->m2 / (double)(s->n - 1)) : 0.0;
            if (s->sigma_cl < 1e-6) s->sigma_cl = 1e-6;
            s->frozen = 1;
        }
        return SPC_OK;   /* no control decisions during Phase I */
    }

    /* Phase-II: classify into the ring and run the rules */
    classify(s, x, &side, &level);
    s->r_side[s->ring_i]  = side;
    s->r_level[s->ring_i] = level;
    s->ring_i = (s->ring_i + 1) % SPC_HIST;
    if (s->ring_fill < SPC_HIST) s->ring_fill++;

    rule = check_we(s);
    if (rule != SPC_OK) {
        s->n_alarms++;
        if (s->first_rule == SPC_OK) { s->first_rule = rule; s->first_alarm_n = s->n; }
    }
    return rule;
}

void spc_finalize(const spc_t *s, spc_result_t *r)
{
    double sigma = (s->n > 1) ? sqrt(s->m2 / (double)(s->n - 1)) : 0.0;
    double cp = 0.0, cpu = 0.0, cpl = 0.0, cpk = 0.0;

    long phase2 = (s->n > s->warmup_n) ? (s->n - s->warmup_n) : 0;

    r->n = s->n;
    r->mean = (float)s->mean;
    r->sigma = (float)sigma;
    r->ewma = s->ewma;
    r->n_alarms = s->n_alarms;
    r->alarm_rate = (phase2 > 0) ? (float)s->n_alarms / (float)phase2 : 0.0f;
    r->first_rule = s->first_rule;
    r->first_alarm_n = s->first_alarm_n;
    /* a batch is condemned only on a SUSTAINED out-of-control pattern; isolated
     * chance trips of the sensitive run-rules (a few %) are investigated, not failed. */
    r->in_control = (r->alarm_rate < 0.10f) ? 1 : 0;

    if (sigma > 1e-6) {
        cp  = ((double)s->usl - (double)s->lsl) / (6.0 * sigma);
        cpu = ((double)s->usl - s->mean) / (3.0 * sigma);
        cpl = (s->mean - (double)s->lsl) / (3.0 * sigma);
        cpk = (cpu < cpl) ? cpu : cpl;
    }
    r->cp  = (float)cp;
    r->cpk = (float)cpk;
    r->capable = (cpk >= 1.33) ? 1 : 0;
}

const char *spc_rule_str(spc_rule_t r)
{
    switch (r) {
        case SPC_OK:                return "OK";
        case SPC_WE1_BEYOND_3SIGMA: return "WE1_beyond_3sigma";
        case SPC_WE2_2OF3_2SIGMA:   return "WE2_2of3_2sigma";
        case SPC_WE3_4OF5_1SIGMA:   return "WE3_4of5_1sigma";
        case SPC_WE4_8_RUN:         return "WE4_8_run";
        default:                    return "?";
    }
}

const char *spc_capability_str(float cpk)
{
    if (cpk >= 1.33f) return "capable";
    if (cpk >= 1.00f) return "marginal";
    return "not capable";
}
