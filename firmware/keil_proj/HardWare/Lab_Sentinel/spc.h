/******************************************************************************
 * spc.h — Statistical Process Control for the sintering soak (AMS2750 / CQI-9)
 *
 * Enterprise heat-treat lines (AMS2750 thermal processing, CQI-9 §3) require
 * each batch's soak to be proven IN STATISTICAL CONTROL and CAPABLE against an
 * engineering tolerance (e.g. soak must hold setpoint +-10C). This module gives
 * the Lab-Sentinel the same QC layer that a furnace MES would, but ON-CHIP:
 *
 *   - online Welford mean/variance       -> Cp / Cpk vs spec window
 *   - EWMA chart (lambda)                -> early drift / slow bias detection
 *   - Western Electric runtime rules     -> out-of-control pattern alarms
 *
 * Everything is O(1) memory (no buffering the whole soak), so it streams during
 * the soak on the GD32: feed (meas - setpoint) each control tick, then finalize
 * at soak end to stamp the batch record with Cpk + control verdict.
 *
 * Pure C (<stdint.h>/<math.h>); host-validated by host_test/spc_test.c.
 ******************************************************************************/
#ifndef SPC_H
#define SPC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SPC_HIST 8   /* ring depth for Western Electric rules (need >= 5) */

/* Western Electric (Nelson) rule that tripped on a given sample. */
typedef enum {
    SPC_OK                 = 0,
    SPC_WE1_BEYOND_3SIGMA  = 1,  /* 1 point beyond 3 sigma                     */
    SPC_WE2_2OF3_2SIGMA    = 2,  /* 2 of 3 in zone A+ on the same side         */
    SPC_WE3_4OF5_1SIGMA    = 3,  /* 4 of 5 in zone B+ on the same side         */
    SPC_WE4_8_RUN          = 4   /* 8 consecutive on one side of the center    */
} spc_rule_t;

typedef struct {
    /* ---- config ---- */
    float  lsl, usl;        /* spec window on the monitored value (e.g. dev C) */
    float  ewma_lambda;     /* EWMA weight (0.2 typical)                       */
    int    warmup_n;        /* Phase-I samples to freeze the control limits    */

    /* ---- full-run Welford (for Cp/Cpk) ---- */
    double mean, m2;
    long   n;

    /* ---- control limits ----
     * center line = the TARGET (we control TO a setpoint, so a sustained offset
     * IS an out-of-control signal); only the spread (sigma) is learned in Phase I. */
    double cl;              /* center line (= target, fixed)                   */
    double sigma_cl;        /* Phase-I sigma -> zone boundaries                */
    int    frozen;

    /* ---- EWMA ---- */
    float  ewma;
    int    ewma_started;

    /* ---- Western Electric ring (most-recent samples) ---- */
    int8_t r_side[SPC_HIST];   /* +1 / -1 relative to center                  */
    int8_t r_level[SPC_HIST];  /* 0=zone C, 1=B, 2=A, 3=beyond 3 sigma         */
    int    ring_i, ring_fill;

    /* ---- results ---- */
    int        n_alarms;
    spc_rule_t first_rule;
    long       first_alarm_n;
} spc_t;

/* spec window = [lsl, usl] on the monitored quantity (deviation from SP, in C);
 * target = the control-chart center line (0 when monitoring deviation-from-SP). */
void spc_init(spc_t *s, float lsl, float usl, float target,
              float ewma_lambda, int warmup_n);

/* Feed one sample (e.g. meas - SP). Returns the rule that fired THIS sample
 * (SPC_OK if none). Control limits are inactive until warmup_n samples seen. */
spc_rule_t spc_update(spc_t *s, float x);

typedef struct {
    long       n;
    float      mean, sigma;
    float      cp, cpk;        /* capability vs spec window                    */
    float      ewma;           /* final EWMA value                             */
    int        n_alarms;       /* total Western Electric alarms                */
    float      alarm_rate;     /* alarms / Phase-II samples                    */
    spc_rule_t first_rule;
    long       first_alarm_n;
    int        in_control;     /* 1 if alarm_rate < 10% (tolerates chance trips) */
    int        capable;        /* 1 if cpk >= 1.33                            */
} spc_result_t;

void spc_finalize(const spc_t *s, spc_result_t *r);
const char *spc_rule_str(spc_rule_t r);
const char *spc_capability_str(float cpk);   /* "capable" / "marginal" / "not capable" */

#ifdef __cplusplus
}
#endif
#endif /* SPC_H */
