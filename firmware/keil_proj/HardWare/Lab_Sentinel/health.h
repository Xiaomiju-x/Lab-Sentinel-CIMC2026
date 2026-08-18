/******************************************************************************
 * health.h — equipment health (PdM) + drift monitoring (edge MLOps)
 *
 * Enterprise furnace lines run condition-based / predictive maintenance and
 * AMS2750 system-accuracy tests; cloud MLOps platforms watch for model input
 * drift. This module gives the Lab-Sentinel the on-chip equivalents:
 *
 *   1) heating-element health  — NO extra sensor: as a SiC/MoSi2 element ages
 *      its resistance rises, so the duty `u` needed to HOLD the soak setpoint
 *      climbs. Tracking the soak duty across batches yields a remaining-life %
 *      and WARN/EOL alarms before the element fails mid-batch.
 *
 *   2) drift monitor (generic)  — one-sided EWMA against a commissioned
 *      baseline (mean, sigma). Reused for (a) thermocouple SAT drift (AMS2750
 *      mandates |meas-ref| stay within tolerance) and (b) AI model input drift
 *      (e.g. the AI-2 autoencoder reconstruction-error baseline shifting ->
 *      "recalibrate"), the edge-MLOps equivalent of a data-drift detector.
 *
 * Pure C, O(1) memory; host-validated by host_test/health_test.c.
 ******************************************************************************/
#ifndef HEALTH_H
#define HEALTH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { HEALTH_OK = 0, HEALTH_WARN = 1, HEALTH_ALARM = 2 } health_status_t;

/* ---- heating-element health from control effort ---- */
typedef struct {
    float u0;          /* commissioned soak duty for a healthy element  */
    float u_warn;      /* soak duty that raises a WARN (e.g. 0.75)       */
    float u_eol;       /* soak duty = end-of-life (e.g. 0.90)            */
    float lambda;      /* cross-batch EWMA weight                       */
    float u_ewma;      /* smoothed soak duty across batches             */
    int   started;
    long  n_batches;
} elem_health_t;

void elem_health_init(elem_health_t *e, float u0, float u_warn, float u_eol, float lambda);
/* call once per completed soak with the MEAN heater duty held during that soak */
health_status_t elem_health_update(elem_health_t *e, float soak_u);
float elem_health_remaining_pct(const elem_health_t *e);   /* 100% healthy .. 0% at EOL */
const char *health_status_str(health_status_t s);

/* ---- generic one-sided EWMA drift monitor (TC SAT / model input drift) ---- */
typedef struct {
    float lambda;      /* EWMA weight                                   */
    float baseline;    /* commissioned mean                             */
    float sigma;       /* commissioned sigma                            */
    float k;           /* alarm threshold = baseline + k*sigma          */
    float ewma;
    int   started;
    int   tripped;     /* latches 1 once EWMA crosses the limit         */
    long  trip_n;      /* sample index where it first tripped           */
    long  n;
} drift_mon_t;

void drift_init(drift_mon_t *d, float baseline, float sigma, float lambda, float k);
/* feed one sample; returns 1 the first time the EWMA crosses baseline + k*sigma. */
int  drift_update(drift_mon_t *d, float x);
float drift_limit(const drift_mon_t *d);   /* baseline + k*sigma */

#ifdef __cplusplus
}
#endif
#endif /* HEALTH_H */
