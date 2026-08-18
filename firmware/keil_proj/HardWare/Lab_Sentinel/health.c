/******************************************************************************
 * health.c — equipment health (PdM) + drift monitoring. See health.h.
 * O(1) memory; host-validated by host_test/health_test.c before hardware.
 ******************************************************************************/
#include "health.h"

/* ───────────── heating-element health ───────────── */
void elem_health_init(elem_health_t *e, float u0, float u_warn, float u_eol, float lambda)
{
    e->u0 = u0; e->u_warn = u_warn; e->u_eol = u_eol;
    e->lambda = (lambda > 0.0f && lambda <= 1.0f) ? lambda : 0.3f;
    e->u_ewma = u0; e->started = 0; e->n_batches = 0;
}

health_status_t elem_health_update(elem_health_t *e, float soak_u)
{
    if (soak_u < 0.0f) soak_u = 0.0f;
    if (soak_u > 1.0f) soak_u = 1.0f;
    if (!e->started) { e->u_ewma = soak_u; e->started = 1; }
    else e->u_ewma = e->lambda * soak_u + (1.0f - e->lambda) * e->u_ewma;
    e->n_batches++;

    if (e->u_ewma >= e->u_eol)  return HEALTH_ALARM;
    if (e->u_ewma >= e->u_warn) return HEALTH_WARN;
    return HEALTH_OK;
}

float elem_health_remaining_pct(const elem_health_t *e)
{
    float span = e->u_eol - e->u0;
    float rem;
    if (span < 1e-6f) return 0.0f;
    rem = 100.0f * (e->u_eol - e->u_ewma) / span;
    if (rem < 0.0f)   rem = 0.0f;
    if (rem > 100.0f) rem = 100.0f;
    return rem;
}

const char *health_status_str(health_status_t s)
{
    switch (s) {
        case HEALTH_OK:    return "OK";
        case HEALTH_WARN:  return "WARN";
        case HEALTH_ALARM: return "ALARM";
        default:           return "?";
    }
}

/* ───────────── generic one-sided EWMA drift monitor ───────────── */
void drift_init(drift_mon_t *d, float baseline, float sigma, float lambda, float k)
{
    d->baseline = baseline;
    d->sigma = (sigma > 1e-9f) ? sigma : 1e-9f;
    d->lambda = (lambda > 0.0f && lambda <= 1.0f) ? lambda : 0.2f;
    d->k = (k > 0.0f) ? k : 3.0f;
    d->ewma = baseline; d->started = 0; d->tripped = 0; d->trip_n = 0; d->n = 0;
}

float drift_limit(const drift_mon_t *d)
{
    return d->baseline + d->k * d->sigma;
}

int drift_update(drift_mon_t *d, float x)
{
    d->n++;
    if (!d->started) { d->ewma = x; d->started = 1; }
    else d->ewma = d->lambda * x + (1.0f - d->lambda) * d->ewma;

    if (!d->tripped && d->ewma > drift_limit(d)) {
        d->tripped = 1; d->trip_n = d->n;
        return 1;
    }
    return 0;
}
