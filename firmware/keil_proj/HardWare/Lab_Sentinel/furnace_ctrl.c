/******************************************************************************
 * furnace_ctrl.c — closed-loop recipe controller + AI safety supervisor.
 * See furnace_ctrl.h. Host-validated (ctrl_test.c) before hardware.
 ******************************************************************************/
#include "furnace_ctrl.h"
#include <math.h>

/* ───────────── recipe: garnet (YAG:Cr), sintering_profiles.json "garnet" ─────
 * calcine 900C 4h, sinter 1500C 6h, ramp 5C/min, cool ~3C/min, Al2O3 crucible,
 * 2 grinding cycles (off-furnace). Source: predict_engine/sintering_profiles.json.
 * Program: ramp1 -> calcine soak -> grind(off) -> ramp2 -> sinter soak -> cool. */
static const recipe_seg_t s_garnet_seg[] = {
    { SEG_RAMP,   900.0f, 5.0f,   0.0f, "ramp1->900"  },
    { SEG_SOAK,   900.0f, 0.0f, 240.0f, "calcine 4h"  },
    { SEG_GRIND,   25.0f, 0.0f,  30.0f, "grind(off)"  },
    { SEG_RAMP,  1500.0f, 5.0f,   0.0f, "ramp2->1500" },
    { SEG_SOAK,  1500.0f, 0.0f, 360.0f, "sinter 6h"   },
    { SEG_COOL,    25.0f, 3.0f,   0.0f, "cool->25"    },
};
const recipe_t RECIPE_GARNET = {
    "garnet(YAG:Cr)", s_garnet_seg,
    (int)(sizeof(s_garnet_seg) / sizeof(s_garnet_seg[0])), 25.0f
};

/* segment duration (s): RAMP/COOL derived from rate, SOAK/GRIND from dur_min. */
static float seg_dur_s(const recipe_t *r, int i, float seg_start_C)
{
    const recipe_seg_t *s = &r->seg[i];
    if (s->kind == SEG_RAMP || s->kind == SEG_COOL) {
        float d = fabsf(s->target_C - seg_start_C);
        float rate = (s->rate_C_min > 0.1f) ? s->rate_C_min : 5.0f;
        return (d / rate) * 60.0f;
    }
    return s->dur_min * 60.0f;
}

/* walk segments tracking the SP at each boundary */
float recipe_setpoint(const recipe_t *r, float t_s, int *seg_idx, seg_kind_t *kind)
{
    float start_C = r->amb_C;
    float t0 = 0.0f;
    int i;
    for (i = 0; i < r->n_seg; i++) {
        float dur = seg_dur_s(r, i, start_C);
        float end_C = r->seg[i].target_C;
        if (t_s <= t0 + dur || i == r->n_seg - 1) {
            float frac = (dur > 1e-3f) ? (t_s - t0) / dur : 1.0f;
            if (frac < 0.0f) frac = 0.0f;
            if (frac > 1.0f) frac = 1.0f;
            if (seg_idx) *seg_idx = i;
            if (kind)    *kind = r->seg[i].kind;
            if (r->seg[i].kind == SEG_SOAK)  return end_C;
            if (r->seg[i].kind == SEG_GRIND) return r->amb_C;            /* heater off */
            return start_C + (end_C - start_C) * frac;                  /* ramp/cool */
        }
        t0 += dur;
        start_C = end_C;
    }
    if (seg_idx) *seg_idx = r->n_seg - 1;
    if (kind)    *kind = SEG_COOL;
    return r->amb_C;
}

float recipe_total_s(const recipe_t *r)
{
    float start_C = r->amb_C, t = 0.0f;
    int i;
    for (i = 0; i < r->n_seg; i++) {
        t += seg_dur_s(r, i, start_C);
        start_C = r->seg[i].target_C;
    }
    return t;
}

/* ───────────── PID ───────────── */
void pid_init(pid_t *p, float kp, float ki, float kd, float out_min, float out_max)
{
    p->kp = kp; p->ki = ki; p->kd = kd;
    p->integ = 0.0f; p->prev_meas = 0.0f;
    p->out_min = out_min; p->out_max = out_max; p->started = 0;
}

float pid_step(pid_t *p, float sp, float meas, float dt_s)
{
    float err = sp - meas;
    float deriv, u, u_unclamped;
    if (!p->started) { p->prev_meas = meas; p->started = 1; }

    /* derivative on measurement (no setpoint-kick) */
    deriv = -(meas - p->prev_meas) / (dt_s > 1e-3f ? dt_s : 1e-3f);
    p->prev_meas = meas;

    /* tentative output with current integral */
    u_unclamped = p->kp * err + p->ki * p->integ + p->kd * deriv;
    u = u_unclamped;
    if (u > p->out_max) u = p->out_max;
    if (u < p->out_min) u = p->out_min;

    /* conditional integration (anti-windup): only integrate when not saturated,
     * or when integrating would pull the output back toward range. */
    if ((u_unclamped >= p->out_min && u_unclamped <= p->out_max) ||
        (err > 0.0f && u <= p->out_min) == 0) {
        if (!(u >= p->out_max && err > 0.0f) && !(u <= p->out_min && err < 0.0f))
            p->integ += err * dt_s;
    }
    return u;
}

/* ───────────── FOPDT plant (sim only) ───────────── */
void plant_init(plant_t *pl, float amb, float gain, float tau, int delay_steps)
{
    int k;
    pl->T = amb; pl->amb = amb; pl->gain = gain; pl->tau = tau;
    pl->delay_n = (delay_steps < 1) ? 1 : (delay_steps > 8 ? 8 : delay_steps);
    pl->delay_i = 0;
    for (k = 0; k < 8; k++) pl->delay_buf[k] = 0.0f;
    pl->rng = 0x1234567u;
}

float plant_step(plant_t *pl, float u, float dt_s)
{
    float u_delayed, drive, noise;
    /* transport delay on the actuation */
    pl->delay_buf[pl->delay_i] = u;
    pl->delay_i = (pl->delay_i + 1) % pl->delay_n;
    u_delayed = pl->delay_buf[pl->delay_i];

    drive = pl->amb + pl->gain * u_delayed;
    pl->T += (drive - pl->T) / pl->tau * dt_s;     /* first-order lag */

    /* small thermocouple-ish noise (deterministic LCG) */
    pl->rng = pl->rng * 1103515245u + 12345u;
    noise = (((pl->rng >> 16) & 0xFFFF) / 65535.0f - 0.5f) * 1.5f;  /* +-0.75C */
    return pl->T + noise;
}

/* ───────────── closed-loop controller + AI safety ───────────── */
void furnace_ctrl_init(furnace_ctrl_t *c, const recipe_t *r,
                       float kp, float ki, float kd)
{
    c->recipe = r;
    pid_init(&c->pid, kp, ki, kd, 0.0f, 1.0f);
    c->state = CTRL_IDLE; c->fault = CTRL_OK;
    c->t_elapsed_s = 0.0f; c->sp_C = r->amb_C; c->meas_C = r->amb_C; c->u = 0.0f;
    c->seg_idx = 0; c->seg_kind = SEG_RAMP;
    c->track_tol_C = 60.0f; c->track_fault_s = 180.0f; c->max_safe_C = 1650.0f;
    c->_track_bad_s = 0.0f;
}

void furnace_ctrl_start(furnace_ctrl_t *c)
{
    c->state = CTRL_RUN; c->fault = CTRL_OK; c->t_elapsed_s = 0.0f;
    c->_track_bad_s = 0.0f;
    pid_init(&c->pid, c->pid.kp, c->pid.ki, c->pid.kd, 0.0f, 1.0f);
}

void furnace_ctrl_abort(furnace_ctrl_t *c, ctrl_fault_t reason)
{
    c->state = CTRL_FAULT; c->fault = reason; c->u = 0.0f;
}

float furnace_ctrl_step(furnace_ctrl_t *c, float meas_C, int ai_risk,
                        int tc_sensor_fault, float dt_s)
{
    float total;
    c->meas_C = meas_C;

    if (c->state != CTRL_RUN) { c->u = 0.0f; return 0.0f; }

    /* ---- safety supervisor (independent of the PID loop) ---- */
    if (tc_sensor_fault)              { furnace_ctrl_abort(c, CTRL_FAULT_SENSOR);   return 0.0f; }
    if (meas_C > c->max_safe_C)       { furnace_ctrl_abort(c, CTRL_FAULT_OVERTEMP); return 0.0f; }
    if (ai_risk >= 3)                 { furnace_ctrl_abort(c, CTRL_FAULT_AI_CRITICAL); return 0.0f; }

    /* ---- recipe setpoint ---- */
    c->t_elapsed_s += dt_s;
    c->sp_C = recipe_setpoint(c->recipe, c->t_elapsed_s, &c->seg_idx, &c->seg_kind);

    total = recipe_total_s(c->recipe);
    if (c->t_elapsed_s >= total) { c->state = CTRL_DONE; c->u = 0.0f; return 0.0f; }

    /* ---- control-tracking fault watch (only while actively heating) ---- */
    if (c->seg_kind == SEG_RAMP || c->seg_kind == SEG_SOAK) {
        if (fabsf(c->sp_C - meas_C) > c->track_tol_C) {
            c->_track_bad_s += dt_s;
            if (c->_track_bad_s > c->track_fault_s) {
                furnace_ctrl_abort(c, CTRL_FAULT_TRACK); return 0.0f;
            }
        } else {
            c->_track_bad_s = 0.0f;
        }
    } else {
        c->_track_bad_s = 0.0f;
    }

    /* ---- PID, or heater forced off during grinding/cooling ---- */
    if (c->seg_kind == SEG_GRIND || c->seg_kind == SEG_COOL) {
        c->u = 0.0f;
        /* keep integral from winding during open-loop phases */
        c->pid.integ = 0.0f; c->pid.prev_meas = meas_C;
    } else {
        c->u = pid_step(&c->pid, c->sp_C, meas_C, dt_s);
    }
    return c->u;
}

const char *furnace_ctrl_fault_str(ctrl_fault_t f)
{
    switch (f) {
        case CTRL_OK:                 return "OK";
        case CTRL_FAULT_AI_CRITICAL:  return "AI_CRITICAL";
        case CTRL_FAULT_TRACK:        return "TRACK_FAULT";
        case CTRL_FAULT_OVERTEMP:     return "OVERTEMP";
        case CTRL_FAULT_SENSOR:       return "TC_SENSOR";
        case CTRL_FAULT_OPERATOR:     return "OPERATOR";
        default:                      return "?";
    }
}
