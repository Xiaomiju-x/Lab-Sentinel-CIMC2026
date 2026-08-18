/******************************************************************************
 * furnace_ctrl.h — closed-loop recipe controller for the sintering furnace
 *
 * Turns the Lab-Sentinel from a MONITOR into an "AI-supervised closed-loop
 * furnace controller" (vs Eurotherm/Honeywell process programmers):
 *
 *   recipe programmer  -> setpoint SP(t)  (multi-segment ramp/soak/grind/cool,
 *                          values from sintering_profiles.json — real garnet/YAG)
 *   PID (anti-windup)  -> heater duty u in [0,1], driven on a relay/SSR as a
 *                          slow time-proportioning PWM (thermal mass is slow)
 *   AI safety layer    -> the 5 on-chip models supervise; on CRITICAL risk or a
 *                          control-tracking fault or over-temp -> SAFE STATE
 *                          (heater OFF, batch FAULT). Independent of the PID loop.
 *
 * Pure C (<stdint.h>/<math.h> only) so the loop is host-validated on an FOPDT
 * plant model BEFORE touching hardware; on the GD32 the plant is replaced by a
 * MAX31855 thermocouple read and `u` drives the heater relay (slow PWM).
 ******************************************************************************/
#ifndef FURNACE_CTRL_H
#define FURNACE_CTRL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- recipe programmer ---- */
typedef enum {
    SEG_RAMP = 0,   /* drive SP from prev target to target_C at rate_C_min      */
    SEG_SOAK = 1,   /* hold SP = target_C for dur_min                           */
    SEG_GRIND = 2,  /* off-furnace grinding: heater forced OFF for dur_min      */
    SEG_COOL = 3    /* SP ramps down to target_C at rate_C_min (heater ~off)    */
} seg_kind_t;

typedef struct {
    seg_kind_t kind;
    float target_C;       /* end setpoint of this segment                      */
    float rate_C_min;     /* ramp/cool rate (ignored for SOAK/GRIND)           */
    float dur_min;        /* soak/grind duration (ignored for RAMP/COOL)       */
    const char *label;
} recipe_seg_t;

typedef struct {
    const char        *name;
    const recipe_seg_t *seg;
    int                n_seg;
    float              amb_C;     /* ambient / start temperature               */
} recipe_t;

/* Garnet (YAG:Cr) program — sintering_profiles.json "garnet":
 *   calcine 900C 4h, sinter 1500C 6h, ramp 5C/min, cool ~3C/min, Al2O3 crucible. */
extern const recipe_t RECIPE_GARNET;

/* Setpoint at elapsed time (s). Also returns the active segment index + its kind. */
float recipe_setpoint(const recipe_t *r, float t_s, int *seg_idx, seg_kind_t *kind);
/* Total program length (s). */
float recipe_total_s(const recipe_t *r);

/* ---- PID (output 0..1 heater duty, anti-windup, derivative-on-measurement) ---- */
typedef struct {
    float kp, ki, kd;
    float integ;          /* integral accumulator                             */
    float prev_meas;      /* for derivative-on-measurement                    */
    float out_min, out_max;
    int   started;
} pid_t;

void  pid_init(pid_t *p, float kp, float ki, float kd, float out_min, float out_max);
float pid_step(pid_t *p, float sp, float meas, float dt_s);

/* ---- FOPDT plant model (SIM ONLY; on HW replaced by thermocouple read) ----
 *   dT/dt = ((amb + gain*u) - T) / tau   with a small transport delay.        */
typedef struct {
    float T;              /* current temperature (C)                          */
    float amb, gain, tau; /* steady-state T(u=1)=amb+gain ; time constant      */
    float delay_buf[8];   /* transport-delay ring (dead time ~ N*dt)          */
    int   delay_n, delay_i;
    uint32_t rng;
} plant_t;

void  plant_init(plant_t *pl, float amb, float gain, float tau, int delay_steps);
float plant_step(plant_t *pl, float u, float dt_s);   /* returns new T (with sensor noise) */

/* ---- closed-loop controller + AI safety supervisor ---- */
typedef enum {
    CTRL_IDLE = 0, CTRL_RUN = 1, CTRL_DONE = 2, CTRL_FAULT = 3
} ctrl_state_t;

typedef enum {
    CTRL_OK = 0,
    CTRL_FAULT_AI_CRITICAL = 1,   /* AI-4 risk = CRITICAL -> abort             */
    CTRL_FAULT_TRACK = 2,         /* |SP-meas| > tol too long -> element/door  */
    CTRL_FAULT_OVERTEMP = 3,      /* meas > max_safe_C                         */
    CTRL_FAULT_SENSOR = 4,        /* thermocouple open/short (HW fault bit)    */
    CTRL_FAULT_OPERATOR = 5       /* operator/voice abort to safe state        */
} ctrl_fault_t;

typedef struct {
    const recipe_t *recipe;
    pid_t  pid;
    ctrl_state_t state;
    ctrl_fault_t fault;
    float  t_elapsed_s;
    float  sp_C, meas_C, u;       /* setpoint, measured temp, heater duty 0..1 */
    int    seg_idx;
    seg_kind_t seg_kind;
    /* safety config */
    float  track_tol_C;           /* allowed |SP-meas| before counting fault   */
    float  track_fault_s;         /* sustained over-tol time -> TRACK fault    */
    float  max_safe_C;            /* hard over-temp cutoff                     */
    float  _track_bad_s;          /* internal: accumulated over-tol time       */
} furnace_ctrl_t;

void  furnace_ctrl_init(furnace_ctrl_t *c, const recipe_t *r,
                        float kp, float ki, float kd);
void  furnace_ctrl_start(furnace_ctrl_t *c);
/* One control step: given measured temp + AI risk (0..3, 3=CRITICAL), advance the
 * recipe, run PID, apply safety, return heater duty u (0 in FAULT/IDLE/DONE). */
float furnace_ctrl_step(furnace_ctrl_t *c, float meas_C, int ai_risk,
                        int tc_sensor_fault, float dt_s);
/* External AI / operator abort to safe state. */
void  furnace_ctrl_abort(furnace_ctrl_t *c, ctrl_fault_t reason);
const char *furnace_ctrl_fault_str(ctrl_fault_t f);

#ifdef __cplusplus
}
#endif
#endif /* FURNACE_CTRL_H */
