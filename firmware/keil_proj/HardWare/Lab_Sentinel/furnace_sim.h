/******************************************************************************
 * furnace_sim.h  —  NIR-phosphor sintering furnace temperature simulator
 *
 * WHY a simulator: the bench has SHT30 (room T/RH), MQ-135 (gas) and ADXL345
 * (vibration) — real sensors — but NO 1600 C furnace thermocouple. AI-2/AI-3
 * are trained on the 0-1600 C furnace temperature trajectory. So the furnace
 * temperature channel is *played back* from a real sintering profile (garnet /
 * YAG:Cr, 900 C calcine + 1500 C sinter, ramp 5 C/min — sintering_profiles.json
 * "garnet"), while room/gas/vibration come from the real sensors. This is the
 * standard way to demo a furnace-anomaly detector without burning phosphor at
 * 1600 C on the judging table, and is labelled as such in the UI/report.
 *
 * Reproduces simulate_host() (CIMC/model/ai2_env_ae/synth_data.py) Group-A
 * temperature math exactly, so AI-2/AI-3 see in-distribution features.
 * Injectable anomalies mirror synth_data_ai3.py (fast_ramp / undertemp /
 * temp_drift / slow_ramp) for live demo of AI-3 + AI-2 + AI-4 reaction.
 ******************************************************************************/
#ifndef FURNACE_SIM_H
#define FURNACE_SIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { FURN_IDLE = 0, FURN_RUNNING = 1, FURN_DONE = 2 } furnace_state_t;

/* mirror synth_data_ai3.py anomaly classes (AI-3 labels 1..4) */
typedef enum {
    FURN_ANOM_NONE       = 0,
    FURN_ANOM_FAST_RAMP  = 1,
    FURN_ANOM_UNDERTEMP  = 2,
    FURN_ANOM_TEMP_DRIFT = 3,
    FURN_ANOM_SLOW_RAMP  = 4
} furnace_anomaly_t;

typedef struct {
    float temp_feat[8];   /* Group-A normalised features (= AI-3 input row)  */
    float vis_proxy[4];   /* Group-E stage-conditioned vision probs [emp,load,sint,done] */
    int   stage_id;       /* 0 ramp1 1 calcine 2 grind 3 ramp2 4 sinter 5 cool */
    int   atm_code;       /* atmosphere (0 air for garnet)                   */
    float grinding_flag;  /* 1.0 during grinding stage                       */
    float progress;       /* 0..1 across the whole run                       */
    float t_current_C;    /* raw furnace temperature (for LCD/voice)         */
    float t_target_C;
    furnace_state_t state;
} furnace_out_t;

void              furnace_sim_init(void);
void              furnace_sim_start(void);                 /* run from minute 0 */
void              furnace_sim_stop(void);                  /* back to idle      */
void              furnace_sim_set_anomaly(furnace_anomaly_t a);
furnace_anomaly_t furnace_sim_get_anomaly(void);
furnace_state_t   furnace_sim_state(void);

/* Advance `n_minutes` simulated minutes (recording each into the 60-min
 * history) and fill `out` with the state at the final minute. */
void furnace_sim_advance(int n_minutes, furnace_out_t *out);

#ifdef __cplusplus
}
#endif

#endif /* FURNACE_SIM_H */
