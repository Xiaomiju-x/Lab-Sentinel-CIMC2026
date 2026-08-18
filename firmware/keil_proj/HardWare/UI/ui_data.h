/******************************************************************************
 * ui_data.h — HMI data layer (decouples the LVGL UI from the control loop).
 *
 * ctrl_task (lab_sentinel.c) fills these snapshots under a critical section;
 * ui_task reads them to refresh the multi-screen HMI. The UI never touches
 * furnace_ctrl/spc internals directly — clean, thread-safe, render-anytime.
 ******************************************************************************/
#ifndef __UI_DATA_H__
#define __UI_DATA_H__

#include <stdint.h>

/* Live closed-loop controller snapshot. */
typedef struct {
    uint8_t  state;          /* ctrl_state_t: 0 idle 1 run 2 done 3 fault   */
    uint8_t  fault;          /* ctrl_fault_t                                 */
    uint8_t  seg_idx;
    uint8_t  risk;           /* AI-4 verdict 0..3 (s_eth_risk)               */
    char     seg_label[16];
    int      sp_c;           /* setpoint (C)                                 */
    int      meas_c;         /* measured / plant temp (C)                    */
    int      u_pct;          /* heater duty 0..100                           */
    uint32_t elapsed_s;
    uint32_t total_s;
    uint32_t batch_id;
    /* sinter-soak SPC live stats (0 until the soak accumulates samples)     */
    int      cpk_x100;       /* Cpk ×100                                     */
    int      spc_mean_x100;  /* soak deviation mean ×100 (C)                 */
    int      spc_sigma_x100; /* soak deviation sigma ×100 (C)                */
    uint8_t  spc_in_control; /* 1 = in statistical control                   */
    uint16_t spc_alarms;
    int      elem_pct;       /* heating-element remaining health %           */
    /* MAX31855 real K-type thermocouple channel (read every control step;    */
    /* its fault bit feeds the safety supervisor, see ctrl_task).             */
    int      probe_c;        /* real thermocouple temperature (C)            */
    uint8_t  tc_fault;       /* bit0 OC / bit1 SCG / bit2 SCV (0 = healthy)  */
    uint8_t  tc_present;     /* 1 = a plausible MAX31855 reading this cycle  */
} ctrl_snapshot_t;

/* Copy the live controller snapshot (thread-safe). */
void lab_ctrl_get(ctrl_snapshot_t *out);

/* Sinter-soak deviation series (meas-SP, C), oldest first, for the SPC chart.
 * Returns the number of samples copied (<= max). */
int  lab_spc_series(int16_t *out, int max);

/* SPC spec window for the chart limit lines: LSL/USL (C) + center line. */
void lab_spc_limits(int *lsl, int *usl, int *cl);

/* ---- Recipe profile (read-only immutable; for the Recipe screen) ---- */
typedef struct {
    uint8_t kind;        /* 0 ramp  1 soak  2 grind  3 cool          */
    int     target_c;    /* segment end setpoint (C)                 */
    int     rate_c_min;  /* ramp/cool rate (C/min), 0 for soak/grind */
    int     dur_min;     /* soak/grind duration (min), 0 for ramp    */
    char    label[20];
} recipe_seg_view_t;

/* Copy the active recipe's segments (returns n_seg, <= max). */
int         lab_recipe_segs(recipe_seg_view_t *out, int max);
const char *lab_recipe_name(void);
int         lab_recipe_total_min(void);

/* ---- Batch ledger (read-only window of the SHA-256 hash chain) ---- */
typedef struct {
    uint32_t batch_id;
    int      peak_c;
    int      cpk_x100;
    int      elem_pct;
    uint16_t ai_alarms;
    uint8_t  in_control;   /* SPC control verdict          */
    uint8_t  capable;      /* Cpk >= 1.33                  */
    uint8_t  final_state;  /* ctrl_state_t (2 done 3 fault)*/
    uint8_t  fault;        /* ctrl_fault_t                 */
    char     hash12[14];   /* first 12 hex of this_hash    */
} batch_view_t;

/* Copy the retained batch records, NEWEST FIRST (returns count, <= max). */
int lab_ledger_get(batch_view_t *out, int max);
/* Number of batches sealed since boot (may exceed the retained window). */
int lab_ledger_total(void);
/* Re-verify the retained hash chain: 1 intact, 0 tampered, -1 empty. */
int lab_ledger_chain_ok(void);

/* ---- Recipe pre-flight AI (AI-6 optical / AI-7 thermal / AI-8 energy /
 *      AI-9 analog retrieval). Computed once for the active recipe preset
 *      (deterministic from the formula), shown on the Pre-flight screen.    */
typedef struct {
    uint8_t valid;          /* 1 once computed                              */
    char    recipe[20];     /* preset name (e.g. "YAG:Cr garnet")           */
    int     lambda_nm;      /* AI-6 predicted emission peak (nm)            */
    int     fwhm_nm;        /* AI-6 predicted bandwidth (nm)                */
    int     thermal_pct;    /* AI-7 thermal stability % @150C (coarse)      */
    uint8_t thermal_band;   /* AI-7 0 poor / 1 marginal / 2 good            */
    int     kwh_x10;        /* AI-8 energy (kWh ×10)                        */
    int     co2_x10;        /* AI-8 carbon (kg CO2 ×10)                     */
    int     analog_idx;     /* AI-9 nearest historical recipe index         */
    char    analog_name[20];/* AI-9 nearest recipe label                    */
    int     analog_dist_x100;/* AI-9 normalised distance ×100               */
    /* ---- AI-11 phase-purity pre-flight prior (24-D descriptor, 37 real labels) */
    uint8_t purity_cls;     /* 0 impure / 1 pure (LOO 70% vs 60% base)      */
    int     p_pure_pct;     /* P(pure) % from the softmax                   */
    /* ---- derived crystal-field read-out (NOT a model: exact algebra from   */
    /*      AI-6 lambda_em: 10*Dq = 1e7/lambda + Stokes; B = corpus prior).    */
    int     dq_cm1;         /* derived Dq (crystal-field splitting)         */
    int     b_cm1;          /* Racah B (corpus prior ~650, near-constant)   */
    int     dq_over_b_x100; /* Dq/B ×100 -> field class                     */
    uint8_t field_class;    /* 0 weak / 1 weak-inter / 2 intermediate / 3 strong */
} recipe_ai_t;
void lab_get_recipe_ai(recipe_ai_t *out);

/* ---- Live camera view (OV5640 -> AI-1 crucible CNN + classical blob box).
 *      vision_task publishes a coarse luminance grid + the bright-region
 *      bounding box (in grid cells) + the CNN verdict; the default Camera
 *      screen renders the grid as lv_obj tiles with the box. The "LIVE"
 *      button opens a separate real-frame overlay (lv_img) — see below.    */
#define CAM_GW 8
#define CAM_GH 6
#define CAM_CELLS (CAM_GW * CAM_GH)
typedef struct {
    uint8_t valid;          /* 1 = a real camera frame (else pattern)        */
    uint8_t lum[CAM_CELLS]; /* per-cell luminance 0..255, row-major (gy*GW+gx)*/
    uint8_t blob_ok;        /* 1 = a bright crucible-like region was found   */
    uint8_t bx, by, bw, bh; /* bounding box in grid cells                    */
    int     cls;            /* AI-1 class 0 empty / 1 loaded / 2 done        */
    int     conf_pct;       /* top-class confidence %                        */
    float   probs[3];       /* AI-1 softmax (empty/loaded/done)              */
    uint32_t frames;        /* frames captured since boot                    */
    uint8_t cam_ok;         /* 1 = B3 CAM heatmap valid                      */
    uint8_t cam[16];        /* B3 Class Activation Map 4x4 (row-major), 0..255*/
} cam_view_t;
void lab_get_cam(cam_view_t *out);

/* ---- Camera LIVE real-frame view (Wave C). lab_camview_render() invalidates the
 *      OV5640 framebuffer's D-cache and writes a stable RGB565 320x240 copy into
 *      SDRAM_CAM_VIEW for the Camera page's lv_img: mode 1 = true colour (direct
 *      copy), mode 2 = thermal false-colour LUT (luminance -> blue..red). The
 *      buffer pointer is fetched once via lab_camview_buf(). All framebuffer +
 *      CMSIS-cache coupling stays here so the HMI layer needs no MCU headers. */
void           lab_camview_render(int mode);
const uint8_t *lab_camview_buf(void);

/* ---- AI-10 vibration PdM (live ADXL345 64-sample window) ---- */
typedef struct {
    uint8_t valid;          /* 1 once a window has been classified           */
    uint8_t running;        /* 1 = motor running (above idle gate, classified)*/
    int     cls;            /* 0 normal / 1 imbalance / 2 bearing / 3 looseness*/
    int     conf_pct;       /* top-class confidence %                         */
    int     rms_mg;         /* window AC RMS (mg) for context                 */
} vib_view_t;
void lab_get_vib(vib_view_t *out);

/* ---- PL-spectrum view: AI-12 dopant classifier + AI-13 QC autoencoder.
 *      The sentinel has no spectrometer, so the PL screen REPLAYS stored real
 *      Fluoromax emission spectra (ai12_demo_spectra.h) — same honest replay as
 *      furnace_sim — cycling one per class so both PL models can be demonstrated. */
#define PL_SPEC_N 64
typedef struct {
    uint8_t valid;          /* 1 once a spectrum has been classified         */
    int     cls;            /* AI-12 class 0 Cr / 1 Ni / 2 Cr+Ni             */
    int     conf_pct;       /* AI-12 top-class confidence %                  */
    int     probs_pct[3];   /* AI-12 softmax %                               */
    uint8_t anomaly;        /* AI-13 1 = recon MSE > q_hat (QC anomaly)      */
    int     mse_x1e5;       /* AI-13 reconstruction MSE x1e5                 */
    int     qhat_x1e5;      /* AI-13 conformal q_hat x1e5                    */
    int     demo_idx;       /* which replayed spectrum (0..2)                */
    float   spec[PL_SPEC_N];/* the 64-pt normalised spectrum (for the chart) */
    /* ---- AI-15 host-ID + AI-16 lambda_em + AI-17 few-shot (read same spectrum) */
    int     host_cls;       /* AI-15 0 NaY2Ga2InGe2O12 / 1 Y3ZnGa3GeO12      */
    int     host_conf_pct;  /* AI-15 host-class confidence %                 */
    int     lambda_nm;      /* AI-16 emission peak read from the spectrum (nm)*/
    int     fewshot_cls;    /* AI-17 nearest registered sample class         */
    int     fewshot_nclass; /* AI-17 number of registered sample classes     */
} pl_view_t;
void lab_get_pl(pl_view_t *out);

/* ---- AI-14 furnace-temperature forecast (live, on the running batch).
 *      ctrl_task keeps a window of recent measured temps, runs the AI-14 multi-step
 *      forecaster each step, and publishes the predicted next-N temps for the Trend
 *      screen to overlay (setpoint vs measured vs forecast). */
#define FC_HORIZON 12
typedef struct {
    uint8_t valid;              /* 1 once the window has filled                 */
    int     n;                  /* number of forecast steps (= FC_HORIZON)      */
    int     next_c[FC_HORIZON]; /* predicted temps (C), +1..+N minutes ahead    */
    int     reach_sp;           /* 1 if forecast trends toward setpoint, 0 stall*/
} fc_view_t;
void lab_get_forecast(fc_view_t *out);

/* ---- AI-19 sintering RUL/ETA + AI-20 thermocouple-integrity (live, env_task).
 *      Both read furnace_sim's per-minute features (exact training match):
 *      AI-19 = minutes-to-firing-complete; AI-20 = sensor-integrity verdict.
 *      env_task fills this under a critical section; the HMI reads it. */
typedef struct {
    uint8_t rul_valid;      /* 1 once the 24-min window has filled + running   */
    int     rul_min;        /* AI-19 estimated minutes to firing-complete      */
    uint8_t tc_valid;       /* 1 once a window has been classified             */
    int     tc_cls;         /* AI-20 0 healthy / 1 open-circuit / 2 erratic    */
    int     tc_conf_pct;    /* AI-20 top-class confidence %                    */
} ai_extra_view_t;
void lab_get_ai_extra(ai_extra_view_t *out);

/* AI-20 live-demo thermocouple-fault injection (HMI button): 0 none / 1 open-circuit /
 * 2 erratic. Non-destructive (only the classifier's input window is perturbed). */
void lab_set_tc_inject(int mode);
int  lab_get_tc_inject(void);

/* ---- On-chip AI latency table (us), measured once at boot by the DWT probe.
 *      Real M7 cycles/600. The Benchmark HMI panel reads this for genuine numbers. */
enum {
    AI_LAT_AI1 = 0, AI_LAT_AI2, AI_LAT_AI3, AI_LAT_AI4,
    AI_LAT_AI11, AI_LAT_AI12, AI_LAT_AI12Q, AI_LAT_AI13, AI_LAT_AI14,
    AI_LAT_AI15, AI_LAT_AI16, AI_LAT_AI17, AI_LAT_AI19, AI_LAT_AI20, AI_LAT_CAM,
    AI_LAT_N
};
int lab_get_ai_lat(int *out, int max);   /* copies up to max; returns count */

/* ---- Cloud uplink (ESP32-C3 over the I2C2 bus) status for the Control+Cloud
 *      screen (tab 3). env_task (1 Hz) pushes a telemetry block to the ESP32-C3
 *      I2C slave (addr 0x42) and reads back the WiFi/cloud link state + the last
 *      diagnosis the bridge fetched from a remote telemetry host.
 *      lab_get_cloud() copies the snapshot under a critical section. When no
 *      ESP32 is wired the I2C reads NACK -> link stays 0 (offline), graceful. */
typedef struct {
    uint8_t  link;        /* 0 offline / 1 wifi-up / 2 cloud-online           */
    int      rssi_dbm;    /* WiFi RSSI (negative dBm); 0 if unknown            */
    uint32_t uplinks;     /* telemetry blocks the ESP32 acked since boot       */
    uint32_t fails;       /* consecutive I2C link failures (0 = healthy)       */
    uint8_t  r1_valid;    /* 1 = a diagnosis string is present                 */
    char     r1[64];      /* last R1 diagnosis text from the XRD AI brain      */
} cloud_view_t;
void lab_get_cloud(cloud_view_t *out);

/* ---- Edge nano-LM diagnosis (generative, on-chip) + edge-cloud cascade.
 *  nlm_task builds a 12-slot control-token context from the live sentinel state
 *  and runs the ~0.6M-param INT8 GPT (ai_nanolm.c) to GENERATE a one-sentence
 *  Chinese diagnosis on the M7 — distilled from DeepSeek, works offline. It sets
 *  a cascade `escalate` flag when confidence is low or risk is critical, asking
 *  a remote review endpoint for a second opinion via the ESP32 uplink (cloud_view.r1).   */
typedef struct {
    uint8_t  valid;       /* 1 once a sentence has been generated              */
    char     text[96];    /* UTF-8 generated diagnosis                          */
    int      conf_pct;    /* mean top-1 token probability % (0..100)            */
    uint8_t  escalate;    /* 1 = low conf / critical -> escalate to cloud R1    */
    uint32_t gens;        /* generations since boot                             */
} nlm_view_t;
void lab_get_nlm(nlm_view_t *out);
void lab_nlm_request(void);   /* HMI: force a fresh generation now              */
/* Active generative LM across the size curve: 0=internal x1p9 (1.8M), 1..N=SPI
 * bank (m1p35 1.26M / s0p6 0.6M). lab_lm_cycle() advances it (HMI "NEXT LM"). */
void lab_lm_cycle(void);
int  lab_lm_active(void);

/* ---- On-device online-learning risk head (TinyML continual learning).
 *  Predicts risk from a compact state feature; the operator CONFIRMS/CORRECTS it
 *  on the HMI and each correction is one on-chip SGD step (it learns this
 *  furnace/operator). lab_online_teach(true_risk) applies a correction. */
typedef struct {
    uint8_t  valid;
    int      pred;        /* predicted risk 0 good / 1 warn / 2 bad / 3 crit    */
    int      conf_pct;
    uint32_t teaches;     /* operator corrections applied (SGD steps)           */
    int      acc_pct;     /* running accuracy over recent feedback (0..100)     */
} online_view_t;
void lab_get_online(online_view_t *out);
void lab_online_teach(int true_risk);   /* operator asserts the true risk       */

/* ---- Edge LLM CLUSTER: 5 role-specialized nano-LM experts swap-loaded one at a
 *  time from 8MB SPI flash into SDRAM (the MCU mirror of a server-class BPU swap-load
 *  cluster). The router picks an expert from the furnace state; the HMI can also
 *  manually cycle experts to demo the swap-load. */
typedef struct {
    uint8_t  valid;
    uint8_t  provisioned; /* 1 = cluster image present in SPI flash              */
    int      expert;      /* active expert id 0..4 (-1 none)                     */
    char     role[8];     /* active role tag: diag/recipe/energy/qc/brief        */
    char     text[96];    /* generated role sentence (UTF-8, shown in CJK font)  */
    int      conf_pct;    /* mean top-1 token probability %                      */
    int      swap_ms;     /* last expert swap-load time from SPI flash (ms)      */
    uint32_t gens;        /* generations since boot                              */
} cluster_view_t;
void lab_get_cluster(cluster_view_t *out);
void lab_cluster_next(void);          /* HMI: manually swap to the next expert   */

/* ---- Reliability / Robustness (the Robust HMI page, rubric "可靠性评估").
 *  LIVE multimodal perturbation injection demonstrating GRACEFUL DEGRADATION:
 *  the operator perturbs a sensing modality on-chip and watches the model's
 *  output degrade (clean vs perturbed verdict + confidence) while the SAFETY
 *  core (AI-4, debounced + motor-gated) holds. vision_task perturbs the AI-1
 *  crucible-CNN input; pl_refresh perturbs the AI-12 spectrum input; the AI-20
 *  thermocouple injection reuses lab_set_tc_inject. Non-destructive — only the
 *  classifier's own input copy is perturbed; the live pipeline is untouched.
 *  Perturbation modes (shared with the host robustness regression):
 *    vision : 0 clean / 1 noise / 2 dark / 3 bright / 4 occlusion
 *    spectrum: 0 clean / 1 noise / 2 occlusion / 3 baseline-drift              */
typedef struct {
    uint8_t vis_inject;     /* active vision perturbation mode 0..4              */
    int     vis_clean_cls;  /* AI-1 class on the clean frame                     */
    int     vis_clean_conf; /* AI-1 clean top confidence %                       */
    int     vis_pert_cls;   /* AI-1 class on the perturbed frame                 */
    int     vis_pert_conf;  /* AI-1 perturbed top confidence %                   */
    uint8_t vis_valid;      /* 1 once vision_task has run a robustness pair       */
    uint8_t pl_inject;      /* active spectrum perturbation mode 0..3            */
    int     pl_clean_cls;   /* AI-12 class on the clean spectrum                 */
    int     pl_clean_conf;  /* AI-12 clean top confidence %                      */
    int     pl_pert_cls;    /* AI-12 class on the perturbed spectrum             */
    int     pl_pert_conf;   /* AI-12 perturbed top confidence %                  */
    uint8_t pl_valid;       /* 1 once pl_refresh has run a robustness pair        */
} rob_view_t;
void lab_get_rob(rob_view_t *out);
void lab_set_vis_inject(int mode);   int lab_get_vis_inject(void);   /* 0..4 */
void lab_set_pl_inject(int mode);    int lab_get_pl_inject(void);    /* 0..3 */

/* ---- Long-run stability snapshot (the Robust page "LONG-RUN" panel).
 *  Genuine liveness/health for the rubric's "长时间稳定" item: how long the
 *  system has run, how many AI inferences it has served, and whether any reset
 *  was a fault (FWDGT/hard) vs a clean power-on/pin reset. */
typedef struct {
    uint32_t uptime_s;      /* seconds since boot                                */
    uint32_t inferences;    /* AI inferences served since boot (AI-1/2/4/12 ...)  */
    uint8_t  reset_cause;   /* 0 unknown / 1 watchdog-fault / 2 software / 3 power-on / 4 ext-pin */
    uint8_t  wdg_armed;     /* 1 = FWDGT functional-safety watchdog is armed      */
} health_view_t;
void lab_get_health(health_view_t *out);

#endif /* __UI_DATA_H__ */
