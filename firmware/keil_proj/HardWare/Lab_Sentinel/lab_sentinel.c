/******************************************************************************
 * lab_sentinel.c
 *
 * CIMC Lab-Sentinel - Task Framework Implementation (Phase 0 baseline)
 *
 * All 8 tasks are heartbeat stubs at this stage. They print to USART0 (CH340)
 * once per period to prove the FreeRTOS scheduler is alive and prioritization
 * is correct. Each Phase 1+ subsystem replaces its stub body with real work.
 *
 * Build dependency: HeaderFiles.h, FreeRTOS, USART driver (usart.h).
 *
 ******************************************************************************/

#include "lab_sentinel.h"

#include <stdio.h>
#include <string.h>
#include <math.h>           /* expf — vision_task crucible softmax */
#include "usart.h"
#include "LED.h"
#include "network.h"
#include "ci1302.h"
#include "actuator.h"
#include "sensors_i2c.h"     /* ESP32-C3 IIoT bridge (shares the PH7/PH8 I2C2 bus) */

/* Set 0 if the Keil CMSIS pack's core_cm7.h DWT/CoreDebug symbols are missing
 * (only disables the boot latency printout — the AI pipeline is unaffected). */
#define AI_LATENCY_PROBE 1

/* AI inference engines (firmware/ai_models_c) + furnace sim + feature builder */
#include "ai1_cnn.h"
#include "ai1_crucible.h"   /* AI-1 deployed task model: 3-class crucible CNN */
#include "ai1b_ncm.h"
#include "ai2_ae.h"
#include "ai3_transformer.h"
#include "ai4_fusion.h"
#include "ai5_diagnose.h"    /* AI-5 root-cause diagnoser (live, UART log only) */
#include "ai_new_models.h"   /* AI-6..AI-10 edge surrogates / retrieval / vib PdM */
#include "ai_ext_models.h"   /* AI-11 purity / AI-12 PL class / AI-13 PL-QC AE + AI-14..17 + online head */
#include "ai_nanolm.h"       /* edge flagship LM: on-chip generative diagnosis (DeepSeek-V4-distilled) */
/* The flagship LM (d192/4L, nanolm_vocab.h) and the 7-expert cluster (d128/3L,
 * nlm_cluster_vocab.h) deliberately ship DIFFERENT sizes but share the NLM_ macro
 * namespace. Each engine compiles in its OWN translation unit with the right dims;
 * this integrator TU only uses the control-token ids and NLM_NCTX (identical in
 * both). Drop the 5 per-model dim macros before the cluster header re-defines
 * them, so there is no (otherwise last-wins, but noisy) redefinition warning. */
#undef NLM_DMODEL
#undef NLM_NLAYER
#undef NLM_NHEAD
#undef NLM_DFF
#undef NLM_MAXSEQ
#include "ai_llm_cluster.h"  /* edge LLM cluster: 5 swap-loaded experts (SPI flash -> SDRAM)   */
#include "ai_lm_bank.h"      /* flagship LM size bank: s0p6/m1p35 swap-loaded vs internal x1p9 */
#include "cimc_spiflash.h"   /* 8MB SPI flash (SPI4) backing store for the cluster             */
#include "flash_provision.h" /* one-time UART provisioning of the cluster image + LM bank       */
#include "lab_build_config.h"/* LAB_LM_ENABLE: comment out to drop the LM stack for fast UI-dev flash */
#include "ai14_forecast_weights.h" /* AI14_WIN / AI14_HOR / AI14_TNORM for the forecaster */
#include "ai19_rul_weights.h"      /* AI19_WIN / AI19_NX / AI19_RNORM */
#include "ai20_tcfault_weights.h"  /* AI20_L for the TC-integrity window */
#include "ai12_demo_spectra.h" /* replayed real PL emission spectra (AI-12/13)    */
#include "recipe_presets.h"  /* PC-precomputed feature vectors for AI-6/7/8/9     */
#include "gas_safety.h"      /* formula-aware furnace gas-evolution supervisor */
#include "ai_selftest.h"
#include "furnace_sim.h"
#include "feature_build.h"

/* Closed-loop recipe controller + quality/traceability stack (host-validated):
 *   furnace_ctrl — recipe programmer + PID + AI safety supervisor
 *   spc          — soak SPC / Cpk (AMS2750/CQI-9 control + capability)
 *   health       — heating-element PdM + drift monitor
 *   batch_record — SHA-256 hash-chained electronic batch ledger */
#include "furnace_ctrl.h"
#include "spc.h"
#include "health.h"
#include "batch_record.h"
#include "sha256.h"

/* Phase B — LCD / Touch / LVGL */
#include "sdram.h"
#include "st7796.h"
#include "gt911.h"
#include "lv_port_disp.h"
#include "lv_port_indev.h"
#include "ui_screen.h"
#include "ui_data.h"
#include "lvgl.h"

/* Phase C — environmental + motion sensors */
#include "sensors_i2c.h"
#include "sht30.h"
#include "adxl345.h"
#include "mq135.h"

/* Phase D — OV5640 camera (DCI + DMA + SCCB) */
#include "ov5640.h"

/* Phase E — relay + smoke sensor + INMP441 mic */
#include "relay.h"
#include "smoke_sensor.h"
#include "inmp441.h"

/* Phase F — L298N motor (烧结炉风扇/搅拌 demo) */
#include "motor.h"

/* Phase 3 — Network: lwIP + Modbus TCP + HTTP POST to XRD AI brain */
#include "lwip/tcpip.h"
#include "lwip/netif.h"
#include "lwip/ip4_addr.h"
#include "ethernetif.h"
#include "modbus_tcp.h"
#include "http_client.h"
#include "lwipopts.h"

/* -------------------------------------------------------------------------- */
/* Stack sizes (in WORDS, FreeRTOS convention - 1 word = 4 bytes on M7)       */
/* -------------------------------------------------------------------------- */
#define STACK_INIT    2560   /* boot + 20-AI golden self-test. History: 512 (5-AI) ->
                              * 2048 (18-AI, after a boot HardFault when the summed local
                              * arrays overflowed the 512 frame) -> 2560 (20-AI: AI-19/20
                              * added feat26/win24/h0/h1 blocks to ai_selftest_run). RULE:
                              * every model added to the boot self-test re-checks this. The
                              * boot task self-deletes so the 10KB is reclaimed; AI engines
                              * themselves use SDRAM/.bss buffers, not stack. */
#define STACK_SENSOR  1280   /* boot MNIST smoke test ai_input[784] ~3.2KB + AI-10 win[64]/probs per-second block */
#define STACK_VISION  1024
#define STACK_ENV     1408   /* AI-2 + AI-3 (activations in SDRAM) + AI-19/20 rings/feat + snprintf
                              * + robustness PL perturbation pair (pl_refresh sp[64] copy). Was 1280;
                              * +128w margin per "bump the task stack whenever a task grows". */
#define STACK_FUSION   768   /* AI-4 fusion + risk alert + snprintf */
#define STACK_UI      3584   /* 14KB: LVGL render depth + AI-5 label update + Wave B (20-card Models page + per-model detail / benchmark overlays). Was 12KB (verified clean @18 models); +2KB margin per the rule "bump the task stack whenever the HMI grows". 2048(8KB) once overran when AI-5 HMI was added → silent heap corruption; configCHECK_FOR_STACK_OVERFLOW=2 guards it. */
#define STACK_VOICE    512
#define STACK_ETH     1024
#define STACK_DOA      512
#define STACK_CTRL    1024   /* furnace_ctrl + spc + health + sha256 batch seal */
#define STACK_WDG      384   /* tiny: heartbeat scan + FWDGT reload + snprintf */
#define STACK_NLM      1024  /* edge generative LM gen (KV cache in SDRAM, weights in Flash
                              * for x1p9 / SDRAM blob for the swap-load bank, activations in
                              * static .bss) + online head + LM-switch path + snprintf trace.
                              * Engine uses no big stack arrays; 4KB is ample. */
#define STACK_CLUSTER  768   /* edge LLM cluster: same engine (KV+blob in SDRAM,
                              * scratch in static .bss), router + swap-load + trace. */

/* IPC queue depths */
#define Q_DEPTH_VISION  4
#define Q_DEPTH_VIB     8
#define Q_DEPTH_ENV     4
#define Q_DEPTH_RISK    8

/* -------------------------------------------------------------------------- */
/* Globals                                                                    */
/* -------------------------------------------------------------------------- */
QueueHandle_t       xQueue_VisionResult = NULL;
QueueHandle_t       xQueue_VibResult    = NULL;
QueueHandle_t       xQueue_EnvResult    = NULL;
QueueHandle_t       xQueue_RiskAlert    = NULL;
SemaphoreHandle_t   xSem_VoiceCmd       = NULL;
static SemaphoreHandle_t xMutex_UART    = NULL;

/* Shared eth-task snapshot: written by env/sensor tasks (≥1 Hz),
 * read by eth_task for Modbus registers and HTTP POST. volatile is
 * sufficient — no atomic multi-field consistency needed at these rates. */
static volatile uint16_t s_eth_temp_q8  = 0U;
static volatile uint16_t s_eth_humid_q8 = 0U;
static volatile uint16_t s_eth_mq135    = 0U;
static volatile uint16_t s_eth_vib_rms  = 0U;
static volatile uint8_t  s_eth_risk     = 0U;
static volatile uint8_t  s_eth_smoke    = 0U;

/* -------------------------------------------------------------------------- */
/* AI pipeline shared state — produced by env_task (AI-2 + AI-3) @1 Hz,       */
/* consumed by fusion_task (AI-4). Copied under a brief critical section so    */
/* fusion never sees a half-updated frame.                                    */
/* -------------------------------------------------------------------------- */
typedef struct {
    float   ai1_probs[4];   /* vision proxy (or real AI-1 when camera works)  */
    float   ai2_ratio;      /* AI-2 anomaly ratio = mse/q_hat (clip 0..6)     */
    float   ai2_resid[3];   /* AI-2 temp/vib/gas residuals                    */
    float   ai3_probs[5];   /* AI-3 softmax                                   */
    float   progress;       /* sintering progress 0..1                        */
    int     ai3_cls;        /* AI-3 argmax                                    */
    int     stage;          /* furnace stage 0..5                             */
    float   temp_c;         /* simulated furnace temperature (display)        */
    uint8_t ready;          /* set once env_task has produced ≥1 result       */
    /* live AI-1 crucible 3-class CNN result from the OV5640 frame (vision_task,
     * 5 Hz). Real on-chip inference on real pixels, weights trained on real
     * phone crucible photos (CV 90.7%). Only [0:3] used (empty/loaded/done); the
     * AI-4 risk fusion still uses the well-characterised 4-stage furnace proxy
     * (process-stage signal), not this static visual readout. */
    float   ai1_cam_probs[4];
    int     ai1_cam_cls;    /* argmax class (valid only when ai1_cam_valid=1)  */
    uint8_t ai1_cam_valid;  /* 1 = a real camera frame was classified          */
    float   ai3_attn[AI3_SEQ_LEN];  /* AI-3 attention saliency (explainability) */
} ai_state_t;
static ai_state_t s_ai;

/* simulated furnace minutes advanced per 1 Hz env tick (keeps per-minute ramp
 * gradients exactly as AI-3 was trained, while reaching hold in ~1 min wall). */
#define FURNACE_STEP_MIN  4

/* AI-3 / AI-4 class names for the boot/diag log */
static const char *AI3_NAMES[5] = {"normal", "fast_ramp", "undertemp", "temp_drift", "slow_ramp"};
static const char *AI4_NAMES[4] = {"good", "suspected", "bad", "critical"};

/* Thread-safe snapshot of the live AI pipeline state for the LVGL dashboard. */
void lab_sentinel_get_ai(lab_ai_snapshot_t *out)
{
    int i;
    if (out == NULL) return;
    taskENTER_CRITICAL();
    for (i = 0; i < 4; i++) out->ai1_probs[i] = s_ai.ai1_probs[i];
    for (i = 0; i < 3; i++) out->ai2_resid[i] = s_ai.ai2_resid[i];
    for (i = 0; i < 5; i++) out->ai3_probs[i] = s_ai.ai3_probs[i];
    out->ai2_ratio = s_ai.ai2_ratio;
    out->progress  = s_ai.progress;
    out->ai3_cls   = s_ai.ai3_cls;
    out->stage     = s_ai.stage;
    out->temp_c    = s_ai.temp_c;
    out->ready     = s_ai.ready;
    for (i = 0; i < AI3_SEQ_LEN; i++) out->ai3_attn[i] = s_ai.ai3_attn[i];
    taskEXIT_CRITICAL();
}

/* -------------------------------------------------------------------------- */
/* Forward declarations                                                       */
/* -------------------------------------------------------------------------- */
static void task_init(void *pv);
static void sensor_task(void *pv);
static void vision_task(void *pv);
static void env_task(void *pv);
static void fusion_task(void *pv);
static void ui_task(void *pv);
static void voice_task(void *pv);
static void eth_task(void *pv);
static void doa_task(void *pv);
static void ctrl_task(void *pv);
static void wdg_task(void *pv);
static void nlm_task(void *pv);
static void cluster_task(void *pv);

static void boot_print(const char *s);

/* Closed-loop controller command, set by voice_dispatch, consumed by ctrl_task. */
#define CTRL_CMD_NONE   0
#define CTRL_CMD_START  1
#define CTRL_CMD_ABORT  2
static volatile int s_ctrl_cmd = CTRL_CMD_NONE;
/* Batch ventilation fan (physical fan on the relay, IN=PA1): ON during a sinter
 * batch (START) — the in-process cooling/exhaust fan. The sensor_task relay block
 * keeps the relay = (s_batch_fan || smoke_alarm) so START and the smoke-alarm
 * ventilation don't fight each other. */
static volatile uint8_t s_batch_fan = 0u;

/* Voice-driven screen navigation request: voice_task sets a target tab here,
 * ui_task consumes it and calls ui_screen_set_nav. LVGL is touched ONLY by
 * ui_task — voice_task must never call into LVGL directly (not thread-safe). */
static volatile int s_nav_req = -1;   /* -1 = none; else tab 0..5 */

/* AI-5 root-cause display state (set by env_task at gas events, read by ui_task
 * via lab_get_ai5). Plain volatiles instead of a snapshot-struct field so the
 * HMI path stays decoupled from lab_ai_snapshot_t. */
static volatile int s_ai5_cls = 0;    /* ai5_rootcause_t (0 = NORMAL)            */
static volatile int s_ai5_pct = 0;    /* top-class confidence %                   */
static volatile int s_ncm_cls = -1;   /* AI-1b few-shot NCM nearest class (-1 = none) */

void lab_get_ai5(int *cls, int *pct)
{
    if (cls) *cls = s_ai5_cls;
    if (pct) *pct = s_ai5_pct;
}

void lab_get_ncm(int *cls)
{
    if (cls) *cls = s_ncm_cls;
}

/* -------------------------------------------------------------------------- */
/* AI-6..AI-10 runtime state (decoupled getters, AI-5 pattern: plain module    */
/* storage + a thread-safe copy-out, kept off the s_ai/Home snapshot path).    */
/* -------------------------------------------------------------------------- */
/* The deployed recipe is RECIPE_GARNET == preset 0 ("YAG:Cr garnet"); its     */
/* AI-6/7/8/9 feature vectors are PC-precomputed in recipe_presets.h.          */
#define ACTIVE_PRESET 0

static recipe_ai_t s_recipe_ai;     /* AI-6/7/8/9 pre-flight (computed once)   */
static cam_view_t  s_cam;           /* AI-1 vision + blob box (vision_task)    */
static vib_view_t  s_vib;           /* AI-10 vibration PdM (sensor_task)       */

void lab_get_recipe_ai(recipe_ai_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_recipe_ai;
    taskEXIT_CRITICAL();
}

void lab_get_cam(cam_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_cam;
    taskEXIT_CRITICAL();
}

/* ---- Camera LIVE real-frame view (Wave C) ----------------------------------
 * The HMI's lv_img renders SDRAM_CAM_VIEW; we keep all OV5640 framebuffer access
 * + D-cache management here (mirrors vision_task) so ui_screen.c stays MCU-header
 * free. mode 1 = true colour (verbatim RGB565), mode 2 = thermal LUT (luminance
 * -> blue/cyan/green/yellow/red). CPU writes the dst buffer that the CPU (LVGL)
 * later reads, so it is self-coherent; only the DMA-written source needs invalidate. */
const uint8_t *lab_camview_buf(void) { return (const uint8_t *)SDRAM_CAM_VIEW; }

static uint16_t _thermal_565(int y)
{
    int r, g, b;
    if (y < 0)   y = 0;
    if (y > 255) y = 255;
    if (y < 64)       { r = 0;             g = y * 4;            b = 255; }
    else if (y < 128) { r = 0;             g = 255;             b = 255 - (y - 64) * 4; }
    else if (y < 192) { r = (y - 128) * 4; g = 255;             b = 0; }
    else              { r = 255;           g = 255 - (y - 192) * 4; b = 0; }
    if (g > 255) g = 255;  if (g < 0) g = 0;
    return (uint16_t)((((unsigned)r >> 3) << 11) | (((unsigned)g >> 2) << 5) | ((unsigned)b >> 3));
}

void lab_camview_render(int mode)
{
    uint16_t *dst = (uint16_t *)SDRAM_CAM_VIEW;
    int i, n = (int)OV5640_QVGA_PIXELS;
    SCB_InvalidateDCache_by_Addr((uint32_t *)ov5640_framebuf, (int32_t)OV5640_QVGA_BYTES);
    if (mode == 2) {                 /* thermal false-colour */
        for (i = 0; i < n; i++) {
            uint16_t pix = ov5640_framebuf[i];
            int r = (int)((pix >> 11) & 0x1Fu) << 3;
            int g = (int)((pix >> 5)  & 0x3Fu) << 2;
            int b = (int)( pix        & 0x1Fu) << 3;
            dst[i] = _thermal_565((r * 77 + g * 150 + b * 29) >> 8);
        }
    } else {                         /* true colour (verbatim RGB565) */
        for (i = 0; i < n; i++) dst[i] = ov5640_framebuf[i];
    }
}

void lab_get_vib(vib_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_vib;
    taskEXIT_CRITICAL();
}

/* Run AI-6/7/8/9 once for a recipe preset and publish the pre-flight result.
 * Cheap (four tiny MLPs / one kNN sweep, all sub-ms); the recipe is immutable
 * so this is called once at boot and the result is cached for the HMI. */
static void recipe_ai_refresh(int preset)
{
    recipe_ai_t r;
    float lam = 0.0f, fwhm = 0.0f, pct = 0.0f, kwh = 0.0f, co2 = 0.0f, dist = 0.0f;
    int   band, nn, i;

    if (preset < 0 || preset >= N_PRESET) preset = 0;
    memset(&r, 0, sizeof(r));

    ai6_optical(&preset_desc[preset * PRESET_DESC], &lam, &fwhm);
    band = ai7_thermal(&preset_desc[preset * PRESET_DESC], &pct);
    ai8_energy(&preset_e5[preset * 5], &kwh, &co2);
    nn = ai9_retrieve(&preset_r18[preset * PRESET_R18], &dist);

    for (i = 0; i < 19 && preset_name[preset][i]; i++) r.recipe[i] = preset_name[preset][i];
    r.lambda_nm     = (int)(lam + 0.5f);
    r.fwhm_nm       = (int)(fwhm + 0.5f);
    r.thermal_pct   = (int)(pct + 0.5f);
    r.thermal_band  = (uint8_t)band;
    r.kwh_x10       = (int)(kwh * 10.0f + 0.5f);
    r.co2_x10       = (int)(co2 * 10.0f + 0.5f);
    r.analog_idx    = nn;
    {
        const char *nm = ai9_recipe_name(nn);
        for (i = 0; i < 19 && nm[i]; i++) r.analog_name[i] = nm[i];
    }
    r.analog_dist_x100 = (int)(dist * 100.0f + 0.5f);

    /* AI-11 phase-purity pre-flight prior (same 24-D descriptor as AI-6/7). Honest
     * edge triage (LOO ~70% vs 60% base); the deep compositional call is off-device. */
    {
        float p_pure = 0.0f;
        r.purity_cls = (uint8_t)ai11_purity(&preset_desc[preset * PRESET_DESC], &p_pure);
        r.p_pure_pct = (int)(p_pure * 100.0f + 0.5f);
    }

    /* Derived crystal-field read-out (NOT a model — exact algebra from AI-6's lambda_em:
     * 10*Dq = E_4T2 = 1e7/lambda_em + Stokes(1800); B = corpus prior ~650, near-constant). */
    {
        float dq = 0.0f, dqb;
        if (lam > 1.0f) dq = (1.0e7f / lam + 1800.0f) / 10.0f;
        r.dq_cm1 = (int)(dq + 0.5f);
        r.b_cm1  = 650;                          /* corpus prior (real B range 640-660) */
        dqb = dq / 650.0f;
        r.dq_over_b_x100 = (int)(dqb * 100.0f + 0.5f);
        r.field_class = (dqb > 2.60f) ? 3u : (dqb > 2.35f) ? 2u : (dqb > 2.05f) ? 1u : 0u;
    }
    r.valid = 1u;

    taskENTER_CRITICAL();
    s_recipe_ai = r;
    taskEXIT_CRITICAL();
}

/* -------------------------------------------------------------------------- */
/* AI-12 PL dopant classifier + AI-13 PL-QC autoencoder runtime (replayed).    */
/* The sentinel has no spectrometer; the PL screen replays stored real Fluoromax */
/* emission spectra (ai12_demo_spectra.h), cycling one per class, and runs both  */
/* PL models on each — same honest replay as furnace_sim. Decision-support QC.   */
/* -------------------------------------------------------------------------- */
static pl_view_t s_pl;

void lab_get_pl(pl_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_pl;
    taskEXIT_CRITICAL();
}

/* AI-14 furnace-temperature forecast snapshot (filled by ctrl_task each tick). */
static fc_view_t s_fc;

void lab_get_forecast(fc_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_fc;
    taskEXIT_CRITICAL();
}

/* AI-19 RUL/ETA + AI-20 thermocouple-integrity snapshot (filled by env_task, which
 * owns furnace_sim — the exact per-minute features both models were trained on). */
static ai_extra_view_t s_aix;
static float    s_rul_win[AI19_WIN];    /* last 24 t_current/1600 (AI-19 window)      */
static float    s_tc_meas[AI20_L];      /* last 12 measured/1600  (AI-20 window)      */
static float    s_tc_setp[AI20_L];      /* last 12 setpoint/1600  (AI-20 window)      */
static int      s_aix_fill;             /* minutes accumulated (window-fill counter)  */

/* AI-20 live demo: operator-triggerable thermocouple-fault injection (furnace_sim
 * itself only models PROCESS faults, never SENSOR faults, so without this the monitor
 * just reads healthy). 0 none / 1 open-circuit / 2 erratic. Applied non-destructively
 * to the classifier's input window only — the real rings/feature builders are untouched. */
static volatile int s_tc_inject = 0;
void lab_set_tc_inject(int mode) { s_tc_inject = (mode < 0 || mode > 2) ? 0 : mode; }
int  lab_get_tc_inject(void)     { return s_tc_inject; }

void lab_get_ai_extra(ai_extra_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_aix;
    taskEXIT_CRITICAL();
}

/* On-chip AI latency table (us), measured once at boot by the DWT probe (real M7
 * cycles / 600). Index order documented in ui_data.h (AI_LAT_* ); 0 = not measured.
 * The Benchmark HMI panel reads this so it shows genuine on-device numbers. */
static int g_ai_lat[AI_LAT_N];
int lab_get_ai_lat(int *out, int max)
{
    int i, n = (max < AI_LAT_N) ? max : AI_LAT_N;
    for (i = 0; i < n; i++) out[i] = g_ai_lat[i];
    return n;
}

/* ---- Reliability / Robustness live-injection state (the Robust HMI page) ----
 * vision_task perturbs a private copy of the AI-1 crucible-CNN input; pl_refresh
 * perturbs a private copy of the AI-12 spectrum; the thermocouple injection reuses
 * s_tc_inject (above, feeding AI-20). All non-destructive: the live pipeline and
 * the Camera/PL screens keep their CLEAN verdicts — only the robustness pair is
 * recomputed on the perturbed copy, demonstrating graceful degradation. */
static volatile int s_vis_inject = 0;   /* 0 clean/1 noise/2 dark/3 bright/4 occlusion */
static volatile int s_pl_inject  = 0;   /* 0 clean/1 noise/2 occlusion/3 baseline      */
void lab_set_vis_inject(int m) { s_vis_inject = (m < 0 || m > 4) ? 0 : m; }
int  lab_get_vis_inject(void)  { return s_vis_inject; }
void lab_set_pl_inject(int m)  { s_pl_inject  = (m < 0 || m > 3) ? 0 : m; }
int  lab_get_pl_inject(void)   { return s_pl_inject; }

static rob_view_t s_rob;
void lab_get_rob(rob_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_rob;
    taskEXIT_CRITICAL();
}

/* AI inferences served since boot + boot reset cause -> the long-run stability panel
 * (rubric "长时间稳定"): genuine liveness + "was any reset a fault?" evidence. */
static volatile uint32_t s_inf_count = 0;
static uint8_t s_reset_cause = 0;       /* 1 wdg-fault / 2 software / 3 power-on / 4 ext-pin */
void lab_get_health(health_view_t *out)
{
    if (out == NULL) return;
    out->uptime_s    = (uint32_t)(xTaskGetTickCount() / configTICK_RATE_HZ);
    out->inferences  = s_inf_count;
    out->reset_cause = s_reset_cause;
    out->wdg_armed   = 1u;   /* wdg_task arms the FWDGT once the safety tasks spawn */
}

static void pl_refresh(int demo_idx)
{
    pl_view_t v;
    float probs[3], mse = 0.0f;
    int i, cls, top;

    if (demo_idx < 0 || demo_idx >= 3) demo_idx = 0;
    memset(&v, 0, sizeof(v));
    for (i = 0; i < PL_SPEC_N && i < DEMO_SPEC_N; i++)
        v.spec[i] = demo_spec[demo_idx * DEMO_SPEC_N + i];

    cls = ai12_plclass(v.spec, probs);
    top = cls;
    v.cls = cls;
    v.conf_pct = (int)(probs[top] * 100.0f + 0.5f);
    for (i = 0; i < 3; i++) v.probs_pct[i] = (int)(probs[i] * 100.0f + 0.5f);

    v.anomaly  = (uint8_t)ai13_plqc(v.spec, NULL, &mse);
    v.mse_x1e5 = (int)(mse * 1.0e5f + 0.5f);
    v.qhat_x1e5 = (int)(ai13_plqc_qhat() * 1.0e5f + 0.5f);
    v.demo_idx = demo_idx;

    /* ---- AI-15 PL host-ID + AI-16 lambda_em (read the same replayed spectrum) ---- */
    {
        float hp[2];
        int hc = ai15_hostid(v.spec, hp);
        v.host_cls = hc;
        v.host_conf_pct = (int)(hp[hc] * 100.0f + 0.5f);
        v.lambda_nm = (int)(ai16_lambda(v.spec) + 0.5f);
    }

    /* ---- AI-17 PL few-shot NCM: lazily register the 3 replayed spectra as 3
     * "sample classes" the first time, then classify the current spectrum live.
     * Demonstrates on-device few-shot registration of a new phosphor sample type. */
    {
        static int s_pl_seeded = 0;
        float emb[AI17_EMB_DIM];
        int d;
        if (!s_pl_seeded) {
            float e2[AI17_EMB_DIM]; int j;
            ai17_pl_reset();
            for (d = 0; d < 3 && d < DEMO_SPEC_N; d++) {
                float sp[PL_SPEC_N];
                for (j = 0; j < PL_SPEC_N && j < DEMO_SPEC_N; j++)
                    sp[j] = demo_spec[d * DEMO_SPEC_N + j];
                ai12_embed(sp, e2);
                ai17_pl_add_sample(d, e2);
            }
            s_pl_seeded = 1;
        }
        ai12_embed(v.spec, emb);
        v.fewshot_cls = ai17_pl_classify(emb, NULL);
        v.fewshot_nclass = ai17_pl_num_classes();
    }
    v.valid = 1u;

    /* ---- robustness: AI-12 graceful-degradation pair (clean vs perturbed) ----
     * Record the clean verdict; if a spectral perturbation is selected, classify a
     * perturbed COPY (cheap, 64 floats) so the Robust page shows the confidence
     * degrade live. The clean v.cls drives the PL page (unchanged). */
    s_inf_count++;
    {
        int pinj = s_pl_inject;
        int pcls = cls, pconf = v.conf_pct;
        if (pinj != 0) {
            float sp[PL_SPEC_N], pp[3]; uint32_t seed = 0x2C0FFEE5u + (uint32_t)demo_idx * 7919u;
            int j;
            for (j = 0; j < PL_SPEC_N; j++) sp[j] = v.spec[j];
            rob_perturb_spec(sp, PL_SPEC_N, pinj, &seed);
            pcls  = ai12_plclass(sp, pp);
            pconf = (int)(pp[pcls] * 100.0f + 0.5f);
            s_inf_count++;
        }
        taskENTER_CRITICAL();
        s_rob.pl_inject     = (uint8_t)pinj;
        s_rob.pl_clean_cls  = cls;
        s_rob.pl_clean_conf = v.conf_pct;
        s_rob.pl_pert_cls   = pcls;
        s_rob.pl_pert_conf  = pconf;
        s_rob.pl_valid      = 1u;
        taskEXIT_CRITICAL();
    }

    taskENTER_CRITICAL();
    s_pl = v;
    taskEXIT_CRITICAL();
}

/* -------------------------------------------------------------------------- */
/* Windowed task-supervision watchdog (hardware FWDGT-backed)                 */
/* -------------------------------------------------------------------------- */
/* Functional-safety layer (IEC 61508-style external supervision). Each safety-
 * relevant task bumps its heartbeat slot once per loop. A high-priority
 * wdg_task verifies every slot advanced within its deadline and only then
 * reloads the free watchdog timer. If any task deadlocks, starves, or crashes,
 * the kicker stops reloading and the FWDGT hard-resets the MCU into a safe
 * (heater-off) cold boot — defence in depth behind the controller's own
 * software safety abort. Slots cover the closed safety loop:
 *   CTRL   - drives the heater (most critical actuator path)
 *   FUSION - produces the AI-4 risk verdict feeding the safety supervisor
 *   ENV    - runs AI-2/AI-3, the inputs to the fusion verdict
 *   SENSOR - acquires the vibration channel
 * (vision/ui/voice are cosmetic and have variable timing, so they are left
 *  out to avoid false trips.) */
enum { WDG_CTRL = 0, WDG_FUSION, WDG_ENV, WDG_SENSOR, WDG_N };
static volatile uint32_t s_wdg_hb[WDG_N];

/* UI/touch control hook (declared in lab_sentinel.h). Mirrors the voice
 * CI1302_CMD_START / CMD_STOP path so a touch button drives the same closed
 * loop AND the AI furnace sim (which gates env/fusion) — touch == voice. */
void lab_ctrl_request(int cmd)
{
    if (cmd == 1) {
        /* START = begin sintering ONLY: furnace sim + closed-loop ctrl +
         * ventilation fan. The stirring/vibration MOTOR is now a SEPARATE control
         * (lab_motor_request / MOTOR button) so AI-10 isn't always running. */
        furnace_sim_set_anomaly(FURN_ANOM_NONE);
        fb_reset();
        furnace_sim_start();
        s_batch_fan = 1u; relay_on();        /* ventilation fan ON for the batch */
        s_ctrl_cmd = CTRL_CMD_START;
    } else if (cmd == 2) {
        furnace_sim_stop();
        s_batch_fan = 0u; relay_off();        /* ventilation fan OFF */
        heater_off();                         /* real PTC heater OFF immediately (safe) */
        s_ctrl_cmd = CTRL_CMD_ABORT;
    }
}

/* Motor control, SEPARATE from the sinter START/STOP. Drives the L298N stirring/
 * vibration motor that AI-10 does PdM on — on its own MOTOR button / voice so the
 * vibration demo runs only when wanted (not every batch). */
void lab_motor_request(int on)
{
    if (on) motor_set(MOTOR_FORWARD);
    else    motor_set(MOTOR_STOP);
}

/* UART log for other modules (forwards to the mutex-guarded boot_print). */
void lab_log(const char *s)
{
    boot_print(s);
}

/* Verbosity gate. When 0 (default), high-frequency task chatter (vision frames,
 * env/fusion polls, ctrl periodic status) is suppressed so the touch/UI/ctrl-
 * event lines stay readable during interactive debugging. Boot, touch, button,
 * and batch START/DONE/FAULT lines always print. Set to 1 to restore full log. */
static volatile uint8_t s_log_verbose = 0U;
static void vbprint(const char *s) { if (s_log_verbose) boot_print(s); }

/* -------------------------------------------------------------------------- */
/* HMI data layer (ui_data.h): controller snapshot + SPC soak trend ring.     */
/* ctrl_task fills these (additive, read-only of its own state); ui_task reads.*/
/* -------------------------------------------------------------------------- */
#define SPC_RING_N 120
static ctrl_snapshot_t s_ui_ctrl;
static int16_t         s_spc_ring[SPC_RING_N];
static int             s_spc_head = 0;   /* next write index */
static int             s_spc_n    = 0;   /* valid samples 0..SPC_RING_N */

/* MAX31855 real-thermocouple channel (defined here so the ctrl snapshots can
 * mirror it; the driver itself is below, before lab_sentinel_main). Polled by
 * ctrl_task; its fault bit feeds the safety supervisor. */
static int     s_tc_c       = 0;   /* whole degC */
static uint8_t s_tc_fault   = 0u;  /* bit0 OC / bit1 SCG / bit2 SCV (0 healthy) */
static uint8_t s_tc_present = 0u;  /* 1 = a plausible frame (device wired)      */

void lab_ctrl_get(ctrl_snapshot_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_ui_ctrl;
    taskEXIT_CRITICAL();
}

int lab_spc_series(int16_t *out, int max)
{
    int n, i, idx;
    if (out == NULL || max <= 0) return 0;
    taskENTER_CRITICAL();
    n = (s_spc_n < max) ? s_spc_n : max;
    for (i = 0; i < n; i++) {
        idx = (s_spc_head - n + i + 2 * SPC_RING_N) % SPC_RING_N;
        out[i] = s_spc_ring[idx];
    }
    taskEXIT_CRITICAL();
    return n;
}

void lab_spc_limits(int *lsl, int *usl, int *cl)
{
    if (lsl) *lsl = -10;   /* SP +-10C spec window (matches spc_init in ctrl_task) */
    if (usl) *usl = 10;
    if (cl)  *cl  = 0;
}

static void spc_ring_push(int16_t dev)
{
    taskENTER_CRITICAL();
    s_spc_ring[s_spc_head] = dev;
    s_spc_head = (s_spc_head + 1) % SPC_RING_N;
    if (s_spc_n < SPC_RING_N) s_spc_n++;
    taskEXIT_CRITICAL();
}

/* Build + commit the controller snapshot. spc may be NULL (idle/no soak yet). */
static void ctrl_snap_commit(const furnace_ctrl_t *c, float meas, spc_t *spc,
                             int elem_pct, uint32_t batch_id, uint32_t total_s)
{
    ctrl_snapshot_t s;
    int i;

    s.state   = (uint8_t)c->state;
    s.fault   = (uint8_t)c->fault;
    s.seg_idx = (uint8_t)c->seg_idx;
    s.risk    = s_eth_risk;
    for (i = 0; i < 15 && c->recipe->seg[c->seg_idx].label[i]; i++)
        s.seg_label[i] = c->recipe->seg[c->seg_idx].label[i];
    s.seg_label[i] = '\0';
    s.sp_c      = (int)c->sp_C;
    s.meas_c    = (int)meas;
    s.u_pct     = (int)(c->u * 100.0f);
    s.elapsed_s = (uint32_t)c->t_elapsed_s;
    s.total_s   = total_s;
    s.batch_id  = batch_id;
    s.elem_pct  = elem_pct;
    s.probe_c   = s_tc_c;        /* MAX31855 real thermocouple channel */
    s.tc_fault  = s_tc_fault;
    s.tc_present= s_tc_present;

    if (spc != NULL) {
        spc_result_t sr;
        spc_finalize(spc, &sr);          /* non-destructive: live stats */
        if (sr.n >= 5) {
            s.cpk_x100       = (int)(sr.cpk   * 100.0f);
            s.spc_mean_x100  = (int)(sr.mean  * 100.0f);
            s.spc_sigma_x100 = (int)(sr.sigma * 100.0f);
            s.spc_in_control = (uint8_t)sr.in_control;
            s.spc_alarms     = (uint16_t)((sr.n_alarms > 65535) ? 65535 : sr.n_alarms);
        } else {
            s.cpk_x100 = 0; s.spc_mean_x100 = 0; s.spc_sigma_x100 = 0;
            s.spc_in_control = 0; s.spc_alarms = 0;
        }
    } else {
        s.cpk_x100 = 0; s.spc_mean_x100 = 0; s.spc_sigma_x100 = 0;
        s.spc_in_control = 0; s.spc_alarms = 0;
    }

    taskENTER_CRITICAL();
    s_ui_ctrl = s;
    taskEXIT_CRITICAL();
}

/* -------------------------------------------------------------------------- */
/* HMI data layer: ESP32-C3 cloud-uplink status (Control+Cloud screen, tab 3). */
/* env_task (1 Hz) pushes a telemetry block to the ESP32-C3 I2C slave and reads */
/* back the link state + last R1 diagnosis; this snapshot is the UI read view.  */
/* Zero-init = offline, so the screen shows "OFFLINE" until env_task connects.  */
/* -------------------------------------------------------------------------- */
static cloud_view_t s_ui_cloud;

void lab_get_cloud(cloud_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_ui_cloud;
    taskEXIT_CRITICAL();
}

/* -------------------------------------------------------------------------- */
/* HMI data layer: edge nano-LM diagnosis + online-learning risk head.        */
/* nlm_task (0.5 Hz) builds a 12-slot context from the live state, GENERATES a */
/* Chinese diagnosis with the on-chip INT8 GPT, and runs the online head; the  */
/* HMI reads these snapshots. lab_online_teach()/lab_nlm_request() are the      */
/* operator hooks (one on-chip SGD step / force a regeneration).               */
/* -------------------------------------------------------------------------- */
static nlm_view_t    s_ui_nlm;
static online_view_t s_ui_online;
static volatile int  s_online_teach_req = -1;   /* HMI: true risk to learn; -1 none */
static volatile int  s_nlm_req = 0;             /* HMI: force a regeneration         */
/* Active generative LM: 0 = internal x1p9 (1.8M, ai_nanolm); 1..N = SPI-flash bank
 * model (s0p6/m1p35, ai_lm_bank). The HMI "SWITCH LM" button cycles it to demo the
 * hardware-ceiling curve live; nlm_task swap-loads the bank model on switch. */
static volatile int  s_lm_active = 0;
static volatile int  s_lm_cycle_req = 0;

void lab_get_nlm(nlm_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_ui_nlm;
    taskEXIT_CRITICAL();
}

void lab_get_online(online_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_ui_online;
    taskEXIT_CRITICAL();
}

void lab_online_teach(int true_risk)
{
    if (true_risk >= 0 && true_risk < 4) s_online_teach_req = true_risk;
}

void lab_nlm_request(void) { s_nlm_req = 1; }

/* HMI: cycle the active generative LM (x1p9 -> m1p35 -> s0p6 -> ...). */
void lab_lm_cycle(void)   { s_lm_cycle_req = 1; }
int  lab_lm_active(void)  { return s_lm_active; }

/* ---- Edge LLM CLUSTER (5 swap-loaded experts) HMI data layer -------------- */
static cluster_view_t s_ui_cluster;
static volatile int   s_cluster_next_req = 0;   /* HMI: cycle to next expert    */

void lab_get_cluster(cluster_view_t *out)
{
    if (out == NULL) return;
    taskENTER_CRITICAL();
    *out = s_ui_cluster;
    taskEXIT_CRITICAL();
}

void lab_cluster_next(void) { s_cluster_next_req = 1; }

/* -------------------------------------------------------------------------- */
/* HMI data layer: recipe profile view (Recipe screen).                       */
/* The recipe is immutable const data, so this just translates RECIPE_GARNET  */
/* into UI-friendly integers (no %f, no float exposure to the LVGL layer).    */
/* -------------------------------------------------------------------------- */
int lab_recipe_segs(recipe_seg_view_t *out, int max)
{
    const recipe_t *r = &RECIPE_GARNET;
    int n, i, j;
    if (out == NULL || max <= 0) return 0;
    n = (r->n_seg < max) ? r->n_seg : max;
    for (i = 0; i < n; i++) {
        const recipe_seg_t *sg = &r->seg[i];
        out[i].kind       = (uint8_t)sg->kind;
        out[i].target_c   = (int)sg->target_C;
        out[i].rate_c_min = (int)sg->rate_C_min;
        out[i].dur_min    = (int)sg->dur_min;
        for (j = 0; j < 19 && sg->label[j]; j++) out[i].label[j] = sg->label[j];
        out[i].label[j] = '\0';
    }
    return n;
}

const char *lab_recipe_name(void)   { return RECIPE_GARNET.name; }
int         lab_recipe_total_min(void) { return (int)(recipe_total_s(&RECIPE_GARNET) / 60.0f); }

/* -------------------------------------------------------------------------- */
/* HMI data layer: batch ledger (Quality screen).                             */
/* ctrl_task retains the last LEDGER_N sealed records in a chronological ring  */
/* so the UI can show batch history + re-verify the SHA-256 chain live. Each   */
/* record already chains into the previous one's hash (batch_record_seal);     */
/* the windowed verify checks recs[0] internal integrity then adjacent links.  */
/* -------------------------------------------------------------------------- */
#define LEDGER_N 8
static batch_record_t s_ledger[LEDGER_N];
static int            s_ledger_head = 0;   /* next write slot                 */
static int            s_ledger_n    = 0;   /* valid records 0..LEDGER_N       */
static int            s_ledger_total = 0;  /* lifetime sealed count           */

/* Push a freshly sealed record into the retained ring (ctrl_task only). */
static void ledger_push(const batch_record_t *rec)
{
    taskENTER_CRITICAL();
    s_ledger[s_ledger_head] = *rec;
    s_ledger_head = (s_ledger_head + 1) % LEDGER_N;
    if (s_ledger_n < LEDGER_N) s_ledger_n++;
    s_ledger_total++;
    taskEXIT_CRITICAL();
}

int lab_ledger_total(void) { return s_ledger_total; }

int lab_ledger_get(batch_view_t *out, int max)
{
    int n, i, idx;
    char hx[65];
    if (out == NULL || max <= 0) return 0;
    taskENTER_CRITICAL();
    n = (s_ledger_n < max) ? s_ledger_n : max;
    for (i = 0; i < n; i++) {
        /* newest first: walk back from the most recently written slot */
        const batch_record_t *r;
        idx = (s_ledger_head - 1 - i + 2 * LEDGER_N) % LEDGER_N;
        r = &s_ledger[idx];
        out[i].batch_id    = r->batch_id;
        out[i].peak_c      = (int)r->peak_C;
        out[i].cpk_x100    = (int)(r->soak_cpk * 100.0f);
        out[i].elem_pct    = (int)r->elem_remaining_pct;
        out[i].ai_alarms   = r->n_ai_alarms;
        out[i].in_control  = r->in_control;
        out[i].capable     = r->capable;
        out[i].final_state = r->final_state;
        out[i].fault       = r->fault;
        sha256_hex(r->this_hash, hx);
        memcpy(out[i].hash12, hx, 12);
        out[i].hash12[12] = '\0';
        out[i].hash12[13] = '\0';
    }
    taskEXIT_CRITICAL();
    return n;
}

int lab_ledger_chain_ok(void)
{
    static batch_record_t snap[LEDGER_N];   /* ui_task-only; SHA done lock-free */
    int n, i, oldest, ok = 1;

    /* snapshot the ring chronologically (oldest..newest), lock held briefly */
    taskENTER_CRITICAL();
    n = s_ledger_n;
    if (n == 0) { taskEXIT_CRITICAL(); return -1; }
    oldest = (s_ledger_head - n + 2 * LEDGER_N) % LEDGER_N;
    for (i = 0; i < n; i++) snap[i] = s_ledger[(oldest + i) % LEDGER_N];
    taskEXIT_CRITICAL();

    /* verify outside the critical section (SHA-256 per record) */
    if (!batch_record_verify(&snap[0], snap[0].prev_hash)) ok = 0;
    for (i = 1; ok && i < n; i++)
        if (!batch_record_verify(&snap[i], snap[i - 1].this_hash)) ok = 0;
    return ok;
}

/* Idle snapshot (no active batch / controller not yet initialised). */
static void ctrl_snap_idle(uint32_t batch_id)
{
    ctrl_snapshot_t s;
    int i;
    const char *lbl = "idle";
    s.state = 0U; s.fault = 0U; s.seg_idx = 0U; s.risk = s_eth_risk;
    for (i = 0; i < 4; i++) s.seg_label[i] = lbl[i];
    s.seg_label[4] = '\0';
    s.sp_c = 25; s.meas_c = 25; s.u_pct = 0;
    s.elapsed_s = 0U; s.total_s = 0U; s.batch_id = batch_id;
    s.cpk_x100 = 0; s.spc_mean_x100 = 0; s.spc_sigma_x100 = 0;
    s.spc_in_control = 0; s.spc_alarms = 0; s.elem_pct = 100;
    s.probe_c = s_tc_c; s.tc_fault = s_tc_fault; s.tc_present = s_tc_present;
    taskENTER_CRITICAL();
    s_ui_ctrl = s;
    taskEXIT_CRITICAL();
}

/* ========================================================================== */
/* MAX31855 K-type thermocouple — bit-bang SPI (read-only), real furnace PV.   */
/*                                                                            */
/* 模块(2).docx wiring:  CLK = PB10 (SPI1_SCK)   SO = PC2 (SPI1_MISO)   CS = PG3 */
/*                                                                            */
/* ⚠ PC2 was MQ-135's ADC0 input — a thermocouple SO line and an analog gas    */
/*   sensor CANNOT share PC2. Per the new wiring doc PC2 now carries MAX31855   */
/*   SO; if MQ-135 is still wired there, move it (the gas reading is advisory). */
/*   MAX_SO_* below is a single #define so the SO pin is trivial to relocate.   */
/*                                                                            */
/* Read-only 32-bit frame, SPI mode 0, polled ~10 Hz from ctrl_task. A software */
/* bit-bang (~1 MHz, max 5 MHz) is ample and avoids claiming the SPI1 block.    */
/* Frame layout (MAX31855 datasheet):                                          */
/*   [31:18] 14-bit signed TC temp, 0.25C/LSB   [16] global fault              */
/*   [15:4]  12-bit signed internal temp        [2] SCV [1] SCG [0] OC         */
/* ========================================================================== */
#define MAX_CLK_PORT GPIOB
#define MAX_CLK_PIN  GPIO_PIN_10
#define MAX_SO_PORT  GPIOC
#define MAX_SO_PIN   GPIO_PIN_2
#define MAX_CS_PORT  GPIOG
#define MAX_CS_PIN   GPIO_PIN_3

/* 2026-06-07 ✅ CS repointed PF8 -> PG3 (模块(2).docx update; user confirmed the CS
 * dupont wire is now on PG3). This frees PF8 to be the cluster SPI-flash MISO (SPI4,
 * board-fixed) exclusively, and brings the thermocouple back on. PG3 was LED2 —
 * retired in LED.{c,h} (its status is redundant with LED1 + the LCD risk bar) so
 * max31855_init owns PG3 alone. CLK=PB10 / SO=PC2 unchanged. With ENABLED 1 the
 * channel is polled every control step; a not-yet-wired module reads present=0 and
 * the loop safely stays on the FOPDT sim furnace (no false fault — see ctrl_task). */
#define MAX31855_ENABLED 0
#define MAX31855_DIAG_VERBOSE 0
#define CI1302_VOICE_ENABLED 0

/* (s_tc_c / s_tc_fault / s_tc_present are declared up by the SPC ring so the
 *  ctrl snapshots, defined earlier, can mirror them — see above.)            */

#if MAX31855_ENABLED
static void max31855_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOB);   /* CLK = PB10 */
    rcu_periph_clock_enable(RCU_GPIOC);   /* SO  = PC2  */
    rcu_periph_clock_enable(RCU_GPIOG);   /* CS  = PG3  (was PF8; freed PF8 for SPI4 flash MISO) */
    gpio_mode_set(MAX_CLK_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, MAX_CLK_PIN);
    gpio_output_options_set(MAX_CLK_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, MAX_CLK_PIN);
    gpio_mode_set(MAX_CS_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, MAX_CS_PIN);
    gpio_output_options_set(MAX_CS_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, MAX_CS_PIN);
    gpio_mode_set(MAX_SO_PORT, GPIO_MODE_INPUT, GPIO_PUPD_PULLUP, MAX_SO_PIN);
    gpio_bit_reset(MAX_CLK_PORT, MAX_CLK_PIN);   /* SCK idle low (mode 0) */
    gpio_bit_set(MAX_CS_PORT, MAX_CS_PIN);       /* CS idle high (deselect) */
}

/* Bit-bang SPI half-period. 30 NOPs gave ~1.5 MHz which is unreliable on long
 * unshielded dupont wiring (corrupted bits -> impossible/jumping temperatures,
 * present=1 fault=0 but garbage). 300 NOPs -> ~150 kHz, rock-solid on dupont,
 * still only tens of microseconds per 32-bit read (negligible). */
static void _max_dly(void) { volatile int i = 300; while (i--) { __NOP(); } }

static uint32_t max31855_read_raw(void)
{
    uint32_t v = 0u;
    int i;
    gpio_bit_reset(MAX_CS_PORT, MAX_CS_PIN);     /* select -> D31 presented */
    _max_dly();
    for (i = 0; i < 32; i++) {
        gpio_bit_reset(MAX_CLK_PORT, MAX_CLK_PIN);   /* SO valid while SCK low */
        _max_dly();
        v <<= 1;
        if (gpio_input_bit_get(MAX_SO_PORT, MAX_SO_PIN) != RESET) v |= 1u;
        gpio_bit_set(MAX_CLK_PORT, MAX_CLK_PIN);     /* clock out the next bit */
        _max_dly();
    }
    gpio_bit_set(MAX_CS_PORT, MAX_CS_PIN);       /* deselect */
    return v;
}

/* Decode the 32-bit frame -> thermocouple temp in centi-degC (×100).
 * *fault gets OC/SCG/SCV bits (0 healthy); *present=0 if the bus floats
 * all-ones / all-zeros (no device). Pure integer math (host-golden tested). */
static int max31855_decode(uint32_t v, uint8_t *fault, uint8_t *present)
{
    int32_t t14;
    uint8_t f = 0u;
    if (present) *present = (v != 0xFFFFFFFFu && v != 0x00000000u) ? 1u : 0u;
    if (v & 0x00010000u) {                /* global fault bit set */
        f = (uint8_t)(v & 0x7u);          /* OC | SCG | SCV */
        if (f == 0u) f = 0x8u;            /* flagged but no specific bit */
    }
    if (fault) *fault = f;
    t14 = (int32_t)(v >> 18);             /* 14-bit signed */
    if (t14 & 0x2000) t14 -= 0x4000;      /* sign-extend */
    return (int)(t14 * 25);               /* 0.25C/LSB ×100 = ×25 */
}

#if MAX31855_DIAG_VERBOSE
/* Extra UART diagnostics for the bench bring-up build. This deliberately stays
 * inside lab_sentinel.c because the MAX31855 driver is embedded in this module. */
static int max31855_decode_internal(uint32_t v)
{
    int32_t t12 = (int32_t)((v >> 4) & 0x0FFFu);
    if (t12 & 0x0800) t12 -= 0x1000;
    return (int)((t12 * 625) / 100);      /* 0.0625C/LSB -> centi-degC */
}

static unsigned max_pin_level(uint32_t port, uint32_t pin)
{
    return (gpio_input_bit_get(port, pin) != RESET) ? 1u : 0u;
}

static void max31855_diag_print(const char *tag, int samples)
{
    uint32_t first = 0u, raw = 0u;
    int i, all_ff = 1, all_00 = 1, changed = 0, present_ok = 0, fault_seen = 0;
    unsigned cs_idle, clk_idle, so_idle, cs_sel, so_sel;
    char b[176];

    gpio_bit_set(MAX_CS_PORT, MAX_CS_PIN);
    gpio_bit_reset(MAX_CLK_PORT, MAX_CLK_PIN);
    _max_dly();
    cs_idle = max_pin_level(MAX_CS_PORT, MAX_CS_PIN);
    clk_idle = max_pin_level(MAX_CLK_PORT, MAX_CLK_PIN);
    so_idle = max_pin_level(MAX_SO_PORT, MAX_SO_PIN);
    gpio_bit_reset(MAX_CS_PORT, MAX_CS_PIN);
    _max_dly();
    cs_sel = max_pin_level(MAX_CS_PORT, MAX_CS_PIN);
    so_sel = max_pin_level(MAX_SO_PORT, MAX_SO_PIN);
    gpio_bit_set(MAX_CS_PORT, MAX_CS_PIN);

    {
        int n = snprintf(b, sizeof(b),
            "[maxdiag] %s pins idle: CS=%u CLK=%u SO=%u; selected: CS=%u SO=%u (PB10/PC2/PG3)\r\n",
            tag, cs_idle, clk_idle, so_idle, cs_sel, so_sel);
        if (n > 0) boot_print(b);
    }

    if (samples <= 0) samples = 1;
    for (i = 0; i < samples; i++) {
        uint8_t f = 0u, p = 0u;
        int tc_x100, cj_x100;
        raw = max31855_read_raw();
        if (i == 0) first = raw;
        if (raw != first) changed = 1;
        if (raw != 0xFFFFFFFFu) all_ff = 0;
        if (raw != 0x00000000u) all_00 = 0;
        tc_x100 = max31855_decode(raw, &f, &p);
        cj_x100 = max31855_decode_internal(raw);
        if (p) present_ok++;
        if (f) fault_seen++;
        {
            int n = snprintf(b, sizeof(b),
                "[maxdiag] %s read[%02d] raw=0x%08lX tc_x100=%d cj_x100=%d present=%u fault=0x%X\r\n",
                tag, i, (unsigned long)raw, tc_x100, cj_x100, (unsigned)p, (unsigned)f);
            if (n > 0) boot_print(b);
        }
        _max_dly();
    }

    {
        const char *verdict = "UNKNOWN";
        if (all_ff) {
            verdict = "ALL_FF_SO_HIGH_OR_FLOAT_CHECK_VCC_GND_SO_PC2_CS_PG3";
        } else if (all_00) {
            verdict = "ALL_00_SO_LOW_OR_UNPOWERED_CHECK_VCC_GND_SO_PC2";
        } else if (changed) {
            verdict = "RAW_UNSTABLE_CHECK_DUPONT_LENGTH_GND_CLK_PB10_SO_PC2";
        } else if (present_ok > 0 && fault_seen > 0) {
            verdict = "DEVICE_SEEN_TC_FAULT_CHECK_T_PLUS_T_MINUS_PROBE";
        } else if (present_ok > 0) {
            verdict = "DEVICE_SEEN_NO_FAULT";
        }
        {
            int n = snprintf(b, sizeof(b),
                "[maxdiag] %s verdict=%s samples=%d present_ok=%d fault_seen=%d changed=%d\r\n",
                tag, verdict, samples, present_ok, fault_seen, changed);
            if (n > 0) boot_print(b);
        }
    }
}
#endif /* MAX31855_DIAG_VERBOSE */

/* Read once -> whole degC; refresh the shared s_tc_* channel. */
static uint32_t s_tc_raw = 0u;   /* last raw 32-bit frame (signal-integrity diag) */
static void ctrl_poll_tc(void)
{
    uint8_t f = 0u, p = 0u;
    uint32_t raw = max31855_read_raw();
    int centi = max31855_decode(raw, &f, &p);
    s_tc_raw     = raw;
    s_tc_c       = centi / 100;
    s_tc_fault   = f;
    s_tc_present = p;
}
#endif /* MAX31855_ENABLED */

/* -------------------------------------------------------------------------- */
/* Public entry                                                               */
/* -------------------------------------------------------------------------- */
void lab_sentinel_main(void)
{
    /* IPC primitives must be created BEFORE any task that touches them runs. */
    xQueue_VisionResult = xQueueCreate(Q_DEPTH_VISION, sizeof(vision_result_t));
    xQueue_VibResult    = xQueueCreate(Q_DEPTH_VIB,    sizeof(vibration_result_t));
    xQueue_EnvResult    = xQueueCreate(Q_DEPTH_ENV,    sizeof(env_result_t));
    xQueue_RiskAlert    = xQueueCreate(Q_DEPTH_RISK,   sizeof(risk_alert_t));
    xSem_VoiceCmd       = xSemaphoreCreateBinary();
    xMutex_UART         = xSemaphoreCreateMutex();

    /* Boot task creates the rest then deletes itself. */
    xTaskCreate(task_init, "init", STACK_INIT, NULL, PRIO_BOOT, NULL);

    vTaskStartScheduler();

    /* Should never return. If it does, the scheduler ran out of heap. */
    for (;;) { }
}

/* -------------------------------------------------------------------------- */
/* Boot task - hardware init + spawn workers                                  */
/* -------------------------------------------------------------------------- */
static void task_init(void *pv)
{
    (void)pv;

    /* NVIC priority grouping: 4 bits preempt, 0 sub. Recommended >= 5.       */
    nvic_priority_group_set(NVIC_PRIGROUP_PRE4_SUB0);

    /* UART4 boot_print 先 ready, 这样后面 SDRAM 失败也能输出诊断信息.
     * UART4 在 PB13/PB5, 跟 SDRAM 完全不在同一个 GPIO bank, 不会干扰.     */
    LED_Init();
    my_usart_init();

    boot_print("[boot] task_init entered\r\n");

    /* Report what caused the last reset, then clear the latches. A FWDGT reset
     * here means the task-supervision watchdog tripped on the previous run
     * (a hung task) and recovered the system — surface it instead of booting
     * silently, so a field watchdog event is visible in the log. */
    if (rcu_flag_get(RCU_FLAG_FWDGTRST) != RESET) {
        boot_print("[boot] *** recovered from WATCHDOG reset (a task had hung) ***\r\n");
        s_reset_cause = 1u;   /* watchdog fault (surfaced on the Robust long-run panel) */
    } else if (rcu_flag_get(RCU_FLAG_SWRST) != RESET) {
        boot_print("[boot] reset cause: software\r\n");
        s_reset_cause = 2u;
    } else if (rcu_flag_get(RCU_FLAG_PORRST) != RESET) {
        boot_print("[boot] reset cause: power-on\r\n");
        s_reset_cause = 3u;
    } else if (rcu_flag_get(RCU_FLAG_EPRST) != RESET) {
        boot_print("[boot] reset cause: external pin\r\n");
        s_reset_cause = 4u;
    }
    rcu_all_reset_flag_clear();

    /* 2026-05-28: 切到 CIMC 官方主板 + RGB 4.3" 800×480 LCD (FPC, TLI 硬件刷新).
     * 旧 8080 NT35510 bring-up 全部作废:
     *   - PA15 WR-line diag block (旧 WR strobe, 新屏 PA15=B6 AF14, 改 GPIO output
     *     直接覆盖 LCD 数据位)
     *   - PB0 LCD RST pre-reset (旧 NT35510 controller reset, RGB 面板无 controller,
     *     无 RST pin)
     * 新 LCD 走 rgb_lcd_init() (在 ui_task 里调), 此处不再做任何 LCD 预初始化. */
    boot_print("[boot] LCD switched to RGB TLI 800x480 (CIMC official panel)\r\n");

    /* SDRAM init — 必须在 ci1302_init (PH13/PH14) 之前, 给 SDRAM 控制脚
     * PH2/3/5 一个最安静的 GPIOH bank 环境完成 JEDEC init 序列.
     * 2026-05-17 排查: 即使 sdram_init 在前, 之后 ci1302_init 配置 PH13 为
     * OSPEED_100_220MHZ 的 push-pull 还是会通过 GPIOH bank 持续 push 大电流
     * 干扰 SDRAM 控制脚的稳态保持 → 花屏. 修法在 ci1302.c 把 PH13 降到
     * OSPEED_12MHZ (已改). */
    sdram_init();
    boot_print("[boot] sdram_init returned\r\n");

    /* SDRAM smoke test: 写若干 magic, 立即读回. 不一致 → SDRAM 没正常工作,
     * 后面再 init 啥都白搭. 即使被 D-Cache 命中, 写 16MB+ 远超 cache 容量,
     * 早期 word 已被 evict 到 SDRAM 实体, 后续读回会真去 SDRAM 取. */
    {
        volatile uint32_t *p = (volatile uint32_t *)0xC0000000U;
        uint32_t fail = 0U;
        uint32_t i;
        /* 测 256 KB / 64K words, 远大于 D-Cache 16KB, 保证测到 SDRAM 实体 */
        for (i = 0U; i < 65536U; i++) {
            p[i] = (uint32_t)(0xA5A50000U ^ i);
        }
        SCB_CleanInvalidateDCache();   /* 把 cache 全部回写 + 清空,
                                          强制下一次读真去 SDRAM */
        for (i = 0U; i < 65536U; i++) {
            if (p[i] != (uint32_t)(0xA5A50000U ^ i)) {
                fail++;
            }
        }
        if (fail == 0U) {
            boot_print("[boot] SDRAM smoke test PASS (256KB OK)\r\n");
        } else {
            char buf[64];
            int n = snprintf(buf, sizeof(buf),
                             "[boot] SDRAM smoke test FAIL: %u/65536\r\n",
                             (unsigned)fail);
            if (n > 0) boot_print(buf);
        }
    }

    /* AI activation scratch probe @ SDRAM_AI_SCRATCH (0xC0400000, +4MB).
     * AI-3's ~100 KB of activations live here. Verify it's reachable BEFORE
     * the AI self-test runs there — a bus fault on an unmapped address would
     * otherwise look like a silent crash right after the OV5640 line. */
    {
        volatile uint32_t *q = (volatile uint32_t *)SDRAM_AI_SCRATCH;
        uint32_t i, fail = 0U;
        for (i = 0U; i < 4096U; i++) q[i] = (uint32_t)(0x5A5A0000U ^ i);
        SCB_CleanInvalidateDCache();
        for (i = 0U; i < 4096U; i++) if (q[i] != (uint32_t)(0x5A5A0000U ^ i)) fail++;
        if (fail == 0U) {
            boot_print("[boot] SDRAM AI-scratch probe PASS (0xC0400000)\r\n");
        } else {
            char buf[64];
            int n = snprintf(buf, sizeof(buf),
                             "[boot] SDRAM AI-scratch probe FAIL: %u/4096\r\n",
                             (unsigned)fail);
            if (n > 0) boot_print(buf);
        }
    }

#if CI1302_VOICE_ENABLED
    /* 给 SDRAM 一段稳定时间, 再启动 CI1302 (它会在 PC10/PC11 上活动).      */
    vTaskDelay(pdMS_TO_TICKS(50U));

    /* CI1302 UART3 — 2026-05-28 迁到 PC10/PC11 (per 模块(2).docx 新接线).
     * PC bank 跟 SDRAM 控制脚 PH2/3/5 不同 bank, 不再有 bounce 干扰风险.
     * 旧 PH13/PH14 隔离实验的理由 (2026-05-17) 已不适用, 恢复启用. */
    ci1302_init();
    boot_print("[boot] ci1302_init OK (UART3 PC10/PC11)\r\n");
#endif

    /* 2026-05-17 热复位花屏新假说: SDRAM LVGL pool 残留 TLSF state, lv_mem_init
     * 可能没完全清干净 → widget allocate 返回错位地址 → render 错位 → LCD 半花.
     * 修法: 在 sdram_init 之后强制 zero-fill 整个 LVGL 用到的 SDRAM 区域.
     *
     * 2026-05-22 升级 480×800 NT35510: 区域扩大
     *   fb1   [0xC0000000 .. 0xC00BB7FF]  framebuffer (768,000 B)
     *   pool  [0xC00C0000 .. 0xC01BFFFF]  LVGL widget heap (1,048,576 B)
     * 总共 0xC0000000 .. 0xC01C0000 = 1792 KB.
     * 2026-05-23: pool 从 256KB 扩到 1MB (256KB 装不下 480×800 widget tree
     * + 18/28pt font glyph cache → 全白屏). 同步把 zero-fill 范围从 1024KB
     * 扩到 1792KB. 后面 0xC01C0000 .. 0xC0200000 还有 256KB 安全 margin. */
    {
        volatile uint32_t *p = (volatile uint32_t *)0xC0000000U;
        uint32_t n_words = 0x1C0000U / 4U;  /* 1792 KB / 4 = 458752 words   */
        uint32_t i;
        for (i = 0U; i < n_words; i++) {
            p[i] = 0U;
        }
        SCB_CleanInvalidateDCache();   /* 强制 cache writeback + invalidate,
                                          保证 EXMC 读 SDRAM 拿到 0          */
        boot_print("[boot] SDRAM LVGL region zero-filled (1792KB)\r\n");
    }

    /* Risk LEDs (LED1=PE2 / LED2=PG3 / LED3=PH7 — 均不撞 SDRAM/RGB LCD).
     * 2026-05-29: 蜂鸣器(旧 PG2=SDRAM A12) + 震动(旧 PA3=RGB B5) 已移除, 不再
     * 配置/翻转这两脚 → SDRAM A12 + RGB 蓝色 B5 完整. LED 自检 ~750ms. */
    actuator_init();
    actuator_self_test();

    /* Sensor I2C bus = software bit-bang on PH7 (SCL) / PH8 (SDA) — see
     * sensors_i2c.c (the old PC10/PC11 note was stale; ci1302 owns PC10/PC11).
     * SHT30 + ADXL345 share this bus (module 4.7k pull-ups to 3V3).
     * MQ-135 is analog on PC2 (ADC0 ch12). Each sensor's _init() probes its
     * device, so wiring problems show up at boot. */
    sensors_i2c_init();
    sht30_init();
    {   /* 2026-06-02: print the ADXL345 init result so an I2C/wiring fault is
         * visible (acRMS stuck at 0 = reads failing). rc: 0 OK / 1 I2C no-ACK /
         * 2 wrong DEVID (addr or wrong chip) / 3 config-write fail. */
        uint8_t adxl_rc = adxl345_init();
        char ab[64]; int an;
        an = snprintf(ab, sizeof(ab),
            "[sensor] ADXL345 init rc=%u (0=OK 1=noACK 2=badID 3=cfg) PH7/PH8\r\n",
            (unsigned)adxl_rc);
        if (an > 0) boot_print(ab);
    }
    /* The MQ-135 (= the 烟雾气敏传感器) moved 2026-06-03 per 模块(2).docx: it now
     * wires AO=PG13 / DO=PC3, NOT the old PC2/ADC0. PC2 is the MAX31855 SO now.
     * So we read the MQ-135 through its DO on PC3 (smoke_sensor.c, ADC1) and retire
     * the old PC2/ADC0 path: mq135.c stays compiled but is never init'd/read ->
     * PC2 belongs solely to the MAX31855 (no conflict). */
    /* (mq135_init removed — PC2 freed for MAX31855) */

    /* MQ-135 gas sensor via its DO on PC3 (ADC1_CH13). AO read dropped — the doc's
     * AO=PG13, but the firmware used PA9 which is the RGB-LCD R5 line; the demo only
     * needs the DO analog threshold, so we leave all AO pins untouched. */
    relay_init();
    heater_init();      /* PTC heating-plate relay on PD12 (real-furnace demo), OFF on boot */
    smoke_sensor_init();

    /* MAX31855 K-type thermocouple — real furnace PV (bit-bang SPI on
     * CLK=PB10 / SO=PC2 / CS=PG3, per 模块(2).docx). Probed once at boot so a
     * wiring fault is visible; the live channel feeds ctrl_task + the safety
     * supervisor + the Control screen. present=0 simply means none wired. */
#if MAX31855_ENABLED
    max31855_init();
    {
        uint8_t f = 0u, p = 0u;
        int tc = max31855_decode(max31855_read_raw(), &f, &p) / 100;
        char tb[80]; int tn;
        tn = snprintf(tb, sizeof(tb),
            "[ctrl] MAX31855 init: present=%u T=%dC fault=0x%X (CLK PB10/SO PC2/CS PG3)\r\n",
            (unsigned)p, tc, (unsigned)f);
        if (tn > 0) boot_print(tb);
#if MAX31855_DIAG_VERBOSE
        max31855_diag_print("boot", 8);
#endif
    }
#else
    boot_print("[ctrl] furnace PV: simulation profile active\r\n");
#endif

    /* L298N motor (IN3=PG11, IN4=PG6) — 2026-05-28 per 模块(2).docx.
     * 2026-06-02 ★ 删除上电自检脉冲 (原 300ms 正转 + 300ms 反转): 马达现在粘了
     * ADXL345 做 AI-10 振动 demo, 开机自己转一下会被 AI-10 当成真振动 + 用户嫌干扰.
     * motor_init() 已把 IN3/IN4 驱动到 LOW (coast/停), 开机静止, 等 START/语音再转. */
    motor_init();
    motor_set(MOTOR_STOP);
    /* slow-test speed: full speed shakes too hard for the demo. 25% duty (IN3
     * chopped @50 Hz) + 300 ms full kickstart to beat stiction. 4-step quantiser
     * → 25/50/75/100%. If the motor stalls/stutters after the kick, raise this.
     * sensor_task @200 Hz drives motor_pwm_tick. NOTE: AI-10 was first captured at
     * 50%; re-capture at this 25% speed (AI10_CAPTURE below) and retrain so
     * 'running' is recognised at the lower vibration level. */
    motor_set_speed(25u);
    boot_print("[Lab-Sentinel] motor OK (L298N PG11/PG6, PWM speed 25%)\r\n");

    /* INMP441 I²S microphone: CK=PA5, SD=PA7, WS=PA0, L/R=PD5 on SPI0/I2S0.
     * 2026-05-28 ★ 暂禁: PA7 跟新 RGB LCD VSYNC=PA7 AF14 直冲. doa_task 也不
     * spawn (见下面). 后续要捡回 DOA 必须先把 INMP441 SD 挪到 PB5(占 UART4 RX)
     * 或 PD7 (空闲, 推荐) 上, 改 inmp441 driver 的 GPIO_PIN_7 → GPIO_PIN_7@GPIOD
     * + AF 切到 SPI0_MOSI 的备选 AF5. */
    /* inmp441_init(); */
    boot_print("[Lab-Sentinel] relay + smoke OK (mic disabled: PA7 conflict with LCD VSYNC)\r\n");

    /* OV5640 摄像头 — DCI 8-bit 并口 + DMA1 CH7 → SDRAM_CAMERA_FB (QVGA RGB565).
     * 2026-05-29 ★ 启用: 驱动按官方板 30-pin 排针重写 (PCLK=PE3 / D4=PE4 / D5=PB6 /
     * D7=PE6 / D6=PB8), 不再碰 PD0(=SDRAM D2). PB8(D6) 由禁用 GT911 触摸释放
     * (见下面 ui_task). 失败 (chip-id/SCCB) 只打印 rc, 不阻塞启动. */
    {
        uint8_t cam_rc = ov5640_init();
        if (cam_rc == 0U) {
            boot_print("[Lab-Sentinel] OV5640 OK (DCI QVGA RGB565)\r\n");
        } else {
            /* 细粒度诊断: SCCB 已在 ov5640_init 里初始化, 直接重读 chip-id.
             *   ack=0 → 设备地址没 ACK → 接线(SCL=PB4/SDA=PB7)/供电(3V3)/XCLK(PG7)/
             *           PWDN 待机. (rc=1 且 ack=0 = 设备根本没在总线上应答)
             *   ack=1 id=FFFF → ACK 了但读全 1 → SDA 读不回 / 模组没真启动
             *   ack=1 id=0000 → 读全 0 → SDA 短地 / 时序
             *   ack=1 id=其他 → SCCB 半通, 时序问题 */
            uint16_t cid  = 0xEEEEU;
            uint8_t  idrc = ov5640_read_chip_id(&cid);   /* 0=ACK, 1=NACK */
            char _cm[72];
            int _cn = snprintf(_cm, sizeof(_cm),
                "[Lab-Sentinel] OV5640 FAIL rc=%u ack=%u id=%04X\r\n",
                (unsigned)cam_rc, (unsigned)(idrc == 0U), (unsigned)cid);
            if (_cn > 0) boot_print(_cm);
        }
    }

    /* ---- 5 AI models: golden-vector on-chip self-test ----
     * Each C float engine runs a fixed input baked from PyTorch and is checked
     * against the PyTorch expected output. Tiny max|err| ⇒ the engines reproduce
     * training byte-for-byte on the M7 FPU. (SDRAM is up → AI-3 scratch valid.) */
#ifdef LAB_AI_BOOT_SELFTEST
    {
        ai_selftest_result_t st;
        char sb[100];
        int n;
        ai_selftest_run(&st);
        n = snprintf(sb, sizeof(sb),
            "[AI] selftest AI1=%d AI2=%d AI3=%d AI4=%d AI5=%d NCM=%d GAS=%d  ALL=%s\r\n",
            st.ai1_pass, st.ai2_pass, st.ai3_pass, st.ai4_pass, st.ai5_pass,
            st.ncm_pass, st.gas_pass, st.all_pass ? "PASS" : "FAIL");
        if (n > 0) boot_print(sb);
        /* edge models 6->10: optical / thermal / energy / retrieval / vib PdM */
        n = snprintf(sb, sizeof(sb),
            "[AI] selftest AI6=%d AI7=%d AI8=%d AI9=%d AI10=%d\r\n",
            st.ai6_pass, st.ai7_pass, st.ai8_pass, st.ai9_pass, st.ai10_pass);
        if (n > 0) boot_print(sb);
        /* new models 11->13: phase-purity / PL dopant classifier / PL-QC autoencoder */
        n = snprintf(sb, sizeof(sb),
            "[AI] selftest AI11=%d AI12=%d AI13=%d\r\n",
            st.ai11_pass, st.ai12_pass, st.ai13_pass);
        if (n > 0) boot_print(sb);
        /* new models 14->17: temp forecaster / PL host-ID / PL lambda / PL few-shot */
        n = snprintf(sb, sizeof(sb),
            "[AI] selftest AI14=%d AI15=%d AI16=%d AI17=%d\r\n",
            st.ai14_pass, st.ai15_pass, st.ai16_pass, st.ai17_pass);
        if (n > 0) boot_print(sb);
        /* new models 18->20: sintering RUL/ETA + thermocouple-integrity classifier */
        n = snprintf(sb, sizeof(sb),
            "[AI] selftest AI19=%d AI20=%d  (20 models on-chip)\r\n",
            st.ai19_pass, st.ai20_pass);
        if (n > 0) boot_print(sb);
        /* B-track depth: CAM explainability / INT8 path / adaptive conformal */
        n = snprintf(sb, sizeof(sb),
            "[AI] selftest CAM=%d INT8=%d ACONF=%d  ALL=%s\r\n",
            st.cam_pass, st.int8_pass, st.aconf_pass, st.all_pass ? "PASS" : "FAIL");
        if (n > 0) boot_print(sb);
        /* GD32 Embedded AI Tool deployment: AI-4 TFLite output re-run on-chip,
         * byte-verified vs the tool's flatbuffer golden (weights traced to the .tflite) */
        n = snprintf(sb, sizeof(sb),
            "[AI] selftest TFLITE=%d (GD32-AI-Tool AI-4 deployed, err x1e6=%ld)\r\n",
            st.tflite_pass, (long)(st.tflite_err * 1e6f));
        if (n > 0) boot_print(sb);
        /* report max errors in ppm-ish (×1e6) so we can read them without %f */
        n = snprintf(sb, sizeof(sb),
            "[AI] maxerr(x1e6) AI1=%ld AI2=%ld AI3=%ld AI4=%ld AI5=%ld\r\n",
            (long)(st.ai1_logit_err * 1e6f), (long)(st.ai2_recon_err * 1e6f),
            (long)(st.ai3_logit_err * 1e6f), (long)(st.ai4_logit_err * 1e6f),
            (long)(st.ai5_logit_err * 1e6f));
        if (n > 0) boot_print(sb);
    }
#else
    boot_print("[AI] boot self-test SKIPPED (fast bring-up build); 20 models still run at runtime\r\n");
#endif /* LAB_AI_BOOT_SELFTEST */

#ifdef LAB_LM_ENABLE
    /* ---- edge nano-LM (generative GPT) + online-learning risk head self-test ----
     * NLM: reproduce the deployed INT8 nano-LM token-for-token on 3 demo contexts
     *      (the on-chip generative diagnosis engine, DeepSeek-distilled).
     * OL : reproduce the numpy online SGD update stream (on-device learning head).
     * These are the new 21st/22nd on-chip AI subsystems. */
    {
        float nlm_err = -1.0f, ol_err = -1.0f;
        char sb[128];
        int n, nlm_ok, ol_ok;
        nlm_ok = nanolm_selftest(&nlm_err);
        ol_ok  = online_selftest(&ol_err);
        online_reset();                 /* runtime starts from the PC seed weights */
        n = snprintf(sb, sizeof(sb),
            "[AI] selftest NLM=%d (logit err x1e6=%ld) OL=%d (w err x1e6=%ld)  [edge GPT + online head]\r\n",
            nlm_ok, (long)(nlm_err * 1e6f), ol_ok, (long)(ol_err * 1e6f));
        if (n > 0) boot_print(sb);
    }

    /* ---- edge LLM CLUSTER: 5 role-specialized experts in 8MB SPI flash,
     * swap-loaded one at a time into SDRAM (MCU mirror of a server-class BPU swap-load
     * cluster). First boot after flashing provisions the image over UART
     * (provision_cluster.py); later boots find the magic and self-test. */
    {
        char sb[128]; int n; uint32_t fid;
        cl_spiflash_init();
        fid = cl_spiflash_id();
        if (!flash_cluster_present()) {
            n = snprintf(sb, sizeof(sb),
                "[AI] cluster: SPI flash id=0x%06lX, no image -> provisioning over UART\r\n",
                (unsigned long)fid);
            if (n > 0) boot_print(sb);
            flash_cluster_provision();        /* waits briefly for PC; skips if none */
        }
        if (flash_cluster_present()) {
            int per[NLM_CL_NEXPERT], ok; float cerr = -1.0f;
            ok = cluster_selftest(per, &cerr);
            n = snprintf(sb, sizeof(sb),
                "[AI] selftest CLUSTER=%d (%d experts swap-load, logit err x1e6=%ld) spi=0x%06lX\r\n",
                ok, NLM_CL_NEXPERT, (long)(cerr * 1e6f), (unsigned long)fid);
            if (n > 0) boot_print(sb);
        } else {
            boot_print("[AI] cluster: not provisioned (run provision_cluster.py); disabled\r\n");
        }
    }

    /* ---- LM size bank: the two SMALLER swept flagship sizes (s0p6 0.6M, m1p35
     * 1.26M) in SPI flash, swap-loaded against the always-on internal x1p9 (1.8M).
     * Lets the operator switch the on-chip generative LM across the hardware-
     * ceiling curve at runtime (HMI "SWITCH LM"). Provisioned over UART on first
     * boot (provision_cluster.py --img lmbank_image.bin); SPI already init'd above. */
    {
        char sb[128]; int n;
        if (!flash_bank_present()) {
            boot_print("[AI] lm bank: no image -> provisioning over UART (lmbank_image.bin)\r\n");
            flash_bank_provision();           /* waits briefly for PC; skips if none */
        }
        if (flash_bank_present()) {
            int per[8], ok; float berr = -1.0f;
            ok = bank_selftest(per, &berr);
            n = snprintf(sb, sizeof(sb),
                "[AI] selftest BANK=%d (%d sizes swap-load, logit err x1e6=%ld)\r\n",
                ok, lm_roster_count() - 1, (long)(berr * 1e6f));
            if (n > 0) boot_print(sb);
        } else {
            boot_print("[AI] lm bank: not provisioned; only internal x1p9 LM active\r\n");
        }
    }
#else
    boot_print("[AI] LM stack DISABLED (UI-dev fast build): NLM/OL/CLUSTER/BANK skipped, flagship weights out of image\r\n");
#endif /* LAB_LM_ENABLE */

    /* ---- recipe pre-flight: run AI-6/7/8/9 for the active recipe preset ----
     * Deterministic from the formula, so computed once here and cached for the
     * Pre-flight HMI screen. AI-6 emission peak/FWHM (distilled Tanabe-Sugano),
     * AI-7 thermal-quench band, AI-8 energy/carbon, AI-9 nearest historical recipe. */
    {
        recipe_ai_t pf;
        char pb[160];
        int  m;
        recipe_ai_refresh(ACTIVE_PRESET);
        lab_get_recipe_ai(&pf);
        m = snprintf(pb, sizeof(pb),
            "[AI] preflight '%s': AI6 lam=%dnm fwhm=%dnm | AI7 %d%% band%d | AI8 %d.%dkWh %d.%dkgCO2 | AI9 #%d %s\r\n",
            pf.recipe, pf.lambda_nm, pf.fwhm_nm, pf.thermal_pct, pf.thermal_band,
            pf.kwh_x10 / 10, pf.kwh_x10 % 10, pf.co2_x10 / 10, pf.co2_x10 % 10,
            pf.analog_idx, pf.analog_name);
        if (m > 0) boot_print(pb);
        /* AI-11 phase-purity prior + derived crystal-field read-out (from AI-6 lambda). */
        m = snprintf(pb, sizeof(pb),
            "[AI] preflight AI11 purity=%s P(pure)=%d%% | derived Dq=%dcm-1 B=%dcm-1 Dq/B=%d.%02d field=%d\r\n",
            pf.purity_cls ? "PURE" : "IMPURE", pf.p_pure_pct,
            pf.dq_cm1, pf.b_cm1, pf.dq_over_b_x100 / 100, pf.dq_over_b_x100 % 100, pf.field_class);
        if (m > 0) boot_print(pb);
    }

    /* ---- on-chip inference latency (DWT cycle counter @600 MHz) ----
     * Concrete per-model timing for the report/defence ("Transformer on M7 = X us").
     * Inputs are dummy (latency is data-independent for these dense nets). */
#if AI_LATENCY_PROBE && defined(LAB_AI_BOOT_SELFTEST)
    {
        static float din[784];
        static float dcin[3 * 64 * 64];   /* crucible 3x64x64 dummy (boot-only) */
        float dlog[10], demb[32], drec[32], d3[5], d4[4];
        uint32_t c1, c2, c3, c4;
        int i;
        char tb[100];
        for (i = 0; i < 784; i++) din[i] = 0.1f;
        for (i = 0; i < 3 * 64 * 64; i++) dcin[i] = 0.1f;

        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

        /* AI-1 = the deployed crucible 4-class CNN (3x64x64), not the MNIST engine. */
        DWT->CYCCNT = 0u; ai1_crucible_forward(dcin, dlog, demb); c1 = DWT->CYCCNT;
        DWT->CYCCNT = 0u; ai2_ae_reconstruct(din, drec);    c2 = DWT->CYCCNT;
        DWT->CYCCNT = 0u; ai3_forward_norm(din, d3);        c3 = DWT->CYCCNT;
        DWT->CYCCNT = 0u; ai4_forward(din, d4);             c4 = DWT->CYCCNT;

        i = snprintf(tb, sizeof(tb),
            "[AI] latency(us) AI1=%lu AI2=%lu AI3=%lu AI4=%lu\r\n",
            (unsigned long)(c1 / 600u), (unsigned long)(c2 / 600u),
            (unsigned long)(c3 / 600u), (unsigned long)(c4 / 600u));
        if (i > 0) boot_print(tb);
        g_ai_lat[AI_LAT_AI1] = (int)(c1 / 600u); g_ai_lat[AI_LAT_AI2] = (int)(c2 / 600u);
        g_ai_lat[AI_LAT_AI3] = (int)(c3 / 600u); g_ai_lat[AI_LAT_AI4] = (int)(c4 / 600u);

        /* ---- new models AI-11..17 + B1 INT8 + B3 CAM latency (us) ---- */
        {
            float sp[64], probs[3], hp[2], mse, p, desc24[24], fcin[24], fcout[12], emb[16], camf[16];
            float feat26[26], win24[24];
            uint32_t c11, c12, c12q, c13, c14, c15, c16, c17, ccam, c19, c20;
            int j;
            for (j = 0; j < 64; j++) sp[j]   = demo_spec[j % DEMO_SPEC_N];
            for (j = 0; j < 24; j++) { fcin[j] = 0.5f; desc24[j] = 0.5f; }
            for (j = 0; j < 26; j++) feat26[j] = 0.5f;
            for (j = 0; j < 24; j++) win24[j]  = 0.5f;
            DWT->CYCCNT = 0u; (void)ai11_purity(desc24, &p);        c11  = DWT->CYCCNT;
            DWT->CYCCNT = 0u; (void)ai12_plclass(sp, probs);        c12  = DWT->CYCCNT;
            DWT->CYCCNT = 0u; (void)ai12_plclass_int8(sp, probs);   c12q = DWT->CYCCNT;
            DWT->CYCCNT = 0u; (void)ai13_plqc(sp, NULL, &mse);      c13  = DWT->CYCCNT;
            DWT->CYCCNT = 0u; ai14_forecast(fcin, fcout);           c14  = DWT->CYCCNT;
            DWT->CYCCNT = 0u; (void)ai15_hostid(sp, hp);            c15  = DWT->CYCCNT;
            DWT->CYCCNT = 0u; (void)ai16_lambda(sp);                c16  = DWT->CYCCNT;
            DWT->CYCCNT = 0u; ai12_embed(sp, emb);
                              (void)ai17_pl_classify(emb, NULL);    c17  = DWT->CYCCNT;
            DWT->CYCCNT = 0u; (void)ai1_crucible_cam(dcin, camf);   ccam = DWT->CYCCNT;
            DWT->CYCCNT = 0u; (void)ai19_rul(feat26);               c19  = DWT->CYCCNT;
            DWT->CYCCNT = 0u; (void)ai20_tcfault(win24, probs);     c20  = DWT->CYCCNT;
            i = snprintf(tb, sizeof(tb),
                "[AI] latency(us) AI11=%lu AI12=%lu AI12int8=%lu AI13=%lu AI14=%lu\r\n",
                (unsigned long)(c11/600u),(unsigned long)(c12/600u),
                (unsigned long)(c12q/600u),(unsigned long)(c13/600u),(unsigned long)(c14/600u));
            if (i > 0) boot_print(tb);
            i = snprintf(tb, sizeof(tb),
                "[AI] latency(us) AI15=%lu AI16=%lu AI17=%lu AI19=%lu AI20=%lu CAM=%lu\r\n",
                (unsigned long)(c15/600u),(unsigned long)(c16/600u),(unsigned long)(c17/600u),
                (unsigned long)(c19/600u),(unsigned long)(c20/600u),(unsigned long)(ccam/600u));
            if (i > 0) boot_print(tb);
            g_ai_lat[AI_LAT_AI11] = (int)(c11/600u);  g_ai_lat[AI_LAT_AI12] = (int)(c12/600u);
            g_ai_lat[AI_LAT_AI12Q] = (int)(c12q/600u);g_ai_lat[AI_LAT_AI13] = (int)(c13/600u);
            g_ai_lat[AI_LAT_AI14] = (int)(c14/600u);  g_ai_lat[AI_LAT_AI15] = (int)(c15/600u);
            g_ai_lat[AI_LAT_AI16] = (int)(c16/600u);  g_ai_lat[AI_LAT_AI17] = (int)(c17/600u);
            g_ai_lat[AI_LAT_AI19] = (int)(c19/600u);  g_ai_lat[AI_LAT_AI20] = (int)(c20/600u);
            g_ai_lat[AI_LAT_CAM]  = (int)(ccam/600u);
        }

        /* ---- AI-3 section breakdown (us), summed over both blocks ---- */
        {
            extern unsigned int ai3_prof_proj, ai3_prof_ln, ai3_prof_qkv,
                ai3_prof_attn, ai3_prof_op, ai3_prof_ffn, ai3_prof_final;
            i = snprintf(tb, sizeof(tb),
                "[AI] AI3us proj=%lu ln=%lu qkv=%lu attn=%lu op=%lu ffn=%lu fin=%lu\r\n",
                (unsigned long)(ai3_prof_proj  / 600u), (unsigned long)(ai3_prof_ln   / 600u),
                (unsigned long)(ai3_prof_qkv   / 600u), (unsigned long)(ai3_prof_attn / 600u),
                (unsigned long)(ai3_prof_op    / 600u), (unsigned long)(ai3_prof_ffn  / 600u),
                (unsigned long)(ai3_prof_final / 600u));
            if (i > 0) boot_print(tb);
        }
    }
#endif /* AI_LATENCY_PROBE */

    /* Furnace temperature playback + feature builder (see furnace_sim.h). */
    furnace_sim_init();
    fb_reset();
    boot_print("[AI] furnace sim ready (garnet/YAG profile, idle)\r\n");

    boot_print("[Lab-Sentinel] boot OK\r\n");
    boot_print("[Lab-Sentinel] FreeRTOS up. Spawning runtime tasks.\r\n");

    /* Spawn order: high-prio first so they get a deterministic startup. */
    xTaskCreate(sensor_task, "sensor", STACK_SENSOR, NULL, PRIO_HIGH, NULL);
    xTaskCreate(fusion_task, "fusion", STACK_FUSION, NULL, PRIO_HIGH, NULL);
    xTaskCreate(vision_task, "vision", STACK_VISION, NULL, PRIO_MED,  NULL);
    xTaskCreate(ui_task,     "ui",     STACK_UI,     NULL, PRIO_MED,  NULL);
#if CI1302_VOICE_ENABLED
    /* 2026-05-28: CI1302 迁到 PC10/PC11 后恢复 voice_task. */
    xTaskCreate(voice_task,  "voice",  STACK_VOICE,  NULL, PRIO_MED,  NULL);
#endif
    /* doa_task disabled 2026-05-28: depends on INMP441, INMP441 SD=PA7 conflicts
     * with new RGB LCD VSYNC=PA7. Re-enable after relocating mic SD off PA7. */
    /* xTaskCreate(doa_task, "doa", STACK_DOA, NULL, PRIO_MED, NULL); */
    /* 2026-05-29: env_task PRIO_LOW → PRIO_MED. 它现在跑 AI-2 + AI-3 (1 Hz),
     * 若留 PRIO_LOW 会被 PRIO_MED 的 ui_task (LVGL + VBlank busy-wait) 永久饿死
     * → s_ai.ready 永不置位 → fusion 一直 "waiting for AI pipeline". 提到 MED 后
     * 跟 vision/ui/voice 同级时间片轮转 (configUSE_TIME_SLICING 默认开), 每 1 Hz
     * 跑几 ms AI 突发再 sleep 1 s. RGB TLI 硬件扫描 + LVGL 渲染到 back buffer,
     * env 不碰 framebuffer → 旧 8080 SPI 屏的 H5/H6 花屏约束已不适用. */
    xTaskCreate(env_task,    "env",    STACK_ENV,    NULL, PRIO_MED,  NULL);
    /* ctrl_task: closed-loop recipe controller (monitor->controller pivot).
     * PRIO_MED like env_task — the RGB-TLI LCD removed the old PRIO_LOW
     * starvation hazard (H5); it sleeps CTRL_TICK_MS each cycle so it never
     * starves vision/ui/voice. Idle until voice "kai shi shao jie" starts a batch. */
    xTaskCreate(ctrl_task,   "ctrl",   STACK_CTRL,   NULL, PRIO_MED,  NULL);
#ifdef LAB_LM_ENABLE
    /* nlm_task: edge nano-LM generative diagnosis (0.5 Hz, ~1s/gen on trigger) +
     * online-learning risk head. PRIO_LOW — it is event-driven inference, never on
     * the safety path, and must yield to control/HMI. Not watchdog-supervised
     * (variable timing, like vision/ui/voice). */
    xTaskCreate(nlm_task,    "nlm",    STACK_NLM,    NULL, PRIO_LOW,  NULL);
    /* cluster_task: 5-expert edge-LLM cluster, swap-loaded from SPI flash. PRIO_LOW,
     * event-driven; self-disables if the SPI-flash image was never provisioned. */
    xTaskCreate(cluster_task, "cluster", STACK_CLUSTER, NULL, PRIO_LOW, NULL);
#endif /* LAB_LM_ENABLE — UI-dev fast build skips the generative-LM tasks */
    /* eth_task disabled: Ethernet downgraded to Phase 6 optional add-on.
     * Re-spawn here AND move INMP441 SD (PA7) off CRS_DV before re-enabling. */
    /* xTaskCreate(eth_task, "eth", STACK_ETH, NULL, PRIO_LOW, NULL); */
    /* wdg_task LAST: arms the FWDGT only after every monitored task exists, so
     * the heavy one-shot boot init above (SDRAM/LVGL/camera/AI self-test) runs
     * watchdog-free. PRIO_HIGH so the kicker can never be starved by MED work. */
    xTaskCreate(wdg_task,    "wdg",    STACK_WDG,    NULL, PRIO_HIGH, NULL);

    boot_print("[Lab-Sentinel] all tasks spawned (eth disabled). init -> exit.\r\n");

    vTaskDelete(NULL);
}

/* -------------------------------------------------------------------------- */
/* sensor_task - 200 Hz ADXL345 sampling, RMS over 1-second window, post a   */
/*               vibration_result_t once per second to xQueue_VibResult.      */
/*                                                                            */
/* Also runs the AI-1 forward-pass smoke test once at boot to confirm the    */
/* model graph loaded without HardFault (Phase 3 will replace this with the  */
/* actual AI-3 vibration classifier on the windowed RMS feature vector).     */
/* -------------------------------------------------------------------------- */
#define SENSOR_HZ           200U
#define SENSOR_WINDOW_N     SENSOR_HZ   /* 1-second window, 200 samples */
/* (AI-10 idle threshold gate removed 2026-06-02: the real-data 2-class model
 * now decides stopped vs running directly from vibration energy.) */

/* 2026-06-02 AI-10 REAL-DATA capture mode. Set to 1, flash, then: let the motor
 * sit STOPPED ~12 s (collects run=0 windows) then press START / say "kai shi shao
 * jie" so it spins ~12 s (collects run=1 windows); copy every "[cap]" serial line
 * and hand it back. Each line is one 1-s, 64-sample ADXL345 window dumped as the
 * raw magnitude stream in mg (chronological, oldest first) + the motor-commanded
 * label, which retrains AI-10 as a real 2-class stopped/running model on THIS
 * motor. Set back to 0 for the normal build. */
#define AI10_CAPTURE        0

static void sensor_task(void *pv)
{
    (void)pv;

    {
        float ai_input[1 * 28 * 28];
        float ai_output[10];
        int i;
        for (i = 0; i < (int)(sizeof(ai_input)/sizeof(ai_input[0])); i++) {
            ai_input[i] = 0.0f;
        }
        network_init();
        network_run(ai_input, ai_output);
    }
    boot_print("[sensor] AI-1 forward OK; ADXL345 200Hz starting\r\n");

    uint32_t   sum_sq    = 0U;     /* Σ (mg)²  over current window */
    uint16_t   peak_mg   = 0U;
    uint16_t   sample_n  = 0U;
    TickType_t last      = xTaskGetTickCount();

    /* AI-10 vibration PdM: rolling 64-sample acceleration ring (g), classified
     * once per 1-s window. .bss-backed so it never touches the task stack. */
    static float   vib_ring[64];
    static uint8_t vw_head = 0u, vw_n = 0u;

    for (;;) {
        s_wdg_hb[WDG_SENSOR]++;            /* watchdog heartbeat */
        motor_pwm_tick();                  /* software PWM speed control @200 Hz */
        uint16_t mg = adxl345_read_magnitude_mg();
        sum_sq    += (uint32_t)mg * (uint32_t)mg;
        if (mg > peak_mg) peak_mg = mg;
        sample_n++;

        vib_ring[vw_head] = (float)mg / 1000.0f;     /* g (includes ~1g DC) */
        vw_head = (uint8_t)((vw_head + 1u) & 63u);
        if (vw_n < 64u) vw_n++;

        if (sample_n >= SENSOR_WINDOW_N) {
            /* RMS = sqrt(Σ x² / N) — same _isqrt used in adxl345.c style */
            uint32_t mean_sq = sum_sq / SENSOR_WINDOW_N;
            uint32_t r = mean_sq;
            for (uint8_t i = 0; i < 16U; i++) {
                if (r == 0U) break;
                r = (r + mean_sq / r) >> 1;
            }
            vibration_result_t res = {
                .timestamp_ms     = (uint32_t)(xTaskGetTickCount() *
                                               (1000U / configTICK_RATE_HZ)),
                .predicted_class  = 0U,    /* Phase 3 will fill */
                .confidence_q7    = 0U,
                .rms_mg           = (uint16_t)((r > 65535U) ? 65535U : r),
            };
            (void)xQueueSend(xQueue_VibResult, &res, 0);
            s_eth_vib_rms = res.rms_mg;   /* eth_task snapshot */

            /* ---- AI-10 vibration PdM (11th model) ---- once per 1-s window.
             * Mean-remove the 64-sample ring -> AC acceleration in g, then run the
             * 2-class model RETRAINED ON REAL DATA from this motor + ADXL345
             * (capture_real.txt 2026-06-02): class 0 = stopped, 1 = running. The
             * model keys on vibration energy (rms/ebb), so it decides stopped vs
             * running directly — no idle threshold gate needed. Decision-support
             * (not safety chain); honest 2-class, no faked fault classes. */
            if (vw_n >= 64u) {
                float   win[64], probs[4], mean = 0.0f, ac_ss = 0.0f;
                int     j, idx, vcls, conf, rms_mg;
                uint8_t running;
                static int     last_vcls = -2;
                static uint8_t last_run  = 2u;
#if AI10_CAPTURE
                /* Dump the raw 64-sample magnitude window (mg, chronological,
                 * oldest first) + motor-commanded label, for real-data retrain. */
                {
                    char cb[440]; int cn = 0, cj, cidx, mgv;
                    uint8_t cap_run = (uint8_t)((motor_state() == MOTOR_FORWARD ||
                                                 motor_state() == MOTOR_REVERSE) ? 1 : 0);
                    cn += snprintf(cb + cn, sizeof(cb) - cn, "[cap] run=%u w=", (unsigned)cap_run);
                    for (cj = 0; cj < 64; cj++) {
                        cidx = (vw_head + cj) & 63;
                        mgv  = (int)(vib_ring[cidx] * 1000.0f + 0.5f);
                        cn  += snprintf(cb + cn, sizeof(cb) - cn, "%d%s", mgv, (cj < 63) ? "," : "");
                        if (cn >= (int)sizeof(cb) - 8) break;
                    }
                    (void)snprintf(cb + cn, sizeof(cb) - cn, "\r\n");
                    boot_print(cb);
                }
#endif
                for (j = 0; j < 64; j++) { idx = (vw_head + j) & 63; win[j] = vib_ring[idx]; mean += win[j]; }
                mean /= 64.0f;
                for (j = 0; j < 64; j++) { win[j] -= mean; ac_ss += win[j] * win[j]; }
                rms_mg = (int)(sqrtf(ac_ss / 64.0f) * 1000.0f + 0.5f);
                /* Real 2-class model decides stopped(0)/running(1) from the live
                 * window — trained on this motor's actual rest + vibration data. */
                vcls    = ai10_vibration(win, probs);   /* 0 stopped / 1 running */
                running = (uint8_t)(vcls == 1);
                conf    = (int)(probs[vcls] * 100.0f + 0.5f);
                taskENTER_CRITICAL();
                s_vib.valid = 1u; s_vib.running = running; s_vib.cls = vcls;
                s_vib.conf_pct = conf; s_vib.rms_mg = rms_mg;
                taskEXIT_CRITICAL();
                if (vcls != last_vcls || running != last_run) {   /* change-gated */
                    static const char *const VN[2] = { "stopped", "running" };
                    char vb[96];
                    int  vn;
                    vn = snprintf(vb, sizeof(vb),
                        "[ai10] vib=%s p=%d%% acRMS=%dmg (real 2-class, this motor)\r\n",
                        VN[vcls & 1], conf, rms_mg);
                    if (vn > 0) boot_print(vb);
                    last_vcls = vcls; last_run = running;
                }
            }

            sum_sq   = 0U;
            peak_mg  = 0U;
            sample_n = 0U;
        }

        vTaskDelayUntil(&last, pdMS_TO_TICKS(1000U / SENSOR_HZ));
    }
}

/* -------------------------------------------------------------------------- */
/* vision_task - 5 Hz OV5640 capture + AI-1 / AI-1b inference                 */
/*                                                                            */
/* Triggers a single-frame capture (~30 ms exposure + DMA), waits up to       */
/* 200 ms for the FRAME flag, then computes a tiny 4×4 luminance signature   */
/* across the QVGA frame and prints it as a sanity probe. AI-1b classifier    */
/* will replace the signature step in Phase 3.                                */
/*                                                                            */
/* Falls back to a 2 s heartbeat if ov5640_init() previously failed (we       */
/* detect that by checking ov5640_capture_one's effect: frame_ready stays 0). */
/* -------------------------------------------------------------------------- */
static void vision_task(void *pv)
{
    (void)pv;
    TickType_t last = xTaskGetTickCount();
    char       msg[128];
    uint32_t   stale_streak = 0U;

    for (;;) {
        ov5640_capture_one();

        /* Wait for the DCI end-of-frame latch (300 ms cap). EF — not the DMA
         * full-transfer flag — is the real frame boundary: the sensor delivers a
         * touch under 320x240x2/4 words, so FTF would never fire. */
        uint32_t waited = 0U;
        while (dci_flag_get(DCI_FLAG_EF) == RESET) {
            vTaskDelay(pdMS_TO_TICKS(5U));
            waited += 5U;
            if (waited >= 300U) break;
        }

        uint8_t frame_ok = (dci_flag_get(DCI_FLAG_EF) != RESET) ? 1U : 0U;
        if (frame_ok) {
            dci_capture_disable();        /* snapshot complete; stop the engine */
            /* DMA filled the cacheable AXI-SRAM buffer; invalidate D-cache so the
             * CPU reads the freshly-DMA'd pixels rather than stale cache lines. */
            SCB_InvalidateDCache_by_Addr((uint32_t *)ov5640_framebuf,
                                         (int32_t)OV5640_QVGA_BYTES);
            stale_streak = 0U;

            /* 4×4 luminance histogram — sample one pixel per 80×60 cell. */
            uint32_t lum_sum = 0U;
            for (uint16_t cy = 0U; cy < 4U; cy++) {
                for (uint16_t cx = 0U; cx < 4U; cx++) {
                    uint16_t y = (uint16_t)(cy * 60U + 30U);
                    uint16_t x = (uint16_t)(cx * 80U + 40U);
                    uint16_t pix = ov5640_framebuf[y * 320U + x];
                    /* RGB565 → quick-and-dirty luminance: G6 channel ×2. */
                    uint8_t g = (uint8_t)((pix >> 5) & 0x3FU);
                    lum_sum += g;
                }
            }
            uint16_t lum_avg = (uint16_t)(lum_sum / 16U);
            uint32_t words   = (OV5640_QVGA_BYTES / 4U) - DMA_CH7CNT(DMA1);

            int n = snprintf(msg, sizeof(msg),
                             "[vision] frame OK  lum_avg=%u words=%lu\r\n",
                             (unsigned)lum_avg, (unsigned long)words);
            if (n > 0) vbprint(msg);
        } else {
            stale_streak++;
            if (stale_streak == 1U || (stale_streak % 5U) == 0U) {
                /* DVP capture diagnostic: dma_rem = words still pending (init
                 * 38400). 38400 => 0 words landed (no PCLK/sync). 0<rem<38400 =>
                 * partial frame (HSYNC/size). 0 => transfer done but FTF missed.
                 * stat0 bits: b0 HS-line b1 VS-line b2 FIFO-valid. ef/ovr/ese =
                 * DCI end-of-frame / FIFO-overrun / sync-error latches. */
                int n = snprintf(msg, sizeof(msg),
                    "[vision] to streak=%u dma_rem=%u stat0=0x%lX ef=%u ovr=%u ese=%u\r\n",
                    (unsigned)stale_streak,
                    (unsigned)DMA_CH7CNT(DMA1),
                    (unsigned long)(DCI_STAT0 & 0x7UL),
                    (unsigned)(dci_flag_get(DCI_FLAG_EF)  != RESET),
                    (unsigned)(dci_flag_get(DCI_FLAG_OVR) != RESET),
                    (unsigned)(dci_flag_get(DCI_FLAG_ESE) != RESET));
                if (n > 0) boot_print(msg);
            }
        }

        /* ---- camera preview: 8x6 luminance grid + bright-blob crucible box ----
         * LV_USE_IMG/CANVAS are off, so the Camera HMI page renders the live frame
         * as a coarse lv_obj luminance-tile grid. The bounding box is a CLASSICAL
         * CV localiser (the ceramic crucible is far brighter than the dark furnace/
         * bench): threshold the cell map, take the extent of the bright cells. The
         * AI-1 CNN (below) supplies the LABEL (what); this supplies WHERE. */
        {
            uint8_t grid[CAM_GW * CAM_GH];
            int gx, gy, gmax = 0, gmin = 255, thr;
            int bx0 = CAM_GW, by0 = CAM_GH, bx1 = -1, by1 = -1, blob = 0;
            for (gy = 0; gy < CAM_GH; gy++) {
                for (gx = 0; gx < CAM_GW; gx++) {
                    long gsum = 0; int sx, sy, ns = 0, v;
                    for (sy = 0; sy < 4; sy++) {
                        for (sx = 0; sx < 4; sx++) {
                            int px = gx * (320 / CAM_GW) + sx * ((320 / CAM_GW) / 4) + 2;
                            int py = gy * (240 / CAM_GH) + sy * ((240 / CAM_GH) / 4) + 2;
                            int lum;
                            if (frame_ok) {
                                uint16_t pix = ov5640_framebuf[py * 320 + px];
                                int r8 = (int)((pix >> 11) & 0x1Fu) << 3;
                                int g8 = (int)((pix >> 5)  & 0x3Fu) << 2;
                                int b8 = (int)( pix        & 0x1Fu) << 3;
                                lum = (r8 * 77 + g8 * 150 + b8 * 29) >> 8;  /* Y 0..255 */
                            } else {
                                lum = ((gx + gy) * 28) & 0xFF;              /* test pattern */
                            }
                            gsum += lum; ns++;
                        }
                    }
                    v = (int)(gsum / ns);
                    grid[gy * CAM_GW + gx] = (uint8_t)v;
                    if (v > gmax) gmax = v;
                    if (v < gmin) gmin = v;
                }
            }
            thr = gmin + (gmax - gmin) * 6 / 10;      /* 60% peak = bright-region cut */
            if (gmax - gmin >= 24) {                   /* enough contrast for a blob */
                for (gy = 0; gy < CAM_GH; gy++)
                    for (gx = 0; gx < CAM_GW; gx++)
                        if (grid[gy * CAM_GW + gx] >= thr) {
                            if (gx < bx0) bx0 = gx; if (gx > bx1) bx1 = gx;
                            if (gy < by0) by0 = gy; if (gy > by1) by1 = gy; blob = 1;
                        }
            }
            taskENTER_CRITICAL();
            for (gx = 0; gx < CAM_GW * CAM_GH; gx++) s_cam.lum[gx] = grid[gx];
            s_cam.valid = frame_ok;
            if (blob) {
                s_cam.blob_ok = 1u;
                s_cam.bx = (uint8_t)bx0;            s_cam.by = (uint8_t)by0;
                s_cam.bw = (uint8_t)(bx1 - bx0 + 1); s_cam.bh = (uint8_t)(by1 - by0 + 1);
            } else {
                s_cam.blob_ok = 0u;
            }
            s_cam.frames++;
            taskEXIT_CRITICAL();
        }

        /* ---- AI-1 crucible 3-class CNN + AI-1b NCM (once per second) ----
         * The DEPLOYED vision task model: a 3-class crucible-state CNN
         * (0 empty / 1 loaded / 2 done) running on real OV5640 pixels. The frame
         * is area-downsampled QVGA RGB565 -> 3x64x64 RGB[0,1] CHW. The 32-D GAP
         * embedding feeds the AI-1b few-shot NCM. Weights are trained on REAL
         * phone-shot crucible photos (CIMC/手机拍摄数据, stratified 5-fold CV
         * 90.7%); the engine is byte-verified vs PyTorch (ai_selftest, 2.4e-6).
         * Note: "sintering" (in-furnace glow) is NOT a visual class — it can't be
         * seen from outside the 1500C furnace; the 4-stage furnace proxy (HMI /
         * AI-4 / AI-5) carries that process-stage signal separately. */
        {
            static float cin[3 * 64 * 64];   /* CHW RGB [0,1] (static .bss, ~49KB) */
            static float clog[3];
            static float cemb[32];
            static uint8_t enrolled = 0u;
            static uint32_t vcnt = 0u;

            if ((vcnt++ % 5u) == 0u) {
                int oy, ox, k;
                uint8_t have_frame = frame_ok;   /* set above from DCI end-of-frame */
                for (oy = 0; oy < 64; oy++) {
                    int sy = oy * 240 / 64;
                    for (ox = 0; ox < 64; ox++) {
                        float r, g, b;
                        if (have_frame) {
                            int sx = ox * 320 / 64;
                            uint16_t pix = ov5640_framebuf[sy * 320 + sx];
                            r = (float)((pix >> 11) & 0x1Fu) / 31.0f;  /* R5 */
                            g = (float)((pix >> 5)  & 0x3Fu) / 63.0f;  /* G6 */
                            b = (float)( pix        & 0x1Fu) / 31.0f;  /* B5 */
                        } else {
                            r = g = b = (float)((ox + oy) & 0x3F) / 63.0f;  /* pattern */
                        }
                        cin[0 * 4096 + oy * 64 + ox] = r;
                        cin[1 * 4096 + oy * 64 + ox] = g;
                        cin[2 * 4096 + oy * 64 + ox] = b;
                    }
                }
                int cls = ai1_crucible_forward(cin, clog, cemb);

                /* softmax over the 3 logits -> probs (for HMI + honest confidence) */
                float mx = clog[0], sum = 0.0f, probs[3];
                for (k = 1; k < 3; k++) if (clog[k] > mx) mx = clog[k];
                for (k = 0; k < 3; k++) { probs[k] = expf(clog[k] - mx); sum += probs[k]; }
                for (k = 0; k < 3; k++) probs[k] = probs[k] / sum;

                /* B3 CAM: forward-only 4x4 class-activation heatmap for the predicted
                 * class (AI-1 is GAP->FC). Scaled 0..255 for the Camera-screen overlay. */
                {
                    static float camf[16];
                    int ci;
                    (void)ai1_crucible_cam(cin, camf);
                    for (ci = 0; ci < 16; ci++) {
                        int cv = (int)(camf[ci] * 255.0f + 0.5f);
                        s_cam.cam[ci] = (uint8_t)(cv < 0 ? 0 : (cv > 255 ? 255 : cv));
                    }
                    s_cam.cam_ok = 1u;
                }

                if (!enrolled) {     /* seed two NCM demo classes from live embeddings */
                    int d; float shifted[32];
                    for (d = 0; d < 32; d++) shifted[d] = cemb[d] + 4.0f;
                    ai1b_reset();
                    ai1b_add_sample(0, cemb);      /* class 0 = current view   */
                    ai1b_add_sample(1, shifted);   /* class 1 = a distinct view */
                    enrolled = 1u;
                }
                float dist = 0.0f;
                int ncm = ai1b_classify(cemb, &dist);
                s_ncm_cls = ncm;   /* AI-1b few-shot head result for the HMI */

                /* publish live crucible result (HMI / future AI-4 fusion read it) */
                taskENTER_CRITICAL();
                for (k = 0; k < 3; k++) s_ai.ai1_cam_probs[k] = probs[k];
                s_ai.ai1_cam_cls   = cls;
                s_ai.ai1_cam_valid = have_frame;
                s_cam.cls          = cls;                       /* Camera page label  */
                s_cam.conf_pct     = (int)(probs[cls] * 100.0f + 0.5f);
                for (k = 0; k < 3; k++) s_cam.probs[k] = probs[k];
                taskEXIT_CRITICAL();

                /* ---- robustness: AI-1 graceful-degradation pair (clean vs perturbed) --
                 * The clean verdict above stays on the Camera page. If a vision
                 * perturbation is selected, perturb cin IN PLACE (it is rebuilt fresh
                 * next frame) and re-run AI-1 so the Robust page shows the degraded
                 * verdict + confidence drop. Decision-support model, off the abort path. */
                s_inf_count++;
                {
                    int vinj  = s_vis_inject;
                    int pcls  = cls;
                    int pconf = (int)(probs[cls] * 100.0f + 0.5f);
                    if (vinj != 0) {
                        uint32_t seed = 0x51ED5EEDu + (vcnt << 3);
                        float p2[3], mx2, sum2; int kk;
                        rob_perturb_img(cin, 3 * 64 * 64, vinj, &seed);
                        pcls = ai1_crucible_forward(cin, clog, 0);
                        mx2 = clog[0]; for (kk = 1; kk < 3; kk++) if (clog[kk] > mx2) mx2 = clog[kk];
                        sum2 = 0.0f; for (kk = 0; kk < 3; kk++) { p2[kk] = expf(clog[kk] - mx2); sum2 += p2[kk]; }
                        pconf = (int)(p2[pcls] / sum2 * 100.0f + 0.5f);
                        s_inf_count++;
                    }
                    taskENTER_CRITICAL();
                    s_rob.vis_inject     = (uint8_t)vinj;
                    s_rob.vis_clean_cls  = cls;
                    s_rob.vis_clean_conf = (int)(probs[cls] * 100.0f + 0.5f);
                    s_rob.vis_pert_cls   = pcls;
                    s_rob.vis_pert_conf  = pconf;
                    s_rob.vis_valid      = 1u;
                    taskEXIT_CRITICAL();
                }

                {
                    static const char *CN[3] = { "empty", "loaded", "done" };
                    static uint8_t announced = 0u;
                    int n = snprintf(msg, sizeof(msg),
                        "[vision] crucible=%s p=%u%% ncm=%d %s\r\n",
                        CN[cls], (unsigned)(probs[cls] * 100.0f + 0.5f), ncm,
                        have_frame ? "(cam)" : "(pattern)");
                    /* announce the first live classification once (proves the
                     * camera->crucible path even when verbose logging is off);
                     * steady-state readouts stay on the verbose-gated channel. */
                    if (n > 0) { if (!announced) { boot_print(msg); announced = 1u; }
                                 else vbprint(msg); }
                }
            }
        }

        vTaskDelayUntil(&last, pdMS_TO_TICKS(200U));   /* ~5 fps */
    }
}

/* -------------------------------------------------------------------------- */
/* env_task - 1 Hz SHT30 (T/RH) + MQ-135 (gas ADC) → xQueue_EnvResult.       */
/* AI-2 anomaly autoencoder will plug in at Phase 3 (placeholder zeros now). */
/* -------------------------------------------------------------------------- */
static void env_task(void *pv)
{
    (void)pv;
    TickType_t last = xTaskGetTickCount();
    char       _emsg[112];

    /* SHT30 软 I2C 在多任务环境下偶发 NACK / CRC 失败 —— 三重防御:
     *   (1) 本帧 retry 3 次 (每次间 15ms), 提高单帧成功率
     *   (2) last-good cache 兜底, 避免 UI 闪 0
     *   (3) 连续 10 帧全失败 → soft reset SHT30 (传感器内部状态机被打断时恢复)
     */
    static int16_t  s_last_temp_q8 = 0;
    static uint16_t s_last_hum_q8  = 0;
    static uint8_t  s_last_valid   = 0U;
    static uint8_t  s_fail_streak  = 0U;

    for (;;) {
        s_wdg_hb[WDG_ENV]++;              /* watchdog heartbeat */
        env_result_t res = {0};
        int16_t  t_q8 = 0;
        uint16_t h_q8 = 0;
        uint8_t  sht_ok = 0U;
        uint8_t  try_n;

        for (try_n = 0U; try_n < 3U; try_n++) {
            if (sht30_read(&t_q8, &h_q8) == 0U) {
                /* Plausibility gate: the soft-I2C bus is shared with the 200 Hz
                 * sensor_task (no mutex), so a preempted SHT30 read can pass CRC
                 * on a garbage value (e.g. T=-45 H=0). Reject implausible lab
                 * readings and retry/keep-last-good instead of feeding the AE
                 * a wild outlier. */
                int tc = (int)(t_q8 >> 8);
                int hc = (int)(h_q8 >> 8);
                if (tc >= 0 && tc <= 60 && hc >= 0 && hc <= 100) {
                    sht_ok = 1U;
                    break;
                }
            }
            vTaskDelay(pdMS_TO_TICKS(15));   /* short backoff between retries */
        }

        if (sht_ok) {
            s_last_temp_q8 = t_q8;
            s_last_hum_q8  = h_q8;
            s_last_valid   = 1U;
            s_fail_streak  = 0U;
        } else {
            if (s_fail_streak < 255U) s_fail_streak++;
            if (s_fail_streak >= 10U) {
                /* 传感器似乎卡死, 软复位让它从初始状态重来 */
                sht30_init();
                s_fail_streak = 0U;
                boot_print("[env] SHT30 soft-reset (10 consecutive failures)\r\n");
            }
        }
        if (s_last_valid) {
            res.temp_c_q8  = s_last_temp_q8;
            res.humidity_q8 = s_last_hum_q8;
        }
        res.timestamp_ms         = (uint32_t)(xTaskGetTickCount() *
                                              (1000U / configTICK_RATE_HZ));

        /* Smoke sensor: read for status/telemetry ONLY — it no longer drives the
         * relay. The relay = the batch ventilation fan, controlled solely by
         * START/STOP (s_batch_fan): boot OFF (relay_init drives PA1 low = off on
         * this active-high module), ON during a batch, OFF after. 2026-06-02:
         * smoke→auto-ventilation removed because the cold MQ sensor false-alarmed
         * and held the fan on permanently; re-add `if(sm.alarm) relay_on();` here
         * once the smoke sensor is verified (warmup-gated). */
        {
            smoke_result_t sm = {0};
            smoke_sensor_read(&sm);
            s_eth_smoke = sm.alarm;        /* eth_task snapshot (display only) */
            /* Gas channel = the MQ-135 (= 烟雾气敏传感器), read via its DO on PC3
             * (ADC1). Per 模块(2).docx the MQ-135 moved off PC2 (now the MAX31855
             * SO), so we use its DO raw ADC (0..4095) here — the mq135_adc field is
             * therefore still literally the MQ-135, just sampled through PC3 not PC2. */
            res.mq135_adc = sm.do_adc_raw;
        }

        /* ---- AI-2 (env AE anomaly) + AI-3 (sintering-curve transformer) ----
         * Furnace temperature is played back by furnace_sim (no 1600 C probe on
         * the bench); room/gas/vibration come from the real SHT30 / MQ-135 DO
         * (PC3, the 烟雾气敏传感器) / ADXL345.
         * Advance the sim minute-by-minute so AI-3 sees per-minute ramp gradients,
         * then build the 32-D feature vector and run both models. */
        {
            furnace_out_t fo;
            fb_sensors_t  fs;
            float feat32[32];
            float resid3[3] = {0.0f, 0.0f, 0.0f};
            float ratio = 0.0f, mse;
            float ai3_probs[5];
            uint8_t attr = 0u;
            int ai3_cls, k;
            int attn_peak = 0;        /* AI-3 explainability: peak-attention minute */
            furnace_state_t fst;

            fs.temp_c_q8  = res.temp_c_q8;
            fs.hum_q8     = res.humidity_q8;
            fs.mq135_adc  = res.mq135_adc;
            fs.vib_rms_mg = s_eth_vib_rms;        /* from sensor_task (200 Hz RMS) */

            for (k = 0; k < FURNACE_STEP_MIN; k++) {
                furnace_sim_advance(1, &fo);
                fb_push_ai3(fo.temp_feat);
                /* AI-19/20 per-minute rings (exact furnace_sim feature scale, /1600). */
                {
                    int w;
                    for (w = 0; w < AI19_WIN - 1; w++) s_rul_win[w] = s_rul_win[w + 1];
                    s_rul_win[AI19_WIN - 1] = fo.temp_feat[0];          /* t_current/1600 */
                    for (w = 0; w < AI20_L - 1; w++) {
                        s_tc_meas[w] = s_tc_meas[w + 1];
                        s_tc_setp[w] = s_tc_setp[w + 1];
                    }
                    s_tc_meas[AI20_L - 1] = fo.temp_feat[0];            /* measured/1600 */
                    s_tc_setp[AI20_L - 1] = fo.t_target_C / AI20_TNORM; /* setpoint/1600 */
                    if (s_aix_fill < AI19_WIN) s_aix_fill++;
                }
            }
            fb_build_ai2(&fo, &fs, feat32);
            mse     = ai2_ae_score(feat32, resid3, &attr, &ratio);
            ai3_cls = ai3_classify(fb_ai3_window(), ai3_probs);
            fst     = furnace_sim_state();

            /* AI-3 explainability: which minute of the 64-step temperature window
             * the transformer attended to most (peak last-block attention). */
            {
                static float att[AI3_SEQ_LEN];
                float mx = -1.0f; int a;
                ai3_get_attention(att);
                for (a = 0; a < AI3_SEQ_LEN; a++) if (att[a] > mx) { mx = att[a]; attn_peak = a; }
            }

            /* Adaptive Conformal (upgrade H): recalibrate AI-2 q_hat online, but
             * ONLY on confirmed-normal samples (running + AI-3 normal + no fault
             * injected) so injected anomalies are scored, not absorbed. */
            if (fst == FURN_RUNNING && ai3_cls == 0 &&
                furnace_sim_get_anomaly() == FURN_ANOM_NONE) {
                ai2_ae_adapt(mse);
            }

            /* Idle gating: AI-2/AI-3 are trained on an ACTIVE sinter. When the
             * furnace sim is idle (standby), the room-temperature furnace
             * features are out-of-distribution, so suppress the anomaly output
             * rather than raise a spurious alarm. The engines still ran (liveness
             * proven) — we just don't score "no sinter in progress" as anomalous. */
            if (fst != FURN_RUNNING) {
                ratio = 0.0f; resid3[0] = resid3[1] = resid3[2] = 0.0f; attr = 0u;
                mse = 0.0f; ai3_cls = 0;
                ai3_probs[0] = 1.0f;
                ai3_probs[1] = ai3_probs[2] = ai3_probs[3] = ai3_probs[4] = 0.0f;
            }

            {
                uint32_t q4 = (uint32_t)(mse * 16.0f);
                res.reconstruction_mse_q4 = (uint16_t)((q4 > 65535U) ? 65535U : q4);
                res.attribution_mask      = attr;
            }

            /* ---- AI-19 RUL/ETA + AI-20 thermocouple-integrity (live) ----
             * Both read furnace_sim's per-minute features (the exact training scale).
             * AI-19 feat[26] = window/1600 + hold_cum/600 + stage/5 -> minutes to done.
             * AI-20 win[2L]  = [measured/1600 xL, setpoint/1600 xL] -> sensor verdict.
             * RUL only published while the sinter is RUNNING (idle has no trajectory);
             * the TC monitor runs whenever a window exists (the sensor can fault idle). */
            {
                ai_extra_view_t ax;
                ax.rul_valid = 0u; ax.rul_min = 0;
                ax.tc_valid = 0u;  ax.tc_cls = 0; ax.tc_conf_pct = 0;
                if (s_aix_fill >= AI19_WIN) {
                    float win2l[2 * AI20_L]; float pr3[3]; int j;
                    if (fst == FURN_RUNNING) {
                        float feat[AI19_NX];
                        for (j = 0; j < AI19_WIN; j++) feat[j] = s_rul_win[j];
                        feat[AI19_WIN]     = fo.temp_feat[6];   /* hold_cum/600 */
                        feat[AI19_WIN + 1] = fo.temp_feat[7];   /* stage/5      */
                        ax.rul_min   = (int)(ai19_rul(feat) + 0.5f);
                        ax.rul_valid = 1u;
                    }
                    for (j = 0; j < AI20_L; j++) { win2l[j] = s_tc_meas[j]; win2l[AI20_L + j] = s_tc_setp[j]; }
                    if (s_tc_inject == 1) {              /* open-circuit: reading -> ~0 */
                        for (j = 0; j < AI20_L; j++) win2l[j] = 0.0f;
                    } else if (s_tc_inject == 2) {       /* erratic: large alternating HF noise */
                        for (j = 0; j < AI20_L; j++) win2l[j] += (j & 1) ? 0.028f : -0.028f;
                    }
                    ax.tc_cls = ai20_tcfault(win2l, pr3);
                    ax.tc_conf_pct = (int)(pr3[ax.tc_cls] * 100.0f + 0.5f);
                    ax.tc_valid = 1u;
                }
                taskENTER_CRITICAL();
                s_aix = ax;
                taskEXIT_CRITICAL();
            }

            /* publish to fusion_task */
            taskENTER_CRITICAL();
            s_ai.ai1_probs[0] = fo.vis_proxy[0]; s_ai.ai1_probs[1] = fo.vis_proxy[1];
            s_ai.ai1_probs[2] = fo.vis_proxy[2]; s_ai.ai1_probs[3] = fo.vis_proxy[3];
            s_ai.ai2_ratio    = ratio;
            s_ai.ai2_resid[0] = resid3[0]; s_ai.ai2_resid[1] = resid3[1]; s_ai.ai2_resid[2] = resid3[2];
            for (k = 0; k < 5; k++) s_ai.ai3_probs[k] = ai3_probs[k];
            s_ai.ai3_cls   = ai3_cls;
            s_ai.progress  = fo.progress;
            s_ai.stage     = fo.stage_id;
            s_ai.temp_c    = fo.t_current_C;
            ai3_get_attention(s_ai.ai3_attn);   /* explainability strip for HMI */
            s_ai.ready     = 1u;
            taskEXIT_CRITICAL();

            /* ---- formula-aware gas-evolution supervision (gas_safety.c) ----
             * The furnace KNOWS its charge chemistry, so we predict which
             * hazardous gas should be evolving at the current temperature and
             * cross-check the live MQ-135. This lifts the gas channel from a
             * bare threshold to a domain-informed safety supervisor (the
             * chemistry table is distilled from raw_materials.json). Demo
             * charge = garnet:Cr (Y2O3/Al2O3/Cr2O3); Cr2O3 -> Cr6+ aerosol
             * caution >900C. Logged on change only (1 Hz loop else spams). */
            {
                static const char *const k_charge[] = { "Y2O3", "Al2O3", "Cr2O3" };
                static uint32_t mq_base = 0u;   /* self-calibrating clean-air EWMA baseline */
                static int      last_sev = -1;
                static int      last_gas = -1;
                static uint8_t  gas_latched = 0u;   /* hysteresis: hazard is "sticky" */
                gas_status_t    gst;
                int             mq_rise, xc, eval_c;

                if (mq_base == 0u) mq_base = res.mq135_adc;
                else               mq_base = (mq_base * 31u + res.mq135_adc) / 32u;
                mq_rise = (res.mq135_adc > (mq_base + (mq_base >> 3))) ? 1 : 0;  /* > +12.5% */

                /* +30C hysteresis once a hazard is latched: stops the reading from
                 * chattering when the furnace dithers right on a gas onset (the
                 * 900C calcine hold sits exactly on the Cr6+ onset). Enter at the
                 * true onset, clear only ~30C below it. */
                eval_c = (int)fo.t_current_C + (gas_latched ? 30 : 0);
                gas_safety_eval(k_charge, 3, eval_c, &gst);
                gas_latched = (gst.max_sev >= 2) ? 1u : 0u;
                xc = gas_safety_crosscheck(&gst, mq_rise);

                if (fst == FURN_RUNNING &&
                    ((int)gst.max_sev != last_sev || (int)gst.top_gas != last_gas)) {
                    const char *xcs = (xc == GAS_XC_CONFIRMED)  ? "sensor-confirmed" :
                                      (xc == GAS_XC_UNEXPECTED) ? "UNEXPECTED-leak"  :
                                      (xc == GAS_XC_SENSORFLAT) ? "sensor-flat?"     : "predicted";
                    int _gn = snprintf(_emsg, sizeof(_emsg),
                        "[gas] %s @%dC sev%d %s: %s\r\n",
                        GAS_NAME[gst.top_gas], (int)fo.t_current_C, (int)gst.max_sev, xcs,
                        gst.top_reason ? gst.top_reason : "-");
                    if (_gn > 0) boot_print(_emsg);   /* safety advisory: always print */

                    /* ---- AI-5 live root-cause diagnosis (6th on-chip model) ----
                     * AI-1/2/3/4 flag THAT a run is anomalous; AI-5 names WHY +
                     * the corrective action. Assemble the 27-D diagnostic vector
                     * from the in-scope runtime values (exact layout =
                     * model/ai5_rootcause/synth_data_ai5.py) and log it. Stack-only
                     * (v27/p9 ~144B within STACK_ENV), NO persistent state — it
                     * piggybacks this deduped gas-change gate so it prints once per
                     * chemistry transition during a RUN, not every second. */
                    {
                        float v27[AI5_IN_DIM];
                        float p9[AI5_N_CLASS];
                        float conf = 0.0f;
                        int   rc, gi;
                        v27[0] = fo.vis_proxy[0]; v27[1] = fo.vis_proxy[1];   /* [0:4] AI-1 stage proxy */
                        v27[2] = fo.vis_proxy[2]; v27[3] = fo.vis_proxy[3];
                        v27[4] = (ratio < 0.0f) ? 0.0f : (ratio > 6.0f ? 6.0f : ratio);  /* [4] AI-2 ratio (clip 0..6) */
                        v27[5] = resid3[0]; v27[6] = resid3[1]; v27[7] = resid3[2];      /* [5:8] AI-2 resid temp/vib/gas */
                        v27[8]  = ai3_probs[0]; v27[9]  = ai3_probs[1]; v27[10] = ai3_probs[2];  /* [8:13] AI-3 softmax */
                        v27[11] = ai3_probs[3]; v27[12] = ai3_probs[4];
                        v27[13] = fo.t_current_C / 1600.0f;                              /* [13] temp_norm */
                        v27[14] = (fo.temp_feat[3] < 0.0f) ? -fo.temp_feat[3] : fo.temp_feat[3];  /* [14] ramp_norm |dT/dt|/20 */
                        v27[15] = (float)gst.max_sev / 3.0f;                             /* [15] gas_sev/3 */
                        for (gi = 0; gi < 7; gi++) v27[16 + gi] = 0.0f;                  /* [16:23] gas one-hot (GAS_* enum) */
                        if (gst.top_gas < 7) v27[16 + gst.top_gas] = 1.0f;
                        v27[23] = (float)(res.humidity_q8 >> 8) / 100.0f;               /* [23] humidity RH/100 */
                        v27[24] = (float)mq_rise;                                        /* [24] MQ-135 rise */
                        v27[25] = (fo.atm_code == 0) ? 1.0f : 0.0f;                      /* [25] atmosphere oxidizing (garnet air=0) */
                        v27[26] = fo.progress;                                           /* [26] progress */

                        rc = ai5_diagnose(v27, p9, &conf);
                        {
                            int _an = snprintf(_emsg, sizeof(_emsg),
                                "[ai5] root-cause=%s conf=%d%% -> ",
                                ai5_name(rc), (int)(conf * 100.0f + 0.5f));
                            if (_an > 0) boot_print(_emsg);
                            boot_print(ai5_action(rc));   /* action may exceed _emsg; print raw */
                            boot_print("\r\n");
                        }
                        /* Publish for the HMI. AI-5 is a *fault* root-cause
                         * diagnoser, so only surface a non-NORMAL cause when an
                         * anomaly is actually flagged (AI-2 ratio >= 1 = mse>=q_hat,
                         * or AI-4 risk >= bad); otherwise show NORMAL so a clean run
                         * doesn't display a scary label at an ordinary gas event. */
                        s_ai5_cls = (ratio >= 1.0f || (int)s_eth_risk >= 2)
                                        ? rc : (int)AI5_RC_NORMAL;
                        s_ai5_pct = (int)(conf * 100.0f + 0.5f);
                    }

                    last_sev = (int)gst.max_sev;
                    last_gas = (int)gst.top_gas;
                }
            }

            {
                int _n = snprintf(_emsg, sizeof(_emsg),
                    "[env] T=%d H=%u furn=%dC st=%d %s | AI2 mse_m=%d r%%=%d qh_m=%d a=%X | AI3=%s att=t%d\r\n",
                    (int)(res.temp_c_q8 >> 8), (unsigned)(res.humidity_q8 >> 8),
                    (int)fo.t_current_C, fo.stage_id,
                    (fst == FURN_RUNNING) ? "RUN" : "idle",
                    (int)(mse * 1000.0f), (int)(ratio * 100.0f),
                    (int)(ai2_ae_qhat() * 1000.0f), attr,
                    AI3_NAMES[ai3_cls], attn_peak);
                if (_n > 0) vbprint(_emsg);
            }
        }

        /* eth_task snapshot */
        s_eth_temp_q8  = (uint16_t)res.temp_c_q8;
        s_eth_humid_q8 = res.humidity_q8;
        s_eth_mq135    = res.mq135_adc;

        (void)xQueueSend(xQueue_EnvResult, &res, 0);

        /* AI-12/13 PL-stage QC: cycle the replayed real emission spectra (one per
         * dopant class) every ~3 s and run the dopant classifier + QC autoencoder.
         * Replayed (no on-board spectrometer) — same honest approach as furnace_sim. */
        {
            static uint32_t pl_tick = 0u;
            pl_refresh((int)((pl_tick++ / 3u) % 3u));
        }

        /* ---- ESP32-C3 cloud uplink (1 Hz, IIoT bridge over the PH7/PH8 I2C2 bus) --
         * Pack a compact telemetry block from the live controller snapshot, push it
         * to the ESP32-C3 (I2C slave 0x42), then read back the WiFi/cloud link state
         * and the last R1 diagnosis the bridge fetched from the XRD AI brain. The
         * three transactions reuse the bus-mutexed primitives so they interleave
         * safely with the SHT30 / ADXL345 traffic. If no ESP32 is wired the push
         * NACKs and the link snapshot just stays offline (one failed transaction per
         * loop, graceful). The R1 string lives in this BSS static, not the 1 Hz
         * stack, and is held across a single failed read so the screen doesn't flicker. */
        {
            static char     cl_r1[ESP32_DIAG_MAX + 1] = {0};
            static uint8_t  cl_r1_valid = 0u;
            static uint32_t cl_up = 0u, cl_fail = 0u;
            ctrl_snapshot_t cc;
            cloud_view_t    cv;
            uint8_t  tx[12], stt[ESP32_STATUS_LEN];
            int      tval;

            lab_ctrl_get(&cc);
            tval = cc.tc_present ? cc.probe_c : cc.meas_c;   /* prefer the real probe */
            /* tx[0]: risk in bits0-1; bit2 = edge-cloud cascade escalate flag (the
             * nano-LM was low-confidence / risk critical -> ESP32 forwards to a remote review endpoint). */
            {
                nlm_view_t _nv;
                lab_get_nlm(&_nv);
                tx[0] = (uint8_t)((cc.risk & 0x3u) | (_nv.escalate ? 0x4u : 0u));
            }
            tx[1]  = (uint8_t)(cc.state & 0x7u);
            tx[2]  = (uint8_t)(((uint16_t)tval >> 8) & 0xFFu);
            tx[3]  = (uint8_t)((uint16_t)tval & 0xFFu);
            tx[4]  = (uint8_t)(cc.u_pct & 0xFFu);
            tx[5]  = (uint8_t)(cc.seg_idx & 0xFFu);
            tx[6]  = (uint8_t)(cc.tc_fault & 0x7u);
            tx[7]  = (uint8_t)(((uint16_t)cc.cpk_x100 >> 8) & 0xFFu);
            tx[8]  = (uint8_t)((uint16_t)cc.cpk_x100 & 0xFFu);
            tx[9]  = (uint8_t)(cc.elem_pct & 0xFFu);
            tx[10] = (uint8_t)(cc.batch_id & 0xFFu);
            tx[11] = (uint8_t)(s_eth_smoke ? 1u : 0u);

            /* A bare I2C ACK is NOT proof an ESP32 is there: with no bridge wired,
             * the shared soft-I2C bus can float/false-ACK, which would make the push
             * "succeed" and the uplink counter climb while the link is really offline
             * (seen on-screen as uplinks=15 yet LINK: OFFLINE). So require a VALID
             * status read — the ESP32 must report a live link byte (1=wifi / 2=cloud)
             * AND a plausible diag length (rejects 0x00 / 0xFF bus garbage). Only then
             * is it a real uplink. This keeps the display consistent: OFFLINE <=>
             * uplinks frozen. */
            {
                uint8_t ok = 0u;
                cv.rssi_dbm = 0;
                if (esp32_push_telemetry(tx, sizeof(tx)) == 0u &&
                    esp32_read_status(stt) == 0u &&
                    (stt[0] == 1u || stt[0] == 2u) &&
                    stt[2] <= ESP32_DIAG_MAX) {
                    ok = 1u;
                    cv.link     = stt[0];
                    cv.rssi_dbm = (stt[1] == 0u) ? 0 : -(int)stt[1];
                    if (stt[2] > 0u) {
                        uint8_t db[ESP32_DIAG_MAX + 1];
                        if (esp32_read_diag(db, (uint16_t)(stt[2] + 1u)) == 0u) {
                            db[stt[2]] = '\0';
                            memcpy(cl_r1, db, (size_t)stt[2] + 1u);
                            cl_r1_valid = 1u;
                        }
                    }
                }
                if (ok) {
                    cl_up++;
                    cl_fail = 0u;
                } else {
                    if (cl_fail < 0xFFFFFFFFu) cl_fail++;
                    cv.link = 0u;       /* offline: no bridge / wifi down / bus garbage */
                }
            }
            cv.uplinks  = cl_up;
            cv.fails    = cl_fail;
            cv.r1_valid = cl_r1_valid;
            memcpy(cv.r1, cl_r1, sizeof(cv.r1));

            taskENTER_CRITICAL();
            s_ui_cloud = cv;
            taskEXIT_CRITICAL();
        }

        vTaskDelayUntil(&last, pdMS_TO_TICKS(1000));
    }
}

/* -------------------------------------------------------------------------- */
/* nlm_task — edge nano-LM generative diagnosis + online-learning risk head.  */
/* Builds a 12-slot control-token context (and a 16-D feature) from the live   */
/* sentinel state; runs the on-chip INT8 GPT to GENERATE a Chinese diagnosis;  */
/* runs/teaches the online head. Event-driven (risk change / HMI / periodic).  */
/* -------------------------------------------------------------------------- */

/* Map the live controller snapshot to the nano-LM 12-slot control-token context
 * (SLOT_ORDER: stage,temp,risk,ramp,drift,tc,gas,ae,vib,energy,host,elem). The
 * mapping is the on-device twin of gen_corpus.SLOTS — same token ids, so the LM
 * sees the distribution it was distilled on. */
static void nlm_build_ctx(const ctrl_snapshot_t *cc, short ctx[NLM_NCTX])
{
    int t = cc->tc_present ? cc->probe_c : cc->meas_c;
    int dev = cc->meas_c - cc->sp_c;
    vib_view_t vv;
    short stage, temp, risk, ramp, drift, tc, gas, ae, vib, energy, host, elem;
    lab_get_vib(&vv);

    /* Clean standby: pin a canonical idle context so the standby diagnosis is the
     * stable, situational "炉冷待机，可启动升温程序…新相烧结" every refresh, instead
     * of momentarily dropping to a plainer line when a transient flag flickers.
     * Anomaly-while-idle (e.g. the test-alarm: risk!=good, or a TC fault / smoke)
     * still flows through the computed path below so real issues surface. */
    if (cc->state == 0u && cc->risk == 0u &&
        !(cc->tc_present && cc->tc_fault) && !s_eth_smoke) {
        ctx[0]  = NLM_CTX_STAGE_IDLE; ctx[1]  = NLM_CTX_TEMP_RT;   ctx[2]  = NLM_CTX_RISK_GOOD;
        ctx[3]  = NLM_CTX_RAMP_OK;    ctx[4]  = NLM_CTX_DRIFT_OK;  ctx[5]  = NLM_CTX_TC_OK;
        ctx[6]  = NLM_CTX_GAS_OK;     ctx[7]  = NLM_CTX_AE_OK;     ctx[8]  = NLM_CTX_VIB_OK;
        ctx[9]  = NLM_CTX_ENERGY_OK;  ctx[10] = NLM_CTX_HOST_YAG;  ctx[11] = NLM_CTX_ELEM_OK;
        return;
    }

    if (cc->state == 0u)      stage = NLM_CTX_STAGE_IDLE;
    else if (cc->state == 2u) stage = NLM_CTX_STAGE_DONE;
    else {
        switch (cc->seg_idx) {                  /* garnet recipe segment -> stage */
            case 0:  stage = NLM_CTX_STAGE_PREHEAT; break;
            case 1:  stage = NLM_CTX_STAGE_CALCINE; break;
            case 2:  stage = NLM_CTX_STAGE_CALCINE; break;   /* grind during calcine hold */
            case 3:  stage = NLM_CTX_STAGE_RAMP;    break;
            case 4:  stage = NLM_CTX_STAGE_SINTER;  break;
            default: stage = NLM_CTX_STAGE_COOL;    break;
        }
    }
    if      (t < 100)   temp = NLM_CTX_TEMP_RT;
    else if (t < 300)   temp = NLM_CTX_TEMP_T200;
    else if (t < 500)   temp = NLM_CTX_TEMP_T400;
    else if (t < 700)   temp = NLM_CTX_TEMP_T600;
    else if (t < 850)   temp = NLM_CTX_TEMP_T800;
    else if (t < 950)   temp = NLM_CTX_TEMP_T900;
    else if (t < 1100)  temp = NLM_CTX_TEMP_T1000;
    else if (t < 1300)  temp = NLM_CTX_TEMP_T1200;
    else if (t < 1450)  temp = NLM_CTX_TEMP_T1400;
    else if (t <= 1530) temp = NLM_CTX_TEMP_T1500;
    else                temp = NLM_CTX_TEMP_OVER;

    risk  = (cc->risk == 0u) ? NLM_CTX_RISK_GOOD :
            (cc->risk == 1u) ? NLM_CTX_RISK_WARN :
            (cc->risk == 2u) ? NLM_CTX_RISK_BAD  : NLM_CTX_RISK_CRIT;

    ramp = NLM_CTX_RAMP_OK;
    if (cc->state == 1u) {
        if (dev > 40) ramp = NLM_CTX_RAMP_FAST;
        else if (dev < -120 && cc->u_pct >= 95) ramp = NLM_CTX_RAMP_SLOW;
    }
    drift = (!cc->spc_in_control && cc->spc_alarms > 0u) ? NLM_CTX_DRIFT_HI : NLM_CTX_DRIFT_OK;
    tc    = (cc->tc_present && (cc->tc_fault & 1u)) ? NLM_CTX_TC_OPEN :
            (cc->tc_present && (cc->tc_fault & 6u)) ? NLM_CTX_TC_SHORT : NLM_CTX_TC_OK;
    gas = NLM_CTX_GAS_OK;
    if (s_eth_smoke) {
        if (t > 900)      gas = NLM_CTX_GAS_CR6;     /* Cr3+ -> Cr6+ oxidation > 900C */
        else if (t > 700) gas = NLM_CTX_GAS_CO2;     /* carbonate decomposition       */
        else              gas = NLM_CTX_GAS_HF;      /* low-temp fluoride off-gas      */
    }
    ae     = (cc->risk >= 2u) ? NLM_CTX_AE_ANOM : NLM_CTX_AE_OK;
    vib    = (vv.running && vv.cls != 0) ? NLM_CTX_VIB_ABN :
             (vv.running)                ? NLM_CTX_VIB_RUN : NLM_CTX_VIB_OK;
    energy = (cc->u_pct >= 92) ? NLM_CTX_ENERGY_HIGH : NLM_CTX_ENERGY_OK;
    host   = NLM_CTX_HOST_YAG;                       /* active preset = garnet YAG:Cr */
    elem   = (cc->elem_pct >= 40) ? NLM_CTX_ELEM_OK :
             (cc->elem_pct >= 25) ? NLM_CTX_ELEM_WARN : NLM_CTX_ELEM_ALARM;

    ctx[0] = stage; ctx[1] = temp;   ctx[2] = risk; ctx[3] = ramp;
    ctx[4] = drift; ctx[5] = tc;     ctx[6] = gas;  ctx[7] = ae;
    ctx[8] = vib;   ctx[9] = energy; ctx[10] = host; ctx[11] = elem;
}

/* Compact 16-D feature for the online-learning risk head. */
static void online_build_feat(const ctrl_snapshot_t *cc, float f[16])
{
    int t = cc->tc_present ? cc->probe_c : cc->meas_c;
    int dev = cc->meas_c - cc->sp_c;
    vib_view_t vv;
    lab_get_vib(&vv);
    f[0]  = (float)cc->risk / 3.0f;
    f[1]  = (float)cc->u_pct / 100.0f;
    f[2]  = (float)dev / 200.0f;
    f[3]  = (cc->tc_present && cc->tc_fault) ? 1.0f : 0.0f;
    f[4]  = (cc->tc_present && (cc->tc_fault & 1u)) ? 1.0f : 0.0f;
    f[5]  = (cc->tc_present && (cc->tc_fault & 6u)) ? 1.0f : 0.0f;
    f[6]  = s_eth_smoke ? 1.0f : 0.0f;
    f[7]  = (float)t / 1600.0f;
    f[8]  = (float)cc->elem_pct / 100.0f;
    f[9]  = (float)cc->state / 3.0f;
    /* out-of-control only means something DURING a soak (state==1 run); at idle
     * spc_in_control is 0 simply because no soak is active -> not a risk signal. */
    f[10] = (cc->state == 1u && !cc->spc_in_control) ? 1.0f : 0.0f;
    f[11] = vv.running ? 1.0f : 0.0f;
    f[12] = (float)cc->cpk_x100 / 700.0f;
    f[13] = (float)cc->seg_idx / 6.0f;
    f[14] = (cc->meas_c > cc->sp_c) ? 1.0f : 0.0f;
    f[15] = 1.0f;
}

static void nlm_task(void *pv)
{
    TickType_t last = xTaskGetTickCount();
    ctrl_snapshot_t cc;
    short ctx12[NLM_NCTX];
    float f16[16], probs4[4], conf;
    char  txt[96];
    nlm_view_t nv;
    online_view_t ov;
    uint8_t  last_risk = 0xFFu;
    uint32_t gens = 0u, teaches = 0u, since = 0u;
    int acc_n = 0, acc_ok = 0, pred, teach, trig;
    (void)pv;

    memset(&nv, 0, sizeof(nv));
    memset(&ov, 0, sizeof(ov));

    for (;;) {
        lab_ctrl_get(&cc);

        /* online-learning risk head: predict; learn one SGD step on operator feedback */
        online_build_feat(&cc, f16);
        pred = online_predict(f16, probs4);
        teach = s_online_teach_req;
        if (teach >= 0) {
            s_online_teach_req = -1;
            acc_n++;
            if (pred == teach) acc_ok++;
            online_update(f16, teach);            /* on-chip forward+backward+SGD */
            teaches++;
            pred = online_predict(f16, probs4);   /* refresh prediction after learning */
        }
        ov.valid    = 1u;
        ov.pred     = pred;
        ov.conf_pct = (int)(probs4[(pred >= 0 && pred < 4) ? pred : 0] * 100.0f);
        ov.teaches  = teaches;
        ov.acc_pct  = acc_n ? (acc_ok * 100 / acc_n) : 0;
        taskENTER_CRITICAL();
        s_ui_online = ov;
        taskEXIT_CRITICAL();

        /* HMI: switch the active generative LM across the size curve (x1p9 -> bank) */
        if (s_lm_cycle_req) {
            s_lm_cycle_req = 0;
            s_lm_active = (s_lm_active + 1) % lm_roster_count();
            s_nlm_req = 1;                      /* force an immediate regeneration */
            {
                char lb[96];
                int n = snprintf(lb, sizeof(lb), "[lm] active=%s (%s, ppl %d.%02d, ~%d.%dx)\r\n",
                                 lm_roster_tag_s(s_lm_active), lm_roster_label_s(s_lm_active),
                                 lm_roster_ppl_x100(s_lm_active) / 100, lm_roster_ppl_x100(s_lm_active) % 100,
                                 lm_roster_lat_x10(s_lm_active) / 10, lm_roster_lat_x10(s_lm_active) % 10);
                if (n > 0) boot_print(lb);
            }
        }

        /* edge nano-LM: regenerate on risk change, HMI request, or ~16s into a run */
        since++;
        trig = (cc.risk != last_risk) || s_nlm_req || (cc.state == 1u && since >= 8u);
        if (trig) {
            int active = s_lm_active;
            last_risk = cc.risk;
            s_nlm_req = 0;
            since = 0u;
            nlm_build_ctx(&cc, ctx12);
            conf = 0.0f;
            if (active == 0) {
                nanolm_generate(ctx12, txt, sizeof(txt), &conf);     /* internal x1p9 (1.8M) */
            } else if (bank_load(active - 1) == 0) {                 /* SPI-flash bank model */
                bank_generate(ctx12, txt, sizeof(txt), &conf);
            } else {
                strncpy(txt, "(LM bank load failed - SPI flash)", sizeof(txt) - 1);
                txt[sizeof(txt) - 1] = '\0'; conf = 0.0f;
            }
            nv.valid    = 1u;
            strncpy(nv.text, txt, sizeof(nv.text) - 1);
            nv.text[sizeof(nv.text) - 1] = '\0';
            nv.conf_pct = (int)(conf * 100.0f);
            nv.escalate = (conf < 0.80f || cc.risk >= 3u) ? 1u : 0u;
            nv.gens     = ++gens;
            taskENTER_CRITICAL();
            s_ui_nlm = nv;
            taskEXIT_CRITICAL();
            {
                char sb[160];
                int n = snprintf(sb, sizeof(sb), "[nlm] %s (conf=%d%% esc=%u)\r\n",
                                 nv.text, nv.conf_pct, (unsigned)nv.escalate);
                if (n > 0) boot_print(sb);
            }
        }

        vTaskDelayUntil(&last, pdMS_TO_TICKS(2000));   /* 0.5 Hz; ~1s gen only on trigger */
    }
}

/* -------------------------------------------------------------------------- */
/* cluster_task — 7-expert edge-LLM cluster, swap-loaded from SPI flash.       */
/* A state router picks the role-expert that fits the current furnace context  */
/* (mirrors a server-class cluster's division of labour); the expert is streamed */
/* from SPI flash into SDRAM (~0.3s) and generates one role sentence. The HMI  */
/* "NEXT" button manually cycles experts to demo the swap-load.                */
/* -------------------------------------------------------------------------- */
static int cluster_route(const ctrl_snapshot_t *cc)
{
    if (cc->risk >= 2u)     return 0;   /* anomaly/critical     -> E1 diagnosis      */
    if (cc->elem_pct <= 40) return 6;   /* heating element worn -> E7 maintenance     */
    if (cc->state == 2u)    return 3;   /* batch done           -> E4 QC verdict      */
    if (cc->state == 0u)    return 4;   /* idle/standby         -> E5 operator brief  */
    if (cc->u_pct >= 92)    return 2;   /* running, high duty   -> E3 energy/carbon   */
    return 1;                           /* running normal       -> E2 recipe advice   */
    /* E6 chemistry (idx 5) has no furnace-state trigger - it is a design/recipe   */
    /* specialist the operator pulls up via the HMI "NEXT EXPERT" cycle.           */
}

static void cluster_task(void *pv)
{
    TickType_t last = xTaskGetTickCount();
    ctrl_snapshot_t cc;
    short ctx12[NLM_NCTX];
    char  txt[96];
    cluster_view_t cv;
    float conf;
    int   want, cur = -1, manual = -1, rc;
    uint32_t gens = 0u, since = 0u;
    (void)pv;

    memset(&cv, 0, sizeof(cv));
    cv.expert = -1;
    cv.provisioned = (uint8_t)flash_cluster_present();
    taskENTER_CRITICAL(); s_ui_cluster = cv; taskEXIT_CRITICAL();
    if (!cv.provisioned) {
        boot_print("[cluster] no SPI-flash image; task idle (provision then reboot)\r\n");
        for (;;) vTaskDelay(pdMS_TO_TICKS(5000));
    }

    for (;;) {
        lab_ctrl_get(&cc);
        since++;
        if (s_cluster_next_req) {
            s_cluster_next_req = 0;
            manual = (cur < 0) ? 0 : (cur + 1) % NLM_CL_NEXPERT;
        }
        want = (manual >= 0) ? manual : cluster_route(&cc);

        if (want != cur || manual >= 0 || since >= 10u) {
            TickType_t t0 = xTaskGetTickCount();
            rc = cluster_load_expert(want);
            cv.swap_ms = (int)((xTaskGetTickCount() - t0) * portTICK_PERIOD_MS);
            since = 0u; manual = -1;
            if (rc == 0) {
                cur = want;
                nlm_build_ctx(&cc, ctx12);
                conf = 0.0f;
                cluster_generate(ctx12, txt, sizeof(txt), &conf);
                cv.valid = 1u; cv.provisioned = 1u; cv.expert = cur;
                strncpy(cv.role, nlm_cl_role[cur], sizeof(cv.role) - 1);
                cv.role[sizeof(cv.role) - 1] = '\0';
                strncpy(cv.text, txt, sizeof(cv.text) - 1);
                cv.text[sizeof(cv.text) - 1] = '\0';
                cv.conf_pct = (int)(conf * 100.0f);
                cv.gens = ++gens;
                taskENTER_CRITICAL(); s_ui_cluster = cv; taskEXIT_CRITICAL();
                {
                    char sb[160];
                    int n = snprintf(sb, sizeof(sb), "[cluster] E%d %s swap=%dms conf=%d%% %s\r\n",
                                     cur + 1, cv.role, cv.swap_ms, cv.conf_pct, cv.text);
                    if (n > 0) boot_print(sb);
                }
            }
        }
        vTaskDelayUntil(&last, pdMS_TO_TICKS(2000));   /* 0.5 Hz */
    }
}

/* Consecutive CRITICAL fusion verdicts (each cycle ~1.5s) required before the
 * AI-4 verdict reaches the controller as SEVERE and trips the safety abort.
 * Debounces nuisance trips from one-off transients; a real/sustained fault still
 * aborts after ~AI_CRIT_PERSIST*1.5s. */
#define AI_CRIT_PERSIST  3u

/* -------------------------------------------------------------------------- */
/* fusion_task - waits on 3 input queues, runs AI-4, posts to xQueue_RiskAlert*/
/* -------------------------------------------------------------------------- */
static void fusion_task(void *pv)
{
    (void)pv;
    lab_risk_level_t prev_risk = RISK_NORMAL;
    uint8_t prev_abstain = 0u;
    char fmsg[100];

    for (;;) {
        ai_state_t snap;
        uint8_t    ready;

        s_wdg_hb[WDG_FUSION]++;          /* watchdog heartbeat */

        /* snapshot the shared AI state without tearing */
        taskENTER_CRITICAL();
        snap  = s_ai;
        ready = s_ai.ready;
        taskEXIT_CRITICAL();

        if (ready) {
            float feat16[16];
            float probs4[4];
            float ai4_conf = 1.0f;
            uint8_t ai4_abstain = 0u;
            int   cls;
            lab_risk_level_t risk;

            /* Standby gate: only fuse/alarm during an active sinter. Idle =
             * NORMAL (the furnace isn't running, so there is no batch to judge). */
            if (furnace_sim_state() != FURN_RUNNING) {
                actuator_set_risk(RISK_NORMAL);
                s_eth_risk = 0u;
                prev_risk  = RISK_NORMAL;
                vbprint("[fusion] standby (furnace idle; say 'kai shi shao jie')\r\n");
                vTaskDelay(pdMS_TO_TICKS(1500));
                continue;
            }

            fb_build_ai4(snap.ai1_probs, snap.ai2_ratio, snap.ai2_resid,
                         snap.ai3_probs, snap.progress, feat16);
            cls  = ai4_fuse_calibrated(feat16, probs4, &ai4_conf, &ai4_abstain);
            risk = (lab_risk_level_t)cls;     /* 0 good..3 critical maps to RISK_* */
            s_inf_count++;                    /* safety-core inference (long-run panel) */

            /* Selective abstention (ai4_calib.h): when the fusion model is not
             * confident (top prob < tau), do NOT auto-clear the batch. Floor an
             * uncertain GOOD up to WARNING (operator review); never downgrade
             * a genuine bad/critical, so abstention can neither hide a real
             * shutdown nor raise a spurious one. */
            if (ai4_abstain && risk == RISK_NORMAL) risk = RISK_WARNING;
            if (ai4_abstain && !prev_abstain) {
                boot_print("[fusion] AI uncertain -> flag for operator review\r\n");
            }
            prev_abstain = ai4_abstain;

            /* Auxiliary MOTOR (stirring/vibration test rig for the AI-10 PdM demo)
             * running injects EMI + mechanical vibration that corrupts the env /
             * vision sensors -> AI-4 would false-trip CRITICAL and abort the sinter
             * (observed as the burn-in progress snapping to 0, AIalarms spiking into
             * the thousands). Treat the AI verdict as UNRELIABLE while the motor runs
             * AND for a short settle window after it stops -- otherwise the trailing
             * contaminated readings fire a spurious CRITICAL the instant the motor is
             * switched off (seen as the batch faulting right on MOTOR-off). Cap risk
             * at WARNING during this window so a deliberate motor/AI-10 demo cannot
             * shut the furnace down. The genuine AI safety abort still fires normally
             * with the motor off and settled (its real use case). */
            {
                static TickType_t motor_active_tick = 0;
                int motor_on = (motor_state() == MOTOR_FORWARD ||
                                motor_state() == MOTOR_REVERSE);
                if (motor_on) motor_active_tick = xTaskGetTickCount();
                if ((motor_on ||
                     (motor_active_tick != 0 &&
                      (xTaskGetTickCount() - motor_active_tick) < pdMS_TO_TICKS(3000u)))
                    && risk > RISK_WARNING) {
                    if (prev_risk <= RISK_WARNING) {
                        boot_print("[fusion] motor active/settling -> AI gated (EMI), no auto-abort\r\n");
                    }
                    risk = RISK_WARNING;
                }
            }

            /* AI-critical TRIP DEBOUNCE (anti-nuisance-trip): furnace_ctrl aborts on
             * a SINGLE risk>=SEVERE, so one transient false CRITICAL (a sensor blip /
             * a single jittery AI-4 verdict) would needlessly kill the batch. Require
             * the CRITICAL verdict to persist AI_CRIT_PERSIST consecutive fusion
             * cycles (~1.5s each) before it reaches the controller as SEVERE. A
             * genuine fault and the deliberate "test alarm" demo both hold critical
             * across cycles and still trip (~AI_CRIT_PERSIST*1.5s later, imperceptible
             * in a demo); a one-off transient is reported as ANOMALY (logged + HMI)
             * but does NOT abort. Hard physical limits (over-temp, TC fault) stay
             * instant in furnace_ctrl -- only the AI verdict is debounced. */
            {
                static uint8_t crit_run = 0u;
                if (risk >= RISK_SEVERE) {
                    if (crit_run < 255u) crit_run++;
                    if (crit_run < AI_CRIT_PERSIST) {
                        if (crit_run == 1u) {
                            boot_print("[fusion] CRITICAL seen -> debouncing "
                                       "(must persist to trip; transient won't abort)\r\n");
                        }
                        risk = RISK_ANOMALY;        /* hold one notch below the trip */
                    }
                } else {
                    crit_run = 0u;                  /* reset on any non-critical verdict */
                }
            }

            actuator_set_risk(risk);
            s_eth_risk = (uint8_t)risk;

            {
                risk_alert_t alert;
                alert.timestamp_ms  = (uint32_t)(xTaskGetTickCount() *
                                                 (1000U / configTICK_RATE_HZ));
                alert.risk          = risk;
                alert.trigger_source = (uint8_t)(((snap.ai2_ratio >= 1.0f) ? 0x04u : 0u) |
                                                 ((snap.ai3_cls   != 0)    ? 0x02u : 0u) |
                                                 (ai4_abstain               ? 0x01u : 0u));
                alert.doa_angle_deg = (int16_t)0x8000;   /* none (mic disabled) */
                (void)xQueueSend(xQueue_RiskAlert, &alert, 0);
            }

            /* proactive voice alarm on rising edge into SEVERE (critical) */
            if (risk >= RISK_SEVERE && prev_risk < RISK_SEVERE) {
#if CI1302_VOICE_ENABLED
                ci1302_play(CI1302_CMD_EMERGENCY);   /* CI1302 announces 已紧急停止 */
#endif
            }
            prev_risk = risk;

            {
                int _n = snprintf(fmsg, sizeof(fmsg),
                    "[fusion] AI4 risk=%d(%s) conf%%=%d%s | ai3=%s ratio%%=%d\r\n",
                    cls, AI4_NAMES[cls], (int)(ai4_conf * 100.0f),
                    ai4_abstain ? " ABSTAIN" : "",
                    AI3_NAMES[snap.ai3_cls], (int)(snap.ai2_ratio * 100.0f));
                if (_n > 0) vbprint(fmsg);
            }
        } else {
            vbprint("[fusion] waiting for AI pipeline (env_task)...\r\n");
        }

        vTaskDelay(pdMS_TO_TICKS(1500));
    }
}

/* -------------------------------------------------------------------------- */
/* ui_task - ST7796 LCD init + GT911 touch + LVGL 8.3 dashboard               */
/*                                                                            */
/* Init order (all in this task to serialise hardware access):                */
/*   1. st7796_init()       — EXMC NOR/SRAM region 1 + LCD register seq       */
/*   2. st7796_set_backlight(1) — turn on display                             */
/*   3. gt911_init()        — soft-I2C touch controller reset + address sel   */
/*   4. lv_init()           — LVGL kernel (uses FreeRTOS tick via lv_conf.h)  */
/*   5. lv_port_disp_init() — register ST7796 flush driver (SDRAM FB)         */
/*   6. lv_port_indev_init()— register GT911 pointer driver                   */
/*   7. ui_screen_init()    — create dashboard widgets                        */
/*                                                                            */
/* Main loop (33 ms = ~30 fps):                                               */
/*   - Drain xQueue_RiskAlert → ui_screen_update_risk()                       */
/*   - Check xSem_VoiceCmd   → ui_screen_set_voice_status()                   */
/*   - lv_timer_handler()    — LVGL rendering + input polling                 */
/*   - uptime counter every 1 second                                          */
/* -------------------------------------------------------------------------- */
static void ui_task(void *pv)
{
    (void)pv;

    /* 2026-05-17 reliability: 等其他 hw init 完成 + SDRAM 完全稳定 + actuator
     * self-test 走完 (~2s) 之后再 init LCD. zero-fill 解决了 SDRAM 残留, 这里
     * 再加 1s settle 保证 hw 全静默. 之前"偶尔半花"原因之一: ui_task 跟其他
     * task 同时启动时, sensor_task 200Hz I2C / env_task SHT30 read 等可能跟
     * LCD bit-bang 的 SDRAM read 在 EXMC bus 上竞争 → 首帧 render 时序漂. */
    vTaskDelay(pdMS_TO_TICKS(1000U));

    /* --- hardware init --- */
    st7796_init();
    st7796_set_backlight(1U);
    /* 2026-05-30 ★ GT911 触摸重新启用: 旧"PB8 跟相机 D6 冲突"是退役的 8080
     * NT35510 屏的脚位; 官方 RGB 板触摸在 PD5(SCL)/PD7(SDA)/PH13(RST)/PH15(INT),
     * 跟相机/SDRAM/RGB 全不撞, 且 PH13 在 CI1302 迁到 PC10/PC11 后已空闲 →
     * 相机和触摸可共存. gt911.c 已按官方脚位重写 (含自带 RST 脉冲). */
    {
        uint8_t trc = gt911_init();
        uint8_t id5[4], id14[4];
        char    gb[80];
        int     gn;
        boot_print(trc == 0U ? "[touch] GT911 present (probe OK)\r\n"
                             : "[touch] GT911 NOT found (check PD5/PD7/PH13/PH15)\r\n");
        /* raw product-ID bytes: 00=NACK, FF=bus stuck high, "911"=alive */
        gt911_debug_ids(id5, id14);
        gn = snprintf(gb, sizeof(gb),
            "[touch] PID@5D=%02X %02X %02X %02X  @14=%02X %02X %02X %02X\r\n",
            id5[0], id5[1], id5[2], id5[3], id14[0], id14[1], id14[2], id14[3]);
        if (gn > 0) boot_print(gb);
    }

    /* --- LVGL init --- */
    lv_init();
    lv_port_disp_init();
    lv_port_indev_init();   /* register GT911 pointer device (touch enabled) */
    ui_screen_init();

    /* 强制立即渲染两次. LVGL 第一帧偶尔不完整 render (dirty area 算错 / TLSF
     * 第一次分配 widget object 跟 render 同步问题), 第二次 render 一定能把
     * framebuffer 完整覆盖. 中间加 50ms 让 LVGL 内部状态稳定.               */
    lv_obj_invalidate(lv_scr_act());
    lv_refr_now(NULL);
    vTaskDelay(pdMS_TO_TICKS(50U));
    lv_obj_invalidate(lv_scr_act());
    lv_refr_now(NULL);

    boot_print("[ui] LCD+LVGL ready\r\n");

    TickType_t last_tick   = xTaskGetTickCount();
    TickType_t last_uptime = last_tick;

    /* Touch coordinate trace: one line per tap (rising edge) so it stays calm.
     * Shows where taps land relative to the buttons. Remove once confirmed. */
    extern volatile int     g_touch_raw_x, g_touch_raw_y;
    extern volatile uint8_t g_touch_down;
    uint8_t prev_touch_down = 0U;

    /* Single-loop design (proven flicker-free): lv_timer_handler() below both
     * renders the screen (full_refresh, ~33 ms) AND polls the GT911 via the
     * indev read_cb. A dedicated high-priority touch sampler was tried to make
     * quick nav-tab taps land more reliably, but it starved this loop (all LVGL
     * event dispatch happens here) -> worse flicker + less responsive buttons,
     * so it was reverted. The correct improvement is interrupt-driven touch
     * (GT911 INT on PH15), kept as a separate change. */
    for (;;) {
        /* --- consume pending risk alerts (non-blocking) --- */
        risk_alert_t alert;
        while (xQueueReceive(xQueue_RiskAlert, &alert, 0) == pdTRUE) {
            ui_screen_update_risk(&alert);
            actuator_set_risk(alert.risk);   /* drive buzzer/fan/vib/LEDs */
        }

        /* --- drain env / vib queues (kept empty so they never overflow; the
         *     dashboard now reads the richer AI state via the snapshot below) - */
        env_result_t env;
        while (xQueueReceive(xQueue_EnvResult, &env, 0) == pdTRUE) { (void)env; }
        vibration_result_t vib;
        while (xQueueReceive(xQueue_VibResult, &vib, 0) == pdTRUE) { (void)vib; }

        /* --- live AI panel (4-model verdicts + confidence + AE attribution) --- */
        {
            static lab_ai_snapshot_t snap;   /* static: keep the 64-pt attention array off the ui_task stack */
            lab_sentinel_get_ai(&snap);
            ui_screen_update_ai(&snap);
        }

        /* --- voice-driven screen navigation (set by voice_dispatch) --- */
        if (s_nav_req >= 0) {
            ui_screen_set_nav(s_nav_req);
            s_nav_req = -1;
        }

        /* --- refresh the active screen from the controller snapshot
         *     (status bar temp/batch + Home + Trend SPC chart) --- */
        ui_screen_tick();

        /* --- voice command notification --- */
        if (xSemaphoreTake(xSem_VoiceCmd, 0) == pdTRUE) {
            ui_screen_set_voice_status("CMD");
        }

        /* --- LVGL render pass (also polls GT911 via the indev read_cb) --- */
        lv_timer_handler();

        /* --- touch coordinate trace: one line per tap (rising edge) --- */
        if (g_touch_down && !prev_touch_down) {
            char tb[48];
            int  n = snprintf(tb, sizeof(tb), "[touch] x=%d y=%d\r\n",
                              g_touch_raw_x, g_touch_raw_y);
            if (n > 0) boot_print(tb);
        }
        prev_touch_down = g_touch_down;


        /* --- uptime counter (every ~1 s, catch-up loop) --- */
        /* 2026-05-28: 改 if→while. 双缓冲 + VBlank busy-wait 后 ui_task 单次
         * loop 可能远超 1s, 用 if 时一个 loop 内只 +1 但 wall clock 已走 N s,
         * 显示秒数变慢 N×. while 一次 catch-up 多个 1s 槽位, 跟 wall clock 同步. */
        TickType_t now = xTaskGetTickCount();
        while ((now - last_uptime) >= pdMS_TO_TICKS(1000U)) {
            last_uptime += pdMS_TO_TICKS(1000U);
            ui_screen_tick_uptime();
            {   /* push the CONTROLLER temperature into the Home 升温曲线 (1 Hz) */
                ctrl_snapshot_t cs;
                lab_ctrl_get(&cs);
                ui_screen_push_temp((float)cs.meas_c);
            }
        }

        /* 33 ms render cadence (flicker-free, proven). lv_timer_handler above
         * also polls the GT911 via the indev read_cb on this same cadence. */
        vTaskDelayUntil(&last_tick, pdMS_TO_TICKS(33U));
    }
}

/* -------------------------------------------------------------------------- */
/* voice_task — CI1302 UART 8-byte frame parser + command dispatcher          */
/*                                                                            */
/* Chipintelli ICS 平台默认 8 字节协议:                                       */
/*   [A5][FA][SEQ=00][TYPE][CMD][DATA=00][CHK][FB]                            */
/*                                                                            */
/*   TYPE=0x81 上行: CI1302 → MCU (识别到命令或唤醒)                          */
/*   TYPE=0x82 下行: MCU → CI1302 (主动让模块播 TTS, voice_task 不解析此方向) */
/*   CHK = (0x20 + CMD) & 0xFF  当 TYPE=0x81                                  */
/*                                                                            */
/* CMD=0x01 是唤醒 "你好小亚" (CI1302 内部自动播 "我在", MCU 只点亮 UI).      */
/* CMD=0x02-0x11 是 16 个业务命令, dispatch 到 voice_dispatch.                */
/* CI1302 自动播报识别到的命令对应的 TTS, MCU 不需要主动调 ci1302_play().    */
/* -------------------------------------------------------------------------- */
typedef enum {
    VS_IDLE = 0,    /* waiting for 0xA5                                       */
    VS_HDR1,        /* waiting for 0xFA                                       */
    VS_SEQ,         /* reading SEQ (byte 2)                                   */
    VS_TYPE,        /* reading TYPE (byte 3) — 0x81 / 0x82                    */
    VS_CMD,         /* reading CMD_ID (byte 4)                                */
    VS_DATA,        /* reading DATA (byte 5)                                  */
    VS_CHK,         /* reading CHK (byte 6) — validated against type+cmd      */
    VS_TAIL         /* waiting for 0xFB (byte 7), then fire                   */
} voice_parse_state_t;

static void voice_dispatch(uint8_t cmd_id);

static void voice_task(void *pv)
{
    (void)pv;

    voice_parse_state_t state = VS_IDLE;
    uint8_t  type   = 0U;
    uint8_t  cmd    = 0U;
    uint8_t  chk_rx = 0U;
    uint8_t  b;

    /* Boot announcement: give the module 500 ms to finish its own init.
     * Note: 8 字节协议下没有 "ONLINE" 概念 — CI1302 识别什么就播什么.
     * 上电不主动播报, 等用户念 "你好小亚" 唤醒. */
    vTaskDelay(pdMS_TO_TICKS(500));
    boot_print("[voice] CI1302 ready (8-byte protocol)\r\n");

    for (;;) {
        if (xQueueReceive(xQueue_CI1302Rx, &b, pdMS_TO_TICKS(10)) != pdTRUE) {
            continue;
        }

        switch (state) {
            case VS_IDLE:
                state = (b == CI1302_HDR0) ? VS_HDR1 : VS_IDLE;
                break;

            case VS_HDR1:
                state = (b == CI1302_HDR1) ? VS_SEQ : VS_IDLE;
                break;

            case VS_SEQ:
                /* SEQ byte — 平台目前固定 0x00, 不校验直接吃掉 */
                state = VS_TYPE;
                break;

            case VS_TYPE:
                type  = b;
                state = (type == CI1302_TYPE_RECOG) ? VS_CMD : VS_IDLE;
                /* TYPE=0x82 是我们自己发出去的下行回响, 直接丢弃 */
                break;

            case VS_CMD:
                cmd   = b;
                state = VS_DATA;
                break;

            case VS_DATA:
                /* DATA byte — 平台目前固定 0x00, 不校验 */
                state = VS_CHK;
                break;

            case VS_CHK:
                chk_rx = b;
                state  = VS_TAIL;
                break;

            case VS_TAIL:
                if (b == CI1302_TAIL) {
                    /* 8 字节帧完整, 校验 checksum */
                    uint8_t chk_calc = ci1302_checksum_recog(cmd);
                    if (chk_rx == chk_calc) {
                        char _dbg[40];
                        int _n = snprintf(_dbg, sizeof(_dbg),
                                          "[voice] frame cmd=%02X chk=OK\r\n",
                                          cmd);
                        if (_n > 0) boot_print(_dbg);

                        if (cmd == CI1302_CMD_WAKE) {
                            /* 唤醒事件: CI1302 内部已播 "我在", MCU 点亮 UI */
                            xSemaphoreGive(xSem_VoiceCmd);
                        } else {
                            /* 业务命令: dispatch (CI1302 自动播报对应 TTS) */
                            xSemaphoreGive(xSem_VoiceCmd);
                            voice_dispatch(cmd);
                        }
                    } else {
                        char _dbg[60];
                        int _n = snprintf(_dbg, sizeof(_dbg),
                                          "[voice] frame cmd=%02X CHK FAIL rx=%02X want=%02X\r\n",
                                          cmd, chk_rx, chk_calc);
                        if (_n > 0) boot_print(_dbg);
                    }
                }
                state = VS_IDLE;
                break;

            default:
                state = VS_IDLE;
                break;
        }
    }
}

/* Dispatch a recognised CI1302 command word to actions.
 *
 * 2026-05-28 ★ 8 字节协议简化版: CI1302 识别命令后自动播报对应 TTS
 * (ICS 平台已把 "播报语句" 跟每个命令绑定), MCU 不需要主动调 TTS API,
 * 只做 actuator/motor/relay 等执行器动作. 想主动播报用 ci1302_play().
 */
static void voice_dispatch(uint8_t cmd_id)
{
    switch (cmd_id) {

        /* === 控制类: 烧结流程 ============================================ */
        case CI1302_CMD_START:
            /* 开始烧结: 启动炉温仿真 (AI-2/AI-3/AI-4 闭环随之活跃) + 搅拌电机.
             * fb_reset 清 AI-3 时序环 + 重新校准 room/gas 基线 → 从干净窗口起跑. */
            /* 开始烧结 = 仿真 + 控制器 + 通风风扇 (不含电机, 电机走 FAN/MOTOR 命令) */
            furnace_sim_set_anomaly(FURN_ANOM_NONE);
            fb_reset();
            furnace_sim_start();
            s_batch_fan = 1u; relay_on();   /* ventilation fan ON */
            s_ctrl_cmd = CTRL_CMD_START;   /* also run the closed-loop controller */
            break;

        case CI1302_CMD_STOP:
            /* 结束烧结: 停炉温仿真 + 停风扇 + 停控制器 (电机独立, 不在此停) */
            furnace_sim_stop();
            s_batch_fan = 0u; relay_off();
            s_ctrl_cmd = CTRL_CMD_ABORT;
            break;

        case CI1302_CMD_PAUSE:
        case CI1302_CMD_RESUME:
            /* TTS 由 CI1302 自动播, UI 暗淡靠 xSem_VoiceCmd 通知 ui_task */
            break;

        case CI1302_CMD_EMERGENCY:
            /* 紧急停止: 停炉 + motor 制动 + 继电器关 + 全 LED + 控制器安全态
             * (TTS "已紧急停止" 由 CI1302 自动播报) */
            furnace_sim_stop();
            motor_set(MOTOR_BRAKE);
            s_batch_fan = 0u; relay_off();
            actuator_set_risk(RISK_SEVERE);
            s_ctrl_cmd = CTRL_CMD_ABORT;
            break;

        case CI1302_CMD_ACK_ALARM:
            /* 复位报警: 清注入异常 + 回到 NORMAL (AI-4 随实际特征自然回落) */
            furnace_sim_set_anomaly(FURN_ANOM_NONE);
            actuator_set_risk(RISK_NORMAL);
            break;

        /* === 查询类: 语音导航跳屏 (TTS "数据已显示 请查看屏幕" 由 CI1302 自动播)
         * voice_task 只置 s_nav_req, 真正的 LVGL 切屏由 ui_task 消费 (线程安全).
         * 屏序: 0 Home 1 Recipe 2 Trend 3 AI 4 Quality 5 System              */
        case CI1302_CMD_QUERY_TEMP:    s_nav_req = 2; break;  /* 查询温度 → 趋势 SPC */
        case CI1302_CMD_QUERY_GAS:                            /* 查询气体 → AI 诊断  */
        case CI1302_CMD_QUERY_HUMI:    s_nav_req = 3; break;  /* 查询湿度 → AI 诊断  */
        case CI1302_CMD_QUERY_STATUS:  s_nav_req = 5; break;  /* 查询状态 → 系统     */

        /* === 执行器类 ====================================================== */
        case CI1302_CMD_FAN_ON:
            motor_set(MOTOR_FORWARD);
            break;

        case CI1302_CMD_FAN_OFF:
            motor_set(MOTOR_STOP);
            break;

        case CI1302_CMD_VENT_ON:
            relay_on();
            break;

        case CI1302_CMD_VENT_OFF:
            relay_off();
            break;

        /* === 调试类: 答辩 demo ============================================ */
        case CI1302_CMD_TEST_LED:
            actuator_self_test();          /* LED1→2→3 sweep, ~750ms (蜂鸣器/震动已移除) */
            break;

        case CI1302_CMD_TEST_ALARM: {
            /* 测试报警 = 演示完整 AI→报警链: 每次说一次循环注入一种烧结异常
             * (升温过快→不达温→温漂→正常), 由 AI-3 识别 + AI-2 打分 + AI-4 升级
             * 风险等级, fusion_task 自动点 LED + 让 CI1302 播 "已紧急停止".
             * 不是硬编码闪灯 — 是真·闭环演示. 需先 "开始烧结" 让炉温仿真运行. */
            static uint8_t anom_cycle = 0u;
            furnace_anomaly_t a;
            anom_cycle = (uint8_t)((anom_cycle + 1u) & 0x03u);
            switch (anom_cycle) {
                case 1:  a = FURN_ANOM_FAST_RAMP;  break;
                case 2:  a = FURN_ANOM_UNDERTEMP;  break;
                case 3:  a = FURN_ANOM_TEMP_DRIFT; break;
                default: a = FURN_ANOM_NONE;       break;
            }
            furnace_sim_set_anomaly(a);
            actuator_led_set(2U, 1U); vTaskDelay(pdMS_TO_TICKS(150));
            actuator_led_set(2U, 0U);
            break;
        }

        default:
            break;   /* unrecognised command — ignore silently */
    }
}

/* -------------------------------------------------------------------------- */
/* eth_task - lwIP init + Modbus TCP server + periodic HTTP POST              */
/* -------------------------------------------------------------------------- */
static void eth_task(void *pv)
{
    (void)pv;

    struct netif gnetif;
    ip4_addr_t   ip_addr, netmask, gw;
    char         msg[64];
    uint32_t     link_retries;

    /* ── lwIP init: starts tcpip thread internally ── */
    tcpip_init(NULL, NULL);

    IP4_ADDR(&ip_addr,
             CIMC_IP_ADDR0, CIMC_IP_ADDR1, CIMC_IP_ADDR2, CIMC_IP_ADDR3);
    IP4_ADDR(&netmask,
             CIMC_NETMASK0, CIMC_NETMASK1, CIMC_NETMASK2, CIMC_NETMASK3);
    IP4_ADDR(&gw,
             CIMC_GW_ADDR0, CIMC_GW_ADDR1, CIMC_GW_ADDR2, CIMC_GW_ADDR3);

    netif_add(&gnetif, &ip_addr, &netmask, &gw,
              NULL, ethernetif_init, tcpip_input);
    netif_set_default(&gnetif);
    netif_set_up(&gnetif);

    /* ── Wait for PHY link (up to 10 s) ── */
    link_retries = 100U;
    while (!ethernetif_link_up() && link_retries--) {
        vTaskDelay(pdMS_TO_TICKS(100U));
    }
    if (link_retries == 0U) {
        boot_print("[eth] PHY link timeout — Ethernet offline\r\n");
    } else {
        int n = snprintf(msg, sizeof(msg),
                         "[eth] link up — %u.%u.%u.%u\r\n",
                         CIMC_IP_ADDR0, CIMC_IP_ADDR1,
                         CIMC_IP_ADDR2, CIMC_IP_ADDR3);
        if (n > 0) boot_print(msg);
    }

    /* ── Start Modbus TCP server (its own task, port 502) ── */
    modbus_tcp_server_start();
    boot_print("[eth] Modbus TCP server started on port 502\r\n");

    /* ── Main loop: drive RX + periodic HTTP POST (5 s) ── */
    TickType_t last_post = xTaskGetTickCount();

    for (;;) {
        /* Feed received frames to lwIP (non-blocking poll) */
        ethernetif_input(&gnetif);

        /* Update Modbus holding registers from latest sensor snapshot */
        modbus_tcp_update_regs(s_eth_temp_q8, s_eth_humid_q8, s_eth_mq135,
                               s_eth_vib_rms, s_eth_risk, s_eth_smoke);

        /* HTTP POST every 5 seconds if link is up */
        if ((xTaskGetTickCount() - last_post) >= pdMS_TO_TICKS(5000U)) {
            last_post = xTaskGetTickCount();
            if (ethernetif_link_up()) {
                uint8_t rc = http_client_post(
                    s_eth_temp_q8, s_eth_humid_q8, s_eth_mq135,
                    s_eth_vib_rms, s_eth_risk,     s_eth_smoke);
                if (rc == 0U) {
                    boot_print("[eth] POST OK\r\n");
                } else {
                    boot_print("[eth] POST fail\r\n");
                }
            }
        }

        vTaskDelay(pdMS_TO_TICKS(10U));   /* 100 Hz RX poll ceiling */
    }
}

/* -------------------------------------------------------------------------- */
/* doa_task - INMP441 sound-level monitor, 1 Hz report.                       */
/*                                                                            */
/* Reads RMS of 256 samples (~16 ms of audio) every second.                   */
/* Prints loudness to UART. Triggers RISK_ANOMALY if RMS > threshold.        */
/* DOA cross-correlation can be added in Phase 4 (second mic needed).        */
/* -------------------------------------------------------------------------- */
#define SOUND_ALARM_RMS   8000U   /* adjust via measurement; ~60 dB SPL proxy */

static void doa_task(void *pv)
{
    (void)pv;
    char msg[48];
    TickType_t last = xTaskGetTickCount();

    for (;;) {
        uint16_t rms = inmp441_rms_256();
        int n = snprintf(msg, sizeof(msg), "[mic] RMS=%u%s\r\n",
                         (unsigned)rms, (rms > SOUND_ALARM_RMS) ? " LOUD!" : "");
        if (n > 0) boot_print(msg);

        vTaskDelayUntil(&last, pdMS_TO_TICKS(1000U));
    }
}

/* -------------------------------------------------------------------------- */
/* ctrl_task — closed-loop recipe controller + AI safety supervisor.          */
/*                                                                            */
/* The monitor->controller pivot: the Lab-Sentinel no longer only WATCHES a   */
/* furnace, it DRIVES one. ctrl_task runs the garnet sintering recipe          */
/* (predict_engine/sintering_profiles.json) through a PID loop, supervised by  */
/* the 5 on-chip AI models: it reads fusion_task's AI-4 verdict (s_eth_risk)   */
/* every step, and on CRITICAL risk / over-temp / TC fault / sustained         */
/* tracking error it drops to a SAFE STATE (heater off, batch FAULT) —         */
/* independent of the PID loop. During the sinter soak it streams deviations   */
/* into the SPC engine (Cpk + control verdict, AMS2750/CQI-9 style), updates   */
/* heating-element health from the soak duty, and on completion seals a        */
/* SHA-256 hash-chained electronic batch record (tamper-evident traceability). */
/*                                                                            */
/* SIM mode (CTRL_HW_PLANT 0): an FOPDT plant model stands in for the furnace  */
/* so the full controller is demonstrable on the bench with no 1600C heater.   */
/* HW mode: meas = MAX31855 thermocouple read; `u` drives the heater relay as  */
/* a slow time-proportioning PWM (both marked below).                          */
/*                                                                            */
/* Time is accelerated (CTRL_SUBSTEPS steps of CTRL_DT_S sim-seconds per       */
/* CTRL_TICK_MS wall tick) so a ~24 h program plays in ~70 s — fast enough to  */
/* demo, slow enough to say "ce shi bao jing" mid-run and watch the AI abort   */
/* the controller live. dt stays 5 s (the host-validated value; 10 s degrades  */
/* ramp tracking — see ctrl_test.c).                                           */
/* -------------------------------------------------------------------------- */
#define CTRL_HW_PLANT   0          /* 0 = FOPDT sim, 1 = MAX31855 TC + relay heater */
#define CTRL_DT_S       5.0f       /* control step (host-validated at 5 s)          */
#define CTRL_SUBSTEPS   5          /* control steps per wall tick                    */
#define CTRL_TICK_MS    20U        /* wall period -> garnet program in ~70 s         */

/* Real-heater over-temperature SAFETY CUT (the bench "real furnace" demo).
 * While a batch runs the PTC plate relay (PD12, heater_*) is energised; the moment
 * the REAL MAX31855 thermocouple reads >= this limit the safety supervisor cuts the
 * heater and faults the batch (CTRL_FAULT_OVERTEMP) — a genuine sensor -> actuator
 * -> safety closed loop on real hardware (no 1500C furnace needed). This complements
 * the AI-4 learned-anomaly trip and the TC open/short (unplug) trip.
 *   50C PTC plate self-limits ~50C -> trip 42C gives margin above room ambient and
 *   below the cap. For the 70C plate raise this (~60). Boot prints the idle probe
 *   temp so you can confirm margin on the day. Honest framing: a hard over-temp
 *   cutoff is a deterministic functional-safety rule (like max_safe_C), distinct
 *   from the AI's learned process-anomaly judgement.                              */
#define TC_PROBE_TRIP_C 42         /* real-probe hard over-temp trip (C)            */

static void ctrl_task(void *pv)
{
    (void)pv;
    elem_health_t elem;
    uint8_t       last_hash[32];
    uint32_t      batch_id = 0U;
    char          cb[160];

    /* element-health commissioning baseline: the SIM plant holds the 1500C
     * sinter at ~0.88 duty; WARN/EOL above that. Real element aging (rising
     * duty across batches) only shows once a real heater/TC is wired (HW mode). */
    elem_health_init(&elem, 0.88f, 0.94f, 0.98f, 0.3f);
    memcpy(last_hash, BR_GENESIS_PREV, 32);

#if CI1302_VOICE_ENABLED
    boot_print("[ctrl] closed-loop furnace controller ready (say 'kai shi shao jie')\r\n");
#else
    boot_print("[ctrl] closed-loop furnace controller ready (touch START)\r\n");
#endif

#if MAX31855_ENABLED
    /* Heat-demo banner: idle real-probe temp + the over-temp trip point, so the
     * margin is visible on the day (tune TC_PROBE_TRIP_C if the lab runs hot). */
    ctrl_poll_tc();
    {
        int n = snprintf(cb, sizeof(cb),
            "[ctrl] heat-demo: idle probe=%dC present=%u -> real over-temp cut at %dC (PD12 PTC heater)\r\n",
            s_tc_c, (unsigned)s_tc_present, TC_PROBE_TRIP_C);
        if (n > 0) boot_print(cb);
    }
    /* TC signal-integrity probe: sample 6x so wiring noise is obvious. If the raw
     * frames jump around -> bad SPI wiring (re-seat SO=PC2 / SSK=PB10, common GND).
     * A stable frame near room temp (~25C) = good. */
    {
        int k;
        for (k = 0; k < 6; k++) {
            ctrl_poll_tc();
            {
                int n = snprintf(cb, sizeof(cb),
                    "[ctrl] TC probe[%d] raw=0x%08lX T=%dC present=%u fault=0x%X\r\n",
                    k, (unsigned long)s_tc_raw, s_tc_c, (unsigned)s_tc_present, s_tc_fault);
                if (n > 0) boot_print(cb);
            }
            vTaskDelay(pdMS_TO_TICKS(50));
        }
    }
#endif

    for (;;) {
        furnace_ctrl_t c;
        plant_t        pl;
        spc_t          spc;
        float          meas, peak, soak_u_sum;
        long           soak_u_n;
        int            n_ai_alarms;
        TickType_t     last, last_log;

        s_wdg_hb[WDG_CTRL]++;            /* watchdog heartbeat (covers idle poll) */

        /* ---- idle until a start request ---- */
        if (s_ctrl_cmd != CTRL_CMD_START) {
#if MAX31855_ENABLED
            ctrl_poll_tc();             /* keep the real TC channel live when idle */
#endif
            ctrl_snap_idle(batch_id);   /* HMI: show idle furnace */
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }
        s_ctrl_cmd = CTRL_CMD_NONE;
        batch_id++;

        /* PID: gentle SIMC/IMC tuning for the slow FOPDT furnace (tau=180s, dead
         * time ~10s). loop gain Kp*plant_gain ~= 8, not 67 — no limit cycle.
         * host ctrl_spc_test: sinter soak holds 1500C +-0.5C, Cpk 7.0 in-control. */
        furnace_ctrl_init(&c, &RECIPE_GARNET, 0.005f, 0.0001f, 0.0f);
        plant_init(&pl, 25.0f, 1675.0f, 180.0f, 2);     /* SIM furnace (HW: MAX31855) */
        spc_init(&spc, -10.0f, +10.0f, 0.0f, 0.2f, 40); /* soak spec SP+-10C, target 0 */
        furnace_ctrl_start(&c);
#if MAX31855_ENABLED
        heater_on();    /* energise the real PTC plate for the run (real-furnace demo).
                         * Connect the heater for the over-temp-cut demo; leave it
                         * disconnected for a clean 1500C sim completion. */
#endif

        meas = 25.0f; peak = 25.0f; soak_u_sum = 0.0f; soak_u_n = 0; n_ai_alarms = 0;
        last = xTaskGetTickCount(); last_log = last;

        {
            int n = snprintf(cb, sizeof(cb),
                "[ctrl] batch #%lu START recipe='%s' total=%ldmin\r\n",
                (unsigned long)batch_id, RECIPE_GARNET.name,
                (long)(recipe_total_s(&RECIPE_GARNET) / 60.0f));
            if (n > 0) boot_print(cb);
        }
        /* AI pre-flight echo: the predicted optical/thermal/energy outcome of THIS
         * recipe before the furnace fires (AI-6/7/8/9, cached at boot). */
        {
            recipe_ai_t pf;
            int n;
            lab_get_recipe_ai(&pf);
            n = snprintf(cb, sizeof(cb),
                "[ctrl] preflight: target lam~%dnm thermal~%d%% energy~%d.%dkWh analog=#%d %s\r\n",
                pf.lambda_nm, pf.thermal_pct, pf.kwh_x10 / 10, pf.kwh_x10 % 10,
                pf.analog_idx, pf.analog_name);
            if (n > 0) boot_print(cb);
        }

        /* ---- closed loop ---- */
        while (c.state == CTRL_RUN) {
            int s;
            s_wdg_hb[WDG_CTRL]++;       /* watchdog heartbeat (covers active run) */
#if MAX31855_ENABLED
            ctrl_poll_tc();             /* refresh the real MAX31855 channel once/tick */
#endif
            for (s = 0; s < CTRL_SUBSTEPS && c.state == CTRL_RUN; s++) {
                int     risk     = (int)s_eth_risk;   /* AI-4 verdict from fusion_task */
                /* A wired thermocouple's fault (OC/SCG/SCV) trips the safety
                 * supervisor (CTRL_FAULT_SENSOR) even in SIM mode — pull the TC
                 * lead during a demo and the controller drops to safe state.
                 * No TC wired (present=0) -> 0, no effect. */
                uint8_t tc_fault = (s_tc_present ? s_tc_fault : 0U);
                float   u;
#if CTRL_HW_PLANT
                meas = (float)s_tc_c;     /* HW mode: real thermocouple IS the plant PV */
#endif
                u = furnace_ctrl_step(&c, meas, risk, tc_fault, CTRL_DT_S);
#if CTRL_HW_PLANT
                /* heater_relay_set_duty(u);  // needs a dedicated heater SSR (not the
                 * exhaust relay) — drive as slow time-proportioning PWM */
#else
                meas = plant_step(&pl, u, CTRL_DT_S);
#endif
                if (meas > peak) peak = meas;
                if (risk >= 2) n_ai_alarms++;
                /* SPC + element-health sampled on the high-temp sinter soak */
                if (c.seg_kind == SEG_SOAK && c.sp_C > 1200.0f) {
                    spc_update(&spc, meas - c.sp_C);
                    soak_u_sum += u; soak_u_n++;
                }
            }

            /* operator/voice abort -> safe state (only if still running) */
            if (s_ctrl_cmd == CTRL_CMD_ABORT && c.state == CTRL_RUN) {
                s_ctrl_cmd = CTRL_CMD_NONE;
                furnace_ctrl_abort(&c, CTRL_FAULT_OPERATOR);
                boot_print("[ctrl] operator abort -> safe state (heater off)\r\n");
            }

#if MAX31855_ENABLED
            /* REAL-heater over-temperature SAFETY CUT: the PTC plate is energised
             * (heater_on above); the moment the real thermocouple crosses the hard
             * limit, cut the heater relay and drop to safe state. Genuine
             * sensor->actuator->safety loop on real hardware. Only fires when a
             * probe is actually wired (present) so it can't false-trip on a float. */
            {
                /* Validate the sample (in plausible K-probe range) and require 3
                 * consecutive over-limit reads, so a single corrupted SPI frame on
                 * the field wiring can't false-trip the safety cut. */
                static int ot_cnt = 0;
                int tc_ok = (s_tc_present && s_tc_c > -50 && s_tc_c < 1100);
                if (c.state == CTRL_RUN && tc_ok && s_tc_c >= TC_PROBE_TRIP_C) {
                    if (++ot_cnt >= 3) {
                        heater_off();
                        furnace_ctrl_abort(&c, CTRL_FAULT_OVERTEMP);
                        {
                            int n = snprintf(cb, sizeof(cb),
                                "[ctrl] REAL probe over-temp %dC >= %dC (x3) -> heater OFF + safe state\r\n",
                                s_tc_c, TC_PROBE_TRIP_C);
                            if (n > 0) boot_print(cb);
                        }
                        ot_cnt = 0;
                    }
                } else {
                    ot_cnt = 0;   /* reset on any in-range/below-limit/invalid read */
                }
            }
#endif

            /* periodic status (~3 s wall) */
            if ((xTaskGetTickCount() - last_log) >= pdMS_TO_TICKS(3000U)) {
                last_log = xTaskGetTickCount();
                int n = snprintf(cb, sizeof(cb),
                    "[ctrl] seg=%s SP=%dC T=%dC u%%=%d AI4risk=%d\r\n",
                    RECIPE_GARNET.seg[c.seg_idx].label,
                    (int)c.sp_C, (int)meas, (int)(c.u * 100.0f), (int)s_eth_risk);
                if (n > 0) vbprint(cb);
            }

            /* HMI snapshot + SPC soak-trend ring (additive; pure reads) */
            ctrl_snap_commit(&c, meas, &spc, (int)elem_health_remaining_pct(&elem),
                             batch_id, (uint32_t)recipe_total_s(&RECIPE_GARNET));
            if (c.seg_kind == SEG_SOAK && c.sp_C > 1200.0f)
                spc_ring_push((int16_t)(meas - c.sp_C));

            /* AI-14 multi-step forecaster: maintain a window of recent measured temps
             * and publish the predicted next-N temps for the Trend overlay (setpoint
             * vs measured vs forecast). Demonstration cadence (1 sample per ctrl tick;
             * trained on per-minute furnace_sim levels, so the plateau-aware forecast
             * tracks the same recipe levels the Trend shows). */
            {
                static float fcwin[AI14_WIN];
                static int   fcn = 0;
                int wi;
                for (wi = 0; wi < AI14_WIN - 1; wi++) fcwin[wi] = fcwin[wi + 1];
                fcwin[AI14_WIN - 1] = meas / AI14_TNORM;
                if (fcn < AI14_WIN) fcn++;
                if (fcn >= AI14_WIN) {
                    float out[AI14_HOR];
                    fc_view_t fv;
                    int oi;
                    ai14_forecast(fcwin, out);
                    fv.valid = 1u;
                    fv.n = AI14_HOR;
                    for (oi = 0; oi < AI14_HOR; oi++)
                        fv.next_c[oi] = (int)(out[oi] * AI14_TNORM + 0.5f);
                    fv.reach_sp = (fv.next_c[AI14_HOR - 1] >= (int)c.sp_C - 30) ? 1 : 0;
                    taskENTER_CRITICAL();
                    s_fc = fv;
                    taskEXIT_CRITICAL();
                }
            }

            vTaskDelayUntil(&last, pdMS_TO_TICKS(CTRL_TICK_MS));
        }

#if MAX31855_ENABLED
        heater_off();   /* run ended (DONE/FAULT/abort): cut the real heater (safe) */
#endif

        /* ---- batch complete: SPC verdict + element health + sealed record ---- */
        {
            spc_result_t    sr;
            health_status_t hs;
            batch_record_t  rec;
            float           mean_soak_u = (soak_u_n > 0) ? (soak_u_sum / (float)soak_u_n) : 0.0f;
            char            hx[65];
            int             i, n;

            spc_finalize(&spc, &sr);
            hs = elem_health_update(&elem, mean_soak_u);

            memset(&rec, 0, sizeof(rec));
            rec.batch_id  = batch_id;
            rec.unix_time = (uint32_t)(xTaskGetTickCount() / configTICK_RATE_HZ);  /* HW: DS3231 RTC */
            for (i = 0; i < BR_RECIPE_LEN - 1 && RECIPE_GARNET.name[i]; i++)
                rec.recipe[i] = RECIPE_GARNET.name[i];
            rec.operator_id[0]='l'; rec.operator_id[1]='a'; rec.operator_id[2]='b'; rec.operator_id[3]='0'; rec.operator_id[4]='1';
            rec.peak_C             = peak;
            rec.soak_cpk           = sr.cpk;
            rec.elem_remaining_pct = elem_health_remaining_pct(&elem);
            rec.n_ai_alarms        = (uint16_t)((n_ai_alarms > 65535) ? 65535 : n_ai_alarms);
            rec.in_control         = (uint8_t)sr.in_control;
            rec.capable            = (uint8_t)sr.capable;
            rec.final_state        = (uint8_t)c.state;
            rec.fault              = (uint8_t)c.fault;

            batch_record_seal(&rec, last_hash);     /* hash folds in previous record */
            memcpy(last_hash, rec.this_hash, 32);
            ledger_push(&rec);                      /* retain for the Quality screen */

            n = snprintf(cb, sizeof(cb),
                "[ctrl] batch #%lu %s fault=%s peak=%dC | Cpkx100=%d %s/%s | elem=%d%% %s | AIalarms=%d\r\n",
                (unsigned long)batch_id,
                (c.state == CTRL_DONE) ? "DONE" : "FAULT",
                furnace_ctrl_fault_str(c.fault), (int)peak,
                (int)(sr.cpk * 100.0f),
                sr.in_control ? "in-ctl" : "OOC",
                sr.capable ? "capable" : "incapable",
                (int)rec.elem_remaining_pct, health_status_str(hs), n_ai_alarms);
            if (n > 0) boot_print(cb);

            sha256_hex(rec.this_hash, hx);
            hx[12] = '\0';                          /* short hash for the log */
            n = snprintf(cb, sizeof(cb),
                "[ctrl] record sealed hash=%s.. chained (tamper-evident ledger)\r\n", hx);
            if (n > 0) boot_print(cb);

            /* final HMI snapshot: DONE/FAULT state + final Cpk (brief, until idle) */
            ctrl_snap_commit(&c, meas, &spc, (int)rec.elem_remaining_pct,
                             batch_id, (uint32_t)recipe_total_s(&RECIPE_GARNET));
        }
    }
}

/* -------------------------------------------------------------------------- */
/* wdg_task - windowed task-supervision watchdog                              */
/*                                                                            */
/* Arms the GD32 free watchdog (FWDGT, clocked by the ~32 kHz IRC32K) for a   */
/* ~4 s hardware timeout, then every WDG_KICK_MS verifies each monitored task */
/* heartbeat advanced within its deadline. While all are healthy the FWDGT is */
/* reloaded; the moment one stalls past its deadline the kicker stops         */
/* reloading and the FWDGT resets the MCU (heater off on cold boot). The next */
/* boot prints "recovered from WATCHDOG reset" via the RCU reset flag.        */
/* -------------------------------------------------------------------------- */
#define WDG_KICK_MS  100U   /* heartbeat scan / FWDGT reload cadence */

static void wdg_task(void *pv)
{
    (void)pv;
    /* Max ticks (WDG_KICK_MS each) a slot may stall before it is a fault.
     * Sized to a few times each task's natural loop period:
     *   ctrl 2.0s (idle 100ms / run 20ms), fusion 4.0s (1.5s loop),
     *   env  5.0s (1Hz + 70ms AI-3),        sensor 2.0s (5ms loop). */
    static const uint16_t    deadline[WDG_N] = { 20U, 40U, 50U, 20U };
    static const char *const name[WDG_N]     = { "ctrl", "fusion", "env", "sensor" };
    uint32_t   last[WDG_N];
    uint16_t   stall[WDG_N];
    TickType_t tw;
    int        i, was_ok = 1;

    for (i = 0; i < WDG_N; i++) { last[i] = s_wdg_hb[i]; stall[i] = 0U; }

    /* FWDGT is clocked by IRC32K; bring it up and wait for stability first, or
     * fwdgt_config() would spin on status flags that only sync in that domain. */
    rcu_osci_on(RCU_IRC32K);
    while (rcu_osci_stab_wait(RCU_IRC32K) == ERROR) { }

    /* FWDGT timeout = reload * prescaler / f_IRC32K = 500 * 256 / 32000 ≈ 4.0 s.
     * (IRC32K ~32 kHz; its tolerance is fine for a coarse safety timeout.) Once
     * enabled the FWDGT cannot be stopped, so this only arms after boot init. */
    fwdgt_config(500U, FWDGT_PSC_DIV256);
    fwdgt_enable();
    fwdgt_counter_reload();
    boot_print("[wdg] FWDGT armed (~4s hw timeout) supervising ctrl/fusion/env/sensor\r\n");

    tw = xTaskGetTickCount();
    for (;;) {
        int ok = 1, culprit = -1;
        for (i = 0; i < WDG_N; i++) {
            uint32_t now = s_wdg_hb[i];
            if (now != last[i]) {
                last[i]  = now;
                stall[i] = 0U;
            } else if (++stall[i] > deadline[i]) {
                ok = 0;
                if (culprit < 0) culprit = i;
            }
        }

        if (ok) {
            fwdgt_counter_reload();         /* every task alive -> kick the dog */
            was_ok = 1;
        } else if (was_ok) {
            /* First miss of this episode: stop kicking. The FWDGT now resets the
             * MCU within ~4 s. Announce which task hung (reset may cut this off). */
            char wb[96];
            int  n = snprintf(wb, sizeof(wb),
                "[wdg] TASK STALL '%s' (no heartbeat ~%ums) -> FWDGT resetting MCU\r\n",
                (culprit >= 0) ? name[culprit] : "?",
                (unsigned)(((culprit >= 0) ? stall[culprit] : 0U) * WDG_KICK_MS));
            if (n > 0) boot_print(wb);
            was_ok = 0;
        }

        vTaskDelayUntil(&tw, pdMS_TO_TICKS(WDG_KICK_MS));
    }
}

/* -------------------------------------------------------------------------- */
/* FreeRTOS hooks                                                             */
/* -------------------------------------------------------------------------- */

/* Called from FreeRTOS SysTick handler at 1 kHz.
 * Forwards the tick to the GD32 systick.c blocking delay counter so that
 * delay_1ms() keeps working after the scheduler takes over the SysTick.      */
void vApplicationTickHook(void)
{
    delay_decrement();
}

/* Called when pvPortMalloc() fails (configUSE_MALLOC_FAILED_HOOK = 1).
 * Indicates configTOTAL_HEAP_SIZE is too small or a leak. Trap for debugger. */
void vApplicationMallocFailedHook(void)
{
    taskDISABLE_INTERRUPTS();
    for (;;) { }
}

/* Functional-safety layer (configCHECK_FOR_STACK_OVERFLOW=2): fires from the
 * scheduler context when a task overruns its stack. We CANNOT take the UART
 * mutex here, so write the offending task's name raw to UART4 (same approach as
 * the HardFault dump), then halt for the FWDGT to reset into a safe cold boot.
 * This is what root-caused the AI-5 HMI hang (ui_task stack overflow) in 2026-06
 * — kept as permanent protection so any future stack regression names its task
 * instead of silently corrupting the shared FreeRTOS heap. */
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName)
{
    const char *p;
    (void)xTask;
    taskDISABLE_INTERRUPTS();
    for (p = "\r\n*** STACK OVERFLOW: "; *p; p++) {
        usart_data_transmit(UART4, (uint8_t)*p);
        while (RESET == usart_flag_get(UART4, USART_FLAG_TBE)) { }
    }
    for (p = pcTaskName; p && *p; p++) {
        usart_data_transmit(UART4, (uint8_t)*p);
        while (RESET == usart_flag_get(UART4, USART_FLAG_TBE)) { }
    }
    for (p = " ***\r\n"; *p; p++) {
        usart_data_transmit(UART4, (uint8_t)*p);
        while (RESET == usart_flag_get(UART4, USART_FLAG_TBE)) { }
    }
    for (;;) { }
}

/* -------------------------------------------------------------------------- */
/* Local helpers                                                              */
/* -------------------------------------------------------------------------- */
/* Self-contained UART send (UART4 PB13/PB5 AF14 / 115200 8N1 — DAP-Link CDC).
 * Board labels the connector "UART0" but MCU peripheral is UART4 (AF14). */
static void boot_print(const char *s)
{
    xSemaphoreTake(xMutex_UART, portMAX_DELAY);
    while (*s != '\0') {
        usart_data_transmit(UART4, (uint8_t)*s);
        while (RESET == usart_flag_get(UART4, USART_FLAG_TBE)) { }
        s++;
    }
    xSemaphoreGive(xMutex_UART);
}

/******************************* End of File *********************************/
