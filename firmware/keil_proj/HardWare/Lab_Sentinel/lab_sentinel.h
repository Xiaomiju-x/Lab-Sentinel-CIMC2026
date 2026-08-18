/******************************************************************************
 * lab_sentinel.h
 *
 * CIMC Lab-Sentinel - Task Framework Public API
 *
 * Phase 0 baseline: 8 task stubs (heartbeat only).
 * Phase 1+: each task is wired to real drivers (OV5640 / ADXL345 / SHT30 /
 *          MQ-135 / INMP441 / ST7796 + GT911 / CI1302 / ENET 8521).
 *
 * Task table:
 *   Name           Priority  Stack  Period       Role
 *   task_init        5        256   once         Boot init (deletes self)
 *   sensor_task      3        512   5  ms 200Hz  ADXL345 + AI-3 vibration
 *   vision_task      2       1024   200ms 5Hz    OV5640 + AI-1 vision
 *   env_task         1        256   1000ms 1Hz   SHT30 + MQ-135 + AI-2 env AE
 *   fusion_task      3        512   event-driven AI-4 fusion + risk
 *   ui_task          2       1024   33ms 30Hz    LVGL + GT911 touch
 *   voice_task       2        512   event-driven CI1302 UART parser
 *   eth_task         1       1024   event-driven HTTP / Modbus uplink
 *   doa_task         2        512   event-driven INMP441 dual-mic DOA
 *
 * Inter-task IPC:
 *   xQueue_VisionResult  - vision_task -> fusion_task
 *   xQueue_VibResult     - sensor_task -> fusion_task
 *   xQueue_EnvResult     - env_task    -> fusion_task
 *   xQueue_RiskAlert     - fusion_task -> ui_task / eth_task / voice_task
 *   xSem_VoiceCmd        - voice_task ISR -> ui_task (mode switch)
 *
 ******************************************************************************/
#ifndef __LAB_SENTINEL_H__
#define __LAB_SENTINEL_H__

#include "HeaderFiles.h"

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "semphr.h"

/* -------------------------------------------------------------------------- */
/* Task priorities (higher number = higher priority in FreeRTOS)              */
/* -------------------------------------------------------------------------- */
typedef enum {
    PRIO_IDLE       = 0,
    PRIO_LOW        = 1,    /* env_task, eth_task                              */
    PRIO_MED        = 2,    /* vision_task, ui_task, voice_task, doa_task      */
    PRIO_HIGH       = 3,    /* sensor_task, fusion_task                        */
    PRIO_BOOT       = 5     /* task_init (deletes self after boot)             */
} lab_sentinel_priority_t;

/* -------------------------------------------------------------------------- */
/* Risk levels (output of AI-4 fusion)                                        */
/* -------------------------------------------------------------------------- */
typedef enum {
    RISK_NORMAL     = 0,
    RISK_WARNING    = 1,
    RISK_ANOMALY    = 2,
    RISK_SEVERE     = 3
} lab_risk_level_t;

/* -------------------------------------------------------------------------- */
/* IPC payloads                                                               */
/* -------------------------------------------------------------------------- */
typedef struct {
    uint32_t timestamp_ms;
    uint8_t  predicted_class;       /* AI-1 or AI-1b NCM result                */
    uint8_t  confidence_q7;         /* 0..127 = 0..1.0                         */
    int16_t  conformal_lower_q7;    /* AI-1 conformal CI lower (Q7)            */
    int16_t  conformal_upper_q7;    /* AI-1 conformal CI upper (Q7)            */
} vision_result_t;

typedef struct {
    uint32_t timestamp_ms;
    uint8_t  predicted_class;       /* AI-3 vibration class                    */
    uint8_t  confidence_q7;
    uint16_t rms_mg;                /* RMS in milli-g (raw feature for AI-2)   */
} vibration_result_t;

typedef struct {
    uint32_t timestamp_ms;
    int16_t  temp_c_q8;             /* SHT30 Q8 fixed-point degC               */
    uint16_t humidity_q8;
    uint16_t mq135_adc;             /* 12-bit ADC raw                          */
    uint16_t reconstruction_mse_q4; /* AI-2 AE MSE (Q4 fixed-point)            */
    uint8_t  attribution_mask;      /* bit0=temp bit1=humid bit2=gas bit3=vib  */
} env_result_t;

typedef struct {
    uint32_t timestamp_ms;
    lab_risk_level_t risk;
    uint8_t  trigger_source;        /* bit0=vision bit1=vib bit2=env bit3=doa  */
    int16_t  doa_angle_deg;         /* -90..+90, 0 = front (or INT16_MIN none) */
} risk_alert_t;

/* -------------------------------------------------------------------------- */
/* AI pipeline snapshot — read-only copy of the live s_ai state (env_task @1Hz)*/
/* for the LVGL dashboard. Filled by lab_sentinel_get_ai() under a critical    */
/* section so the UI never reads a half-updated frame.                         */
/* -------------------------------------------------------------------------- */
typedef struct {
    float   ai1_probs[4];   /* AI-1 crucible-state proxy (camera-less)         */
    float   ai2_ratio;      /* AI-2 anomaly ratio mse/q_hat (0..6)             */
    float   ai2_resid[3];   /* AI-2 residuals: [0]=temp [1]=vib [2]=gas        */
    float   ai3_probs[5];   /* AI-3 sinter-curve softmax                       */
    float   progress;       /* sintering progress 0..1                         */
    int     ai3_cls;        /* AI-3 argmax (0=normal..4)                       */
    int     stage;          /* furnace stage 0..5                              */
    float   temp_c;         /* simulated furnace temperature (degC)            */
    uint8_t ready;          /* 1 once env_task has produced >=1 frame          */
    float   ai3_attn[64];   /* AI-3 attention saliency over the 64-min window  */
                            /* (sums to ~1; explainability strip on the AI tab) */
} lab_ai_snapshot_t;

/* Copy the live AI pipeline state for the UI (thread-safe). */
void lab_sentinel_get_ai(lab_ai_snapshot_t *out);

/* UI/touch hook into the closed-loop furnace controller (same path as voice):
 * cmd 1 = start a garnet batch, cmd 2 = abort to safe state. Consumed by
 * ctrl_task. Safe to call from the LVGL button event callback. */
void lab_ctrl_request(int cmd);

/* Motor control, SEPARATE from the sinter START/STOP (its own MOTOR button):
 * on != 0 spins the stirring/vibration motor (AI-10 PdM target), 0 stops it. */
void lab_motor_request(int on);

/* UART log helper for code outside lab_sentinel.c (e.g. the LVGL button
 * callbacks in ui_screen.c) — forwards to the mutex-guarded boot_print. */
void lab_log(const char *s);

/* AI-5 root-cause for the HMI: *cls = ai5_rootcause_t (0=NORMAL), *pct = conf%.
 * Decoupled from lab_ai_snapshot_t so the AI-5 display path adds no struct field. */
void lab_get_ai5(int *cls, int *pct);

/* AI-1b few-shot NCM nearest-class for the HMI (-1 = not yet classified). */
void lab_get_ncm(int *cls);

/* -------------------------------------------------------------------------- */
/* Globals (defined in lab_sentinel.c)                                        */
/* -------------------------------------------------------------------------- */
extern QueueHandle_t        xQueue_VisionResult;
extern QueueHandle_t        xQueue_VibResult;
extern QueueHandle_t        xQueue_EnvResult;
extern QueueHandle_t        xQueue_RiskAlert;
extern SemaphoreHandle_t    xSem_VoiceCmd;

/* -------------------------------------------------------------------------- */
/* Public entry point - called from HardWare/FreeRTOS/app_main.c              */
/* -------------------------------------------------------------------------- */
void lab_sentinel_main(void);

#endif /* __LAB_SENTINEL_H__ */
