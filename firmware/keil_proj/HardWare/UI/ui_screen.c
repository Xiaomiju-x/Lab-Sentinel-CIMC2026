/******************************************************************************
 * ui_screen.c — Lab-Sentinel industrial HMI (LVGL 8.3, 800x480 RGB-TLI).
 *
 * 2026-05-30: single AI dashboard -> multi-screen touch HMI (Eurotherm/西门子
 *   面板风). 常驻状态栏 + 6 标签导航 + 内容区 (容器 show/hide). Phase 1+2:
 *   框架 + 数据快照层接入 + 概览(Home) + 趋势(Trend SPC 控制图);
 *   AI 诊断 + 系统(报警/维护) 复用既有面板; 配方/质量 先占位.
 *
 * 约束 (踩坑提炼): 可点目标中心落在 y[60,430]/x[40,760] 可靠区(屏幕边缘电容
 *   读数偏短); 目标 >=56px; 抖动触摸用 LV_EVENT_PRESSED. 控件只用 label/list/
 *   bar/line/obj (按钮/标签页用可点 lv_obj 自建). 无 %f, 浮点整数化. 字体 14/20/28.
 *   渲染只读快照 (lab_ctrl_get / lab_sentinel_get_ai), 不碰控制环内部.
 *
 * 布局:
 *   状态栏  (0,0,800,48)    风险色 + 配方/批次 + 大号炉温
 *   标签栏  (0,50,800,58)   [Home][Recipe][Trend][AI][Quality][System]
 *   内容区  (0,110,800,370) 6 容器叠放, 1 可见
 ******************************************************************************/

#include "ui_screen.h"
#include "ui_data.h"
#include "ai2_ae.h"      /* ai2_ae_qhat() */
#include "ai5_diagnose.h" /* ai5_name() + AI5_RC_NORMAL for the root-cause line */
#include "lab_font_cn.h"  /* subset CJK font for the nano-LM Chinese diagnosis */
#include "ai_lm_bank.h"   /* lm_roster_label_s/tag_s accessors for the LM-size switch */
#include "lab_build_config.h"  /* LAB_LM_ENABLE -> honest "LM disabled" state on the Edge LM page */
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Geometry                                                            */
/* ------------------------------------------------------------------ */
#define SCREEN_W   800
#define SCREEN_H   480
#define NAV_W       84            /* left vertical navigation rail (web-dashboard style) */
#define STAT_H      44            /* status bar height (sits right of the rail)          */
#define BODY_X      NAV_W         /* content area starts right of the rail               */
#define BODY_Y      STAT_H        /* content area starts below the status bar            */
#define BODY_W     (SCREEN_W - NAV_W)   /* 716                                           */
#define BODY_H     (SCREEN_H - STAT_H)  /* 436                                           */

#define NPAGE       13   /* Home Recipe Trend Control Quality System Camera Pre-flt PL Models AI-Hub Edge-LM Robust */

/* PL-spectrum chart (AI-12/13 page) + Models-overview sparklines */
#define PLX          10
#define PLY          44
#define PLW         500
#define PLH         190
#define SPARK_N      48          /* sparkline ring length */

/* Home rolling-temperature curve (controller temp, 1 Hz) */
#define CURVE_X      10
#define CURVE_Y      70
#define CURVE_W     470
#define CURVE_H     130
#define CURVE_N      64
#define TEMP_MAX   1650

/* Trend SPC chart (inside the Trend page container) */
#define CHX          10
#define CHY          24
#define CHW         500          /* narrowed for the left rail (stats column moves left) */
#define CHH         312
#define DEV_FS       15          /* chart full-scale deviation +-15 C */
#define SPC_N       120          /* matches SPC_RING_N in lab_sentinel.c */

/* AI tab: AI-3 attention saliency strip (last-block attention over 64 min).
 * 500 wide: stays left of the AE-attribution column (x=540) and fills the space
 * freed when the AI-6..13 roster was moved off this page (that roster duplicated
 * the Models page and overlapped the AI-1b/NCM column). */
#define ATX          10
#define ATY         250
#define ATW         500
#define ATH          92
#define ATTN_N       64

/* ------------------------------------------------------------------ */
/* Palette                                                             */
/* ------------------------------------------------------------------ */
/* dark-dashboard palette (vivid accents on a deep navy-charcoal canvas) */
#define COL_GREEN   0x3DD68CU
#define COL_AMBER   0xF2B83AU
#define COL_ORANGE  0xFF8A3DU
#define COL_RED     0xFF5C5CU
#define COL_BLUE    0x3BA7FFU
#define COL_GRAY    0x8B95A3U
#define COL_DIM     0x1A2230U
/* surface tones (cards / panels / borders / text) — the "web dashboard" base */
#define COL_BG      0x0C1016U   /* app background (top of gradient)     */
#define COL_BG2     0x10161FU   /* app background (bottom of gradient)  */
#define COL_CARD    0x161D27U   /* card / panel fill (top of gradient)  */
#define COL_CARD2   0x1B2430U   /* card / panel fill (bottom)           */
#define COL_BORDER  0x2A3340U   /* subtle 1px card border               */
#define COL_TEXT    0xE6EAF0U   /* primary text                         */
#define COL_TEXT2   0xAEB7C4U   /* secondary text                       */
#define COL_ACCENT  0x9A6BFFU   /* generative-AI / LM accent (purple)   */

/* Cap for the System event/risk log list. The fusion task posts a risk alert
 * every cycle, so an uncapped list grows without bound -> exhausts the LVGL
 * pool -> the renderer starves and the screen flickers (then faults). */
#define LOG_MAX_ENTRIES  40U

/* _risk_bg (old full-bar dark fills) removed: the status bar is now a flat dark
 * strip + a bright rounded risk pill, so risk colour lives on _risk_fill only. */
static const uint32_t _risk_fill[4] = { COL_GREEN, COL_AMBER, COL_ORANGE, COL_RED };
static const char *const _risk_text[4] = { "NORMAL", "WARNING", "ANOMALY", "SEVERE" };

static const char *const AI1N[4]  = { "empty", "loaded", "firing", "done" };
static const char *const AI1N3[3] = { "EMPTY", "LOADED", "DONE" };  /* camera CNN 3-class */
static const char *const AI3N[5] = { "normal", "fast-ramp", "undertemp", "drift", "slow-ramp" };
static const char *const AI4N[4] = { "GOOD", "SUSPECT", "BAD", "CRITICAL" };
static const char *const TABN[NPAGE] = { "Home", "Recipe", "Trend", "Control", "Quality", "System", "Camera", "Pre-flt", "PL", "Models", "E-Twin", "Edge LM", "Robust" };

/* PL dopant-class names (AI-12) */
static const char *const PLN[3] = { "Cr3+", "Ni2+", "Cr+Ni" };
static const uint32_t    PLC[3] = { COL_ORANGE, COL_BLUE, 0xAA66EEU };

/* 20-model roster + per-model detail metadata. EVERY field is real: the metric is
 * the validated number from each model's host report, the architecture is the actual
 * layer stack, the latency is read LIVE from the on-chip DWT probe (lab_get_ai_lat,
 * lat_idx; -1 = not separately timed). No fabricated params/Flash are shown. */
#define N_MODELS 20
#define ARCH_MAX 5
typedef struct {
    const char *id;            /* "AI-1"                                  */
    const char *name;          /* short name                              */
    const char *metric;        /* headline validated metric (roster chip) */
    const char *purpose;       /* one-line what-it-does                   */
    const char *arch[ARCH_MAX];/* layer-block labels (diagram), NULL-pad   */
    const char *input;         /* input descriptor                        */
    const char *data;          /* training-data provenance                */
    int         acc_pct;       /* metric bar 0..100, -1 if N/A (regressor) */
    int         lat_idx;       /* index into lab_get_ai_lat(), -1 if none  */
} model_info_t;

static const model_info_t MODELS[N_MODELS] = {
 {"AI-1","Crucible CNN","CV 90.7%","Crucible state from the OV5640 frame",
  {"img 3x64","Conv 3x3","MaxPool","GAP","FC ->3"},"64x64 crop","52 real phone photos",91,AI_LAT_AI1},
 {"AI-1b","Few-shot NCM","on-device","Register a new crucible type from few shots",
  {"emb 32","class means","NCM dist","-> class",0},"AI-1 embedding","incremental class means",-1,-1},
 {"AI-2","Sinter AE","FPR 10%","Furnace anomaly via reconstruction error",
  {"feat 32","enc 16/8","latent 8","dec 32","MSE>qhat"},"32-D furnace feat","furnace_sim + conformal",-1,AI_LAT_AI2},
 {"AI-3","TinyXformer","69ms/M7","Sintering-curve anomaly classifier",
  {"seq 64","2x attn","mean-pool","FC ->5",0},"64-min temp seq","furnace_sim 5-class",-1,AI_LAT_AI3},
 {"AI-4","Risk Fusion","97.5%","Fuse model votes into one risk verdict",
  {"feats","MLP hidden","->4 risk",0,0},"multi-model feats","fused AI-1/2/3 + calib",97,AI_LAT_AI4},
 {"AI-5","Root-cause","99.9%","Name the most likely fault cause",
  {"state 27","MLP","->9 cause",0,0},"27-D state vector","fault taxonomy",99,-1},
 {"AI-6","Optical TS","MAE 6.2nm","Predict emission peak from the recipe",
  {"desc 24","MLP","-> lam/FWHM",0,0},"24-D descriptor","distilled Tanabe-Sugano",-1,-1},
 {"AI-7","Thermal Q","3-band","Predict thermal-quench retention band",
  {"desc 24","MLP","->3 band",0,0},"24-D descriptor","quench physics distill",-1,-1},
 {"AI-8","Energy/CO2","0.7% err","Estimate firing energy + carbon",
  {"recipe 5","MLP","-> kWh/CO2",0,0},"5-D recipe","grid 0.5703 kgCO2/kWh",-1,-1},
 {"AI-9","Recipe kNN","67-row","Retrieve nearest historical recipe",
  {"recipe vec","kNN 67","-> analog",0,0},"recipe vector","67 observed recipes",-1,-1},
 {"AI-10","Vib PdM","real 2-cls","Motor health from vibration",
  {"vib 64","RMS feats","MLP ->2",0,0},"64-sample ADXL345","real accelerometer",-1,-1},
 {"AI-11","Phase-purity","LOO 70%","XRD phase-purity edge triage",
  {"desc 24","MLP","-> pure?",0,0},"24-D descriptor","37 real XRD labels",70,AI_LAT_AI11},
 {"AI-12","PL Dopant","CV 98.2%","Classify dopant from PL spectrum",
  {"spec 64","FC 24","FC 16","->3 dopant",0},"64-pt PL spectrum","281 real Fluoromax",98,AI_LAT_AI12},
 {"AI-13","PL-QC AE","AE q_hat","PL spectrum QC anomaly detector",
  {"spec 64","encoder","dec 64","MSE>qhat",0},"64-pt PL spectrum","281 real + conformal",-1,AI_LAT_AI13},
 {"AI-14","Temp Forecast","+12 min","Forecast next 12 min of furnace temp",
  {"win 24","FC 32","FC 32","->12 temp",0},"24-min window","furnace_sim, beats lin-extrap",-1,AI_LAT_AI14},
 {"AI-15","PL Host-ID","CV 97.1%","Identify garnet host from PL",
  {"spec 64","FC 24","FC 24","->2 host",0},"64-pt PL spectrum","281 real Fluoromax",97,AI_LAT_AI15},
 {"AI-16","PL Lambda","MAE 18.9nm","Read emission peak off the spectrum",
  {"spec 64","FC 24","FC 24","-> lam nm",0},"64-pt PL spectrum","281 real Fluoromax",-1,AI_LAT_AI16},
 {"AI-17","PL Few-shot","5-shot 87%","Register a new phosphor from few spectra",
  {"emb 16","class means","NCM","-> class",0},"AI-12 embedding","few-shot over AI-12",87,AI_LAT_AI17},
 {"AI-19","RUL / ETA","76 min MAE","Minutes remaining to firing-complete",
  {"win24+dwell","FC 48","FC 48","-> ETA min",0},"24-win+hold+stage","furnace_sim, beats nominal",-1,AI_LAT_AI19},
 {"AI-20","TC-Integrity","acc 97.7%","Thermocouple sensor-fault monitor",
  {"win 12x2","8 feats","FC 16","->3 fault",0},"meas+setpoint","fault-injection trained",98,AI_LAT_AI20},
};

/* per-model category icon for the detail-page title (a lively glyph next to the
 * id/name). Order matches MODELS[]; all codepoints are in the baked montserrat
 * symbol range (vision=IMAGE, anomaly=WARNING, spectra=TINT, forecast=UP, ...). */
static const char *const MODICON[N_MODELS] = {
    LV_SYMBOL_IMAGE,    LV_SYMBOL_COPY,     LV_SYMBOL_WARNING, LV_SYMBOL_BARS,  LV_SYMBOL_OK,
    LV_SYMBOL_LIST,     LV_SYMBOL_TINT,     LV_SYMBOL_DOWN,    LV_SYMBOL_CHARGE,LV_SYMBOL_COPY,
    LV_SYMBOL_LOOP,     LV_SYMBOL_EYE_OPEN, LV_SYMBOL_TINT,    LV_SYMBOL_OK,    LV_SYMBOL_UP,
    LV_SYMBOL_EYE_OPEN, LV_SYMBOL_TINT,     LV_SYMBOL_COPY,    LV_SYMBOL_UP,    LV_SYMBOL_WARNING
};

/* AI-20 thermocouple-integrity class names (Home light + detail live readout) */
static const char *const TCN[3] = { "healthy", "open-ckt", "erratic" };
/* Benchmark page row labels (one per AI_LAT_* index, same order as the enum) */
static const char *const LATN[AI_LAT_N] = {
    "AI-1 CNN", "AI-2 AE", "AI-3 Xformer", "AI-4 fusion", "AI-11 purity",
    "AI-12 PL", "AI-12 INT8", "AI-13 QC", "AI-14 forecast", "AI-15 host",
    "AI-16 lambda", "AI-17 fewshot", "AI-19 RUL", "AI-20 TC", "CAM frame"
};

/* AI-10 vibration PdM (real 2-class) + AI-7 thermal band names (Home + Pre-flight) */
static const char *const VIBN[2] = { "stopped", "running" };
static const char *const BANDN[3] = { "POOR", "MARGINAL", "GOOD" };
static const uint32_t    BANDC[3] = { COL_RED, COL_AMBER, COL_GREEN };

/* recipe segment kinds + ledger state names (mirror furnace_ctrl enums) */
static const char *const SEGKIND[4] = { "RAMP", "SOAK", "GRIND", "COOL" };
static const uint32_t    SEGCOL[4]  = { COL_BLUE, COL_ORANGE, COL_GRAY, 0x33AACCU };
static const char *const SEGICON[4] = { LV_SYMBOL_UP, LV_SYMBOL_PAUSE, LV_SYMBOL_LOOP, LV_SYMBOL_DOWN };
static const char *const STATEN[4]  = { "IDLE", "RUN", "DONE", "FAULT" };

#define QLEDGER_N 8   /* must match LEDGER_N in lab_sentinel.c */

/* ------------------------------------------------------------------ */
/* Widget handles                                                      */
/* ------------------------------------------------------------------ */
/* status bar + nav */
static lv_obj_t *s_statusbar, *s_status_risk, *s_status_batch, *s_status_temp;
static lv_obj_t *s_status_ai5;   /* AI-5 root-cause badge (always-visible, fault only) */
static lv_obj_t *s_status_pill;  /* rounded risk chip (bright fill coloured by AI-4 risk) */
static lv_obj_t *s_tab[NPAGE], *s_page[NPAGE];
static int       s_cur_page = 0;
/* modal confirm overlay (destructive ABORT) — kept built/hidden; ABORT now uses
 * an on-button two-tap confirm instead (the modal's nested buttons were unreliable
 * to tap on this GT911 panel). */
static lv_obj_t *s_modal;
/* MOTOR toggle button (separate from sinter START) + state. */
static lv_obj_t *s_motor_btn;
static uint8_t   s_motor_on = 0u;
/* ABORT two-tap confirm: first tap arms ("TAP AGAIN"), second within the window
 * fires the abort; auto-reverts after ABORT_ARM_MS (checked in ui_screen_tick).
 * 10 s window: 4 s was too short — a careful operator re-aiming for the 2nd tap on
 * this destructive action routinely exceeded it, so the arm expired and every tap
 * merely re-armed instead of confirming. */
static lv_obj_t *s_abort_btn;
static uint8_t   s_abort_armed = 0u;
static uint32_t  s_abort_arm_ms = 0u;
#define ABORT_ARM_MS  10000u
/* Home */
static lv_obj_t *s_home_temp, *s_home_seg, *s_curve, *s_bar_prog, *s_lbl_prog;
/* 20 model lights: [0..3]=AI-1/2/3/4, [4]=AI-1b NCM, [5]=AI-5 root cause,
 * [6]=AI-6 optical, [7]=AI-7 thermal, [8]=AI-8 energy, [9]=AI-9 analog,
 * [10]=AI-10 vib, [11]=AI-11 purity, [12]=AI-12 PL dopant, [13]=AI-13 PL QC,
 * [14]=AI-14 forecast, [15]=AI-15 host-ID, [16]=AI-16 lambda, [17]=AI-17 PL few-shot,
 * [18]=AI-19 RUL/ETA, [19]=AI-20 TC-integrity */
static lv_obj_t *s_home_ai[20];
/* Camera (OV5640 -> AI-1): luminance-tile grid + bright-blob box + verdict */
static lv_obj_t *s_cam_tile[CAM_CELLS], *s_cam_box, *s_cam_boxlbl;
static lv_obj_t *s_cam_cls, *s_cam_conf, *s_cam_bar[3], *s_cam_info;
static lv_obj_t *s_cam_cam[16];   /* B3 CAM 4x4 class-activation mini-heatmap */
/* Camera LIVE real-frame overlay (Wave C): a real RGB565 lv_img of the OV5640
 * frame + true/thermal toggle + back-to-tiles. The DEFAULT Camera page is
 * untouched, so this flicker-prone live path can never regress the tile view. */
static lv_obj_t  *s_camview, *s_cam_img, *s_cam_modebtn, *s_cam_modeinfo;
static lv_img_dsc_t s_cam_dsc;
static int        s_camview_open = 0, s_cam_mode = 1;   /* 1 true-colour, 2 thermal */
/* Pre-flight (AI-6/7/8/9 for the active recipe) + AI-11 purity + derived Dq/B */
static lv_obj_t *s_pf_recipe, *s_pf_lam, *s_pf_fwhm, *s_pf_thermal, *s_pf_band;
static lv_obj_t *s_pf_energy, *s_pf_co2, *s_pf_bar_e, *s_pf_analog;
static lv_obj_t *s_pf_purity, *s_pf_dqb;     /* AI-11 prior + derived crystal field */
/* PL-spectrum page (AI-12 dopant classifier + AI-13 QC autoencoder + AI-15/16/17) */
static lv_obj_t *s_pl_line, *s_pl_cls, *s_pl_conf, *s_pl_bar[3], *s_pl_qc, *s_pl_mse, *s_pl_info;
static lv_obj_t *s_pl_host, *s_pl_lambda, *s_pl_fewshot;   /* AI-15 / AI-16 / AI-17 */
static const char *const HOSTN[2] = { "NaYGaInGe", "Y3ZnGaGe" };  /* short host tags */
static lv_point_t s_pl_pts[PL_SPEC_N];
/* Models page (20 clickable cards) -> per-model detail overlay + Benchmark overlay.
 * Both overlay the body region and are hidden by _show_page() on any tab tap. */
static lv_obj_t *s_card[N_MODELS];
static lv_obj_t *s_detail, *s_bench;
static lv_obj_t *s_det_title, *s_det_purpose, *s_det_io, *s_det_data,
                *s_det_metric, *s_det_lat, *s_det_live, *s_det_bar;
static lv_obj_t *s_det_blk[ARCH_MAX], *s_det_blklbl[ARCH_MAX], *s_det_arr[ARCH_MAX - 1];
static lv_obj_t *s_det_inj[3];               /* AI-20 live fault-inject buttons       */
static lv_obj_t *s_bench_us[AI_LAT_N], *s_bench_bar[AI_LAT_N];
static int       s_detail_open = 0, s_detail_idx = -1, s_det_live_sig = -123456;
/* Edge LLM cluster overlay (7 swap-loaded experts) */
#define CL_NEXP 7
/* ASCII row labels + FontAwesome icon (both in montserrat, no CJK glyph needed);
 * the generated sentence is the only CJK text and is always in the font's vocab. */
static const char *const CL_ROW[CL_NEXP] = {
    LV_SYMBOL_WARNING "  E1 diag    fault diagnosis",
    LV_SYMBOL_LIST    "  E2 recipe  process advice",
    LV_SYMBOL_CHARGE  "  E3 energy  power & carbon",
    LV_SYMBOL_OK      "  E4 qc      batch verdict",
    LV_SYMBOL_FILE    "  E5 brief   operator brief",
    LV_SYMBOL_EYE_OPEN"  E6 chem    formula chemistry",
    LV_SYMBOL_SETTINGS"  E7 maint   equipment PdM",
};
static lv_obj_t *s_cluster_ov, *s_cl_row[CL_NEXP], *s_cl_text, *s_cl_meta;
static int       s_cluster_open = 0;
static unsigned  s_cl_gens_seen = 0xFFFFFFFFu;
/* Recipe */
static lv_obj_t *s_seg_row[8];
static int       s_seg_rows;
/* Trend (SPC) */
static lv_obj_t *s_spc_line;
static lv_obj_t *s_trend_cpk, *s_trend_mean, *s_trend_sigma, *s_trend_inctl;
static lv_obj_t *s_trend_fc, *s_trend_fc_val;   /* AI-14 temp forecast readout */
/* Quality (batch ledger) */
static lv_obj_t *s_q_list, *s_q_chain, *s_q_count;
static int       s_q_built_total = -1;
/* Control (tab 3): MAX31855 closed-loop control telemetry (top half) + on-chip
 * EDGE LM diagnosis + an ESP32 TELEMETRY UPLINK (export only, bottom half). The live AI-1..5 / NCM
 * readouts moved to the Home lights + Models detail page; the AI-2/AI-3
 * explainability (AE attribution + q^, attention peak) is folded into the Models
 * detail "live" line via the stashes below (filled in ui_screen_update_ai). */
static lv_obj_t *s_cc_state, *s_cc_seg, *s_cc_sp, *s_cc_pv, *s_cc_probe;
static lv_obj_t *s_cc_u, *s_cc_ubar, *s_cc_tc, *s_cc_cpk, *s_cc_elem;
static lv_obj_t *s_cc_link, *s_cc_rssi, *s_cc_up, *s_cc_r1;
static lv_obj_t *s_cc_nlm, *s_cc_olrow;   /* edge nano-LM diagnosis + online-head readout */
static lv_obj_t *s_cc_lm;                 /* active generative-LM size indicator */
static int       s_cc_lm_shown = -1;      /* change-gate for the LM indicator */
static int s_ai2_attr_top   = -1;  /* dominant AE residual: 0=T 1=V 2=G  */
static int s_ai2_qhat_milli = -1;  /* AI-2 conformal q^ x1000            */
static int s_ai2_ratio_x10  = -1;  /* AI-2 anomaly ratio x10             */
static int s_ai3_attn_peak  = -1;  /* AI-3 peak-attention minute (0..63) */
/* System */
static lv_obj_t *s_log_list, *s_lbl_voice, *s_lbl_uptime;

/* buffers (lv_line keeps the pointer — must persist) */
static lv_point_t s_pts[CURVE_N];
static float      s_temps[CURVE_N];
static int        s_tcount;
static lv_point_t s_spc_pts[SPC_N];
static int16_t    s_spc_dev[SPC_N];
static lv_point_t s_ucl_pts[2], s_cl_pts[2], s_lcl_pts[2];

static int        s_last_risk;
static uint32_t   s_uptime_s;

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */
static lv_obj_t *_label(lv_obj_t *p, lv_coord_t x, lv_coord_t y,
                        const lv_font_t *f, uint32_t color, const char *t)
{
    lv_obj_t *l = lv_label_create(p);
    lv_label_set_text(l, t);
    lv_obj_set_style_text_color(l, lv_color_hex(color), LV_PART_MAIN);
    lv_obj_set_style_text_font(l, f, LV_PART_MAIN);
    lv_obj_set_pos(l, x, y);
    return l;
}

static lv_obj_t *_bar(lv_obj_t *p, lv_coord_t x, lv_coord_t y,
                      lv_coord_t w, lv_coord_t h)
{
    lv_obj_t *b = lv_bar_create(p);
    lv_obj_set_pos(b, x, y);
    lv_obj_set_size(b, w, h);
    lv_bar_set_range(b, 0, 100);
    lv_bar_set_value(b, 0, LV_ANIM_OFF);
    lv_obj_set_style_bg_color(b, lv_color_hex(COL_DIM), LV_PART_MAIN);
    lv_obj_set_style_radius(b, 7, LV_PART_MAIN);
    lv_obj_set_style_border_width(b, 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(b, lv_color_hex(COL_BORDER), LV_PART_MAIN);
    lv_obj_set_style_bg_color(b, lv_color_hex(COL_BLUE), LV_PART_INDICATOR);
    lv_obj_set_style_radius(b, 7, LV_PART_INDICATOR);
    return b;
}

static void _set_bar(lv_obj_t *b, int v, uint32_t color)
{
    if (v < 0)   v = 0;
    if (v > 100) v = 100;
    lv_bar_set_value(b, v, LV_ANIM_OFF);
    lv_obj_set_style_bg_color(b, lv_color_hex(color), LV_PART_INDICATOR);
}

/* a static line at fixed local y across the chart width (for SPC limits) */
static lv_obj_t *_hline(lv_obj_t *p, lv_point_t *pts, int yy, uint32_t color, int dashed)
{
    lv_obj_t *ln = lv_line_create(p);
    lv_obj_set_pos(ln, CHX, CHY);
    pts[0].x = 0;        pts[0].y = (lv_coord_t)yy;
    pts[1].x = CHW - 1;  pts[1].y = (lv_coord_t)yy;
    lv_line_set_points(ln, pts, 2);
    lv_obj_set_style_line_width(ln, dashed ? 1 : 2, LV_PART_MAIN);
    lv_obj_set_style_line_color(ln, lv_color_hex(color), LV_PART_MAIN);
    return ln;
}

/* ------------------------------------------------------------------ */
/* Buttons / tabs (lv_btn not compiled in -> clickable lv_obj + label) */
/* ------------------------------------------------------------------ */
static void _btn_set_label(lv_obj_t *btn, const char *txt)
{
    if (btn != NULL) {
        lv_obj_t *l = lv_obj_get_child(btn, 0);
        if (l != NULL) lv_label_set_text(l, txt);
    }
}

static void _abort_disarm(void)
{
    s_abort_armed = 0u;
    _btn_set_label(s_abort_btn, LV_SYMBOL_STOP "  ABORT");
    if (s_abort_btn != NULL)
        lv_obj_set_style_bg_color(s_abort_btn, lv_color_hex(COL_RED), LV_PART_MAIN);
}

static void _btn_start_cb(lv_event_t *e)
{
    (void)e;
    lab_log("[ui] touch START button\r\n");
    _abort_disarm();                 /* clear any pending abort arm */
    lab_ctrl_request(1);
    ui_screen_set_voice_status("TOUCH:START");
}

/* MOTOR toggle (separate from sinter START) — drives the stirring/vibration motor
 * that AI-10 does PdM on, so the vibration demo runs only when you want it. */
static void _btn_motor_cb(lv_event_t *e)
{
    (void)e;
    s_motor_on = (uint8_t)(!s_motor_on);
    lab_motor_request((int)s_motor_on);
    _btn_set_label(s_motor_btn, s_motor_on ? LV_SYMBOL_LOOP "  MOTOR ON" : LV_SYMBOL_LOOP "  MOTOR");
    if (s_motor_btn != NULL)
        lv_obj_set_style_bg_color(s_motor_btn,
            lv_color_hex(s_motor_on ? COL_AMBER : COL_BLUE), LV_PART_MAIN);
    lab_log(s_motor_on ? "[ui] MOTOR on\r\n" : "[ui] MOTOR off\r\n");
}

/* ABORT is destructive (heater off + batch FAULT). On-BUTTON two-tap confirm
 * (replaces the old modal whose nested buttons wouldn't register on this GT911):
 * 1st tap arms + relabels "TAP AGAIN"; a 2nd tap within ABORT_ARM_MS fires the
 * abort; otherwise ui_screen_tick auto-reverts. Reuses the proven main-button. */
static void _btn_abort_cb(lv_event_t *e)
{
    (void)e;
    if (!s_abort_armed) {
        s_abort_armed  = 1u;
        s_abort_arm_ms = lv_tick_get();
        _btn_set_label(s_abort_btn, LV_SYMBOL_WARNING "  TAP AGAIN");
        if (s_abort_btn != NULL)
            lv_obj_set_style_bg_color(s_abort_btn, lv_color_hex(COL_AMBER), LV_PART_MAIN);
        lab_log("[ui] ABORT armed - tap again to confirm\r\n");
    } else {
        lab_log("[ui] ABORT confirmed (two-tap)\r\n");
        lab_ctrl_request(2);
        ui_screen_set_voice_status("TOUCH:ABORT");
        _abort_disarm();
    }
}

static void _btn_confirm_cb(lv_event_t *e)
{
    (void)e;
    lab_log("[ui] ABORT confirmed\r\n");
    lab_ctrl_request(2);
    ui_screen_set_voice_status("TOUCH:ABORT");
    if (s_modal != NULL) lv_obj_add_flag(s_modal, LV_OBJ_FLAG_HIDDEN);
}

static void _btn_cancel_cb(lv_event_t *e)
{
    (void)e;
    lab_log("[ui] ABORT cancelled\r\n");
    if (s_modal != NULL) lv_obj_add_flag(s_modal, LV_OBJ_FLAG_HIDDEN);
}

/* Control page: force a fresh nano-LM generation now. */
static void _btn_nlmrefresh_cb(lv_event_t *e)
{
    (void)e;
    lab_nlm_request();
    lab_log("[ui] nano-LM regenerate\r\n");
}

/* Control page: cycle the active generative LM across the hardware-ceiling curve
 * (internal x1p9 1.8M -> SPI-bank m1p35 1.26M -> s0p6 0.6M -> ...). Demonstrates
 * the swap-load size selector live. */
static void _btn_lmcycle_cb(lv_event_t *e)
{
    (void)e;
    lab_lm_cycle();
    lab_log("[ui] switch LM size\r\n");
}

/* Control page: teach the online-learning head that the CURRENT controller risk
 * is correct (one on-chip SGD step). Demonstrates on-device continual learning. */
static void _btn_teach_cb(lv_event_t *e)
{
    ctrl_snapshot_t cc;
    (void)e;
    lab_ctrl_get(&cc);
    lab_online_teach((int)cc.risk);
    lab_log("[ui] online-learn teach (confirm current risk)\r\n");
}

static lv_obj_t *_ctrl_btn(lv_obj_t *p, lv_coord_t x, lv_coord_t y,
                      lv_coord_t w, lv_coord_t h, uint32_t bg,
                      const char *txt, lv_event_cb_t cb)
{
    lv_obj_t *b = lv_obj_create(p);
    lv_obj_set_pos(b, x, y);
    lv_obj_set_size(b, w, h);
    lv_obj_set_style_bg_color(b, lv_color_hex(bg), LV_PART_MAIN);
    lv_obj_set_style_bg_grad_color(b, lv_color_darken(lv_color_hex(bg), LV_OPA_40), LV_PART_MAIN);
    lv_obj_set_style_bg_grad_dir(b, LV_GRAD_DIR_VER, LV_PART_MAIN);
    lv_obj_set_style_border_width(b, 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(b, lv_color_hex(0xFFFFFFU), LV_PART_MAIN);
    lv_obj_set_style_border_opa(b, LV_OPA_30, LV_PART_MAIN);
    lv_obj_set_style_radius(b, 10, LV_PART_MAIN);
    lv_obj_set_style_pad_all(b, 0, LV_PART_MAIN);
    /* subtle press feedback (no animation -> no flicker): darken on PRESSED state */
    lv_obj_set_style_bg_color(b, lv_color_darken(lv_color_hex(bg), LV_OPA_30), LV_STATE_PRESSED);
    lv_obj_clear_flag(b, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_flag(b, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_add_event_cb(b, cb, LV_EVENT_PRESSED, NULL);
    {
        lv_obj_t *l = lv_label_create(b);
        lv_label_set_text(l, txt);
        lv_obj_set_style_text_color(l, lv_color_white(), LV_PART_MAIN);
        lv_obj_set_style_text_font(l, &lv_font_montserrat_20, LV_PART_MAIN);
        lv_obj_center(l);
    }
    return b;
}

/* show page i, hide the rest, highlight tab i */
static void _show_page(int i)
{
    int k;
    if (i < 0 || i >= NPAGE) return;
    /* leaving a page also dismisses the Models detail/benchmark + Camera LIVE overlays */
    if (s_detail  != NULL) lv_obj_add_flag(s_detail,  LV_OBJ_FLAG_HIDDEN);
    if (s_bench   != NULL) lv_obj_add_flag(s_bench,   LV_OBJ_FLAG_HIDDEN);
    if (s_camview != NULL) lv_obj_add_flag(s_camview, LV_OBJ_FLAG_HIDDEN);
    if (s_cluster_ov != NULL) lv_obj_add_flag(s_cluster_ov, LV_OBJ_FLAG_HIDDEN);
    s_detail_open  = 0;
    s_detail_idx   = -1;
    s_camview_open = 0;
    s_cluster_open = 0;
    for (k = 0; k < NPAGE; k++) {
        if (k == i) lv_obj_clear_flag(s_page[k], LV_OBJ_FLAG_HIDDEN);
        else        lv_obj_add_flag(s_page[k], LV_OBJ_FLAG_HIDDEN);
        lv_obj_set_style_bg_color(s_tab[k],
            lv_color_hex((k == i) ? COL_BLUE : COL_DIM), LV_PART_MAIN);
    }
    s_cur_page = i;
}

void ui_screen_set_nav(int tab) { _show_page(tab); }

/* one tiny callback per tab (no pointer<->int casts, matches button style) */
static void _tab0(lv_event_t *e){ (void)e; _show_page(0); }
static void _tab1(lv_event_t *e){ (void)e; _show_page(1); }
static void _tab2(lv_event_t *e){ (void)e; _show_page(2); }
static void _tab3(lv_event_t *e){ (void)e; _show_page(3); }
static void _tab4(lv_event_t *e){ (void)e; _show_page(4); }
static void _tab5(lv_event_t *e){ (void)e; _show_page(5); }
static void _tab6(lv_event_t *e){ (void)e; _show_page(6); }
static void _tab7(lv_event_t *e){ (void)e; _show_page(7); }
static void _tab8(lv_event_t *e){ (void)e; _show_page(8); }
static void _tab9(lv_event_t *e){ (void)e; _show_page(9); }
static void _tab10(lv_event_t *e){ (void)e; _show_page(10); }
static void _tab11(lv_event_t *e){ (void)e; _show_page(11); }
static void _tab12(lv_event_t *e){ (void)e; _show_page(12); }

/* one vertical navigation-rail item (idx 0..NPAGE-1, stacked top-to-bottom).
 * Replaces the old cramped horizontal tab bar: bigger touch target + room for
 * more sections + the left-rail "web dashboard" look. */
static lv_obj_t *_make_tab(lv_obj_t *p, int idx, const char *txt, lv_event_cb_t cb)
{
    lv_obj_t *b = lv_obj_create(p);
    lv_obj_set_pos(b, 4, (lv_coord_t)(6 + idx * 36));
    lv_obj_set_size(b, NAV_W - 8, 33);          /* full rail width, 13 items high (6+12*36+33=471<480) */
    lv_obj_set_style_bg_color(b, lv_color_hex(COL_DIM), LV_PART_MAIN);
    lv_obj_set_style_border_width(b, 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(b, lv_color_hex(COL_BORDER), LV_PART_MAIN);
    lv_obj_set_style_radius(b, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_all(b, 0, LV_PART_MAIN);
    lv_obj_clear_flag(b, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_flag(b, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_set_ext_click_area(b, 2);
    lv_obj_add_event_cb(b, cb, LV_EVENT_PRESSED, NULL);
    {
        lv_obj_t *l = lv_label_create(b);
        lv_label_set_text(l, txt);
        lv_obj_set_style_text_color(l, lv_color_white(), LV_PART_MAIN);
        lv_obj_set_style_text_font(l, &lv_font_montserrat_14, LV_PART_MAIN);
        lv_obj_center(l);
    }
    return b;
}

/* ------------------------------------------------------------------ */
/* Page builders                                                       */
/* ------------------------------------------------------------------ */
static lv_obj_t *_new_page(lv_obj_t *scr)
{
    lv_obj_t *c = lv_obj_create(scr);
    lv_obj_set_pos(c, BODY_X, BODY_Y);
    lv_obj_set_size(c, BODY_W, BODY_H);
    lv_obj_set_style_bg_color(c, lv_color_hex(COL_BG), LV_PART_MAIN);
    lv_obj_set_style_bg_grad_color(c, lv_color_hex(COL_BG2), LV_PART_MAIN);
    lv_obj_set_style_bg_grad_dir(c, LV_GRAD_DIR_VER, LV_PART_MAIN);
    lv_obj_set_style_border_width(c, 0, LV_PART_MAIN);
    lv_obj_set_style_radius(c, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_all(c, 0, LV_PART_MAIN);
    lv_obj_clear_flag(c, LV_OBJ_FLAG_SCROLLABLE);
    return c;
}

static void _build_home(lv_obj_t *c)
{
    int i;
    _label(c, 10, 8, &lv_font_montserrat_20, COL_GRAY, LV_SYMBOL_CHARGE "  FURNACE");
    s_home_temp = _label(c, 170, 2, &lv_font_montserrat_28, COL_ORANGE, "-- C");
    s_home_seg  = _label(c, 10, 44, &lv_font_montserrat_14, 0xCCCCCCU, "idle");

    s_curve = lv_line_create(c);
    lv_obj_set_pos(s_curve, CURVE_X, CURVE_Y);
    lv_obj_set_size(s_curve, CURVE_W, CURVE_H);
    lv_obj_set_style_line_width(s_curve, 2, LV_PART_MAIN);
    lv_obj_set_style_line_color(s_curve, lv_color_hex(COL_GREEN), LV_PART_MAIN);
    lv_obj_set_style_line_rounded(s_curve, true, LV_PART_MAIN);

    _label(c, 10, 210, &lv_font_montserrat_14, COL_GRAY, "PROG");
    s_bar_prog = _bar(c, 70, 212, 360, 14);
    _set_bar(s_bar_prog, 0, COL_BLUE);
    s_lbl_prog = _label(c, 440, 208, &lv_font_montserrat_14, 0xCCCCCCU, "0%");

    /* right column: all 20 AI model lights. [0..3] AI-1/2/3/4, [4] AI-1b NCM,
     * [5] AI-5 root cause, [6..9] AI-6 optical/AI-7 thermal/AI-8 energy/AI-9 analog,
     * [10] AI-10 vib, [11] AI-11 purity, [12] AI-12 PL dopant, [13] AI-13 PL QC,
     * [14] AI-14 forecast, [15] AI-15 host-ID, [16] AI-16 lambda, [17] AI-17 PL few-shot,
     * [18] AI-19 RUL/ETA, [19] AI-20 TC-integrity. 17px pitch fits all 20 in the right
     * column (last line bottom ~363 < 370; the START/MOTOR/ABORT buttons own the LEFT
     * half x<498 so the taller light list never overlaps them). */
    _label(c, 498, 4, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_EYE_OPEN "  DISCRIMINATIVE (20)");
    for (i = 0; i < 20; i++)
        s_home_ai[i] = _label(c, 498, (lv_coord_t)(22 + i * 17),
                              &lv_font_montserrat_14, 0xDDDDDDU, "--");

    /* control buttons. 3 across: START | MOTOR | ABORT, centred on the proven
     * abs y~382 touch band. START = begin sintering only; MOTOR = the separate
     * stirring/vibration motor (AI-10 PdM); ABORT = two-tap confirm.
     * They share the left half (AI-model column owns x>=498), so MOTOR/ABORT were
     * small, adjacent targets and the 2nd tap (toggle off / confirm) was easy to
     * miss. Fix = taller buttons + a 10px EXTENDED CLICK AREA per button so a
     * slightly-off tap still registers; the 24px gaps keep the extended hit zones
     * from overlapping (10+10 < 24) so taps never route to the wrong button. */
    {
        lv_obj_t *bs, *bm, *ba;
        bs = _ctrl_btn(c,   6, 230, 145, 84, COL_GREEN, LV_SYMBOL_PLAY "  START", _btn_start_cb);
        bm = _ctrl_btn(c, 175, 230, 145, 84, COL_BLUE,  LV_SYMBOL_LOOP "  MOTOR", _btn_motor_cb);
        ba = _ctrl_btn(c, 344, 230, 145, 84, COL_RED,   LV_SYMBOL_STOP "  ABORT", _btn_abort_cb);
        lv_obj_set_ext_click_area(bs, 10);
        lv_obj_set_ext_click_area(bm, 10);
        lv_obj_set_ext_click_area(ba, 10);
        s_motor_btn = bm;
        s_abort_btn = ba;
    }

    /* ── GENERATIVE AI panel (free band below the 3 buttons, left half x<490) ──
     * Static capability roster (built once, never touched by the tick) so it
     * adds ZERO risk to the 20-light discriminative list / its change-gating.
     * Showcases the on-chip generative stack the Home page was missing:
     *   - nano-LM bank x3  (x1p9 1.8M / m1p35 1.26M / s0p6 0.6M, SPI swap-load)
     *   - LLM cluster  x7  (E1 diag/E2 recipe/E3 energy/E4 qc/E5 brief/E6 chem/E7 maint)
     * Live switching + generated CN text live on the Control page; here it is the
     * "this device also runs generative LMs on-chip" headline. Accent-purple
     * border marks it as the AI showcase. (In the UI-dev fast build the LM stack
     * is not spawned, but the roster is a true capability statement either way.) */
    {
        lv_obj_t *card = lv_obj_create(c);
        lv_obj_set_pos(card, 6, 318);
        lv_obj_set_size(card, 484, 114);
        lv_obj_set_style_bg_color(card, lv_color_hex(COL_CARD), LV_PART_MAIN);
        lv_obj_set_style_border_width(card, 1, LV_PART_MAIN);
        lv_obj_set_style_border_color(card, lv_color_hex(COL_ACCENT), LV_PART_MAIN);
        lv_obj_set_style_radius(card, 8, LV_PART_MAIN);
        lv_obj_set_style_pad_all(card, 0, LV_PART_MAIN);
        lv_obj_clear_flag(card, LV_OBJ_FLAG_SCROLLABLE);

        _label(card, 10,  6, &lv_font_montserrat_14, COL_ACCENT, LV_SYMBOL_KEYBOARD "  GENERATIVE  AI  -  ON-CHIP");

        _label(card, 10, 30, &lv_font_montserrat_14, COL_TEXT2, "nano-LM bank  x3 (SPI)");
        _label(card, 18, 50, &lv_font_montserrat_14, COL_TEXT,  "x1p9   1.8M");
        _label(card, 18, 68, &lv_font_montserrat_14, COL_TEXT,  "m1p35  1.26M");
        _label(card, 18, 86, &lv_font_montserrat_14, COL_TEXT,  "s0p6   0.6M");

        _label(card, 244, 30, &lv_font_montserrat_14, COL_TEXT2, "LLM cluster  x7 (SPI)");
        _label(card, 252, 48, &lv_font_montserrat_14, COL_TEXT,  "E1 diag     E2 recipe");
        _label(card, 252, 63, &lv_font_montserrat_14, COL_TEXT,  "E3 energy   E4 qc");
        _label(card, 252, 78, &lv_font_montserrat_14, COL_TEXT,  "E5 brief    E6 chem");
        _label(card, 252, 93, &lv_font_montserrat_14, COL_TEXT,  "E7 maint");
    }
}

static void _build_recipe(lv_obj_t *c)
{
    recipe_seg_view_t sg[8];
    char buf[52];
    int  n, i;

    n = lab_recipe_segs(sg, 8);
    snprintf(buf, sizeof(buf), LV_SYMBOL_LIST "  RECIPE: %s    total %d min",
             lab_recipe_name(), lab_recipe_total_min());
    _label(c, 10, 6, &lv_font_montserrat_20, 0xEEEEEEU, buf);

    s_seg_rows = (n > 8) ? 8 : n;
    for (i = 0; i < s_seg_rows; i++) {
        lv_coord_t ry = (lv_coord_t)(40 + i * 52);
        uint8_t    k  = (sg[i].kind > 3) ? 3 : sg[i].kind;
        lv_obj_t  *row = lv_obj_create(c);

        lv_obj_set_pos(row, 10, ry);
        lv_obj_set_size(row, 696, 46);
        lv_obj_set_style_bg_color(row, lv_color_hex(0x141414U), LV_PART_MAIN);
        lv_obj_set_style_border_width(row, 1, LV_PART_MAIN);
        lv_obj_set_style_border_color(row, lv_color_hex(0x2A2A2AU), LV_PART_MAIN);
        lv_obj_set_style_radius(row, 4, LV_PART_MAIN);
        lv_obj_set_style_pad_all(row, 0, LV_PART_MAIN);
        lv_obj_clear_flag(row, LV_OBJ_FLAG_SCROLLABLE);
        s_seg_row[i] = row;

        snprintf(buf, sizeof(buf), "%d", i + 1);
        _label(row, 10, 12, &lv_font_montserrat_20, COL_GRAY, buf);
        snprintf(buf, sizeof(buf), "%s  %s", SEGICON[k], SEGKIND[k]);
        _label(row, 44, 14, &lv_font_montserrat_14, SEGCOL[k], buf);
        _label(row, 140, 4, &lv_font_montserrat_14, 0xDDDDDDU, sg[i].label);

        if (k == 1)       /* soak  */
            snprintf(buf, sizeof(buf), "hold %d C    %d min", sg[i].target_c, sg[i].dur_min);
        else if (k == 2)  /* grind */
            snprintf(buf, sizeof(buf), "heater OFF    %d min", sg[i].dur_min);
        else              /* ramp/cool */
            snprintf(buf, sizeof(buf), "%d C/min  ->  %d C", sg[i].rate_c_min, sg[i].target_c);
        _label(row, 140, 24, &lv_font_montserrat_14, 0x999999U, buf);
    }
}

static void _build_trend(lv_obj_t *c)
{
    int mid = CHH / 2;
    int yu  = mid - 10 * mid / DEV_FS;   /* +10C */
    int yl  = mid + 10 * mid / DEV_FS;   /* -10C */

    _label(c, CHX, 4, &lv_font_montserrat_14, COL_GRAY,
           LV_SYMBOL_BARS "  SINTER SOAK SPC  (dev = T - SP, C)");

    /* limit lines first (under the trace) */
    _hline(c, s_ucl_pts, yu,  COL_RED,  1);
    _hline(c, s_cl_pts,  mid, COL_GRAY, 1);
    _hline(c, s_lcl_pts, yl,  COL_RED,  1);

    /* deviation trace */
    s_spc_line = lv_line_create(c);
    lv_obj_set_pos(s_spc_line, CHX, CHY);
    lv_obj_set_size(s_spc_line, CHW, CHH);
    lv_obj_set_style_line_width(s_spc_line, 2, LV_PART_MAIN);
    lv_obj_set_style_line_color(s_spc_line, lv_color_hex(COL_GREEN), LV_PART_MAIN);

    _label(c, CHX + 4, (lv_coord_t)(CHY + yu - 16), &lv_font_montserrat_14, COL_RED, "+10");
    _label(c, CHX + 4, (lv_coord_t)(CHY + yl + 2),  &lv_font_montserrat_14, COL_RED, "-10");

    /* right stats column */
    s_trend_cpk   = _label(c, 524, 24,  &lv_font_montserrat_28, COL_GREEN,  "Cpk --");
    s_trend_mean  = _label(c, 524, 70,  &lv_font_montserrat_14, 0xCCCCCCU, "mean --");
    s_trend_sigma = _label(c, 524, 96,  &lv_font_montserrat_14, 0xCCCCCCU, "sigma --");
    s_trend_inctl = _label(c, 524, 130, &lv_font_montserrat_20, COL_GRAY,  "--");

    /* AI-14 multi-step temperature forecast (predictive control readout) */
    s_trend_fc    = _label(c, 524, 168, &lv_font_montserrat_14, COL_GRAY,  LV_SYMBOL_UP "  AI-14 FORECAST");
    s_trend_fc_val= _label(c, 524, 188, &lv_font_montserrat_14, 0xCCCCCCU, "+12min --");
}

/* tab 3 — Control "live plant" page. Top half = the MAX31856 closed-loop
 * control telemetry (setpoint vs the sim plant PV vs the REAL K-type probe, PID
 * duty, segment, thermocouple-fault, Cpk, element health); bottom half = the
 * on-chip EDGE nano-LM diagnosis + online-learn head, plus an ESP32 TELEMETRY
 * UPLINK (export only: link state / RSSI / uplink count). The uplink EXPORTS
 * telemetry; it never imports a cloud verdict and offloads no process-model
 * compute. All values come from existing read-only snapshots
 * (lab_ctrl_get + lab_get_cloud), refreshed change-gated in ui_screen_tick.
 * The MAX31856 is read every control step and its fault bit feeds the safety
 * supervisor; the closed loop itself still tracks the FOPDT sim plant (no real
 * 1500C furnace on the bench), so PV and PROBE are shown side by side honestly. */
static void _build_control(lv_obj_t *c)
{
    /* ---- top half: MAX31856 closed-loop control telemetry ---- */
    _label(c, 10, 6, &lv_font_montserrat_20, 0xCCCCCCU, LV_SYMBOL_CHARGE "  CLOSED-LOOP CONTROL  (MAX31856 K-type)");

    s_cc_state = _label(c, 12,  40, &lv_font_montserrat_14, 0xDDDDDDU, "STATE: --");
    s_cc_seg   = _label(c, 12,  68, &lv_font_montserrat_14, 0xDDDDDDU, "SEG: --");
    s_cc_sp    = _label(c, 12,  96, &lv_font_montserrat_14, 0x88CCFFU, "SETPOINT: -- C");
    s_cc_pv    = _label(c, 12, 124, &lv_font_montserrat_14, 0xDDDDDDU, "PV (sim plant): -- C");
    s_cc_probe = _label(c, 12, 152, &lv_font_montserrat_20, COL_GREEN, "PROBE (real TC): -- C");

    _label(c, 400,  40, &lv_font_montserrat_14, COL_GRAY, "HEATER DUTY");
    s_cc_u     = _label(c, 400,  62, &lv_font_montserrat_14, 0xDDDDDDU, "u: --%");
    s_cc_ubar  = _bar(c, 400, 88, 300, 14);
    _set_bar(s_cc_ubar, 0, COL_ORANGE);
    s_cc_tc    = _label(c, 400, 112, &lv_font_montserrat_14, COL_GREEN, "TC: --");
    s_cc_cpk   = _label(c, 400, 140, &lv_font_montserrat_14, 0xDDDDDDU, "Cpk: --");
    s_cc_elem  = _label(c, 400, 168, &lv_font_montserrat_14, 0xDDDDDDU, "ELEMENT: --%");

    /* ---- bottom half: optional telemetry export (disabled by default). The flagship EDGE
     *      generative-LM diagnosis now has its own dedicated "Edge LM" tab (it
     *      earns the room) — here we keep just the uplink + a pointer to it. The ESP
     *      link only EXPORTS risk/temp/Cpk/batch; it never imports a cloud verdict and
     *      offloads no process-model inference.
     *      design-note: do NOT re-introduce a "cloud R1 does the diagnosis" path here. */
    _label(c, 10, 196, &lv_font_montserrat_20, 0xCCCCCCU, LV_SYMBOL_WIFI "  OPTIONAL TELEMETRY EXPORT  (disabled by default)");

    s_cc_link  = _label(c, 12,  232, &lv_font_montserrat_20, COL_GRAY,  "LINK: OFFLINE");
    s_cc_rssi  = _label(c, 320, 238, &lv_font_montserrat_14, 0xCCCCCCU, "WiFi: --");
    s_cc_up    = _label(c, 560, 238, &lv_font_montserrat_14, 0xCCCCCCU, "uplinks: 0");

    s_cc_r1    = _label(c, 12, 272, &lv_font_montserrat_14, 0x88CCFFU,
                        "exports risk / temp / Cpk / batch  -  process-model inference remains on GD32");
    lv_obj_set_width(s_cc_r1, 700);
    lv_label_set_long_mode(s_cc_r1, LV_LABEL_LONG_WRAP);

    /* the flagship on-chip generative diagnosis lives on its own page now */
    _label(c, 12, 320, &lv_font_montserrat_14, COL_ACCENT,
           LV_SYMBOL_KEYBOARD "  flagship generative diagnosis  ->  see the  \"Edge LM\"  tab");
}

static void _build_quality(lv_obj_t *c)
{
    _label(c, 10, 6,  &lv_font_montserrat_20, 0xEEEEEEU, LV_SYMBOL_SD_CARD "  BATCH LEDGER");
    _label(c, 10, 36, &lv_font_montserrat_14, COL_GRAY,
           "SHA-256 hash-chained electronic batch records (21 CFR 11 style)");

    s_q_count = _label(c, 500, 6,  &lv_font_montserrat_14, 0xCCCCCCU, "sealed: 0");
    s_q_chain = _label(c, 500, 30, &lv_font_montserrat_20, COL_GRAY,  "CHAIN: --");

    s_q_list = lv_list_create(c);
    lv_obj_set_pos(s_q_list, 10, 64);
    lv_obj_set_size(s_q_list, 696, 360);
    lv_obj_set_style_bg_color(s_q_list, lv_color_hex(0x101010U), LV_PART_MAIN);
    lv_obj_set_style_border_width(s_q_list, 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(s_q_list, lv_color_hex(0x333333U), LV_PART_MAIN);
    lv_obj_set_style_radius(s_q_list, 4, LV_PART_MAIN);
    lv_obj_set_style_pad_all(s_q_list, 4, LV_PART_MAIN);
    {
        lv_obj_t *e = lv_list_add_text(s_q_list, "no batches sealed yet  -  run a batch (Home: START) to seal record #1");
        lv_obj_set_style_text_color(e, lv_color_hex(0x777777U), LV_PART_MAIN);
    }
}

/* Rebuild the ledger list + chain-verify badge. Called from ui_screen_tick
 * only when a new batch seals (cheap-ish: one SHA pass + a list rebuild). */
static void _quality_refresh(void)
{
    batch_view_t bv[QLEDGER_N];
    char buf[80];
    int  n, i, total, chain;

    total = lab_ledger_total();
    snprintf(buf, sizeof(buf), "sealed: %d", total);
    lv_label_set_text(s_q_count, buf);

    chain = lab_ledger_chain_ok();
    if (chain == 1) {
        lv_label_set_text(s_q_chain, LV_SYMBOL_OK " CHAIN: INTACT");
        lv_obj_set_style_text_color(s_q_chain, lv_color_hex(COL_GREEN), LV_PART_MAIN);
    } else if (chain == 0) {
        lv_label_set_text(s_q_chain, LV_SYMBOL_WARNING " CHAIN: TAMPERED");
        lv_obj_set_style_text_color(s_q_chain, lv_color_hex(COL_RED), LV_PART_MAIN);
    } else {
        lv_label_set_text(s_q_chain, "CHAIN: --");
        lv_obj_set_style_text_color(s_q_chain, lv_color_hex(COL_GRAY), LV_PART_MAIN);
    }

    lv_obj_clean(s_q_list);
    n = lab_ledger_get(bv, QLEDGER_N);
    if (n == 0) {
        lv_obj_t *e = lv_list_add_text(s_q_list, "no batches sealed yet  -  run a batch (Home: START) to seal record #1");
        lv_obj_set_style_text_color(e, lv_color_hex(0x777777U), LV_PART_MAIN);
        return;
    }
    for (i = 0; i < n; i++) {
        int      w = bv[i].cpk_x100 / 100, f = bv[i].cpk_x100 % 100;
        uint32_t col;
        lv_obj_t *e;
        if (f < 0) f = -f;
        col = (bv[i].final_state == 3) ? COL_RED
            : (bv[i].in_control && bv[i].capable) ? COL_GREEN : COL_AMBER;
        snprintf(buf, sizeof(buf),
                 "#%lu %s peak%dC Cpk%d.%02d %s a%d el%d%% %s..",
                 (unsigned long)bv[i].batch_id,
                 STATEN[bv[i].final_state & 3],
                 bv[i].peak_c, w, f,
                 bv[i].in_control ? "in-ctl" : "OOC",
                 (int)bv[i].ai_alarms, bv[i].elem_pct,
                 bv[i].hash12);
        e = lv_list_add_text(s_q_list, buf);
        lv_obj_set_style_text_color(e, lv_color_hex(col), LV_PART_MAIN);
    }
}

static void _build_system(lv_obj_t *c)
{
    _label(c, 10, 6, &lv_font_montserrat_20, 0xEEEEEEU, LV_SYMBOL_SETTINGS "  SYSTEM");

    s_log_list = lv_list_create(c);
    lv_obj_set_pos(s_log_list, 10, 38);
    lv_obj_set_size(s_log_list, 500, 318);
    lv_obj_set_style_bg_color(s_log_list, lv_color_hex(0x101010U), LV_PART_MAIN);
    lv_obj_set_style_border_width(s_log_list, 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(s_log_list, lv_color_hex(0x333333U), LV_PART_MAIN);
    lv_obj_set_style_radius(s_log_list, 4, LV_PART_MAIN);
    lv_obj_set_style_pad_all(s_log_list, 4, LV_PART_MAIN);
    {
        lv_obj_t *e = lv_list_add_text(s_log_list, "[boot] Lab-Sentinel HMI ready");
        lv_obj_set_style_text_color(e, lv_color_hex(COL_GREEN), LV_PART_MAIN);
    }

    _label(c, 540, 12, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_AUDIO "  VOICE");
    s_lbl_voice  = _label(c, 540, 36, &lv_font_montserrat_20, 0xAAFFAAU, "IDLE");
    _label(c, 540, 80, &lv_font_montserrat_14, COL_GRAY, "UPTIME");
    s_lbl_uptime = _label(c, 540, 104, &lv_font_montserrat_14, 0xCCCCCCU, "0s");
    _label(c, 540, 150, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_OK "  WATCHDOG");
    _label(c, 540, 174, &lv_font_montserrat_14, COL_GREEN, "FWDGT armed ~4s");

    /* diagnostics: measured static system facts (boot self-test + architecture). */
    _label(c, 540, 214, &lv_font_montserrat_14, COL_GRAY,  LV_SYMBOL_LIST "  DIAGNOSTICS");
    _label(c, 540, 240, &lv_font_montserrat_14, COL_TEXT2, "AI: 20 models PASS");
    _label(c, 540, 262, &lv_font_montserrat_14, COL_TEXT2, "RTOS: 10 tasks");
    _label(c, 540, 284, &lv_font_montserrat_14, COL_TEXT2, "Disp: TLI 45.7 FPS");
    _label(c, 540, 306, &lv_font_montserrat_14, COL_TEXT2, "Touch: GT911");
    _label(c, 540, 328, &lv_font_montserrat_14, COL_TEXT2, "Safety: fixed interlocks");
    _label(c, 540, 350, &lv_font_montserrat_14, COL_TEXT2, "Edge: process AI local");
    /* GD32 Embedded AI Tool: AI-4 converted to TFLite and re-run on-chip (weights
     * traced to the tool's .tflite flatbuffer; boot self-test prints TFLITE=1). */
    _label(c, 540, 372, &lv_font_montserrat_14, COL_ACCENT, LV_SYMBOL_OK " AI Tool: TFLite AI-4");
}

/* Full-screen modal: confirm the destructive ABORT. Built once on the active
 * screen LAST so it sits on top of every page/status-bar/tab; hidden by default.
 * The dim backdrop is CLICKABLE so taps can't leak through to the tabs behind. */
static void _build_modal(lv_obj_t *scr)
{
    lv_obj_t *panel;

    s_modal = lv_obj_create(scr);
    lv_obj_set_pos(s_modal, 0, 0);
    lv_obj_set_size(s_modal, SCREEN_W, SCREEN_H);
    lv_obj_set_style_bg_color(s_modal, lv_color_hex(0x000000U), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(s_modal, LV_OPA_70, LV_PART_MAIN);
    lv_obj_set_style_border_width(s_modal, 0, LV_PART_MAIN);
    lv_obj_set_style_radius(s_modal, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_all(s_modal, 0, LV_PART_MAIN);
    lv_obj_clear_flag(s_modal, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_add_flag(s_modal, LV_OBJ_FLAG_CLICKABLE);   /* swallow backdrop taps */

    /* Text panel (upper). Buttons are NOT children of it — see below. */
    panel = lv_obj_create(s_modal);
    lv_obj_set_size(panel, 520, 150);
    lv_obj_set_pos(panel, (SCREEN_W - 520) / 2, 110);   /* 140,110 */
    lv_obj_set_style_bg_color(panel, lv_color_hex(0x1A1A1AU), LV_PART_MAIN);
    lv_obj_set_style_border_width(panel, 3, LV_PART_MAIN);
    lv_obj_set_style_border_color(panel, lv_color_hex(COL_RED), LV_PART_MAIN);
    lv_obj_set_style_radius(panel, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_all(panel, 0, LV_PART_MAIN);
    lv_obj_clear_flag(panel, LV_OBJ_FLAG_SCROLLABLE);

    _label(panel, 28, 34, &lv_font_montserrat_28, COL_RED,   "CONFIRM ABORT?");
    _label(panel, 28, 86, &lv_font_montserrat_14, 0xCCCCCCU, "Heater OFF, batch sealed as FAULT.");

    /* CONFIRM/CANCEL placed DIRECTLY on the overlay (abs coords) and ENLARGED,
     * centred at abs y~392 — the same low touch band where START/ABORT register
     * reliably on this GT911 panel (the old y~300 spot was hard to hit). */
    _ctrl_btn(s_modal, 110, 345, 260, 95, COL_RED,  "CONFIRM", _btn_confirm_cb);
    _ctrl_btn(s_modal, 430, 345, 260, 95, COL_GRAY, "CANCEL",  _btn_cancel_cb);

    lv_obj_add_flag(s_modal, LV_OBJ_FLAG_HIDDEN);
}

/* ================================================================== */
/* Camera LIVE real-frame overlay (Wave C). Shows the actual OV5640    */
/* RGB565 320x240 frame via lv_img (true colour or thermal LUT) — the  */
/* live-imaging capability other teams cannot field. It is a SEPARATE  */
/* body overlay so the default tile view is never at flicker risk.     */
/* ================================================================== */
static void _camview_set_mode_label(void)
{
    if (s_cam_modebtn != NULL)
        _btn_set_label(s_cam_modebtn, (s_cam_mode == 2) ? "THERMAL" : "TRUE COLOR");
    if (s_cam_modeinfo != NULL)
        lv_label_set_text(s_cam_modeinfo,
            (s_cam_mode == 2) ? "mode: thermal false-colour (luminance LUT)"
                              : "mode: true colour (real RGB565)");
}

static void _camview_refresh(void)   /* pull one frame into the lv_img */
{
    lab_camview_render(s_cam_mode);
    if (s_cam_img != NULL) lv_obj_invalidate(s_cam_img);
}

static void _cam_tiles_cb(lv_event_t *e) { (void)e; _show_page(6); }   /* back to tiles */

static void _cam_toggle_cb(lv_event_t *e)
{
    (void)e;
    s_cam_mode = (s_cam_mode == 1) ? 2 : 1;
    if (s_cam_modebtn != NULL)
        lv_obj_set_style_bg_color(s_cam_modebtn,
            lv_color_hex(s_cam_mode == 2 ? COL_ORANGE : COL_BLUE), LV_PART_MAIN);
    _camview_set_mode_label();
    _camview_refresh();
}

static void _cam_live_cb(lv_event_t *e)   /* "LIVE" on the Camera page */
{
    int k;
    (void)e;
    for (k = 0; k < NPAGE; k++) lv_obj_add_flag(s_page[k], LV_OBJ_FLAG_HIDDEN);
    lv_obj_clear_flag(s_camview, LV_OBJ_FLAG_HIDDEN);
    s_camview_open = 1;
    _camview_set_mode_label();
    _camview_refresh();
}

static void _build_camview(lv_obj_t *scr)
{
    s_camview = _new_page(scr);
    lv_obj_add_flag(s_camview, LV_OBJ_FLAG_HIDDEN);

    _label(s_camview, 12, 6, &lv_font_montserrat_14, COL_GRAY,
           LV_SYMBOL_IMAGE "  LIVE FRAME  OV5640 320x240 RGB565");

    /* the real frame, native 320x240, top-left */
    s_cam_dsc.header.always_zero = 0;
    s_cam_dsc.header.cf = LV_IMG_CF_TRUE_COLOR;     /* RGB565 (LV_COLOR_DEPTH=16) */
    s_cam_dsc.header.w  = 320;
    s_cam_dsc.header.h  = 240;
    s_cam_dsc.data_size = 320U * 240U * 2U;
    s_cam_dsc.data      = lab_camview_buf();
    s_cam_img = lv_img_create(s_camview);
    lv_img_set_src(s_cam_img, &s_cam_dsc);
    lv_obj_set_pos(s_cam_img, 12, 34);

    /* right column: true/thermal toggle + back-to-tiles + mode note */
    s_cam_modebtn = _ctrl_btn(s_camview, 440, 60, 200, 60, COL_BLUE, "TRUE COLOR", _cam_toggle_cb);
    _ctrl_btn(s_camview, 440, 140, 200, 60, COL_GRAY, "TILES", _cam_tiles_cb);
    s_cam_modeinfo = _label(s_camview, 440, 214, &lv_font_montserrat_14, 0xCCCCCCU,
                            "mode: true colour (real RGB565)");
    _label(s_camview, 440, 240, &lv_font_montserrat_14, 0x777777U,
           "live ~5 fps - real imaging on the edge MCU");

    _ctrl_btn(s_camview, 608, 4, 100, 40, COL_GRAY, "BACK", _cam_tiles_cb);
}

/* ------------------------------------------------------------------ */
/* Camera page: OV5640 -> AI-1 crucible CNN.                           */
/* The default view is a coarse 8x6 luminance-tile grid (lv_obj tiles  */
/* recoloured each refresh) + a classical bright-region bounding box + */
/* the CNN label. "What" = AI-1 CNN; "where" = bright-blob CV. The     */
/* "LIVE" button opens the real-frame overlay above (Wave C).          */
/* ------------------------------------------------------------------ */
#define CAMX     10
#define CAMY     28
#define CAMTILE  52
static void _build_camera(lv_obj_t *c)
{
    int gx, gy, i;

    _label(c, CAMX, 4, &lv_font_montserrat_14, COL_GRAY,
           LV_SYMBOL_IMAGE "  CAMERA  OV5640 -> AI-1 crucible CNN (live)");
    /* open the real-frame overlay (Wave C) */
    _ctrl_btn(c, 520, 2, 184, 26, COL_BLUE, "LIVE FRAME", _cam_live_cb);

    /* 8x6 luminance preview tiles (recoloured grayscale in ui_screen_tick) */
    for (gy = 0; gy < CAM_GH; gy++) {
        for (gx = 0; gx < CAM_GW; gx++) {
            lv_obj_t *t = lv_obj_create(c);
            lv_obj_set_pos(t, (lv_coord_t)(CAMX + gx * CAMTILE),
                              (lv_coord_t)(CAMY + gy * CAMTILE));
            lv_obj_set_size(t, CAMTILE - 2, CAMTILE - 2);
            lv_obj_set_style_bg_color(t, lv_color_hex(0x101010U), LV_PART_MAIN);
            lv_obj_set_style_border_width(t, 0, LV_PART_MAIN);
            lv_obj_set_style_radius(t, 0, LV_PART_MAIN);
            lv_obj_set_style_pad_all(t, 0, LV_PART_MAIN);
            lv_obj_clear_flag(t, LV_OBJ_FLAG_SCROLLABLE);
            lv_obj_clear_flag(t, LV_OBJ_FLAG_CLICKABLE);
            s_cam_tile[gy * CAM_GW + gx] = t;
        }
    }

    /* bright-blob bounding box overlay (built after tiles => drawn on top) */
    s_cam_box = lv_obj_create(c);
    lv_obj_set_pos(s_cam_box, CAMX, CAMY);
    lv_obj_set_size(s_cam_box, CAMTILE, CAMTILE);
    lv_obj_set_style_bg_opa(s_cam_box, LV_OPA_TRANSP, LV_PART_MAIN);
    lv_obj_set_style_border_width(s_cam_box, 3, LV_PART_MAIN);
    lv_obj_set_style_border_color(s_cam_box, lv_color_hex(COL_GREEN), LV_PART_MAIN);
    lv_obj_set_style_radius(s_cam_box, 2, LV_PART_MAIN);
    lv_obj_set_style_pad_all(s_cam_box, 0, LV_PART_MAIN);
    lv_obj_clear_flag(s_cam_box, LV_OBJ_FLAG_SCROLLABLE);
    lv_obj_clear_flag(s_cam_box, LV_OBJ_FLAG_CLICKABLE);
    lv_obj_add_flag(s_cam_box, LV_OBJ_FLAG_HIDDEN);
    s_cam_boxlbl = _label(c, CAMX, CAMY, &lv_font_montserrat_14, COL_GREEN, "");
    lv_obj_add_flag(s_cam_boxlbl, LV_OBJ_FLAG_HIDDEN);

    /* right column: AI-1 verdict + per-class probability bars */
    _label(c, 440, 28, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_EYE_OPEN "  DETECTED (AI-1)");
    s_cam_cls  = _label(c, 440, 48, &lv_font_montserrat_28, COL_ORANGE, "--");
    s_cam_conf = _label(c, 440, 92, &lv_font_montserrat_14, 0xCCCCCCU, "-- %");
    _label(c, 440, 124, &lv_font_montserrat_14, 0xCCCCCCU, "empty");
    s_cam_bar[0] = _bar(c, 520, 126, 180, 12);
    _label(c, 440, 150, &lv_font_montserrat_14, 0xCCCCCCU, "loaded");
    s_cam_bar[1] = _bar(c, 520, 152, 180, 12);
    _label(c, 440, 176, &lv_font_montserrat_14, 0xCCCCCCU, "done");
    s_cam_bar[2] = _bar(c, 520, 178, 180, 12);
    for (i = 0; i < 3; i++) _set_bar(s_cam_bar[i], 0, COL_BLUE);
    s_cam_info = _label(c, 440, 208, &lv_font_montserrat_14, COL_GRAY, "frames 0");
    _label(c, 440, 232, &lv_font_montserrat_14, 0x777777U, "box=CV blob   label=CNN");

    /* B3 CAM 4x4 mini-heatmap (class-activation map for the predicted class):
     * red-hot = where AI-1 looked. Forward-only (GAP->FC), no backward pass. */
    _label(c, 440, 256, &lv_font_montserrat_14, COL_GRAY, "AI-1 CAM (B3, where AI looks)");
    for (i = 0; i < 16; i++) {
        lv_obj_t *t = lv_obj_create(c);
        lv_obj_set_pos(t, (lv_coord_t)(440 + (i % 4) * 40),
                          (lv_coord_t)(278 + (i / 4) * 22));
        lv_obj_set_size(t, 36, 20);
        lv_obj_set_style_bg_color(t, lv_color_hex(0x101010U), LV_PART_MAIN);
        lv_obj_set_style_border_width(t, 1, LV_PART_MAIN);
        lv_obj_set_style_border_color(t, lv_color_hex(0x303030U), LV_PART_MAIN);
        lv_obj_set_style_radius(t, 0, LV_PART_MAIN);
        lv_obj_set_style_pad_all(t, 0, LV_PART_MAIN);
        lv_obj_clear_flag(t, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_clear_flag(t, LV_OBJ_FLAG_CLICKABLE);
        s_cam_cam[i] = t;
    }
}

/* ------------------------------------------------------------------ */
/* Pre-flight page: AI-6/7/8/9 predicted outcome for the active recipe */
/* (computed once at boot). AI-6 optical + AI-7 thermal distil the      */
/* validated XRD physics; AI-8 energy/carbon; AI-9 nearest 67-row recipe*/
/* ------------------------------------------------------------------ */
static void _build_preflight(lv_obj_t *c)
{
    s_pf_recipe = _label(c, 10, 6, &lv_font_montserrat_20, 0xEEEEEEU, LV_SYMBOL_LIST "  PRE-FLIGHT  --");

    /* AI-6 optical (distilled Tanabe-Sugano) — left top */
    _label(c, 10, 44, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_TINT "  AI-6 OPTICAL (distilled TS)");
    s_pf_lam  = _label(c, 10, 64,  &lv_font_montserrat_28, COL_ORANGE, "-- nm");
    s_pf_fwhm = _label(c, 10, 108, &lv_font_montserrat_14, 0xCCCCCCU, "FWHM -- nm");

    /* AI-7 thermal quench — right top */
    _label(c, 410, 44, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_DOWN "  AI-7 THERMAL QUENCH @150C");
    s_pf_band    = _label(c, 410, 64,  &lv_font_montserrat_28, COL_GREEN, "--");
    s_pf_thermal = _label(c, 410, 108, &lv_font_montserrat_14, 0xCCCCCCU, "-- % retained");

    /* AI-8 energy / carbon — left bottom + bar (full-scale 200 kWh) */
    _label(c, 10, 152, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_CHARGE "  AI-8 ENERGY / CARBON");
    s_pf_energy = _label(c, 10, 172, &lv_font_montserrat_28, COL_BLUE, "-- kWh");
    s_pf_co2    = _label(c, 10, 216, &lv_font_montserrat_14, 0xCCCCCCU, "-- kg CO2");
    s_pf_bar_e  = _bar(c, 10, 242, 360, 14);
    _set_bar(s_pf_bar_e, 0, COL_BLUE);

    /* AI-9 nearest historical recipe — right middle */
    _label(c, 410, 152, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_COPY "  AI-9 NEAREST RECIPE (67-row kNN)");
    s_pf_analog = _label(c, 410, 172, &lv_font_montserrat_20, 0x88CCFFU, "--");

    /* AI-11 phase-purity prior + derived crystal-field read-out — right bottom */
    _label(c, 410, 210, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_EYE_OPEN "  AI-11 PHASE-PURITY (37 real, LOO 70%)");
    s_pf_purity = _label(c, 410, 230, &lv_font_montserrat_20, COL_GREEN, "--");
    _label(c, 410, 262, &lv_font_montserrat_14, COL_GRAY, "DERIVED FIELD (from AI-6 lam)");
    s_pf_dqb    = _label(c, 410, 282, &lv_font_montserrat_14, 0x88CCFFU, "Dq -- B -- (--)");

    _label(c, 10, 300, &lv_font_montserrat_14, 0x777777U,
           "predicted before firing; AI-6/7 distil validated XRD physics; AI-11 edge triage");
    _label(c, 10, 320, &lv_font_montserrat_14, 0x777777U,
           "AI-8 grid 0.5703 kgCO2/kWh (China, MEE 2022); Dq/B = algebra, not a model");
}

/* ------------------------------------------------------------------ */
/* PL-spectrum page: AI-12 dopant classifier + AI-13 QC autoencoder.    */
/* Replays stored real Fluoromax emission spectra (no on-board          */
/* spectrometer, same honest replay as furnace_sim) and shows the       */
/* spectrum (lv_line) + AI-12 dopant verdict + AI-13 anomaly verdict.   */
/* ------------------------------------------------------------------ */
static void _build_pl(lv_obj_t *c)
{
    int i;
    _label(c, 10, 6, &lv_font_montserrat_20, 0xCCCCCCU, LV_SYMBOL_TINT "  PL SPECTRUM  (AI-12/13/15/16/17)");

    s_pl_line = lv_line_create(c);
    lv_obj_set_pos(s_pl_line, PLX, PLY);
    lv_obj_set_size(s_pl_line, PLW, PLH);
    lv_obj_set_style_line_width(s_pl_line, 2, LV_PART_MAIN);
    lv_obj_set_style_line_color(s_pl_line, lv_color_hex(COL_GREEN), LV_PART_MAIN);
    lv_obj_set_style_line_rounded(s_pl_line, true, LV_PART_MAIN);
    _label(c, PLX, (lv_coord_t)(PLY + PLH + 6), &lv_font_montserrat_14, 0x777777U,
           "emission 600-1650 nm (normalised) - replayed real Fluoromax spectrum");

    /* AI-12 dopant verdict + per-class bars (right top) */
    _label(c, 528, 40, &lv_font_montserrat_14, COL_GRAY, "DOPANT (AI-12)");
    s_pl_cls  = _label(c, 528, 60,  &lv_font_montserrat_28, COL_ORANGE, "--");
    s_pl_conf = _label(c, 528, 104, &lv_font_montserrat_14, 0xCCCCCCU, "-- %");
    _label(c, 528, 130, &lv_font_montserrat_14, 0xCCCCCCU, "Cr");
    s_pl_bar[0] = _bar(c, 576, 132, 132, 12);
    _label(c, 528, 152, &lv_font_montserrat_14, 0xCCCCCCU, "Ni");
    s_pl_bar[1] = _bar(c, 576, 154, 132, 12);
    _label(c, 528, 174, &lv_font_montserrat_14, 0xCCCCCCU, "C+N");
    s_pl_bar[2] = _bar(c, 576, 176, 132, 12);
    for (i = 0; i < 3; i++) _set_bar(s_pl_bar[i], 0, COL_BLUE);

    /* AI-13 QC verdict (right bottom) */
    _label(c, 528, 200, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_OK "  QC (AI-13 AE)");
    s_pl_qc  = _label(c, 528, 218, &lv_font_montserrat_28, COL_GREEN, "--");
    s_pl_mse = _label(c, 528, 258, &lv_font_montserrat_14, 0xCCCCCCU, "MSE -- / q^ --");

    /* AI-15 host-ID + AI-16 lambda_em + AI-17 few-shot (read the same spectrum) */
    _label(c, 528, 284, &lv_font_montserrat_14, COL_GRAY, "AI-15 / 16 / 17");
    s_pl_host    = _label(c, 528, 304, &lv_font_montserrat_14, 0xCCCCCCU, "host --");
    s_pl_lambda  = _label(c, 528, 324, &lv_font_montserrat_14, 0xCCCCCCU, "lam -- nm");
    s_pl_fewshot = _label(c, 528, 344, &lv_font_montserrat_14, 0xCCCCCCU, "few-shot --");

    s_pl_info = _label(c, 10, 322, &lv_font_montserrat_14, 0x777777U,
           "281 real spectra: AI-12 dopant / AI-13 QC / AI-15 host / AI-16 peak / AI-17 few-shot");
}

/* ================================================================== */
/* Models page: 20 clickable cards -> per-model detail screen (block   */
/* diagram + spec table + metric bar + live readout) + Benchmark page  */
/* (real DWT-measured on-chip latency). Detail/Benchmark are body-area */
/* overlays; _show_page() hides them whenever a tab is tapped.         */
/* ================================================================== */
#define DET_BLKW  124
#define DET_BLKH   46
#define DET_BLKY   96

/* Format the AI-19/AI-20 live readout for the detail screen; returns a change
 * signature so the tick repaints only when the live value actually moves. */
static int _det_live_fmt(int idx, char *buf, int bufsz)
{
    ai_extra_view_t ax;

    if (idx == 2) {   /* AI-2 AE: anomaly ratio + dominant attribution + conformal q^.
                       * (Folded here from the old AI tab; stashes filled in
                       *  ui_screen_update_ai.) */
        static const char *const CH[3] = { "temp", "vib", "gas" };
        int rw = s_ai2_ratio_x10 / 10, rf = s_ai2_ratio_x10 % 10;
        if (rf < 0) rf = 0;
        if (s_ai2_attr_top < 0)
            snprintf(buf, bufsz, "live AE: x%d.%d  q^=0.%03d  (normal)",
                     rw, rf, s_ai2_qhat_milli);
        else
            snprintf(buf, bufsz, "live AE: x%d.%d  top=%s  q^=0.%03d",
                     rw, rf, CH[s_ai2_attr_top], s_ai2_qhat_milli);
        return (s_ai2_attr_top + 2) * 1000000 + s_ai2_ratio_x10 * 1000 + s_ai2_qhat_milli;
    }
    if (idx == 3) {   /* AI-3 transformer: peak-attention minute of the 64-min curve */
        snprintf(buf, bufsz, "live attention: focus at minute t%d / 64", s_ai3_attn_peak);
        return 300000 + s_ai3_attn_peak;
    }

    lab_get_ai_extra(&ax);
    if (idx == 18) {   /* AI-19 RUL/ETA */
        if (ax.rul_valid) {
            snprintf(buf, bufsz, "live ETA: %d min to firing-complete", ax.rul_min);
            return 1000000 + ax.rul_min;
        }
        snprintf(buf, bufsz, "live ETA: idle (start a batch)");
        return -1;
    }
    /* AI-20 TC-integrity */
    if (ax.tc_valid) {
        snprintf(buf, bufsz, "live TC: %s  %d%%", TCN[ax.tc_cls % 3], ax.tc_conf_pct);
        return 2000000 + ax.tc_cls * 1000 + ax.tc_conf_pct;
    }
    snprintf(buf, bufsz, "live TC: --");
    return -2;
}

static void _populate_detail(int idx)
{
    const model_info_t *m = &MODELS[idx];
    char buf[64];
    int  i, nlat, lat[AI_LAT_N];

    snprintf(buf, sizeof(buf), "%s  %s   %s", MODICON[idx], m->id, m->name);
    lv_label_set_text(s_det_title, buf);
    lv_label_set_text(s_det_purpose, m->purpose);

    /* architecture block diagram (show only the populated blocks/arrows) */
    for (i = 0; i < ARCH_MAX; i++) {
        if (m->arch[i] != NULL && m->arch[i][0] != '\0') {
            lv_label_set_text(s_det_blklbl[i], m->arch[i]);
            lv_obj_clear_flag(s_det_blk[i], LV_OBJ_FLAG_HIDDEN);
        } else {
            lv_obj_add_flag(s_det_blk[i], LV_OBJ_FLAG_HIDDEN);
        }
    }
    for (i = 0; i < ARCH_MAX - 1; i++) {
        int two = (m->arch[i] && m->arch[i][0]) && (m->arch[i + 1] && m->arch[i + 1][0]);
        if (two) lv_obj_clear_flag(s_det_arr[i], LV_OBJ_FLAG_HIDDEN);
        else     lv_obj_add_flag(s_det_arr[i], LV_OBJ_FLAG_HIDDEN);
    }

    snprintf(buf, sizeof(buf), LV_SYMBOL_RIGHT "  Input:   %s", m->input);
    lv_label_set_text(s_det_io, buf);
    snprintf(buf, sizeof(buf), LV_SYMBOL_SD_CARD "  Data:    %s", m->data);
    lv_label_set_text(s_det_data, buf);
    snprintf(buf, sizeof(buf), LV_SYMBOL_OK "  Metric:  %s", m->metric);
    lv_label_set_text(s_det_metric, buf);

    nlat = lab_get_ai_lat(lat, AI_LAT_N);
    if (m->lat_idx >= 0 && m->lat_idx < nlat && lat[m->lat_idx] > 0)
        snprintf(buf, sizeof(buf), LV_SYMBOL_CHARGE "  Latency: %d us  (DWT-measured, M7 @600MHz)", lat[m->lat_idx]);
    else
        snprintf(buf, sizeof(buf), LV_SYMBOL_CHARGE "  Latency: sub-ms  (not separately timed)");
    lv_label_set_text(s_det_lat, buf);

    if (m->acc_pct >= 0) {
        _set_bar(s_det_bar, m->acc_pct,
                 (m->acc_pct >= 90) ? COL_GREEN : (m->acc_pct >= 70) ? COL_AMBER : COL_ORANGE);
        lv_obj_clear_flag(s_det_bar, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(s_det_bar, LV_OBJ_FLAG_HIDDEN);
    }

    /* AI-20 (idx 19) gets live fault-inject buttons; AI-2/AI-3/AI-19/AI-20 each get
     * a live readout line (AI-2 = AE attribution + q^, AI-3 = attention peak — both
     * folded here from the old AI tab; AI-19 = RUL/ETA, AI-20 = TC integrity). */
    for (i = 0; i < 3; i++) {
        if (idx == 19) lv_obj_clear_flag(s_det_inj[i], LV_OBJ_FLAG_HIDDEN);
        else           lv_obj_add_flag(s_det_inj[i], LV_OBJ_FLAG_HIDDEN);
    }
    if (idx == 2 || idx == 3 || idx == 18 || idx == 19) {
        s_det_live_sig = _det_live_fmt(idx, buf, sizeof(buf));
        lv_label_set_text(s_det_live, buf);
        lv_obj_clear_flag(s_det_live, LV_OBJ_FLAG_HIDDEN);
    } else {
        lv_obj_add_flag(s_det_live, LV_OBJ_FLAG_HIDDEN);
    }
}

static void _open_detail(int idx)
{
    int k;
    if (idx < 0 || idx >= N_MODELS) return;
    _populate_detail(idx);
    for (k = 0; k < NPAGE; k++) lv_obj_add_flag(s_page[k], LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(s_bench, LV_OBJ_FLAG_HIDDEN);
    lv_obj_clear_flag(s_detail, LV_OBJ_FLAG_HIDDEN);
    s_detail_open = 1;
    s_detail_idx  = idx;
}

static void _card_cb(lv_event_t *e)
{
    lv_obj_t *t = lv_event_get_current_target(e);
    int i;
    for (i = 0; i < N_MODELS; i++)
        if (s_card[i] == t) { _open_detail(i); return; }
}

static void _det_back_cb(lv_event_t *e) { (void)e; _show_page(9); }   /* _show_page hides overlays */

/* AI-20 live thermocouple-fault injection (non-destructive — perturbs only the
 * classifier's input window in env_task; see lab_set_tc_inject). */
static void _inj0(lv_event_t *e) { (void)e; lab_set_tc_inject(0); }   /* healthy */
static void _inj1(lv_event_t *e) { (void)e; lab_set_tc_inject(1); }   /* open-circuit */
static void _inj2(lv_event_t *e) { (void)e; lab_set_tc_inject(2); }   /* erratic */

static void _bench_refresh(void)
{
    int lat[AI_LAT_N], n, i;
    char buf[24];
    n = lab_get_ai_lat(lat, AI_LAT_N);
    for (i = 0; i < AI_LAT_N; i++) {
        int us = (i < n) ? lat[i] : 0;
        int w  = (us > 2000) ? 100 : (us * 100 / 2000);
        if (us > 0) snprintf(buf, sizeof(buf), "%d us", us);
        else        snprintf(buf, sizeof(buf), "--");
        lv_label_set_text(s_bench_us[i], buf);
        _set_bar(s_bench_bar[i], w,
                 (us <= 0)     ? COL_GRAY  :
                 (us < 1000)   ? COL_GREEN :
                 (us < 20000)  ? COL_AMBER : COL_RED);
    }
}

static void _bench_open_cb(lv_event_t *e)
{
    int k;
    (void)e;
    _bench_refresh();
    for (k = 0; k < NPAGE; k++) lv_obj_add_flag(s_page[k], LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(s_detail, LV_OBJ_FLAG_HIDDEN);
    lv_obj_clear_flag(s_bench, LV_OBJ_FLAG_HIDDEN);
    s_detail_open = 0;
    s_detail_idx  = -1;
}

/* per-model detail overlay (built once, repopulated on open) */
static void _build_detail(lv_obj_t *scr)
{
    int i;
    s_detail = _new_page(scr);
    lv_obj_add_flag(s_detail, LV_OBJ_FLAG_HIDDEN);

    s_det_title   = _label(s_detail, 12, 6,  &lv_font_montserrat_28, COL_BLUE,  "--");
    s_det_purpose = _label(s_detail, 12, 46, &lv_font_montserrat_14, 0xCCCCCCU, "--");
    _label(s_detail, 12, 72, &lv_font_montserrat_14, COL_GRAY, LV_SYMBOL_SHUFFLE "  ARCHITECTURE");

    for (i = 0; i < ARCH_MAX; i++) {
        lv_coord_t bx = (lv_coord_t)(12 + i * 136);   /* 136 pitch fits 5 blocks in 716 */
        lv_obj_t  *bl;
        s_det_blk[i] = lv_obj_create(s_detail);
        lv_obj_set_pos(s_det_blk[i], bx, DET_BLKY);
        lv_obj_set_size(s_det_blk[i], DET_BLKW, DET_BLKH);
        lv_obj_set_style_bg_color(s_det_blk[i], lv_color_hex(0x18314AU), LV_PART_MAIN);
        lv_obj_set_style_border_width(s_det_blk[i], 2, LV_PART_MAIN);
        lv_obj_set_style_border_color(s_det_blk[i], lv_color_hex(COL_BLUE), LV_PART_MAIN);
        lv_obj_set_style_radius(s_det_blk[i], 6, LV_PART_MAIN);
        lv_obj_set_style_pad_all(s_det_blk[i], 0, LV_PART_MAIN);
        lv_obj_clear_flag(s_det_blk[i], LV_OBJ_FLAG_SCROLLABLE);
        bl = lv_label_create(s_det_blk[i]);
        lv_obj_set_width(bl, DET_BLKW - 6);
        lv_obj_set_pos(bl, 3, (DET_BLKH - 16) / 2);
        lv_obj_set_style_text_align(bl, LV_TEXT_ALIGN_CENTER, LV_PART_MAIN);
        lv_obj_set_style_text_color(bl, lv_color_white(), LV_PART_MAIN);
        lv_obj_set_style_text_font(bl, &lv_font_montserrat_14, LV_PART_MAIN);
        lv_label_set_text(bl, "");
        s_det_blklbl[i] = bl;
        if (i < ARCH_MAX - 1)
            s_det_arr[i] = _label(s_detail, (lv_coord_t)(bx + DET_BLKW + 2),
                                  (lv_coord_t)(DET_BLKY + 12), &lv_font_montserrat_20,
                                  COL_GRAY, ">");
    }

    s_det_io     = _label(s_detail, 12, 156, &lv_font_montserrat_14, 0xDDDDDDU, "--");
    s_det_data   = _label(s_detail, 12, 180, &lv_font_montserrat_14, 0xDDDDDDU, "--");
    s_det_metric = _label(s_detail, 12, 204, &lv_font_montserrat_14, 0xDDDDDDU, "--");
    s_det_lat    = _label(s_detail, 12, 228, &lv_font_montserrat_14, 0x88CCFFU, "--");
    s_det_bar    = _bar(s_detail, 12, 262, 320, 14);
    _set_bar(s_det_bar, 0, COL_GREEN);
    s_det_live   = _label(s_detail, 12, 290, &lv_font_montserrat_20, COL_GREEN, "live: --");

    /* AI-20 live fault-inject buttons (right column; shown only on AI-20) */
    s_det_inj[0] = _ctrl_btn(s_detail, 320, 250, 128, 44, COL_GREEN, "HEALTHY",  _inj0);
    s_det_inj[1] = _ctrl_btn(s_detail, 452, 250, 128, 44, COL_AMBER, "OPEN-CKT", _inj1);
    s_det_inj[2] = _ctrl_btn(s_detail, 584, 250, 128, 44, COL_RED,   "ERRATIC",  _inj2);
    for (i = 0; i < 3; i++) lv_obj_add_flag(s_det_inj[i], LV_OBJ_FLAG_HIDDEN);

    _ctrl_btn(s_detail, 608, 4, 100, 40, COL_GRAY, "BACK", _det_back_cb);
}

/* on-chip latency benchmark overlay (real DWT numbers, read on open) */
static void _build_bench(lv_obj_t *scr)
{
    int i;
    s_bench = _new_page(scr);
    lv_obj_add_flag(s_bench, LV_OBJ_FLAG_HIDDEN);

    _label(s_bench, 12, 6, &lv_font_montserrat_20, 0xEEEEEEU,
           LV_SYMBOL_BARS "  ON-CHIP LATENCY  (DWT, M7 @600MHz)");

    for (i = 0; i < AI_LAT_N; i++) {
        lv_coord_t y = (lv_coord_t)(36 + i * 20);
        _label(s_bench, 12, y, &lv_font_montserrat_14, 0xDDDDDDU, LATN[i]);
        s_bench_us[i]  = _label(s_bench, 220, y, &lv_font_montserrat_14, 0xFFFFFFU, "--");
        s_bench_bar[i] = _bar(s_bench, 320, (lv_coord_t)(y + 2), 300, 12);
        _set_bar(s_bench_bar[i], 0, COL_GRAY);
    }
    _label(s_bench, 12, (lv_coord_t)(36 + AI_LAT_N * 20 + 4), &lv_font_montserrat_14, 0x88CCFFU,
           LV_SYMBOL_CHARGE "  INT8 weight-only PTQ: AI-12 484->146 us = 3.3x (less Flash traffic)");
    _label(s_bench, 12, (lv_coord_t)(36 + AI_LAT_N * 20 + 26), &lv_font_montserrat_14, COL_GRAY,
           "bar: green <1ms   amber <20ms   red slower");
    _label(s_bench, 12, (lv_coord_t)(36 + AI_LAT_N * 20 + 48), &lv_font_montserrat_14, COL_ACCENT,
           LV_SYMBOL_OK "  20 AI models, all on a no-NPU Cortex-M7 @600MHz - no cloud");

    _ctrl_btn(s_bench, 608, 4, 100, 40, COL_GRAY, "BACK", _det_back_cb);
}

/* fwd decl: _ai_card (flat card + accent bar) is defined further down but the
 * cluster overlay above its definition now uses it. */
static lv_obj_t *_ai_card(lv_obj_t *p, lv_coord_t x, lv_coord_t y,
                          lv_coord_t w, lv_coord_t h,
                          uint32_t bordercol, uint32_t barcol);

/* ---- Edge LLM cluster overlay: 5 swap-loaded experts ---- */
static void _cl_next_cb(lv_event_t *e) { (void)e; lab_cluster_next(); }
/* opened from the Edge LM page now (generative AI all lives there) -> back to it */
static void _cl_back_cb(lv_event_t *e) { (void)e; _show_page(11); }

static void _cl_open_cb(lv_event_t *e)
{
    int k;
    (void)e;
    for (k = 0; k < NPAGE; k++) lv_obj_add_flag(s_page[k], LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(s_detail, LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_flag(s_bench,  LV_OBJ_FLAG_HIDDEN);
    lv_obj_clear_flag(s_cluster_ov, LV_OBJ_FLAG_HIDDEN);
    s_cluster_open = 1; s_detail_open = 0;
    s_cl_gens_seen = 0xFFFFFFFFu;           /* force a repaint on open */
}

static void _build_cluster_ov(lv_obj_t *scr)
{
    lv_obj_t *lcard, *rcard;
    int i;
    s_cluster_ov = _new_page(scr);
    lv_obj_add_flag(s_cluster_ov, LV_OBJ_FLAG_HIDDEN);
    _label(s_cluster_ov, 12, 6, &lv_font_montserrat_20, 0xEEEEEEU,
           LV_SYMBOL_COPY "  EDGE LLM CLUSTER");
    _label(s_cluster_ov, 12, 34, &lv_font_montserrat_14, COL_GRAY,
           "DeepSeek-distilled, 100% on-chip, SPI-flash swap-load (1 expert bound at a time)");

    /* left: the 7 role-specialised experts (the active one is highlighted green by
     * the tick). Carded for structure; labels stay labels so the tick recolour works. */
    lcard = _ai_card(s_cluster_ov, 8, 60, 280, 256, COL_BORDER, COL_ACCENT);
    _label(lcard, 10, 8, &lv_font_montserrat_14, COL_GRAY, "EXPERTS  (swap-load, 1 active)");
    for (i = 0; i < CL_NEXP; i++)
        s_cl_row[i] = _label(lcard, 14, (lv_coord_t)(36 + i * 31),
                             &lv_font_montserrat_14, 0x999999U, CL_ROW[i]);

    /* right: the live on-chip generation from the bound expert */
    rcard = _ai_card(s_cluster_ov, 296, 60, 412, 200, COL_ACCENT, COL_ACCENT);
    _label(rcard, 10, 8, &lv_font_montserrat_14, COL_ACCENT, LV_SYMBOL_KEYBOARD "  LIVE GENERATION (on-chip)");
    s_cl_text = _label(rcard, 10, 34, &lab_font_cn16, COL_GREEN, "(select an expert / generating...)");
    lv_obj_set_width(s_cl_text, 392);
    lv_label_set_long_mode(s_cl_text, LV_LABEL_LONG_WRAP);

    s_cl_meta = _label(s_cluster_ov, 296, 270, &lv_font_montserrat_14, 0x88CCFFU, "expert --");
#ifndef LAB_LM_ENABLE
    _label(s_cluster_ov, 296, 294, &lv_font_montserrat_14, COL_AMBER,
           LV_SYMBOL_WARNING "  LM disabled (enable LAB_LM_ENABLE)");
#endif
    _ctrl_btn(s_cluster_ov, 296, 330, 200, 50, COL_BLUE, "NEXT EXPERT", _cl_next_cb);
    _ctrl_btn(s_cluster_ov, 608, 4, 100, 40, COL_GRAY, "BACK", _cl_back_cb);
}

static void _build_models(lv_obj_t *c)
{
    int i;
    _label(c, 10, 6, &lv_font_montserrat_20, 0xEEEEEEU, LV_SYMBOL_LIST "  AI MODELS (20)  tap a card");
    /* generative AI (LLM cluster) moved to the Edge LM page; Models is the 20
     * discriminative models + their aggregate on-chip latency benchmark. */
    _ctrl_btn(c, 548, 1, 160, 30, COL_BLUE,  "BENCHMARK",   _bench_open_cb);

    /* 5 cols x 4 rows of tappable model cards (id + name + headline metric).
     * 142px pitch x 138px card fits 5 cols inside the 716-wide content area. */
    for (i = 0; i < N_MODELS; i++) {
        int        col = i % 5, row = i / 5;
        lv_coord_t x   = (lv_coord_t)(6  + col * 142);
        lv_coord_t y   = (lv_coord_t)(34 + row * 84);
        lv_obj_t  *card = lv_obj_create(c);
        lv_obj_set_pos(card, x, y);
        lv_obj_set_size(card, 138, 76);
        lv_obj_set_style_bg_color(card, lv_color_hex(0x161616U), LV_PART_MAIN);
        lv_obj_set_style_border_width(card, 2, LV_PART_MAIN);
        lv_obj_set_style_border_color(card, lv_color_hex(0x3A3A3AU), LV_PART_MAIN);
        lv_obj_set_style_radius(card, 6, LV_PART_MAIN);
        lv_obj_set_style_pad_all(card, 0, LV_PART_MAIN);
        lv_obj_clear_flag(card, LV_OBJ_FLAG_SCROLLABLE);
        lv_obj_add_flag(card, LV_OBJ_FLAG_CLICKABLE);
        lv_obj_add_event_cb(card, _card_cb, LV_EVENT_PRESSED, NULL);
        lv_obj_set_ext_click_area(card, 2);   /* < the 6px gap so hits never overlap */
        s_card[i] = card;
        _label(card, 8, 4,  &lv_font_montserrat_20, COL_GREEN, MODELS[i].id);
        _label(card, 8, 32, &lv_font_montserrat_14, 0xFFFFFFU, MODELS[i].name);
        _label(card, 8, 54, &lv_font_montserrat_14, 0x99CCFFU, MODELS[i].metric);
    }
}

/* ------------------------------------------------------------------ */
/* Build the HMI                                                       */
/* ------------------------------------------------------------------ */
/* a rounded FLAT-fill card with a chosen border colour (accent = AI showcase).
 * Flat (not gradient): 7 gradient cards over a gradient page bg pushed this page's
 * full_refresh past the ~22ms VBlank budget -> the two framebuffers beat (ghosting
 * "屏幕闪烁"). A solid fill matches the proven-flicker-free Models/Recipe/Home cards. */
static lv_obj_t *_ai_card(lv_obj_t *p, lv_coord_t x, lv_coord_t y,
                          lv_coord_t w, lv_coord_t h,
                          uint32_t bordercol, uint32_t barcol)
{
    lv_obj_t *bar;
    lv_obj_t *card = lv_obj_create(p);
    lv_obj_set_pos(card, x, y);
    lv_obj_set_size(card, w, h);
    lv_obj_set_style_bg_color(card, lv_color_hex(COL_CARD), LV_PART_MAIN);
    lv_obj_set_style_border_width(card, 1, LV_PART_MAIN);
    lv_obj_set_style_border_color(card, lv_color_hex(bordercol), LV_PART_MAIN);
    lv_obj_set_style_radius(card, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_all(card, 0, LV_PART_MAIN);
    lv_obj_clear_flag(card, LV_OBJ_FLAG_SCROLLABLE);

    /* 3px left accent bar: a flat-fill child (cheap, no gradient -> no VBlank-budget
     * risk). Inset 8px top/bottom so it sits clear of the radius-8 rounded corners. */
    bar = lv_obj_create(card);
    lv_obj_set_pos(bar, 0, 8);
    lv_obj_set_size(bar, 3, (lv_coord_t)(h - 16));
    lv_obj_set_style_bg_color(bar, lv_color_hex(barcol), LV_PART_MAIN);
    lv_obj_set_style_border_width(bar, 0, LV_PART_MAIN);
    lv_obj_set_style_radius(bar, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_all(bar, 0, LV_PART_MAIN);
    lv_obj_clear_flag(bar, LV_OBJ_FLAG_SCROLLABLE);
    return card;
}

/* E-Twin: the on-chip AI showcase page (the competitive edge, finally a whole
 * screen of its own). Static -- built once, the tick never touches it, so
 * it can never flicker and adds no risk to any other page's change-gating.
 * Three blocks: (1) the 3-tier heterogeneous ecosystem -- the headline KILL;
 * (2) what THIS device runs (20 discriminative + 3 nano-LM + 5 cluster = 28);
 * (3) a measured no-NPU MCU evidence strip. Live switching / live CN
 * generation stay on the Control page; this is the at-a-glance map. */
static void _build_ai_overview(lv_obj_t *c)
{
    lv_obj_t *card;

    _label(c, 12,  6, &lv_font_montserrat_20, COL_TEXT,
           LV_SYMBOL_CHARGE "  ON-CHIP AI  -  30 RUNTIME ASSETS");
    _label(c, 12, 34, &lv_font_montserrat_14, COL_TEXT2,
           "GD32H759 Cortex-M7 @600MHz  -  no NPU  -  28 logical models");

    /* ---- the on-chip story: 30 competition runtime assets, locally executed.
     * Deliberately NO off-device/cloud tiles: CIMC scores LOCAL deployment, and the
     * device never offloads inference -- the ESP32 hook is telemetry-only (see card 3),
     * not an AI path. Honest framing per ADR-4 + edge-only positioning. ---- */
    card = _ai_card(c,   8, 58, 222, 92, COL_ACCENT, COL_ACCENT);
    _label(card, 10,  6, &lv_font_montserrat_14, COL_ACCENT, LV_SYMBOL_CHARGE "  GD32 EDGE");
    _label(card, 10, 30, &lv_font_montserrat_14, COL_TEXT,   "20 discr + 3 LM + 7 expert");
    _label(card, 10, 48, &lv_font_montserrat_14, COL_GREEN,  "= 30 runtime assets");
    _label(card, 10, 68, &lv_font_montserrat_14, COL_TEXT2,  "on a single GD32 MCU");

    card = _ai_card(c, 240, 58, 222, 92, COL_GREEN, COL_GREEN);
    _label(card, 10,  6, &lv_font_montserrat_14, COL_GREEN,  LV_SYMBOL_OK "  LOCAL MODEL PATH");
    _label(card, 10, 30, &lv_font_montserrat_14, COL_TEXT,   "no NPU  -  no cloud inference");
    _label(card, 10, 48, &lv_font_montserrat_14, COL_TEXT,   "no external compute");
    _label(card, 10, 68, &lv_font_montserrat_14, COL_TEXT2,  "30-asset inference on M7");

    card = _ai_card(c, 472, 58, 230, 92, COL_BORDER, COL_BLUE);
    _label(card, 10,  6, &lv_font_montserrat_14, COL_BLUE,   LV_SYMBOL_USB "  EXTERNAL I/O");
    _label(card, 10, 30, &lv_font_montserrat_14, COL_TEXT,   "CI1302: fixed voice I/O");
    _label(card, 10, 48, &lv_font_montserrat_14, COL_TEXT2,  "NOT model authority");
    _label(card, 10, 68, &lv_font_montserrat_14, COL_TEXT2,  "no compute offload");

    /* ---- on-chip AI breakdown (what THIS GD32 runs) ---- */
    card = _ai_card(c,   8, 160, 226, 116, COL_BORDER, COL_GREEN);
    _label(card, 10,  6, &lv_font_montserrat_14, COL_GREEN, LV_SYMBOL_EYE_OPEN "  DISCRIMINATIVE  x20");
    _label(card, 10, 30, &lv_font_montserrat_14, COL_TEXT2, "AI-1 vision   AI-2 AE");
    _label(card, 10, 48, &lv_font_montserrat_14, COL_TEXT2, "AI-3 Transf   AI-4 fuse");
    _label(card, 10, 66, &lv_font_montserrat_14, COL_TEXT2, "AI-5..20 spectra/RUL/TC");
    _label(card, 10, 90, &lv_font_montserrat_14, COL_TEXT,  "classify + detect  <1-70ms");

    card = _ai_card(c, 244, 160, 226, 116, COL_ACCENT, COL_ACCENT);
    _label(card, 10,  6, &lv_font_montserrat_14, COL_ACCENT, LV_SYMBOL_KEYBOARD "  nano-LM bank  x3");
    _label(card, 10, 30, &lv_font_montserrat_14, COL_TEXT,   "x1p9 1.8M  m1p35 1.26M");
    _label(card, 10, 48, &lv_font_montserrat_14, COL_TEXT,   "s0p6 0.6M");
    _label(card, 10, 66, &lv_font_montserrat_14, COL_TEXT2,  "INT8 decoder-GPT");
    _label(card, 10, 90, &lv_font_montserrat_14, COL_TEXT2,  "DeepSeek-V4 distil ~2-4s");

    card = _ai_card(c, 480, 160, 226, 116, COL_ACCENT, COL_ACCENT);
    _label(card, 10,  4, &lv_font_montserrat_14, COL_ACCENT, LV_SYMBOL_COPY "  LLM cluster  x7");
    _label(card, 10, 26, &lv_font_montserrat_14, COL_TEXT,   "E1 diag    E2 recipe");
    _label(card, 10, 44, &lv_font_montserrat_14, COL_TEXT,   "E3 energy  E4 qc");
    _label(card, 10, 62, &lv_font_montserrat_14, COL_TEXT,   "E5 brief   E6 chem");
    _label(card, 10, 80, &lv_font_montserrat_14, COL_TEXT,   "E7 maint");
    _label(card, 10, 98, &lv_font_montserrat_14, COL_TEXT2,  "role experts, SPI swap-load");

    /* ---- measured on-device evidence strip ---- */
    card = _ai_card(c, 8, 286, 698, 142, COL_ACCENT, COL_ACCENT);
    _label(card, 12,   8, &lv_font_montserrat_14, COL_ACCENT, LV_SYMBOL_OK "  VERIFIED ON A NO-NPU GD32 MCU");
    _label(card, 12,  32, &lv_font_montserrat_14, COL_TEXT,
           "- INT8 autoregressive generative LLM  (1.8M params, DeepSeek-V4 distilled)");
    _label(card, 12,  54, &lv_font_montserrat_14, COL_TEXT,
           "- SPI-flash swap-load 7-expert LLM cluster  (1 resident in SDRAM)");
    _label(card, 12,  76, &lv_font_montserrat_14, COL_TEXT,
           "- hand-written 24-layer Transformer attention  (AI-3, 17x cache-opt)");
    _label(card, 12, 104, &lv_font_montserrat_14, COL_TEXT2,
           "20 discriminative classify/detect  +  generative LM diagnoses in Chinese");
}

/* ------------------------------------------------------------------ */
/* Edge LM page (idx 11): the flagship on-chip GENERATIVE LM gets its  */
/* own full screen (was a cramped corner of Control). Hero = the LIVE  */
/* Chinese diagnosis the nano-LM generates; right = the active-LM      */
/* selector + swap-load controls; bottom = the measured edge story.   */
/* When LAB_LM_ENABLE is off (UI-dev fast build) the LM is compiled out */
/* of the image, so we say that PLAINLY instead of a fake forever       */
/* "generating..." — the live diagnosis overwrites it once enabled.    */
/* ------------------------------------------------------------------ */
static void _build_edge_lm(lv_obj_t *c)
{
    lv_obj_t *card;

    _label(c, 12,  6, &lv_font_montserrat_20, COL_TEXT,
           LV_SYMBOL_KEYBOARD "  FLAGSHIP EDGE LM");
    _label(c, 12, 34, &lv_font_montserrat_14, COL_TEXT2,
           "generative  -  x1p9 1.8M INT8  -  autoregressive KV-cache  -  ~2-4s/sentence  -  100% on-chip");

    /* hero: the live on-chip Chinese diagnosis (the flagship's actual output) */
    card = _ai_card(c, 8, 58, 472, 150, COL_ACCENT, COL_ACCENT);
    _label(card, 12,  8, &lv_font_montserrat_14, COL_ACCENT,
           LV_SYMBOL_KEYBOARD "  LIVE DIAGNOSIS  (on-chip generation)");
    s_cc_nlm = _label(card, 12, 36, &lab_font_cn16, COL_GREEN, "");
    lv_obj_set_width(s_cc_nlm, 448);
    lv_label_set_long_mode(s_cc_nlm, LV_LABEL_LONG_WRAP);
#ifndef LAB_LM_ENABLE
    /* LM stack compiled out of this fast UI-dev build -> say so in plain ASCII
     * (montserrat, no CJK-subset tofu risk); s_cc_nlm stays empty until enabled. */
    _label(card, 12, 40, &lv_font_montserrat_14, COL_TEXT2,
           "compiled out of this fast UI-dev build.\n"
           "enable LAB_LM_ENABLE + reflash to run the\n"
           "flagship generative diagnosis on-chip.");
#endif

    /* right: active-LM indicator + swap-load controls (x1p9 -> m1p35 -> s0p6) */
    card = _ai_card(c, 488, 58, 218, 150, COL_BORDER, COL_BLUE);
    _label(card, 10,  8, &lv_font_montserrat_14, COL_BLUE, LV_SYMBOL_LOOP "  ACTIVE LM");
    s_cc_lm = _label(card, 10, 32, &lv_font_montserrat_14, 0x88CCFFU, "LM: x1p9 1.8M");
    _ctrl_btn(card,  10,  58, 96, 42, COL_BLUE,   "DIAG",    _btn_nlmrefresh_cb);
    _ctrl_btn(card, 112,  58, 96, 42, COL_GREEN,  "TEACH",   _btn_teach_cb);
    _ctrl_btn(card,  10, 106, 198, 38, COL_ACCENT, "NEXT LM", _btn_lmcycle_cb);

    /* build state (honest in both modes) + online-learn head + escalation note */
#ifdef LAB_LM_ENABLE
    _label(c, 12, 216, &lv_font_montserrat_14, COL_GREEN,
           LV_SYMBOL_OK "  live on-chip generation  (DeepSeek-distilled, runs offline)");
#else
    _label(c, 12, 216, &lv_font_montserrat_14, COL_AMBER,
           LV_SYMBOL_WARNING "  generative LM compiled out (UI-dev fast build) - enable LAB_LM_ENABLE to run it");
#endif
    s_cc_olrow = _label(c, 12, 242, &lv_font_montserrat_14, 0xCCAA66U,
                        "online head: risk=-- taught 0 acc --%");
    _label(c, 12, 266, &lv_font_montserrat_14, COL_TEXT2,
           "uncertain -> flags the case for offline review via the telemetry uplink (no cloud verdict imported)");

    /* measured on-device showcase strip (static) */
    card = _ai_card(c, 8, 292, 698, 140, COL_ACCENT, COL_ACCENT);
    _label(card, 12,   8, &lv_font_montserrat_14, COL_ACCENT,
           LV_SYMBOL_OK "  VERIFIED GENERATIVE LM ON NO-NPU GD32");
    _label(card, 12,  32, &lv_font_montserrat_14, COL_TEXT,
           "- x3 nano-LM bank:  x1p9 1.8M / m1p35 1.26M / s0p6 0.6M  (SPI-flash swap-load)");
    _label(card, 12,  54, &lv_font_montserrat_14, COL_TEXT,
           "- x7 expert cluster:  E1 diag/E2 recipe/E3 energy/E4 qc/E5 brief/E6 chem/E7 maint");
    _label(card, 12,  76, &lv_font_montserrat_14, COL_TEXT,
           "- DeepSeek-V4 distilled  -  INT8 weight-only  -  KV-cache autoregressive");
    _label(card, 12, 104, &lv_font_montserrat_14, COL_TEXT2,
           "host golden logit err ~1e-5  -  ~2-4s/sentence  -  LM inference offline");
    /* open the LIVE 7-expert cluster overlay (moved here from the Models page so all
     * generative AI lives on this one page). */
    _ctrl_btn(card, 506, 86, 184, 44, COL_GREEN, "LLM CLUSTER", _cl_open_cb);
}

/* ------------------------------------------------------------------ */
/* Robust (reliability) page: live multimodal perturbation injection   */
/* (graceful degradation) + bench robustness matrix + long-run panel.  */
/* ------------------------------------------------------------------ */
static lv_obj_t *s_rob_visbtn, *s_rob_plbtn, *s_rob_tcbtn;
static lv_obj_t *s_rob_vislbl, *s_rob_pllbl, *s_rob_tclbl;
static lv_obj_t *s_rob_uptime, *s_rob_infer, *s_rob_reset;

/* one-tap cycle of each modality's perturbation mode (vision 5 / spectrum 4 / TC 3) */
static void _rob_vis_cb(lv_event_t *e){ (void)e; lab_set_vis_inject((lab_get_vis_inject() + 1) % 5); }
static void _rob_pl_cb (lv_event_t *e){ (void)e; lab_set_pl_inject ((lab_get_pl_inject () + 1) % 4); }
static void _rob_tc_cb (lv_event_t *e){ (void)e; lab_set_tc_inject ((lab_get_tc_inject () + 1) % 3); }

static void _build_robust(lv_obj_t *c)
{
    lv_obj_t *card;
    _label(c, 10, 6, &lv_font_montserrat_20, COL_AMBER,
           LV_SYMBOL_WARNING "  RELIABILITY / ROBUSTNESS");

    /* ---- LEFT: live perturbation injection (graceful degradation) ---- */
    card = _ai_card(c, 8, 36, 350, 332, COL_BLUE, COL_BLUE);
    _label(card, 12,  8, &lv_font_montserrat_14, COL_BLUE,
           LV_SYMBOL_CHARGE "  LIVE PERTURBATION");
    _label(card, 12, 28, &lv_font_montserrat_14, COL_TEXT2,
           "tap to perturb -> watch graceful degrade");

    _label(card, 12, 52, &lv_font_montserrat_14, COL_TEXT, "VISION  AI-1  (support)");
    s_rob_visbtn = _ctrl_btn(card, 12, 72, 150, 38, COL_BLUE, "clean", _rob_vis_cb);
    lv_obj_set_ext_click_area(s_rob_visbtn, 8);
    s_rob_vislbl = _label(card, 172, 82, &lv_font_montserrat_14, COL_TEXT2, "d -- -> --");

    _label(card, 12, 124, &lv_font_montserrat_14, COL_TEXT, "SPECTRUM  AI-12  (support)");
    s_rob_plbtn = _ctrl_btn(card, 12, 144, 150, 38, COL_BLUE, "clean", _rob_pl_cb);
    lv_obj_set_ext_click_area(s_rob_plbtn, 8);
    s_rob_pllbl = _label(card, 172, 154, &lv_font_montserrat_14, COL_TEXT2, "-- -> --");

    _label(card, 12, 196, &lv_font_montserrat_14, COL_TEXT, "THERMOCOUPLE  AI-20  (safety)");
    s_rob_tcbtn = _ctrl_btn(card, 12, 216, 150, 38, COL_AMBER, "healthy", _rob_tc_cb);
    lv_obj_set_ext_click_area(s_rob_tcbtn, 8);
    s_rob_tclbl = _label(card, 172, 226, &lv_font_montserrat_14, COL_TEXT2, "verdict --");

    _label(card, 12, 268, &lv_font_montserrat_14, COL_GREEN,
           LV_SYMBOL_OK "  SAFETY CORE (AI-4) holds");
    _label(card, 12, 288, &lv_font_montserrat_14, COL_TEXT2, "debounced ~4.5s + motor-gated:");
    _label(card, 12, 306, &lv_font_montserrat_14, COL_TEXT2, "one glitch never trips the furnace");

    /* ---- RIGHT-TOP: bench robustness matrix (real-data, pre-measured) ---- */
    card = _ai_card(c, 366, 36, 342, 196, COL_GREEN, COL_GREEN);
    _label(card, 12,  8, &lv_font_montserrat_14, COL_GREEN,
           LV_SYMBOL_EYE_OPEN "  ROBUSTNESS  (bench, real data)");
    _label(card, 12, 34, &lv_font_montserrat_14, COL_TEXT,  "AI-1 vis  light.6-2x >=97%");
    _label(card, 12, 54, &lv_font_montserrat_14, COL_TEXT2, "          noise s.1 100% / occ30% 99%");
    _label(card, 12, 78, &lv_font_montserrat_14, COL_TEXT,  "AI-12 PL  noise s.06 94% / occ5b 95%");
    _label(card, 12, 98, &lv_font_montserrat_14, COL_TEXT2, "          baseline.1 90% (clean 98.9%)");
    _label(card, 12, 122, &lv_font_montserrat_14, COL_TEXT, "AI-2 sens adaptive-conformal holds");
    _label(card, 12, 142, &lv_font_montserrat_14, COL_TEXT2,"          FPR~10% @2x drift (vs 55%)");
    _label(card, 12, 166, &lv_font_montserrat_14, COL_GRAY, "281 real spectra | det. RNG | reproducible");

    /* ---- RIGHT-BOTTOM: long-run stability (live) ---- */
    card = _ai_card(c, 366, 240, 342, 128, COL_AMBER, COL_AMBER);
    _label(card, 12,  8, &lv_font_montserrat_14, COL_AMBER,
           LV_SYMBOL_LOOP "  LONG-RUN STABILITY  (live)");
    s_rob_uptime = _label(card, 12, 34, &lv_font_montserrat_14, COL_TEXT, "uptime --");
    s_rob_infer  = _label(card, 12, 56, &lv_font_montserrat_14, COL_TEXT, "AI inferences --");
    s_rob_reset  = _label(card, 12, 78, &lv_font_montserrat_14, COL_TEXT, "reset -- | wdg armed");
    _label(card, 12, 100, &lv_font_montserrat_14, COL_GREEN,
           LV_SYMBOL_OK "  0 fault-resets since boot");
}

void ui_screen_init(void)
{
    lv_obj_t *scr = lv_scr_act();
    static lv_event_cb_t tabcb[NPAGE] = { _tab0, _tab1, _tab2, _tab3, _tab4,
                                          _tab5, _tab6, _tab7, _tab8, _tab9, _tab10, _tab11, _tab12 };
    int i;

    lv_obj_set_style_bg_color(scr, lv_color_hex(0x0B0B0BU), LV_PART_MAIN);
    lv_obj_clear_flag(scr, LV_OBJ_FLAG_SCROLLABLE);

    /* ---- left navigation rail (vertical; replaces the horizontal tab bar) ---- */
    {
        lv_obj_t *rail = lv_obj_create(scr);
        lv_obj_set_pos(rail, 0, 0);
        lv_obj_set_size(rail, NAV_W, SCREEN_H);
        lv_obj_set_style_bg_color(rail, lv_color_hex(COL_CARD), LV_PART_MAIN);
        lv_obj_set_style_bg_grad_color(rail, lv_color_hex(COL_BG), LV_PART_MAIN);
        lv_obj_set_style_bg_grad_dir(rail, LV_GRAD_DIR_VER, LV_PART_MAIN);
        lv_obj_set_style_border_width(rail, 0, LV_PART_MAIN);
        lv_obj_set_style_radius(rail, 0, LV_PART_MAIN);
        lv_obj_set_style_pad_all(rail, 0, LV_PART_MAIN);
        lv_obj_clear_flag(rail, LV_OBJ_FLAG_SCROLLABLE);
        for (i = 0; i < NPAGE; i++)
            s_tab[i] = _make_tab(rail, i, TABN[i], tabcb[i]);
    }

    /* ---- status bar (top strip, right of the rail) ---- */
    s_statusbar = lv_obj_create(scr);
    lv_obj_set_pos(s_statusbar, NAV_W, 0);
    lv_obj_set_size(s_statusbar, BODY_W, STAT_H);
    lv_obj_set_style_bg_color(s_statusbar, lv_color_hex(COL_BG2), LV_PART_MAIN);  /* dark dashboard strip */
    lv_obj_set_style_border_width(s_statusbar, 0, LV_PART_MAIN);
    lv_obj_set_style_radius(s_statusbar, 0, LV_PART_MAIN);
    lv_obj_set_style_pad_all(s_statusbar, 0, LV_PART_MAIN);
    lv_obj_clear_flag(s_statusbar, LV_OBJ_FLAG_SCROLLABLE);

    /* risk chip: a bright rounded pill (coloured by AI-4 risk) with a status icon,
     * instead of flooding the whole bar -> cleaner dark-dashboard look. Dark text on
     * the bright fill stays readable at every level; the pill is the only coloured
     * element so an alarm reads as a distinct chip, not a full-width colour wash. */
    s_status_pill = lv_obj_create(s_statusbar);
    lv_obj_set_pos(s_status_pill, 12, 5);
    lv_obj_set_size(s_status_pill, 172, 34);   /* wide enough for "WARNING"/"ANOMALY" + right padding */
    lv_obj_set_style_bg_color(s_status_pill, lv_color_hex(_risk_fill[0]), LV_PART_MAIN);
    lv_obj_set_style_border_width(s_status_pill, 0, LV_PART_MAIN);
    lv_obj_set_style_radius(s_status_pill, 17, LV_PART_MAIN);
    lv_obj_set_style_pad_all(s_status_pill, 0, LV_PART_MAIN);
    lv_obj_clear_flag(s_status_pill, LV_OBJ_FLAG_SCROLLABLE);
    s_status_risk  = _label(s_status_pill, 14, 6, &lv_font_montserrat_20, COL_BG, LV_SYMBOL_OK "  NORMAL");

    s_status_batch = _label(s_statusbar, 200, 12, &lv_font_montserrat_14, 0xDDDDDDU, LV_SYMBOL_FILE "  garnet idle");
    /* AI-5 root-cause badge: blank when NORMAL, white "AI5:<cause>" when a fault
     * is diagnosed (always visible regardless of which tab is open). */
    s_status_ai5   = _label(s_statusbar, 392, 14, &lv_font_montserrat_14, 0xFFFFFFU, "");
    s_status_temp  = _label(s_statusbar, 600, 4,  &lv_font_montserrat_28, 0xFFFFFFU, "-- C");

    /* ---- content pages ---- */
    for (i = 0; i < NPAGE; i++) s_page[i] = _new_page(scr);
    _build_home    (s_page[0]);
    _build_recipe  (s_page[1]);
    _build_trend   (s_page[2]);
    _build_control (s_page[3]);
    _build_quality (s_page[4]);
    _build_system  (s_page[5]);
    _build_camera  (s_page[6]);
    _build_preflight(s_page[7]);
    _build_pl      (s_page[8]);
    _build_models  (s_page[9]);
    _build_ai_overview(s_page[10]);
    _build_edge_lm (s_page[11]);
    _build_robust  (s_page[12]);

    /* per-model detail + benchmark overlays (body-area, hidden until opened) */
    _build_detail(scr);
    _build_bench(scr);
    _build_cluster_ov(scr);   /* Edge LLM cluster overlay (5 swap-loaded experts) */
    _build_camview(scr);   /* Camera LIVE real-frame overlay (Wave C) */

    /* modal overlay built LAST → topmost; hidden until ABORT is pressed */
    _build_modal(scr);

    s_tcount        = 0;
    s_last_risk     = 0;
    s_uptime_s      = 0U;
    s_q_built_total = -1;   /* force a Quality refresh on first view */

    _show_page(0);   /* start on Home */
}

/* ------------------------------------------------------------------ */
/* AI-4 risk -> status bar + System log                               */
/* ------------------------------------------------------------------ */
/* Append one line to the System risk/event log, then cap the list length so it
 * can never grow without bound (see LOG_MAX_ENTRIES). Shared by the risk-alert
 * and AI-5 root-cause log paths. */
static void _log_add_cap(const char *txt, uint32_t color)
{
    lv_obj_t *e;
    if (s_log_list == NULL) return;
    e = lv_list_add_text(s_log_list, txt);
    if (e != NULL) lv_obj_set_style_text_color(e, lv_color_hex(color), LV_PART_MAIN);
    while (lv_obj_get_child_cnt(s_log_list) > LOG_MAX_ENTRIES) {
        lv_obj_del(lv_obj_get_child(s_log_list, 0));   /* drop oldest */
    }
    lv_obj_scroll_to_y(s_log_list, LV_COORD_MAX, LV_ANIM_OFF);
}

void ui_screen_update_risk(const risk_alert_t *alert)
{
    char buf[56];
    uint8_t lvl;
    static const char *const _src[8] = {
        "-", "VIS", "VIB", "VIS+VIB", "ENV", "VIS+ENV", "VIB+ENV", "ALL"
    };

    if (alert == NULL) return;
    lvl = (uint8_t)alert->risk;
    if (lvl > 3U) lvl = 3U;

    /* The fusion task posts an alert EVERY cycle. Only act on a CHANGE of risk
     * level: re-styling the status bar + appending a log line on every identical
     * alert repaints the whole screen (full_refresh=1) each loop and balloons
     * the log list — both of which the user saw as screen flicker. */
    if ((int)lvl == s_last_risk) return;
    s_last_risk = (int)lvl;

    lv_obj_set_style_bg_color(s_status_pill, lv_color_hex(_risk_fill[lvl]), LV_PART_MAIN);
    snprintf(buf, sizeof(buf), "%s  %s",
             (lvl == 0U) ? LV_SYMBOL_OK : LV_SYMBOL_WARNING, _risk_text[lvl]);
    lv_label_set_text(s_status_risk, buf);   /* dark text set at creation stays readable on the bright pill */

    snprintf(buf, sizeof(buf), "[T+%lus] %s src=%s",
             (unsigned long)(alert->timestamp_ms / 1000U),
             _risk_text[lvl], _src[alert->trigger_source & 0x07U]);
    _log_add_cap(buf, _risk_fill[lvl]);
}

/* ------------------------------------------------------------------ */
/* AI panel: AI-tab detail + Home compact lights                       */
/* ------------------------------------------------------------------ */
void ui_screen_update_ai(const lab_ai_snapshot_t *s)
{
    char buf[40];
    int i, amax, conf, c3, rk;
    float mx, r;
    /* CHANGE-DETECT: this runs every 33ms (ui_task). Re-setting a label every call
     * invalidates the screen -> full_refresh repaints the whole 800x480 -> ghosting.
     * Gate each Home-light update on its value so an idle AI snapshot issues 0
     * invalidations. The AI-1..4 live readouts now live ONLY on the Home compact
     * lights (the old AI-tab bars were repurposed into the Control+Cloud page); the
     * AI-2/AI-3 explainability is stashed (plain ints, no widget) for the Models
     * detail page to surface on its "live" line. */
    static int la1 = -1, la2 = -1, la3 = -1, la4 = -1;

    if (s == NULL) return;

    /* AI-1 vision -> Home light */
    amax = 0; mx = s->ai1_probs[0];
    for (i = 1; i < 4; i++) if (s->ai1_probs[i] > mx) { mx = s->ai1_probs[i]; amax = i; }
    conf = (int)(mx * 100.0f);
    if (amax * 1000 + conf != la1) {
        la1 = amax * 1000 + conf;
        snprintf(buf, sizeof(buf), "AI-1 vision: %s  %d%%", AI1N[amax], conf);
        lv_label_set_text(s_home_ai[0], buf);
    }

    /* AI-2 anomaly ratio -> Home light (+ stash ratio for the Models detail fold) */
    r = s->ai2_ratio;
    {
        int whole = (int)r;
        int frac  = (int)((r - (float)whole) * 10.0f);
        if (frac < 0) frac = 0;
        if (whole * 100 + frac != la2) {
            la2 = whole * 100 + frac;
            snprintf(buf, sizeof(buf), "AI-2 anomaly: x%d.%d", whole, frac);
            lv_label_set_text(s_home_ai[1], buf);
        }
        s_ai2_ratio_x10 = whole * 10 + frac;
    }

    /* AI-3 sinter curve -> Home light */
    c3 = s->ai3_cls; if (c3 < 0) c3 = 0; if (c3 > 4) c3 = 4;
    conf = (int)(s->ai3_probs[c3] * 100.0f);
    if (c3 * 1000 + conf != la3) {
        la3 = c3 * 1000 + conf;
        snprintf(buf, sizeof(buf), "AI-3 sinter: %s  %d%%", AI3N[c3], conf);
        lv_label_set_text(s_home_ai[2], buf);
    }

    /* AI-4 risk -> Home light (latest fusion verdict from ui_screen_update_risk) */
    rk = s_last_risk; if (rk < 0) rk = 0; if (rk > 3) rk = 3;
    if (rk != la4) {
        la4 = rk;
        snprintf(buf, sizeof(buf), "AI-4 risk: %s", AI4N[rk]);
        lv_label_set_text(s_home_ai[3], buf);
    }

    /* ---- explainability stashes for the Models detail "live" line (no widgets,
     * so unconditional int writes here cost nothing — no screen invalidation) ---- */
    /* AI-2 AE per-feature attribution: which channel dominates the residual. */
    {
        int t0 = (int)(s->ai2_resid[0] * 100.0f);
        int t1 = (int)(s->ai2_resid[1] * 100.0f);
        int t2 = (int)(s->ai2_resid[2] * 100.0f);
        int top = 0, best = t0;
        if (t1 > best) { best = t1; top = 1; }
        if (t2 > best) { best = t2; top = 2; }
        s_ai2_attr_top = (best <= 0) ? -1 : top;
    }
    {
        int milliq = (int)(ai2_ae_qhat() * 1000.0f);
        if (milliq < 0)   milliq = 0;
        if (milliq > 999) milliq = 999;
        s_ai2_qhat_milli = milliq;
    }
    /* AI-3 attention saliency: which minute of the 64-min window it focused on. */
    {
        int   peak = 0;
        float amx = 0.0f;
        for (i = 0; i < ATTN_N; i++)
            if (s->ai3_attn[i] > amx) { amx = s->ai3_attn[i]; peak = i; }
        s_ai3_attn_peak = peak;
    }
}

/* ------------------------------------------------------------------ */
/* Rolling temperature curve (Home), fed by ui_task at 1 Hz            */
/* ------------------------------------------------------------------ */
void ui_screen_push_temp(float temp_c)
{
    int i;
    long t;
    static int last_pushed = -99999;

    if (temp_c < 0.0f)            temp_c = 0.0f;
    if (temp_c > (float)TEMP_MAX) temp_c = (float)TEMP_MAX;

    /* Skip the whole re-plot when the curve is full AND the value is unchanged:
     * a constant (idle) temperature would otherwise invalidate the Home page once
     * per second (full_refresh -> whole-screen repaint+swap = a 1 Hz blink). */
    if (s_tcount >= CURVE_N && (int)temp_c == last_pushed) return;
    last_pushed = (int)temp_c;

    if (s_tcount < CURVE_N) {
        s_temps[s_tcount++] = temp_c;
    } else {
        for (i = 1; i < CURVE_N; i++) s_temps[i - 1] = s_temps[i];
        s_temps[CURVE_N - 1] = temp_c;
    }

    for (i = 0; i < s_tcount; i++) {
        long yv;
        t = (long)s_temps[i];
        s_pts[i].x = (lv_coord_t)((long)i * (CURVE_W - 1) / (CURVE_N - 1));
        yv = (CURVE_H - 1) - (t * (CURVE_H - 1) / TEMP_MAX);
        if (yv < 0)           yv = 0;
        if (yv > CURVE_H - 1) yv = CURVE_H - 1;
        s_pts[i].y = (lv_coord_t)yv;
    }

    if (s_tcount >= 2 && s_curve != NULL)
        lv_line_set_points(s_curve, s_pts, (uint16_t)s_tcount);
}

/* ------------------------------------------------------------------ */
/* Voice status (System footer)                                        */
/* ------------------------------------------------------------------ */
void ui_screen_set_voice_status(const char *text)
{
    if (text == NULL || s_lbl_voice == NULL) return;
    lv_label_set_text(s_lbl_voice, text);
}

/* ------------------------------------------------------------------ */
/* Per-frame refresh from the controller snapshot                      */
/* ------------------------------------------------------------------ */
void ui_screen_tick(void)
{
    ctrl_snapshot_t cs;
    char buf[48];
    int n, i;
    /* CHANGE-DETECT caches: repaint only when a value actually changes so an IDLE
     * screen is never re-invalidated. full_refresh=1 repaints the whole 800x480 and
     * swaps both framebuffers on ANY invalidation, so an every-tick label-set made
     * the panel beat between the two buffers (visible as ghosting / "一闪一闪").
     * With these guards an idle page issues 0 invalidations -> static -> no flicker. */
    static int      lc_meas = -99999, lc_sp = -99999, lc_u = -1, lc_pp = -1;
    static int      lc_seg = -1, lc_state = -1;
    static uint32_t lc_batch = 0xFFFFFFFFu;
    static int      lc_cpk = -99999, lc_mean = -99999, lc_sig = -99999;
    static int      lc_inctl = -1, lc_spcn = -1, lc_spclast = -99999;

    /* ABORT two-tap: auto-revert the armed "TAP AGAIN" if no 2nd tap in time.
     * Log the expiry so the serial trace shows whether a failed confirm was the
     * window timing out (-> widen further) vs. a tap not landing. */
    if (s_abort_armed && lv_tick_elaps(s_abort_arm_ms) > ABORT_ARM_MS) {
        lab_log("[ui] ABORT confirm window expired (tap ABORT again to retry)\r\n");
        _abort_disarm();
    }

    /* Camera LIVE overlay: pull a fresh OV5640 frame into the lv_img at ~5 Hz while
     * it's open. Only invalidates when the live view is shown (the deliberate video
     * mode), so the rest of the HMI keeps its change-gated, flicker-free behaviour. */
    if (s_camview_open) {
        static uint32_t cv_div = 0u;
        if (++cv_div >= 6u) { cv_div = 0u; _camview_refresh(); }
    }

    /* Edge LLM cluster overlay: refresh active expert + generated sentence when the
     * cluster_task swaps/regenerates (change-gated by the generation counter). */
    if (s_cluster_open) {
        cluster_view_t cv;
        lab_get_cluster(&cv);
        if (cv.gens != s_cl_gens_seen) {
            s_cl_gens_seen = cv.gens;
            if (!cv.provisioned) {
                lv_label_set_text(s_cl_text, "cluster not provisioned - run provision_cluster.py then reboot");
                lv_label_set_text(s_cl_meta, "no SPI-flash image");
            } else {
                char mb[80]; int i;
                lv_label_set_text(s_cl_text, cv.text[0] ? cv.text : "(generating...)");
                snprintf(mb, sizeof(mb), "active E%d %s   swap %dms   conf %d%%   gen %lu",
                         cv.expert + 1, cv.role, cv.swap_ms, cv.conf_pct, (unsigned long)cv.gens);
                lv_label_set_text(s_cl_meta, mb);
                for (i = 0; i < CL_NEXP; i++)
                    lv_obj_set_style_text_color(s_cl_row[i],
                        lv_color_hex((i == cv.expert) ? COL_GREEN : 0x777777U), LV_PART_MAIN);
            }
        }
    }

    lab_ctrl_get(&cs);

    /* status bar + Home: controller temp (shared label text) — change-gated */
    if (cs.meas_c != lc_meas) {
        snprintf(buf, sizeof(buf), "%d C", cs.meas_c);
        lv_label_set_text(s_status_temp, buf);
        lv_label_set_text(s_home_temp, buf);
        lc_meas = cs.meas_c;
    }
    /* status bar: batch id + state — change-gated */
    if (cs.batch_id != lc_batch || (int)cs.state != lc_state) {
        const char *st = (cs.state == 1) ? "RUN" : (cs.state == 2) ? "DONE" :
                         (cs.state == 3) ? "FAULT" : "idle";
        snprintf(buf, sizeof(buf), LV_SYMBOL_FILE "  garnet #%lu  %s", (unsigned long)cs.batch_id, st);
        lv_label_set_text(s_status_batch, buf);
        lc_batch = cs.batch_id; lc_state = (int)cs.state;
    }
    /* Home: segment line (SP/duty move during a ramp, stable when idle) */
    if ((int)cs.seg_idx != lc_seg || cs.sp_c != lc_sp || cs.u_pct != lc_u) {
        snprintf(buf, sizeof(buf), "seg: %s  SP %dC  u%d%%", cs.seg_label, cs.sp_c, cs.u_pct);
        lv_label_set_text(s_home_seg, buf);
        lc_seg = (int)cs.seg_idx; lc_sp = cs.sp_c; lc_u = cs.u_pct;
    }
    /* Home: progress bar + % — change-gated */
    {
        int pp = (cs.total_s >= 100U) ? (int)(cs.elapsed_s / (cs.total_s / 100U)) : 0;
        if (pp < 0) pp = 0; if (pp > 100) pp = 100;
        if (pp != lc_pp) {
            _set_bar(s_bar_prog, pp, COL_BLUE);
            snprintf(buf, sizeof(buf), "%d%%", pp);
            lv_label_set_text(s_lbl_prog, buf);
            lc_pp = pp;
        }
    }

    /* Trend SPC chart + stats — only when the Trend tab is visible, and each item
     * change-gated (SPC data only moves during a soak), so it adds 0 idle repaint. */
    if (s_cur_page == 2) {
        n = lab_spc_series(s_spc_dev, SPC_N);
        if (n >= 2 && (n != lc_spcn || (int)s_spc_dev[n - 1] != lc_spclast)) {
            int mid = CHH / 2;
            for (i = 0; i < n; i++) {
                long yv;
                s_spc_pts[i].x = (lv_coord_t)((long)i * (CHW - 1) / (n - 1));
                yv = mid - (long)s_spc_dev[i] * mid / DEV_FS;
                if (yv < 0)           yv = 0;
                if (yv > CHH - 1)     yv = CHH - 1;
                s_spc_pts[i].y = (lv_coord_t)yv;
            }
            lv_line_set_points(s_spc_line, s_spc_pts, (uint16_t)n);
            lc_spcn = n; lc_spclast = (int)s_spc_dev[n - 1];
        }
        if (cs.cpk_x100 != lc_cpk) {
            int w = cs.cpk_x100 / 100, f = cs.cpk_x100 % 100; if (f < 0) f = -f;
            snprintf(buf, sizeof(buf), "Cpk %d.%02d", w, f);
            lv_label_set_text(s_trend_cpk, buf);
            lv_obj_set_style_text_color(s_trend_cpk,
                lv_color_hex(cs.cpk_x100 >= 133 ? COL_GREEN : COL_RED), LV_PART_MAIN);
            lc_cpk = cs.cpk_x100;
        }
        if (cs.spc_mean_x100 != lc_mean) {
            int w = cs.spc_mean_x100 / 100, f = cs.spc_mean_x100 % 100; if (f < 0) f = -f;
            snprintf(buf, sizeof(buf), "mean %d.%02d C", w, f);
            lv_label_set_text(s_trend_mean, buf);
            lc_mean = cs.spc_mean_x100;
        }
        if (cs.spc_sigma_x100 != lc_sig) {
            int w = cs.spc_sigma_x100 / 100, f = cs.spc_sigma_x100 % 100; if (f < 0) f = -f;
            snprintf(buf, sizeof(buf), "sigma %d.%02d", w, f);
            lv_label_set_text(s_trend_sigma, buf);
            lc_sig = cs.spc_sigma_x100;
        }
        if ((int)cs.spc_in_control != lc_inctl) {
            lv_label_set_text(s_trend_inctl, cs.spc_in_control ? LV_SYMBOL_OK " IN-CONTROL"
                                                              : LV_SYMBOL_WARNING " OUT-OF-CTRL");
            lv_obj_set_style_text_color(s_trend_inctl,
                lv_color_hex(cs.spc_in_control ? COL_GREEN : COL_RED), LV_PART_MAIN);
            lc_inctl = (int)cs.spc_in_control;
        }
        /* AI-14 multi-step forecast: predicted temp +12 min ahead (change-gated). */
        {
            fc_view_t fc;
            static int lc_fc = -99999;
            lab_get_forecast(&fc);
            if (fc.valid && fc.next_c[fc.n - 1] != lc_fc) {
                lc_fc = fc.next_c[fc.n - 1];
                snprintf(buf, sizeof(buf), "%s +12min: %dC %s",
                         fc.reach_sp ? LV_SYMBOL_UP : LV_SYMBOL_DOWN, fc.next_c[fc.n - 1],
                         fc.reach_sp ? "on track" : "stalling");
                lv_label_set_text(s_trend_fc_val, buf);
                lv_obj_set_style_text_color(s_trend_fc_val,
                    lv_color_hex(fc.reach_sp ? COL_GREEN : COL_AMBER), LV_PART_MAIN);
            }
        }
    }

    /* Control + Cloud page — refresh only when visible, each item change-gated so
     * an idle page issues 0 invalidations. `cs` was already fetched above; the
     * cloud view is a separate read-only snapshot published by env_task (1 Hz). */
    if (s_cur_page == 3) {
        static int cc_state = -2, cc_seg = -2, cc_sp = -99999, cc_pv = -99999,
                   cc_probe = -99999, cc_tcf = -2, cc_tcp = -2, cc_u = -999,
                   cc_cpk = -99999, cc_elem = -2;
        static int cc_link = -2, cc_rssi = 1;
        static uint32_t cc_up = 0xFFFFFFFFu;
        cloud_view_t cv;

        /* ---- closed-loop control telemetry (top half) ---- */
        if ((int)cs.state != cc_state || (int)cs.seg_idx != cc_seg) {
            const char *st = (cs.state == 1) ? "RUN" : (cs.state == 2) ? "DONE" :
                             (cs.state == 3) ? "FAULT" : "IDLE";
            snprintf(buf, sizeof(buf), "STATE: %s", st);
            lv_label_set_text(s_cc_state, buf);
            snprintf(buf, sizeof(buf), "SEG: %s", cs.seg_label);
            lv_label_set_text(s_cc_seg, buf);
            cc_state = (int)cs.state; cc_seg = (int)cs.seg_idx;
        }
        if (cs.sp_c != cc_sp) {
            snprintf(buf, sizeof(buf), "SETPOINT: %d C", cs.sp_c);
            lv_label_set_text(s_cc_sp, buf); cc_sp = cs.sp_c;
        }
        if (cs.meas_c != cc_pv) {
            snprintf(buf, sizeof(buf), "PV (sim plant): %d C", cs.meas_c);
            lv_label_set_text(s_cc_pv, buf); cc_pv = cs.meas_c;
        }
        if (cs.probe_c != cc_probe || (int)cs.tc_present != cc_tcp) {
            if (cs.tc_present) snprintf(buf, sizeof(buf), "PROBE (real TC): %d C", cs.probe_c);
            else               snprintf(buf, sizeof(buf), "PROBE (real TC): -- (not wired)");
            lv_label_set_text(s_cc_probe, buf);
            lv_obj_set_style_text_color(s_cc_probe,
                lv_color_hex(cs.tc_present ? COL_GREEN : COL_GRAY), LV_PART_MAIN);
            cc_probe = cs.probe_c; cc_tcp = (int)cs.tc_present;
        }
        if (cs.u_pct != cc_u) {
            snprintf(buf, sizeof(buf), "u: %d%%", cs.u_pct);
            lv_label_set_text(s_cc_u, buf);
            _set_bar(s_cc_ubar, cs.u_pct, COL_ORANGE);
            cc_u = cs.u_pct;
        }
        {
            /* Gate on present+fault together: a not-wired probe floats all-ones ->
             * fault=0x7, which is NOT a real fault, so show "no probe" (gray) not a
             * scary red triple-fault. Real faults only mean something when present. */
            int tcsig = cs.tc_present ? (1000 + (int)cs.tc_fault) : 0;
            if (tcsig != cc_tcf) {
                uint32_t col;
                if (!cs.tc_present) {
                    snprintf(buf, sizeof(buf), "TC: no probe (not wired)");
                    col = COL_GRAY;
                } else if (cs.tc_fault == 0u) {
                    snprintf(buf, sizeof(buf), "TC: healthy");
                    col = COL_GREEN;
                } else {
                    snprintf(buf, sizeof(buf), "TC FAULT: %s%s%s",
                             (cs.tc_fault & 1u) ? "open " : "",
                             (cs.tc_fault & 2u) ? "short-GND " : "",
                             (cs.tc_fault & 4u) ? "short-VCC" : "");
                    col = COL_RED;
                }
                lv_label_set_text(s_cc_tc, buf);
                lv_obj_set_style_text_color(s_cc_tc, lv_color_hex(col), LV_PART_MAIN);
                cc_tcf = tcsig;
            }
        }
        if (cs.cpk_x100 != cc_cpk) {
            int w = cs.cpk_x100 / 100, f = cs.cpk_x100 % 100; if (f < 0) f = -f;
            snprintf(buf, sizeof(buf), "Cpk: %d.%02d", w, f);
            lv_label_set_text(s_cc_cpk, buf); cc_cpk = cs.cpk_x100;
        }
        if (cs.elem_pct != cc_elem) {
            snprintf(buf, sizeof(buf), "ELEMENT: %d%%", cs.elem_pct);
            lv_label_set_text(s_cc_elem, buf); cc_elem = cs.elem_pct;
        }

        /* ---- ESP32 telemetry uplink, export only (bottom half) ---- */
        lab_get_cloud(&cv);
        if ((int)cv.link != cc_link) {
            const char *ls = (cv.link == 2) ? "UPLINK ON" :
                             (cv.link == 1) ? "WiFi UP" : "OFFLINE";
            snprintf(buf, sizeof(buf), LV_SYMBOL_WIFI " LINK: %s", ls);
            lv_label_set_text(s_cc_link, buf);
            lv_obj_set_style_text_color(s_cc_link,
                lv_color_hex((cv.link == 2) ? COL_GREEN : (cv.link == 1) ? COL_AMBER : COL_GRAY),
                LV_PART_MAIN);
            cc_link = (int)cv.link;
        }
        if (cv.rssi_dbm != cc_rssi) {
            if (cv.rssi_dbm == 0) snprintf(buf, sizeof(buf), "WiFi: --");
            else                  snprintf(buf, sizeof(buf), "WiFi: %d dBm", cv.rssi_dbm);
            lv_label_set_text(s_cc_rssi, buf); cc_rssi = cv.rssi_dbm;
        }
        if (cv.uplinks != cc_up) {
            snprintf(buf, sizeof(buf), "uplinks: %lu", (unsigned long)cv.uplinks);
            lv_label_set_text(s_cc_up, buf); cc_up = cv.uplinks;
        }
        /* NOTE: s_cc_r1 is a static "telemetry payload" descriptor — the ESP32 link
         * is export-only, so we deliberately do NOT render any imported cloud verdict
         * (cv.r1) here. All diagnosis stays on-chip (edge nano-LM above). */

    }

    /* Edge LM page (idx 11) — flagship generative diagnosis + LM selector + online
     * head. Each item change-gated so an idle page issues 0 invalidations. When the
     * LM stack is compiled out (LAB_LM_ENABLE off) lab_get_nlm() returns !valid, so
     * s_cc_nlm keeps the built-in "disabled" note and nothing here repaints. */
    if (s_cur_page == 11) {
        static uint32_t cc_nlm_gen = 0xFFFFFFFFu;   /* gate: nano-LM generation count */
        static long     cc_ol_sig  = -1;            /* gate: online-head pred/teach/acc */

        /* active generative-LM size indicator (gate on selector change) */
        {
            int a = lab_lm_active();
            if (a != s_cc_lm_shown) {
                char lb[48];
                snprintf(lb, sizeof(lb), "LM: %s %s", lm_roster_tag_s(a), lm_roster_label_s(a));
                lv_label_set_text(s_cc_lm, lb);
                s_cc_lm_shown = a;
            }
        }

        /* edge nano-LM generated diagnosis (gate on generation count) */
        {
            nlm_view_t nv;
            char nb[128];
            lab_get_nlm(&nv);
            if (nv.gens != cc_nlm_gen) {
                if (nv.valid) {
                    snprintf(nb, sizeof(nb), "edge AI (%d%%)%s: %s",
                             nv.conf_pct, nv.escalate ? " ->review" : "", nv.text);
                    lv_label_set_text(s_cc_nlm, nb);
                    lv_obj_set_style_text_color(s_cc_nlm,
                        lv_color_hex(nv.escalate ? COL_AMBER : COL_GREEN), LV_PART_MAIN);
                }
                cc_nlm_gen = nv.gens;
            }
        }
        /* on-device online-learning risk head (gate on packed signature) */
        {
            online_view_t ov;
            char nb[80];
            long sig;
            static const char *RN[4] = {"good", "warn", "bad", "crit"};
            lab_get_online(&ov);
            sig = (long)ov.pred * 1000000L + (long)(ov.teaches & 0x3FFu) * 1000L + ov.acc_pct;
            if (sig != cc_ol_sig) {
                snprintf(nb, sizeof(nb), "online head: risk=%s  taught %lu  acc %d%%",
                         RN[(ov.pred >= 0 && ov.pred < 4) ? ov.pred : 0],
                         (unsigned long)ov.teaches, ov.acc_pct);
                lv_label_set_text(s_cc_olrow, nb);
                cc_ol_sig = sig;
            }
        }
    }

    /* Robust page: live perturbation readouts + long-run stability panel. One
     * coarse signature gates the whole refresh -> ~1 update/sec (uptime/inferences)
     * plus an immediate update on any injection tap; idle issues no extra invalidates. */
    if (s_cur_page == 12) {
        static const char *const VISM[5] = {"clean", "noise", "dark", "bright", "occlude"};
        static const char *const PLM[4]  = {"clean", "noise", "occlude", "baseln"};
        static const char *const TCM[3]  = {"healthy", "open-ckt", "erratic"};
        static const char *const VCL[3]  = {"empty", "loaded", "done"};
        static const char *const PCL[3]  = {"Cr", "Ni", "CrNi"};
        static const char *const TCL[3]  = {"healthy", "open", "erratic"};
        static const char *const RST[5]  = {"unknown", "WATCHDOG", "software", "power-on", "ext-pin"};
        static long rob_sig = -1;
        rob_view_t rb; health_view_t hv; ai_extra_view_t ax;
        int vinj = lab_get_vis_inject(), pinj = lab_get_pl_inject(), tcinj = lab_get_tc_inject();
        long sig;
        lab_get_rob(&rb); lab_get_health(&hv); lab_get_ai_extra(&ax);
        sig = (long)vinj + (long)pinj * 5 + (long)tcinj * 23
            + (long)(rb.vis_pert_cls + 1) * 101 + (long)rb.vis_pert_conf * 7
            + (long)(rb.pl_pert_cls + 1) * 1009 + (long)rb.pl_pert_conf * 11
            + (long)(ax.tc_cls + 1) * 9001 + (long)ax.tc_valid * 40000
            + (long)hv.uptime_s * 131 + (long)(hv.inferences & 0xFFFu) * 97;
        if (sig != rob_sig) {
            char b[72];
            int vc  = (rb.vis_pert_cls  >= 0 && rb.vis_pert_cls  < 3) ? rb.vis_pert_cls  : 0;
            int vcc = (rb.vis_clean_cls >= 0 && rb.vis_clean_cls < 3) ? rb.vis_clean_cls : 0;
            int pc  = (rb.pl_pert_cls   >= 0 && rb.pl_pert_cls   < 3) ? rb.pl_pert_cls   : 0;
            int pcc = (rb.pl_clean_cls  >= 0 && rb.pl_clean_cls  < 3) ? rb.pl_clean_cls  : 0;
            int tc  = (ax.tc_cls        >= 0 && ax.tc_cls        < 3) ? ax.tc_cls        : 0;
            _btn_set_label(s_rob_visbtn, VISM[vinj % 5]);
            _btn_set_label(s_rob_plbtn,  PLM[pinj % 4]);
            _btn_set_label(s_rob_tcbtn,  TCM[tcinj % 3]);
            if (vinj == 0) snprintf(b, sizeof(b), "%s %d%%", VCL[vc], rb.vis_pert_conf);
            else snprintf(b, sizeof(b), "%s %d%% -> %s %d%%", VCL[vcc], rb.vis_clean_conf, VCL[vc], rb.vis_pert_conf);
            lv_label_set_text(s_rob_vislbl, b);
            lv_obj_set_style_text_color(s_rob_vislbl,
                lv_color_hex((vinj != 0 && vc != vcc) ? COL_AMBER : COL_TEXT2), LV_PART_MAIN);
            if (pinj == 0) snprintf(b, sizeof(b), "%s %d%%", PCL[pc], rb.pl_pert_conf);
            else snprintf(b, sizeof(b), "%s %d%% -> %s %d%%", PCL[pcc], rb.pl_clean_conf, PCL[pc], rb.pl_pert_conf);
            lv_label_set_text(s_rob_pllbl, b);
            lv_obj_set_style_text_color(s_rob_pllbl,
                lv_color_hex((pinj != 0 && pc != pcc) ? COL_AMBER : COL_TEXT2), LV_PART_MAIN);
            if (!ax.tc_valid) snprintf(b, sizeof(b), "verdict: (run a batch)");
            else snprintf(b, sizeof(b), "verdict: %s %d%%", TCL[tc], ax.tc_conf_pct);
            lv_label_set_text(s_rob_tclbl, b);
            lv_obj_set_style_text_color(s_rob_tclbl,
                lv_color_hex((!ax.tc_valid) ? COL_GRAY : (tc == 0 ? COL_GREEN : COL_RED)), LV_PART_MAIN);
            {
                unsigned long u = (unsigned long)hv.uptime_s;
                if (u < 3600UL) snprintf(b, sizeof(b), "uptime %lum%lus", u / 60UL, u % 60UL);
                else snprintf(b, sizeof(b), "uptime %luh%lum", u / 3600UL, (u % 3600UL) / 60UL);
                lv_label_set_text(s_rob_uptime, b);
            }
            snprintf(b, sizeof(b), "AI inferences %lu", (unsigned long)hv.inferences);
            lv_label_set_text(s_rob_infer, b);
            snprintf(b, sizeof(b), "reset: %s | wdg %s",
                     RST[hv.reset_cause < 5 ? hv.reset_cause : 0], hv.wdg_armed ? "armed" : "off");
            lv_label_set_text(s_rob_reset, b);
            rob_sig = sig;
        }
    }

    /* Recipe: highlight the running segment (only while this page is shown) */
    if (s_cur_page == 1 && s_seg_rows > 0) {
        int act = (cs.state == 1) ? (int)cs.seg_idx : -1;
        for (i = 0; i < s_seg_rows; i++)
            lv_obj_set_style_bg_color(s_seg_row[i],
                lv_color_hex((i == act) ? 0x1E3A1EU : 0x141414U), LV_PART_MAIN);
    }

    /* Quality: rebuild the ledger only when a new batch has been sealed */
    if (s_cur_page == 4) {
        int total = lab_ledger_total();
        if (total != s_q_built_total) { _quality_refresh(); s_q_built_total = total; }
    }

    /* ---- AI-5 root-cause: status-bar badge + System log + Home light (any page).
     * Badge/log fire on a *fault* root-cause change so the operator sees WHY +
     * WHAT-TO-DO from any tab; lab_get_ai5() already gates to NORMAL unless an
     * anomaly is flagged, so a clean run shows a calm green NORMAL. The AI-5 / NCM
     * detail used to also live on the (now repurposed) AI tab — it is fully
     * covered by the Home lights + status badge + System log. ---- */
    {
        static int last_badge = -2, last_ncm_home = -2;
        int cls = 0, pct = 0, ncm = -1;
        lab_get_ai5(&cls, &pct);
        lab_get_ncm(&ncm);
        (void)pct;
        if (cls < 0 || cls >= AI5_N_CLASS) cls = 0;

        /* AI-5: status-bar badge + System log + Home light — fire on fault-class
         * change regardless of which tab is open. */
        if (cls != last_badge) {
            if (cls == (int)AI5_RC_NORMAL) {
                lv_label_set_text(s_status_ai5, "");      /* clear badge */
                snprintf(buf, sizeof(buf), "AI-5 root: NORMAL");
                lv_obj_set_style_text_color(s_home_ai[5], lv_color_hex(COL_GREEN), LV_PART_MAIN);
            } else {
                char lg[160];
                snprintf(buf, sizeof(buf), "AI5:%s", ai5_name(cls));
                lv_label_set_text(s_status_ai5, buf);     /* status-bar badge */
                /* System log: full corrective action, persistent + auto-scroll */
                snprintf(lg, sizeof(lg), "AI-5 %s -> %s", ai5_name(cls), ai5_action(cls));
                _log_add_cap(lg, COL_ORANGE);
                snprintf(buf, sizeof(buf), "AI-5 root: %s", ai5_name(cls));
                lv_obj_set_style_text_color(s_home_ai[5], lv_color_hex(COL_ORANGE), LV_PART_MAIN);
            }
            lv_label_set_text(s_home_ai[5], buf);         /* Home compact light */
            last_badge = cls;
        }
        /* AI-1b NCM — Home light, any page */
        if (ncm != last_ncm_home) {
            if (ncm < 0) snprintf(buf, sizeof(buf), "AI-1b NCM: --");
            else         snprintf(buf, sizeof(buf), "AI-1b NCM: c%d", ncm);
            lv_label_set_text(s_home_ai[4], buf);
            last_ncm_home = ncm;
        }
    }

    /* ---- AI-6..AI-20: Home lights + Camera + Pre-flight + PL + Trend (20-model HMI) ---- */
    {
        recipe_ai_t pf;
        vib_view_t  vv;
        pl_view_t   pl;
        static int  pf_done  = 0;   /* recipe AI is static: fill Home lights once */
        static int  last_vib = -2;
        static int  last_vrun = -1;
        static int  last_pl_home  = -1;   /* Home AI-12/13 lights gate (PL cycles ~3s) */
        static int  last_pl_idx   = -1;   /* PL page repaint gate (changes ~3s) */

        lab_get_recipe_ai(&pf);
        lab_get_vib(&vv);
        lab_get_pl(&pl);

        /* Home compact lights for the recipe-level models (filled once) */
        if (pf.valid && !pf_done) {
            snprintf(buf, sizeof(buf), "AI-6 lam: %dnm", pf.lambda_nm);
            lv_label_set_text(s_home_ai[6], buf);
            snprintf(buf, sizeof(buf), "AI-7 therm: %s", BANDN[pf.thermal_band % 3]);
            lv_label_set_text(s_home_ai[7], buf);
            lv_obj_set_style_text_color(s_home_ai[7],
                lv_color_hex(BANDC[pf.thermal_band % 3]), LV_PART_MAIN);
            snprintf(buf, sizeof(buf), "AI-8 E: %dkWh", pf.kwh_x10 / 10);
            lv_label_set_text(s_home_ai[8], buf);
            snprintf(buf, sizeof(buf), "AI-9 ~#%d", pf.analog_idx);
            lv_label_set_text(s_home_ai[9], buf);
            /* AI-11 phase-purity prior (static for the fixed recipe) */
            snprintf(buf, sizeof(buf), "AI-11 pure: %d%%", pf.p_pure_pct);
            lv_label_set_text(s_home_ai[11], buf);
            lv_obj_set_style_text_color(s_home_ai[11],
                lv_color_hex(pf.purity_cls ? COL_GREEN : COL_AMBER), LV_PART_MAIN);
            pf_done = 1;
        }
        /* AI-10 vibration Home light (change-gated; idle when motor stopped) */
        if (vv.valid && (vv.cls != last_vib || (int)vv.running != last_vrun)) {
            if (!vv.running) {
                lv_label_set_text(s_home_ai[10], "AI-10 vib: stopped");
                lv_obj_set_style_text_color(s_home_ai[10], lv_color_hex(COL_GRAY), LV_PART_MAIN);
            } else {
                snprintf(buf, sizeof(buf), "AI-10 vib: %s", VIBN[vv.cls & 1]);
                lv_label_set_text(s_home_ai[10], buf);
                lv_obj_set_style_text_color(s_home_ai[10],
                    lv_color_hex((vv.cls == 0) ? COL_GREEN : COL_AMBER), LV_PART_MAIN);
            }
            last_vib = vv.cls; last_vrun = (int)vv.running;
        }

        /* AI-12 PL dopant + AI-13 PL QC Home lights. Page-independent (like the
         * other home lights) so they're current the moment you open Home. The PL
         * demo cycles its 3 real spectra ~every 3 s, so this repaints at ~0.33 Hz
         * (one clean VBlank-synced swap) — far below any flicker threshold. */
        {
            int pl_sig = pl.valid ? ((pl.demo_idx + 1) * 1000 + pl.cls * 10
                                     + (pl.anomaly ? 1 : 0) + 1) : 0;
            if (pl_sig != last_pl_home) {
                last_pl_home = pl_sig;
                snprintf(buf, sizeof(buf), "AI-12 PL: %s", pl.valid ? PLN[pl.cls % 3] : "--");
                lv_label_set_text(s_home_ai[12], buf);
                snprintf(buf, sizeof(buf), "AI-13 QC: %s",
                         pl.valid ? (pl.anomaly ? "ANOM" : "OK") : "--");
                lv_label_set_text(s_home_ai[13], buf);
                lv_obj_set_style_text_color(s_home_ai[13],
                    lv_color_hex(pl.valid ? (pl.anomaly ? COL_RED : COL_GREEN) : COL_GRAY),
                    LV_PART_MAIN);
                /* AI-15 host-ID + AI-16 lambda + AI-17 few-shot (same replayed spectrum) */
                snprintf(buf, sizeof(buf), "AI-15 host: %s",
                         pl.valid ? HOSTN[pl.host_cls & 1] : "--");
                lv_label_set_text(s_home_ai[15], buf);
                snprintf(buf, sizeof(buf), "AI-16 lam: %dnm", pl.valid ? pl.lambda_nm : 0);
                lv_label_set_text(s_home_ai[16], buf);
                snprintf(buf, sizeof(buf), "AI-17 fs: cls%d", pl.valid ? pl.fewshot_cls : 0);
                lv_label_set_text(s_home_ai[17], buf);
            }
        }

        /* AI-14 temp-forecast Home light (change-gated). Shows "idle" until a batch
         * runs (the forecaster needs a live temperature window — nothing to predict on
         * an idle furnace), then the predicted +12min temp. -88888 = idle-shown sentinel. */
        {
            fc_view_t fc;
            static int last_fc_home = -99999;
            lab_get_forecast(&fc);
            if (fc.valid && fc.next_c[fc.n - 1] != last_fc_home) {
                last_fc_home = fc.next_c[fc.n - 1];
                snprintf(buf, sizeof(buf), "AI-14 fc: %dC", fc.next_c[fc.n - 1]);
                lv_label_set_text(s_home_ai[14], buf);
                lv_obj_set_style_text_color(s_home_ai[14],
                    lv_color_hex(fc.reach_sp ? COL_GREEN : COL_AMBER), LV_PART_MAIN);
            } else if (!fc.valid && last_fc_home != -88888) {
                last_fc_home = -88888;
                lv_label_set_text(s_home_ai[14], "AI-14 fc: idle");
                lv_obj_set_style_text_color(s_home_ai[14], lv_color_hex(COL_GRAY), LV_PART_MAIN);
            }
        }

        /* AI-19 RUL/ETA + AI-20 TC-integrity Home lights (env_task, change-gated).
         * AI-19 is "idle" on an idle furnace (no live trajectory to extrapolate);
         * AI-20 runs continuously on the meas/setpoint window. */
        {
            ai_extra_view_t ax;
            static int last_rul = -99999, last_tc = -2;
            lab_get_ai_extra(&ax);
            if (ax.rul_valid && ax.rul_min != last_rul) {
                last_rul = ax.rul_min;
                snprintf(buf, sizeof(buf), "AI-19 ETA: %dm", ax.rul_min);
                lv_label_set_text(s_home_ai[18], buf);
                lv_obj_set_style_text_color(s_home_ai[18], lv_color_hex(COL_GREEN), LV_PART_MAIN);
            } else if (!ax.rul_valid && last_rul != -88888) {
                last_rul = -88888;
                lv_label_set_text(s_home_ai[18], "AI-19 ETA: idle");
                lv_obj_set_style_text_color(s_home_ai[18], lv_color_hex(COL_GRAY), LV_PART_MAIN);
            }
            if (ax.tc_valid && ax.tc_cls != last_tc) {
                last_tc = ax.tc_cls;
                snprintf(buf, sizeof(buf), "AI-20 TC: %s", TCN[ax.tc_cls % 3]);
                lv_label_set_text(s_home_ai[19], buf);
                lv_obj_set_style_text_color(s_home_ai[19],
                    lv_color_hex((ax.tc_cls == 0) ? COL_GREEN : COL_RED), LV_PART_MAIN);
            }
        }

        /* AI-19/AI-20 detail-screen live readout (only while that detail is open) */
        if (s_detail_open && (s_detail_idx == 2 || s_detail_idx == 3 ||
                              s_detail_idx == 18 || s_detail_idx == 19)) {
            char lb[48];
            int  sig = _det_live_fmt(s_detail_idx, lb, sizeof(lb));
            if (sig != s_det_live_sig) {
                s_det_live_sig = sig;
                lv_label_set_text(s_det_live, lb);
            }
        }

        /* Pre-flight page (only when visible) */
        if (s_cur_page == 7 && pf.valid) {
            int ep;
            snprintf(buf, sizeof(buf), LV_SYMBOL_LIST "  PRE-FLIGHT  %s", pf.recipe);
            lv_label_set_text(s_pf_recipe, buf);
            snprintf(buf, sizeof(buf), "%d nm", pf.lambda_nm);
            lv_label_set_text(s_pf_lam, buf);
            snprintf(buf, sizeof(buf), "FWHM %d nm", pf.fwhm_nm);
            lv_label_set_text(s_pf_fwhm, buf);
            lv_label_set_text(s_pf_band, BANDN[pf.thermal_band % 3]);
            lv_obj_set_style_text_color(s_pf_band,
                lv_color_hex(BANDC[pf.thermal_band % 3]), LV_PART_MAIN);
            snprintf(buf, sizeof(buf), "%d %% retained", pf.thermal_pct);
            lv_label_set_text(s_pf_thermal, buf);
            snprintf(buf, sizeof(buf), "%d.%d kWh", pf.kwh_x10 / 10, pf.kwh_x10 % 10);
            lv_label_set_text(s_pf_energy, buf);
            snprintf(buf, sizeof(buf), "%d.%d kg CO2", pf.co2_x10 / 10, pf.co2_x10 % 10);
            lv_label_set_text(s_pf_co2, buf);
            ep = pf.kwh_x10 / 20;                       /* 200 kWh full-scale */
            if (ep > 100) ep = 100; if (ep < 0) ep = 0;
            _set_bar(s_pf_bar_e, ep, COL_BLUE);
            snprintf(buf, sizeof(buf), "#%d %s", pf.analog_idx, pf.analog_name);
            lv_label_set_text(s_pf_analog, buf);
            /* AI-11 phase-purity prior */
            snprintf(buf, sizeof(buf), "%s  P(pure) %d%%",
                     pf.purity_cls ? "PURE" : "IMPURE", pf.p_pure_pct);
            lv_label_set_text(s_pf_purity, buf);
            lv_obj_set_style_text_color(s_pf_purity,
                lv_color_hex(pf.purity_cls ? COL_GREEN : COL_AMBER), LV_PART_MAIN);
            /* derived crystal field (algebra from AI-6 lambda, not a model) */
            {
                static const char *const FC[4] = { "weak", "weak-int", "inter", "strong" };
                snprintf(buf, sizeof(buf), "Dq %d B %d  Dq/B %d.%02d (%s)",
                         pf.dq_cm1, pf.b_cm1, pf.dq_over_b_x100 / 100,
                         pf.dq_over_b_x100 % 100, FC[pf.field_class & 3]);
                lv_label_set_text(s_pf_dqb, buf);
            }
        }

        /* Camera page (only when visible): recolour the 8x6 luminance grid,
         * place the bright-blob box, show the AI-1 verdict + per-class bars. */
        if (s_cur_page == 6) {
            cam_view_t cam;
            lab_get_cam(&cam);
            for (i = 0; i < CAM_CELLS; i++) {
                uint32_t v = cam.lum[i];
                lv_obj_set_style_bg_color(s_cam_tile[i],
                    lv_color_hex((v << 16) | (v << 8) | v), LV_PART_MAIN);
            }
            if (cam.blob_ok) {
                lv_obj_set_pos(s_cam_box, (lv_coord_t)(CAMX + cam.bx * CAMTILE),
                                          (lv_coord_t)(CAMY + cam.by * CAMTILE));
                lv_obj_set_size(s_cam_box, (lv_coord_t)(cam.bw * CAMTILE - 2),
                                           (lv_coord_t)(cam.bh * CAMTILE - 2));
                lv_obj_clear_flag(s_cam_box, LV_OBJ_FLAG_HIDDEN);
                lv_obj_set_pos(s_cam_boxlbl, (lv_coord_t)(CAMX + cam.bx * CAMTILE + 3),
                                             (lv_coord_t)(CAMY + cam.by * CAMTILE + 2));
                lv_label_set_text(s_cam_boxlbl, AI1N3[cam.cls % 3]);
                lv_obj_clear_flag(s_cam_boxlbl, LV_OBJ_FLAG_HIDDEN);
            } else {
                lv_obj_add_flag(s_cam_box, LV_OBJ_FLAG_HIDDEN);
                lv_obj_add_flag(s_cam_boxlbl, LV_OBJ_FLAG_HIDDEN);
            }
            lv_label_set_text(s_cam_cls, AI1N3[cam.cls % 3]);
            snprintf(buf, sizeof(buf), "%d %% %s", cam.conf_pct, cam.valid ? "(cam)" : "(pattern)");
            lv_label_set_text(s_cam_conf, buf);
            for (i = 0; i < 3; i++) {
                int p = (int)(cam.probs[i] * 100.0f + 0.5f);
                _set_bar(s_cam_bar[i], p, (i == cam.cls) ? COL_GREEN : COL_BLUE);
            }
            snprintf(buf, sizeof(buf), "frames %lu", (unsigned long)cam.frames);
            lv_label_set_text(s_cam_info, buf);
            /* B3 CAM 4x4 mini-heatmap: red intensity = AI-1 attention for the class */
            if (cam.cam_ok) {
                for (i = 0; i < 16; i++) {
                    uint32_t h = cam.cam[i];               /* 0..255 saliency */
                    uint32_t col = (h << 16) | ((h >> 2) << 8) | 0x10U;  /* red-hot */
                    lv_obj_set_style_bg_color(s_cam_cam[i], lv_color_hex(col), LV_PART_MAIN);
                }
            }
        }

        /* (Tab 3 was repurposed from the AI roster into the Control+Cloud page; the
         * full model catalog is on the Models tab, Home lists all 20 as compact
         * lights, and per-model detail incl. AI-2/AI-3 live explainability is on the
         * Models detail overlay + Pre-flight/PL tabs.) */

        /* PL page (only when visible): emission-spectrum polyline + AI-12 dopant
         * verdict + class bars + AI-13 QC verdict. Repaint only when the replayed
         * spectrum advances (~3 s) -> static between, no flicker. */
        if (s_cur_page == 8 && pl.valid && pl.demo_idx != last_pl_idx) {
            int p;
            last_pl_idx = pl.demo_idx;
            for (i = 0; i < PL_SPEC_N; i++) {
                long yv = (long)((1.0f - pl.spec[i]) * (float)(PLH - 1));
                if (yv < 0)        yv = 0;
                if (yv > PLH - 1)  yv = PLH - 1;
                s_pl_pts[i].x = (lv_coord_t)((long)i * (PLW - 1) / (PL_SPEC_N - 1));
                s_pl_pts[i].y = (lv_coord_t)yv;
            }
            lv_line_set_points(s_pl_line, s_pl_pts, (uint16_t)PL_SPEC_N);
            lv_label_set_text(s_pl_cls, PLN[pl.cls % 3]);
            lv_obj_set_style_text_color(s_pl_cls, lv_color_hex(PLC[pl.cls % 3]), LV_PART_MAIN);
            snprintf(buf, sizeof(buf), "%d %% conf", pl.conf_pct);
            lv_label_set_text(s_pl_conf, buf);
            for (p = 0; p < 3; p++)
                _set_bar(s_pl_bar[p], pl.probs_pct[p], (p == pl.cls) ? COL_GREEN : COL_BLUE);
            lv_label_set_text(s_pl_qc, pl.anomaly ? "ANOMALY" : "OK");
            lv_obj_set_style_text_color(s_pl_qc,
                lv_color_hex(pl.anomaly ? COL_RED : COL_GREEN), LV_PART_MAIN);
            snprintf(buf, sizeof(buf), "MSE %d / q^ %d (x1e5)", pl.mse_x1e5, pl.qhat_x1e5);
            lv_label_set_text(s_pl_mse, buf);
            /* AI-15 host-ID + AI-16 lambda_em + AI-17 few-shot (same replayed spectrum) */
            snprintf(buf, sizeof(buf), "host: %s %d%%",
                     HOSTN[pl.host_cls & 1], pl.host_conf_pct);
            lv_label_set_text(s_pl_host, buf);
            snprintf(buf, sizeof(buf), "lam_em: %d nm", pl.lambda_nm);
            lv_label_set_text(s_pl_lambda, buf);
            snprintf(buf, sizeof(buf), "few-shot: cls %d (%d reg)",
                     pl.fewshot_cls, pl.fewshot_nclass);
            lv_label_set_text(s_pl_fewshot, buf);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Uptime (System footer), call once per second                       */
/* ------------------------------------------------------------------ */
void ui_screen_tick_uptime(void)
{
    char buf[24];
    s_uptime_s++;
    if (s_uptime_s < 60U) {
        snprintf(buf, sizeof(buf), "%lus", (unsigned long)s_uptime_s);
    } else if (s_uptime_s < 3600U) {
        snprintf(buf, sizeof(buf), "%lum%lus",
                 (unsigned long)(s_uptime_s / 60U), (unsigned long)(s_uptime_s % 60U));
    } else {
        snprintf(buf, sizeof(buf), "%luh%lum",
                 (unsigned long)(s_uptime_s / 3600U),
                 (unsigned long)((s_uptime_s % 3600U) / 60U));
    }
    lv_label_set_text(s_lbl_uptime, buf);
}
