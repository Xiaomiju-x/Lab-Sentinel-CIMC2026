/******************************************************************************
 * ui_screen.h
 *
 * Lab-Sentinel AI dashboard — LVGL 8.3, 800x480 landscape (CIMC RGB panel).
 *
 * Layout:
 *   y=  0.. 45  Risk banner       — AI-4 verdict, color-coded
 *   y= 50..255  L: furnace 升温曲线 (lv_line) + stage + progress
 *               R: 4-model verdict panel (AI-1/2/3/4) + confidence bars
 *   y=260..479  L: alert log (scroll)   R: AE per-feature attribution + footer
 *
 * Only widgets whose .c is in the Keil build are used: lv_label / lv_list /
 * lv_bar / lv_line (lv_chart is NOT compiled in). Fonts: montserrat 14/20/28.
 ******************************************************************************/
#ifndef __UI_SCREEN_H__
#define __UI_SCREEN_H__

#include "lvgl.h"
#include "lab_sentinel.h"   /* risk_alert_t, lab_ai_snapshot_t */

/* Build all widgets on the active screen. Call once after lv_init()+disp init. */
void ui_screen_init(void);

/* AI-4 risk → banner color/text + one appended alert-log line. */
void ui_screen_update_risk(const risk_alert_t *alert);

/* Live AI panel: 4-model verdicts + confidence bars + AE attribution + q_hat
 * + furnace stage/progress. Call every UI loop (cheap, label/bar only). */
void ui_screen_update_ai(const lab_ai_snapshot_t *s);

/* Append one furnace-temperature sample to the rolling 升温曲线. Call ~1 Hz. */
void ui_screen_push_temp(float temp_c);

/* Footer voice-status field. */
void ui_screen_set_voice_status(const char *text);

/* Footer uptime counter (call once per second). */
void ui_screen_tick_uptime(void);

/* Refresh the active screen from the controller snapshot (lab_ctrl_get) +
 * SPC trend series. Call every UI loop (cheap; label/bar/line only). */
void ui_screen_tick(void);

/* Switch the active screen (0=Home 1=Recipe 2=Trend 3=AI 4=Quality 5=System).
 * Driven by the touch tab bar or by a voice navigation command. */
void ui_screen_set_nav(int tab);

#endif /* __UI_SCREEN_H__ */
