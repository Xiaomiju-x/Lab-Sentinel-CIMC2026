/******************************************************************************
 * actuator.h
 *
 * CIMC Lab-Sentinel — alarm-side actuator driver.
 *
 *   Buzzer    ---   REMOVED 2026-05-29 [旧 PG2 = SDRAM 地址线 A12, 不能当 GPIO]
 *   Fan       ---   removed      [PA1 is smoke sensor DO; relay.c handles ventilation]
 *   Vibration ---   REMOVED 2026-05-29 [旧 PA3 = RGB LCD 蓝色 B5, 不能当 GPIO]
 *   LED1      PE2   onboard (NORMAL heartbeat) — driven by LED.c primitives
 *   LED2      PG3   onboard (WARNING+) [moved from PE5; PE5 is DCI D6]
 *   LED3      PH7   onboard (SEVERE)
 *
 * 2026-05-29: 蜂鸣器 PG2 (=SDRAM A12) + 震动 PA3 (=RGB LCD B5) 撞硬资源, 一翻转
 * 就花屏 → 彻底移除. 报警闭环改走 LED1/2/3 + LCD 红屏 + CI1302 语音 TTS.
 * actuator_buzzer_set / actuator_vib_set / actuator_buzzer_pulse 保留为 no-op.
 *
 * High-level API: call actuator_set_risk() from fusion_task whenever the
 * AI-4 risk level changes. The driver translates the level into the
 * appropriate LED steady state (audible alarm = CI1302 voice).
 ******************************************************************************/

#ifndef __ACTUATOR_H__
#define __ACTUATOR_H__

#include "HeaderFiles.h"
#include "lab_sentinel.h"   /* lab_risk_level_t */

/* ---- init ---- */
void actuator_init(void);

/* ---- discrete control ---- */
void actuator_buzzer_set(uint8_t on);   /* no-op since 2026-05-29 (buzzer removed) */
void actuator_vib_set(uint8_t on);      /* no-op since 2026-05-29 (vib removed)    */
void actuator_led_set(uint8_t idx, uint8_t on);   /* idx = 0 / 1 / 2 (LED1/2/3) */

/* ---- composite ---- */
/* Set steady-state LED pattern matching the risk level (蜂鸣器/震动已移除,
 * 听觉报警走 CI1302 语音):
 * NORMAL  : LED1 on
 * WARNING : LED2 on
 * ANOMALY : LED2 + LED3 on
 * SEVERE  : all 3 LEDs on (relay_on by fusion_task on smoke)
 */
void actuator_set_risk(lab_risk_level_t lvl);

/* no-op since 2026-05-29 (buzzer removed; kept for call-site compatibility). */
void actuator_buzzer_pulse(uint16_t ms);

/* LED self-test: LED1 → LED2 → LED3, each ~250 ms. Blocks ~750 ms.
 * (蜂鸣器/震动已移除, 不再自检.) */
void actuator_self_test(void);

#endif /* __ACTUATOR_H__ */
