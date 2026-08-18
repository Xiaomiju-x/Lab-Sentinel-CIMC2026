/******************************************************************************
 * actuator.c — see actuator.h for pin map and usage notes.
 *
 * 2026-05-29 ★ 蜂鸣器 + 震动马达彻底移除 (根因: 引脚撞 SDRAM / RGB LCD):
 *
 *   旧 蜂鸣器 PG2 = EXMC SDRAM 地址线 A12 (AF12). 把 PG2 配成 GPIO 并翻转
 *      (蜂鸣器响) → SDRAM A12 被拉乱 → framebuffer 错行 → 永久花屏.
 *   旧 震动   PA3 = TLI RGB 屏 蓝色数据线 B5 (AF14). 配成 GPIO → 蓝色 MSB 恒错.
 *
 *   这两脚是 SDRAM / 显示子系统的硬资源, 物理上不能再当普通 GPIO. 因此:
 *     - actuator_init 不再 gpio_mode_set(PG2/PA3) → PG2 留作 A12, PA3 留作 B5,
 *       SDRAM 全 32MB 可用 + 蓝色通道正确.
 *     - 蜂鸣器/震动 API 保留为空操作 (no-op), 让 dispatch / fusion_task 仍能编译.
 *     - 报警闭环改走: LCD 红屏 + LED1/2/3 (PE2/PG3/PH7, 均安全) + CI1302 语音播报
 *       (TTS "已紧急停止" / "报警已确认" / "自检完成" 由 ICS 平台绑定自动播).
 *   硬件上请把蜂鸣器从 PG2、震动马达从 PA3 物理拔下 (否则 A12 高频翻转会让
 *   无源/有源蜂鸣器持续嗡叫).
 ******************************************************************************/

#include "actuator.h"
#include "LED.h"
#include "FreeRTOS.h"
#include "task.h"

/* ---------- init ---------- */
void actuator_init(void)
{
    /* 蜂鸣器 (旧 PG2=SDRAM A12) 与 震动 (旧 PA3=RGB B5) 已移除 — 不再配置任何
     * GPIO, 保持这两脚的 EXMC/TLI alternate-function 完整. 报警仅用 LED + 语音. */

    /* Onboard risk LEDs (LED1=PE2 / LED2=PG3 / LED3=PH7 — 均不撞 SDRAM/RGB).
     * LED_Init() 幂等, task_init 可能已调过一次, 再调一次无副作用. */
    LED_Init();

    /* Boot state: NORMAL (LED1 on, others off). */
    actuator_set_risk(RISK_NORMAL);
}

/* ---------- discrete control ----------
 * 蜂鸣器 / 震动已物理移除, 这两个函数保留为空操作 (no-op), 仅为兼容旧调用点
 * (actuator_set_risk / actuator_buzzer_pulse / dispatch). 不触碰任何引脚. */
void actuator_buzzer_set(uint8_t on)
{
    (void)on;   /* no-op: buzzer removed (PG2 = SDRAM A12) */
}

void actuator_vib_set(uint8_t on)
{
    (void)on;   /* no-op: vibration removed (PA3 = RGB LCD B5) */
}

void actuator_led_set(uint8_t idx, uint8_t on)
{
    switch (idx) {
        case 0U: if (on) LED1_ON(); else LED1_OFF(); break;
        case 1U: if (on) LED2_ON(); else LED2_OFF(); break;
        case 2U: if (on) LED3_ON(); else LED3_OFF(); break;
        default: break;
    }
}

/* ---------- composite ---------- */
void actuator_set_risk(lab_risk_level_t lvl)
{
    /* LED-only risk indication. Audible alarm = CI1302 TTS (voice_dispatch /
     * fusion_task 主动 ci1302_play). */
    LED1_OFF();
    LED2_OFF();
    LED3_OFF();

    switch (lvl) {
        case RISK_NORMAL:
            LED1_ON();          /* heartbeat: system alive, all clear */
            break;

        case RISK_WARNING:
            LED2_ON();          /* yellow-channel indicator only */
            break;

        case RISK_ANOMALY:
            LED2_ON();
            LED3_ON();
            break;

        case RISK_SEVERE:
            LED1_ON();
            LED2_ON();
            LED3_ON();
            /* relay_on() called by fusion_task when smoke detected */
            break;

        default:
            LED1_ON();
            break;
    }
}

void actuator_buzzer_pulse(uint16_t ms)
{
    (void)ms;   /* no-op: buzzer removed (PG2 = SDRAM A12) */
}

/* ---------- self-test ---------- */
void actuator_self_test(void)
{
    /* LED sweep only — 蜂鸣器/震动已移除, 不再触碰 PG2/PA3. ~750ms blocking. */
    const TickType_t pulse = pdMS_TO_TICKS(250U);

    LED1_ON(); vTaskDelay(pulse); LED1_OFF();
    LED2_ON(); vTaskDelay(pulse); LED2_OFF();
    LED3_ON(); vTaskDelay(pulse); LED3_OFF();

    /* Leave system in NORMAL state. */
    actuator_set_risk(RISK_NORMAL);
}

/****************************End*****************************/
