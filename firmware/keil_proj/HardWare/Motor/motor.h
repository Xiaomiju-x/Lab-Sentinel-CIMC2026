/******************************************************************************
 * motor.h
 *
 * L298N H-bridge channel B driver (single DC motor).
 *
 *   IN3 → PG11   (MCU output, push-pull)
 *   IN4 → PG6    (MCU output, push-pull)
 *   ENB → 短接到 5V (L298N 板载跳线, 全速; 若要 PWM 调速需另接 TIMER 引脚)
 *   VCC → 12V  / GND → GND  (L298N 电源单独走)
 *
 * L298N IN3/IN4 真值表 (channel B):
 *   IN3=0 IN4=0  : 自由停 (free spin / coast)
 *   IN3=1 IN4=0  : 正转 (OUT3=HIGH, OUT4=LOW)
 *   IN3=0 IN4=1  : 反转 (OUT3=LOW,  OUT4=HIGH)
 *   IN3=1 IN4=1  : 制动 (双端短路)
 *
 * 2026-05-28 per 模块(2).docx 新接线. 用于 demo 烧结炉风扇/搅拌器等 actuator.
 ******************************************************************************/

#ifndef __MOTOR_H__
#define __MOTOR_H__

#include "HeaderFiles.h"

typedef enum {
    MOTOR_STOP    = 0,   /* IN3=0 IN4=0  自由停 (coast)        */
    MOTOR_FORWARD = 1,   /* IN3=1 IN4=0  正转                   */
    MOTOR_REVERSE = 2,   /* IN3=0 IN4=1  反转                   */
    MOTOR_BRAKE   = 3    /* IN3=1 IN4=1  制动 (双端短路)        */
} motor_dir_t;

void        motor_init(void);
void        motor_set(motor_dir_t dir);
motor_dir_t motor_state(void);

/* Software PWM speed control (ENB jumpered to 5V → chop the active IN pin).
 * motor_set_speed: 0..100 % duty (quantised); motor_pwm_tick: call from a
 * steady periodic loop (sensor_task @200 Hz → 50 Hz PWM carrier). */
void        motor_set_speed(uint8_t pct);
void        motor_pwm_tick(void);

/* 短时定时驱动 — 阻塞调用 (vTaskDelay), 结束后自动 STOP. 必须在任务上下文调用. */
void        motor_pulse(motor_dir_t dir, uint16_t ms);

#endif /* __MOTOR_H__ */
