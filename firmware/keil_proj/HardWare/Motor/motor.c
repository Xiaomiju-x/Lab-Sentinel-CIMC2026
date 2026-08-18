/******************************************************************************
 * motor.c — L298N H-bridge channel B driver (IN3=PG11, IN4=PG6).
 *
 * 2026-05-28 per 模块(2).docx 新接线.
 *
 * PG11 / PG6 在新 RGB LCD 时代是空闲的 (RGB LCD GPIOG 只用 PG10=G3 / PG12=B4,
 * SDRAM GPIOG 只用 PG0/1/2/4/5/8/15). 旧 8080 ST7796 用过这两个引脚 (D13/D15)
 * 但 ST7796 已废弃.
 ******************************************************************************/

#include "motor.h"
#include "FreeRTOS.h"
#include "task.h"

#define MOTOR_IN3_PORT   GPIOG
#define MOTOR_IN3_PIN    GPIO_PIN_11
#define MOTOR_IN4_PORT   GPIOG
#define MOTOR_IN4_PIN    GPIO_PIN_6

static motor_dir_t s_state = MOTOR_STOP;

/* -------------------------------------------------------------------------- *
 * Software PWM speed control. ENB is hard-jumpered to 5V (no hardware PWM pin),
 * so speed is set by chopping the active IN pin instead: for FORWARD, IN4 stays
 * 0 and IN3 is toggled — IN3=1 drives, IN3=0 coasts (fast decay) — so the duty
 * sets the average drive. motor_pwm_tick() must be called periodically from a
 * steady loop (sensor_task @200 Hz → 200/STEPS Hz carrier). A short full-power
 * kickstart on each direction change beats the small motor's static friction so
 * it reliably starts even at a low slow-test duty. */
#define MOTOR_PWM_STEPS   4              /* 200 Hz / 4 = 50 Hz carrier; 25% steps  */
#define MOTOR_KICK_TICKS  60u            /* ~300 ms full-power kickstart @200 Hz  */

static volatile uint8_t s_pwm_duty  = MOTOR_PWM_STEPS;  /* 0..STEPS (full default)*/
static volatile uint8_t s_pwm_phase = 0u;
static volatile uint8_t s_kick      = 0u;

void motor_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOG);

    /* IN3, IN4 push-pull output, idle LOW (停). */
    gpio_mode_set(MOTOR_IN3_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, MOTOR_IN3_PIN);
    gpio_output_options_set(MOTOR_IN3_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, MOTOR_IN3_PIN);
    gpio_bit_reset(MOTOR_IN3_PORT, MOTOR_IN3_PIN);

    gpio_mode_set(MOTOR_IN4_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, MOTOR_IN4_PIN);
    gpio_output_options_set(MOTOR_IN4_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, MOTOR_IN4_PIN);
    gpio_bit_reset(MOTOR_IN4_PORT, MOTOR_IN4_PIN);

    s_state = MOTOR_STOP;
}

/* duty 0..100 % → quantised to MOTOR_PWM_STEPS. */
void motor_set_speed(uint8_t pct)
{
    uint8_t d;
    if (pct > 100u) pct = 100u;
    d = (uint8_t)(((uint16_t)pct * MOTOR_PWM_STEPS + 50u) / 100u);
    if (d > MOTOR_PWM_STEPS) d = MOTOR_PWM_STEPS;
    s_pwm_duty = d;
}

/* Call at a steady rate (sensor_task 200 Hz). Chops the active IN pin per duty;
 * the first MOTOR_KICK_TICKS after a spin command run full-power to start. */
void motor_pwm_tick(void)
{
    uint8_t on;
    if (s_state != MOTOR_FORWARD && s_state != MOTOR_REVERSE) return;  /* PWM only spins */
    if (s_kick > 0u) { on = 1u; s_kick--; }            /* full-power kickstart window  */
    else             { on = (uint8_t)(s_pwm_phase < s_pwm_duty); }
    if (s_state == MOTOR_FORWARD) {                    /* IN4=0, chop IN3 */
        if (on) gpio_bit_set(MOTOR_IN3_PORT, MOTOR_IN3_PIN);
        else    gpio_bit_reset(MOTOR_IN3_PORT, MOTOR_IN3_PIN);
    } else {                                           /* REVERSE: IN3=0, chop IN4 */
        if (on) gpio_bit_set(MOTOR_IN4_PORT, MOTOR_IN4_PIN);
        else    gpio_bit_reset(MOTOR_IN4_PORT, MOTOR_IN4_PIN);
    }
    if (++s_pwm_phase >= MOTOR_PWM_STEPS) s_pwm_phase = 0u;
}

void motor_set(motor_dir_t dir)
{
    switch (dir) {
        case MOTOR_FORWARD:
            gpio_bit_set  (MOTOR_IN3_PORT, MOTOR_IN3_PIN);
            gpio_bit_reset(MOTOR_IN4_PORT, MOTOR_IN4_PIN);
            break;
        case MOTOR_REVERSE:
            gpio_bit_reset(MOTOR_IN3_PORT, MOTOR_IN3_PIN);
            gpio_bit_set  (MOTOR_IN4_PORT, MOTOR_IN4_PIN);
            break;
        case MOTOR_BRAKE:
            gpio_bit_set  (MOTOR_IN3_PORT, MOTOR_IN3_PIN);
            gpio_bit_set  (MOTOR_IN4_PORT, MOTOR_IN4_PIN);
            break;
        case MOTOR_STOP:
        default:
            gpio_bit_reset(MOTOR_IN3_PORT, MOTOR_IN3_PIN);
            gpio_bit_reset(MOTOR_IN4_PORT, MOTOR_IN4_PIN);
            dir = MOTOR_STOP;
            break;
    }
    /* On a fresh spin command, arm the kickstart + restart the PWM phase so the
     * motor gets full power briefly before dropping to the (possibly low) duty. */
    if ((dir == MOTOR_FORWARD || dir == MOTOR_REVERSE) && s_state != dir) {
        s_kick = MOTOR_KICK_TICKS;
        s_pwm_phase = 0u;
    }
    s_state = dir;
}

motor_dir_t motor_state(void) { return s_state; }

void motor_pulse(motor_dir_t dir, uint16_t ms)
{
    if (dir == MOTOR_STOP) {
        return;
    }
    motor_set(dir);
    vTaskDelay(pdMS_TO_TICKS((TickType_t)ms));
    motor_set(MOTOR_STOP);
}

/****************************End*****************************/
