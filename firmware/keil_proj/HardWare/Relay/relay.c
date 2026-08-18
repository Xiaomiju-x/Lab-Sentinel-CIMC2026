/******************************************************************************
 * relay.c — PA1, HIGH = relay ON (load energised).
 *
 * 2026-05-16: moved from PE3 to PA1 (PE3 now reserved for DCMI_PCLK).
 * PA1 is normally ENET0_REF_CLK (RMII AF11) but Ethernet is downgraded to
 * Phase 6 optional (eth_task not spawned) so PA1 is free.
 ******************************************************************************/

#include "relay.h"

#define RELAY_PORT   GPIOA
#define RELAY_PIN    GPIO_PIN_1

static uint8_t s_state = 0U;

void relay_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOA);
    gpio_mode_set(RELAY_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, RELAY_PIN);
    gpio_output_options_set(RELAY_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, RELAY_PIN);
    gpio_bit_reset(RELAY_PORT, RELAY_PIN);   /* relay open by default */
    s_state = 0U;
}

void relay_on(void)
{
    gpio_bit_set(RELAY_PORT, RELAY_PIN);
    s_state = 1U;
}

void relay_off(void)
{
    gpio_bit_reset(RELAY_PORT, RELAY_PIN);
    s_state = 0U;
}

uint8_t relay_state(void) { return s_state; }

/* ---- Heating-plate relay (PTC), PD12, active HIGH. Separate from the PA1 fan.
 * Energised during a sinter run; the safety supervisor cuts it on a real-probe
 * over-temperature (ctrl_task). Default OFF on boot (safe cold start). */
#define HEATER_PORT  GPIOD
#define HEATER_PIN   GPIO_PIN_12

static uint8_t s_heater = 0U;

void heater_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOD);
    gpio_mode_set(HEATER_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, HEATER_PIN);
    gpio_output_options_set(HEATER_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, HEATER_PIN);
    gpio_bit_reset(HEATER_PORT, HEATER_PIN);   /* heater OFF by default (safe) */
    s_heater = 0U;
}

void heater_on(void)  { gpio_bit_set(HEATER_PORT, HEATER_PIN);   s_heater = 1U; }
void heater_off(void) { gpio_bit_reset(HEATER_PORT, HEATER_PIN); s_heater = 0U; }
uint8_t heater_state(void) { return s_heater; }

/****************************End*****************************/
