/******************************************************************************
 * smoke_sensor.h
 *
 * Smoke / gas sensor with dual output (2026-05-16 pin remap):
 *   AO → PA9   Sensor's "AO" label, wired to MCU as digital GPIO (HIGH = above
 *               on-board comparator threshold). PA9 is USART0_TX by default
 *               but boot_print uses UART4 (PB13/PB5) so USART0 is free.
 *   DO → PC3   Sensor's "DO" label, wired to MCU as analog ADC1_CH13.
 *               Sensor DO swings 0–5 V through a 0.6× voltage divider:
 *               actual_V = measured_V / 0.6  (i.e. × 5/3)
 *               Smoke alarm threshold: actual_V ≥ 3.0 V  (ADC ≥ 2482/4095)
 *
 * smoke_sensor_read() fills a smoke_result_t.  Call once per second from
 * env_task alongside SHT30 and MQ135.
 ******************************************************************************/

#ifndef __SMOKE_SENSOR_H__
#define __SMOKE_SENSOR_H__

#include "HeaderFiles.h"

typedef struct {
    uint8_t  ao_digital;    /* 1 = AO pin HIGH (above threshold on-board pot) */
    uint16_t do_adc_raw;    /* 12-bit ADC reading of DO pin (0..4095) */
    uint16_t do_mv_actual;  /* actual DO voltage in mV (corrected for /0.6 divider) */
    uint8_t  alarm;         /* 1 = smoke detected (either AO high OR DO_V ≥ 3000 mV) */
} smoke_result_t;

void smoke_sensor_init(void);
void smoke_sensor_read(smoke_result_t *out);

#endif /* __SMOKE_SENSOR_H__ */
