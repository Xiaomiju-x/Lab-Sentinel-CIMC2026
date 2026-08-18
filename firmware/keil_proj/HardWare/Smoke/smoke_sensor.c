/******************************************************************************
 * smoke_sensor.c  —  MQ-135 gas sensor (= 模块(2).docx "烟雾气敏传感器")
 *
 * This IS the MQ-135 (one physical sensor). Per 模块(2).docx it wires AO=PG13 /
 * DO=PC3 — NOT the old PC2/ADC0 path that the legacy mq135.c used. We read it
 * here through its DO on PC3 and retire mq135.c (PC2 is now the MAX31855 SO).
 *
 * Pin map:
 *   DO on PC3: ADC1_CH13 — analog gas level (the channel actually used).
 *               Hardware voltage divider: 5 V → 3 V (ratio 0.6).
 *               actual_mV = adc_raw * 3300 / 4095 / 0.6 = adc_raw * 5500 / 4095
 *
 * 2026-06-03: AO digital read DROPPED. The doc lists AO on PG13, but the firmware
 * previously read it on PA9 — and PA9 is the RGB-LCD R5 colour line (rgb_lcd.h),
 * so reconfiguring PA9 to GPIO input here corrupted the display's red channel.
 * The demo only needs the DO analog threshold, so we no longer touch any AO pin
 * (no rewiring; the LCD keeps PA9). out->ao_digital is forced 0.
 ******************************************************************************/

#include "smoke_sensor.h"

#define DO_PORT    GPIOC
#define DO_PIN     GPIO_PIN_3

/* DO alarm threshold: actual voltage ≥ 3000 mV = smoke above sensor preset */
#define DO_ALARM_MV   3000U

/* ADC1 channel for PC3 */
#define DO_ADC_CH      ADC_CHANNEL_13
#define DO_SAMPLE_TIME 480U   /* cycles — same as mq135.c for consistency */

void smoke_sensor_init(void)
{
    /* NOTE: no GPIOA / AO config — PA9 stays the RGB-LCD R5 line (see header). */
    rcu_periph_clock_enable(RCU_GPIOC);   /* DO: PC3 */
    /* ADC0 + ADC1 share one sync-clock register: adc_clock_config(ADC1,...) writes
     * ADC_SYNCCTL(ADC0), so RCU_ADC0 MUST be on or that write is dropped and the
     * ADC sync clock never starts -> adc_calibration_enable(ADC1) spins forever.
     * (Used to be guaranteed by mq135_init's RCU_ADC0 enable; mq135 is now retired,
     * so smoke_sensor_init enables it itself — keeps PC2 free for the MAX31855.) */
    rcu_periph_clock_enable(RCU_ADC0);    /* sync-group master clock (shared SYNCCTL) */
    rcu_periph_clock_enable(RCU_ADC1);

    gpio_mode_set(DO_PORT, GPIO_MODE_ANALOG, GPIO_PUPD_NONE, DO_PIN);

    adc_deinit(ADC1);
    adc_clock_config(ADC1, ADC_CLK_SYNC_HCLK_DIV2);
    adc_resolution_config(ADC1, ADC_RESOLUTION_12B);
    adc_data_alignment_config(ADC1, ADC_DATAALIGN_RIGHT);

    adc_channel_length_config(ADC1, ADC_REGULAR_CHANNEL, 1U);
    adc_regular_channel_config(ADC1, 0U, DO_ADC_CH, DO_SAMPLE_TIME);
    adc_external_trigger_config(ADC1, ADC_REGULAR_CHANNEL, EXTERNAL_TRIGGER_DISABLE);

    adc_enable(ADC1);

    adc_calibration_mode_config(ADC1, ADC_CALIBRATION_OFFSET_MISMATCH);
    adc_calibration_number(ADC1, ADC_CALIBRATION_NUM16);
    adc_calibration_enable(ADC1);
}

void smoke_sensor_read(smoke_result_t *out)
{
    uint32_t timeout;
    uint16_t raw;

    if (out == NULL) return;

    /* AO digital read dropped (PA9 = RGB-LCD R5). Alarm is DO-analog only. */
    out->ao_digital = 0U;

    /* DO: ADC conversion */
    adc_flag_clear(ADC1, ADC_FLAG_EOC);
    adc_software_trigger_enable(ADC1, ADC_REGULAR_CHANNEL);

    timeout = 100000U;
    while (adc_flag_get(ADC1, ADC_FLAG_EOC) == RESET) {
        if (--timeout == 0U) { out->do_adc_raw = 0U; out->do_mv_actual = 0U; goto done; }
    }
    adc_flag_clear(ADC1, ADC_FLAG_EOC);

    raw = (uint16_t)(adc_regular_data_read(ADC1) & 0x0FFFU);
    out->do_adc_raw = raw;
    /* actual_mV = raw * 3300 / 4095 / 0.6 = raw * 5500 / 4095 */
    out->do_mv_actual = (uint16_t)((uint32_t)raw * 5500U / 4095U);

done:
    out->alarm = (out->ao_digital != 0U || out->do_mv_actual >= DO_ALARM_MV) ? 1U : 0U;
}

/****************************End*****************************/
