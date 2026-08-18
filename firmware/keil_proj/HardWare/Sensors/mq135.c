/******************************************************************************
 * mq135.c — ADC0 single conversion on PC2 (channel 12).
 *
 * Originally on PA4 / ADC1_CH18; moved to PC2 / ADC0_CH12 because PA4 is now
 * occupied by the OV5640 DCI HSYNC line. Functionally identical (12-bit
 * single-channel software-trigger), only the pin and ADC instance change.
 ******************************************************************************/

#include "mq135.h"

/* GD32H7 ADC sample_time is a direct cycle count in [0, 809]. 480 cycles
 * gives a slow, well-settled sample appropriate for the high source
 * impedance of the MQ-135 board's voltage divider. */
#define MQ135_SAMPLE_CYCLES   480U

void mq135_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOC);
    rcu_periph_clock_enable(RCU_ADC0);

    /* PC2 → analog mode, no pull, no output options. */
    gpio_mode_set(GPIOC, GPIO_MODE_ANALOG, GPIO_PUPD_NONE, GPIO_PIN_2);

    adc_deinit(ADC0);

    /* HCLK / 2 sync → safe at 600 MHz CPU. */
    adc_clock_config(ADC0, ADC_CLK_SYNC_HCLK_DIV2);

    adc_resolution_config(ADC0, ADC_RESOLUTION_12B);
    adc_data_alignment_config(ADC0, ADC_DATAALIGN_RIGHT);

    /* Single channel, regular group, software trigger. */
    adc_channel_length_config(ADC0, ADC_REGULAR_CHANNEL, 1U);
    adc_regular_channel_config(ADC0, 0U, ADC_CHANNEL_12, MQ135_SAMPLE_CYCLES);
    adc_external_trigger_config(ADC0, ADC_REGULAR_CHANNEL, EXTERNAL_TRIGGER_DISABLE);

    adc_enable(ADC0);

    /* Calibration improves linearity, especially on first power-up. */
    adc_calibration_mode_config(ADC0, ADC_CALIBRATION_OFFSET_MISMATCH);
    adc_calibration_number(ADC0, ADC_CALIBRATION_NUM16);
    adc_calibration_enable(ADC0);
}

uint16_t mq135_read_adc(void)
{
    uint32_t timeout = 100000U;

    adc_flag_clear(ADC0, ADC_FLAG_EOC);
    adc_software_trigger_enable(ADC0, ADC_REGULAR_CHANNEL);

    while (adc_flag_get(ADC0, ADC_FLAG_EOC) == RESET) {
        if (--timeout == 0U) return 0U;
    }
    adc_flag_clear(ADC0, ADC_FLAG_EOC);

    return (uint16_t)(adc_regular_data_read(ADC0) & 0x0FFFU);
}

/****************************End*****************************/
