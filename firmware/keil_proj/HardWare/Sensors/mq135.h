/******************************************************************************
 * mq135.h
 *
 * MQ-135 air quality (NH3 / NOx / CO2 / smoke) — analog AOUT pin into
 * GD32H759 ADC0 channel 12 (PC2).
 *
 * Returns the raw 12-bit ADC value 0..4095. Higher = more contaminants.
 * Calibration to ppm requires a known-clean baseline; we keep the raw value
 * and let env_task / fusion_task interpret deltas.
 *
 * IMPORTANT: MQ-135 needs ≥ 24h pre-heat after first power-on for stable
 * baseline. First-day readings will drift downwards from ~3500 → ~600.
 ******************************************************************************/

#ifndef __MQ135_H__
#define __MQ135_H__

#include "HeaderFiles.h"

void     mq135_init(void);
uint16_t mq135_read_adc(void);   /* 0..4095 */

#endif /* __MQ135_H__ */
