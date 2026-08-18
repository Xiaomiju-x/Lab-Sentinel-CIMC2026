/******************************************************************************
 * feature_build.h  —  assemble AI model input vectors at runtime
 *
 * Bridges real sensors (SHT30 room T/RH, MQ-135 gas, ADXL345 vibration) and
 * the furnace simulator (temperature group + stage + atmosphere + vision proxy)
 * into the exact feature layouts the models were trained on:
 *   - AI-2 : 32-D  (synth_data.py layout)
 *   - AI-3 : 64x8 sliding window of the temperature group
 *   - AI-4 : 16-D  (synth_data_ai4.py layout)
 ******************************************************************************/
#ifndef FEATURE_BUILD_H
#define FEATURE_BUILD_H

#include <stdint.h>
#include "furnace_sim.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int16_t  temp_c_q8;     /* SHT30 room temperature, Q8 fixed-point degC */
    uint16_t hum_q8;        /* SHT30 humidity, Q8 fixed-point %RH          */
    uint16_t mq135_adc;     /* MQ-135 raw 12-bit ADC                        */
    uint16_t vib_rms_mg;    /* ADXL345 RMS magnitude in milli-g             */
} fb_sensors_t;

void fb_reset(void);

/* Build the 32-D AI-2 feature vector (raw, pre-normalisation). */
void fb_build_ai2(const furnace_out_t *f, const fb_sensors_t *s, float feat32[32]);

/* Push the latest 8-D temperature-feature row into the AI-3 window ring. */
void fb_push_ai3(const float temp_feat8[8]);

/* Return a 64*8 row-major copy of the window, ordered oldest..newest. */
const float *fb_ai3_window(void);

/* Build the 16-D AI-4 fusion vector. */
void fb_build_ai4(const float ai1_probs4[4], float ai2_ratio, const float ai2_resid3[3],
                  const float ai3_probs5[5], float progress, float feat16[16]);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_BUILD_H */
