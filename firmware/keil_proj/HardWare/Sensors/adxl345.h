/******************************************************************************
 * adxl345.h
 *
 * Analog Devices ADXL345 3-axis accelerometer over the shared sensor I²C
 * (PB10/PB11). 7-bit address 0x53 (SDO/ALT pin tied to GND).
 *
 * Configured for full-resolution mode (4 mg/LSB) at 100 Hz output data rate.
 * sensor_task polls at 200 Hz which oversamples the sensor — fine, just
 * gets the same value twice occasionally.
 *
 * Reference: ADXL345 datasheet rev G.
 ******************************************************************************/

#ifndef __ADXL345_H__
#define __ADXL345_H__

#include "HeaderFiles.h"

uint8_t adxl345_init(void);

/* Read raw 3-axis sample. Each LSB ≈ 4 mg in full-resolution ±16g mode. */
uint8_t adxl345_read_xyz(int16_t *x, int16_t *y, int16_t *z);

/* Read magnitude in milli-g of |a| - 1g (gravity-compensated single sample).
 * Cheap to call — no buffering, no RMS — useful for 200Hz polling and feeding
 * a windowed RMS computed by the caller. */
uint16_t adxl345_read_magnitude_mg(void);

#endif /* __ADXL345_H__ */
