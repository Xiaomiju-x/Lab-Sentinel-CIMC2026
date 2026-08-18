/******************************************************************************
 * sht30.h
 *
 * Sensirion SHT30/SHT31 temperature + humidity over the shared sensor I²C
 * (PB10/PB11). 7-bit address 0x44 (ADDR pin tied to GND).
 *
 * Outputs Q8 fixed-point: temp_c_q8 = degC * 256, humidity_q8 = %RH * 256.
 *
 * Datasheet ref: Sensirion SHT3x rev 6 (Dec 2019).
 ******************************************************************************/

#ifndef __SHT30_H__
#define __SHT30_H__

#include "HeaderFiles.h"

void    sht30_init(void);
uint8_t sht30_read(int16_t *temp_c_q8, uint16_t *humidity_q8);

#endif /* __SHT30_H__ */
