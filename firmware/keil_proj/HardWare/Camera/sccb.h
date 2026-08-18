/******************************************************************************
 * sccb.h
 *
 * Software-bang SCCB (Serial Camera Control Bus) for OV5640.
 * SCCB is essentially I²C with a quirk: NACK on the last read byte is
 * mandatory and OV5640 uses 16-bit register addresses (0x300A etc.).
 *
 *   SCL → PA5    (open-drain, idle high)  [moved from PB4: PB4 is smoke sensor AO]
 *   SDA → PB7    (open-drain, idle high)
 *   ~50 kHz, plenty for one-shot init + occasional control writes.
 *
 * NOTE on bus separation: this is its OWN soft-I2C, distinct from
 *   - Touch (PD12/PD13)
 *   - Sensors SHT30/ADXL345 (PB10/PB11)
 * That is intentional — OV5640's 7-bit address 0x3C is unique within SCCB,
 * and isolating the camera bus avoids long-trace coupling with the FFC.
 ******************************************************************************/

#ifndef __SCCB_H__
#define __SCCB_H__

#include "HeaderFiles.h"

void    sccb_init(void);

/* Write 1 byte at OV5640's 16-bit register `reg`. Returns 0 on success. */
uint8_t sccb_write_reg(uint16_t reg, uint8_t val);

/* Read 1 byte from OV5640's 16-bit register `reg`. Returns 0 on success. */
uint8_t sccb_read_reg(uint16_t reg, uint8_t *val);

#endif /* __SCCB_H__ */
