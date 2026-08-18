/******************************************************************************
 * sensors_i2c.h
 *
 * Software I²C bus shared by SHT30 + ADXL345 (and any future I²C sensor).
 *   SCL → PB10
 *   SDA → PB11   (open-drain with internal pull-up; add 4.7kΩ external if
 *                 the bus has long traces or 3+ devices)
 *
 * Why software-bang and not hardware I2C: GT911 already runs soft I²C on
 * PD12/PD13. Keeping all sensor I²C on a second soft bus avoids the GT911
 * vs sensor address-collision risk and simplifies bringup. Speed is bounded
 * by `_i2c_delay_us()` ≈ 10 µs/edge → ~50 kHz, plenty for SHT30 1 Hz reads
 * and ADXL345 200 Hz reads.
 ******************************************************************************/

#ifndef __SENSORS_I2C_H__
#define __SENSORS_I2C_H__

#include "HeaderFiles.h"

/* Initialise PB10/PB11 GPIO. Must be called once before any sensor driver. */
void sensors_i2c_init(void);

/* Write to the device at addr_w (8-bit including R/W=0):
 *   START | addr_w | reg | buf[0..len-1] | STOP
 * Returns 0 on success, 1 on NACK / timeout. */
uint8_t sensors_i2c_write(uint8_t addr_w, uint8_t reg, const uint8_t *buf, uint16_t len);

/* Read from the device:
 *   START | addr_w | reg | RESTART | addr_r | buf[0..len-1] | STOP
 * Returns 0 on success, 1 on NACK / timeout. */
uint8_t sensors_i2c_read(uint8_t addr_w, uint8_t addr_r, uint8_t reg,
                         uint8_t *buf, uint16_t len);

/* SHT30 needs to send a 16-bit command then read straight from the device
 * (no register byte). These two helpers expose that style. */
uint8_t sensors_i2c_write_cmd16(uint8_t addr_w, uint16_t cmd);
uint8_t sensors_i2c_read_n(uint8_t addr_r, uint8_t *buf, uint16_t len);

/* -------------------------------------------------------------------------- */
/* ESP32-C3 IIoT bridge — shares this same I2C2 bus (PH7/PH8, "模块(2).docx").  */
/* The ESP32-C3 is an I2C SLAVE at 0x42: the GD32 (master) writes a telemetry  */
/* block to REG_TELEM and reads the link status + the last R1 diagnosis the    */
/* bridge fetched from the XRD AI brain over WiFi. All transactions go through  */
/* the existing bus-mutexed primitives, so they coexist with SHT30/ADXL345.    */
/* -------------------------------------------------------------------------- */
#define ESP32_I2C_ADDR   0x42U
#define ESP32_STATUS_LEN 5U      /* [link, rssi_mag, diag_len, up_lo, up_hi]   */
#define ESP32_DIAG_MAX   63U     /* R1 diagnosis string bytes (NUL-terminated) */

/* Push a telemetry block (risk/temp/batch/cpk/seg/fault...) to the bridge.
 * Returns 0 if the ESP32 acked, 1 on NACK/timeout (bridge absent/asleep). */
uint8_t esp32_push_telemetry(const uint8_t *buf, uint16_t len);

/* Read the 5-byte status block. st must hold >= ESP32_STATUS_LEN bytes.
 * Returns 0 on success, 1 on NACK/timeout. */
uint8_t esp32_read_status(uint8_t *st);

/* Read up to maxlen bytes of the last R1 diagnosis string. Returns 0 ok. */
uint8_t esp32_read_diag(uint8_t *buf, uint16_t maxlen);

#endif /* __SENSORS_I2C_H__ */
