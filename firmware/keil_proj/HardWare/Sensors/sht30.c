/******************************************************************************
 * sht30.c
 ******************************************************************************/

#include "sht30.h"
#include "sensors_i2c.h"
#include "FreeRTOS.h"
#include "task.h"

/* 7-bit addr 0x44 → 8-bit write 0x88, read 0x89 */
#define SHT30_ADDR_W    0x88U
#define SHT30_ADDR_R    0x89U

/* Single-shot, high repeatability, clock stretching disabled.  Datasheet
 * 4.3 table 8: 0x2400. Conversion takes ≤15 ms. */
#define SHT30_CMD_MEAS_HIGH    0x2400U
#define SHT30_CMD_SOFT_RESET   0x30A2U

/* Sensirion CRC-8 over 2-byte words (poly 0x31, init 0xFF). */
static uint8_t _crc8(uint8_t msb, uint8_t lsb)
{
    uint8_t crc = 0xFFU;
    uint8_t data[2] = { msb, lsb };
    uint8_t i, b;
    for (b = 0U; b < 2U; b++) {
        crc ^= data[b];
        for (i = 0U; i < 8U; i++) {
            crc = (crc & 0x80U) ? (uint8_t)((crc << 1) ^ 0x31U)
                                : (uint8_t)(crc << 1);
        }
    }
    return crc;
}

void sht30_init(void)
{
    /* Soft-reset; ignore failure (sensor may take a few ms after power-on). */
    (void)sensors_i2c_write_cmd16(SHT30_ADDR_W, SHT30_CMD_SOFT_RESET);
    vTaskDelay(pdMS_TO_TICKS(10U));
}

uint8_t sht30_read(int16_t *temp_c_q8, uint16_t *humidity_q8)
{
    uint8_t raw[6];

    /* Trigger single-shot high-repeatability measurement. */
    if (sensors_i2c_write_cmd16(SHT30_ADDR_W, SHT30_CMD_MEAS_HIGH) != 0U) {
        return 1U;
    }

    /* Wait for conversion (≤15 ms per datasheet). */
    vTaskDelay(pdMS_TO_TICKS(20U));

    /* Read 6 bytes: T_msb, T_lsb, T_crc, RH_msb, RH_lsb, RH_crc. */
    if (sensors_i2c_read_n(SHT30_ADDR_R, raw, 6U) != 0U) {
        return 1U;
    }

    /* CRC validation — bail on mismatch so we don't post junk. */
    if (_crc8(raw[0], raw[1]) != raw[2]) return 2U;
    if (_crc8(raw[3], raw[4]) != raw[5]) return 2U;

    uint16_t t_raw  = ((uint16_t)raw[0] << 8) | raw[1];
    uint16_t rh_raw = ((uint16_t)raw[3] << 8) | raw[4];

    /* Datasheet 4.13:
     *   T_C = -45 + 175 * t_raw / 65535
     *   RH  = 100 * rh_raw / 65535
     * Express both as Q8 fixed-point (×256) using only 32-bit math. */
    if (temp_c_q8 != NULL) {
        /* T_C * 256  =  -45*256 + (175*256 * t_raw) / 65535
         *           =  -11520 + 44800 * t_raw / 65535          */
        int32_t t_q8 = (int32_t)((44800UL * (uint32_t)t_raw) / 65535UL) - 11520;
        if      (t_q8 >  32767) t_q8 =  32767;
        else if (t_q8 < -32768) t_q8 = -32768;
        *temp_c_q8 = (int16_t)t_q8;
    }
    if (humidity_q8 != NULL) {
        /* RH * 256 = 25600 * rh_raw / 65535 */
        uint32_t rh_q8 = (25600UL * (uint32_t)rh_raw) / 65535UL;
        if (rh_q8 > 65535U) rh_q8 = 65535U;
        *humidity_q8 = (uint16_t)rh_q8;
    }
    return 0U;
}

/****************************End*****************************/
