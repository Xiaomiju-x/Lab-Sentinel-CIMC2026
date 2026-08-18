/******************************************************************************
 * adxl345.c
 ******************************************************************************/

#include "adxl345.h"
#include "sensors_i2c.h"

/* 7-bit addr 0x53 → 8-bit write 0xA6, read 0xA7 */
#define ADXL_ADDR_W     0xA6U
#define ADXL_ADDR_R     0xA7U

/* Register map (datasheet table 19) */
#define ADXL_DEVID          0x00U
#define ADXL_BW_RATE        0x2CU
#define ADXL_POWER_CTL      0x2DU
#define ADXL_DATA_FORMAT    0x31U
#define ADXL_DATAX0         0x32U

/* DEVID always 0xE5 — used as a probe. */
#define ADXL_DEVID_VAL      0xE5U

/* BW_RATE: 0x0A = 100 Hz output, normal-power
 * POWER_CTL: 0x08 = measurement mode
 * DATA_FORMAT: 0x0B = full-res, ±16g, INT active high, right-justified */
#define ADXL_BW_RATE_VAL    0x0AU
#define ADXL_POWER_CTL_VAL  0x08U
#define ADXL_DATA_FMT_VAL   0x0BU

/* Square-root by Newton's method (uint32 → uint16). Avoids sqrt() / float on
 * the M7 hot path. ~6 iters converges. */
static uint16_t _isqrt(uint32_t n)
{
    if (n == 0U) return 0U;
    uint32_t x = n;
    uint32_t y = (x + 1U) >> 1;
    uint8_t  i;
    for (i = 0U; i < 16U && y < x; i++) {
        x = y;
        y = (x + n / x) >> 1;
    }
    return (uint16_t)((x > 65535U) ? 65535U : x);
}

uint8_t adxl345_init(void)
{
    uint8_t id = 0U;
    uint8_t v;

    /* Probe DEVID = 0xE5; bail if mismatch (wiring problem / wrong addr). */
    if (sensors_i2c_read(ADXL_ADDR_W, ADXL_ADDR_R, ADXL_DEVID, &id, 1U) != 0U) {
        return 1U;
    }
    if (id != ADXL_DEVID_VAL) return 2U;

    /* Configure data rate, format, then enable measurement (datasheet 7.5
     * recommends DATA_FORMAT before POWER_CTL). */
    v = ADXL_BW_RATE_VAL;
    if (sensors_i2c_write(ADXL_ADDR_W, ADXL_BW_RATE, &v, 1U) != 0U) return 3U;

    v = ADXL_DATA_FMT_VAL;
    if (sensors_i2c_write(ADXL_ADDR_W, ADXL_DATA_FORMAT, &v, 1U) != 0U) return 3U;

    v = ADXL_POWER_CTL_VAL;
    if (sensors_i2c_write(ADXL_ADDR_W, ADXL_POWER_CTL, &v, 1U) != 0U) return 3U;

    return 0U;
}

uint8_t adxl345_read_xyz(int16_t *x, int16_t *y, int16_t *z)
{
    uint8_t raw[6];
    if (sensors_i2c_read(ADXL_ADDR_W, ADXL_ADDR_R, ADXL_DATAX0, raw, 6U) != 0U) {
        return 1U;
    }
    /* Little-endian per axis */
    if (x) *x = (int16_t)((uint16_t)raw[0] | ((uint16_t)raw[1] << 8));
    if (y) *y = (int16_t)((uint16_t)raw[2] | ((uint16_t)raw[3] << 8));
    if (z) *z = (int16_t)((uint16_t)raw[4] | ((uint16_t)raw[5] << 8));
    return 0U;
}

uint16_t adxl345_read_magnitude_mg(void)
{
    int16_t x, y, z;
    if (adxl345_read_xyz(&x, &y, &z) != 0U) return 0U;

    /* Full-resolution mode: 4 mg / LSB. |a|_mg = 4 * sqrt(x²+y²+z²) */
    int32_t sx = x, sy = y, sz = z;
    uint32_t sq = (uint32_t)(sx * sx + sy * sy + sz * sz);
    uint32_t mag_mg = 4U * (uint32_t)_isqrt(sq);

    /* Subtract 1g (= 1000 mg) so a stationary sensor reads ~0 mg.
     * Negative results clamp to 0 (we report |Δa| not signed offset). */
    if (mag_mg < 1000U) return 0U;
    uint32_t delta = mag_mg - 1000U;
    return (uint16_t)((delta > 65535U) ? 65535U : delta);
}

/****************************End*****************************/
