/******************************************************************************
 * gt911.c
 *
 * GT911 capacitive touch driver — soft I2C on PD5/PD7, INT on PH15, RST on PH13
 * (CIMC official board + FPC RGB 800x480 panel, 2026-05-30).
 *
 * The RGB panel has no LCD controller, so GT911 owns its own RST (PH13) — this
 * pin was freed when CI1302 moved to PC10/PC11. gt911_init() drives the RST
 * pulse itself (INT held to a defined level to latch the I2C address), then
 * auto-probes the address (0x5D first, 0x14 fallback).
 *
 * PD5/PD7/PH13/PH15 are clear of SDRAM, RGB-TLI and OV5640 lines, so touch and
 * the camera coexist (the old PB8/camera-D6 conflict was the retired 8080 panel).
 *
 * Register layout (16-bit address, big-endian register number):
 *   0x814E  GT_STATUS  (touch count in bits[3:0], clear by writing 0)
 *   0x8150  PT1_XL     first point: XL, XH, YL, YH (little-endian), Size_L, Size_H
 *   0x8156  PT2_XL     second point (same layout, 6 bytes each)
 *   ...
 ******************************************************************************/

#include "gt911.h"

/* ---------- soft I2C pin map (official RGB board) ---------- */
#define SCL_PORT    GPIOD
#define SCL_PIN     GPIO_PIN_5
#define SDA_PORT    GPIOD
#define SDA_PIN     GPIO_PIN_7

/* INT on PH15 — GT911 touch-interrupt out (also address-select at reset). */
#define INT_PORT    GPIOH
#define INT_PIN     GPIO_PIN_15

/* RST on PH13 — GT911 reset (own pin; RGB panel has none to share). */
#define RST_PORT    GPIOH
#define RST_PIN     GPIO_PIN_13

/* ---------- I2C addresses (auto-probed) ---------- */
#define GT911_ADDR_W_5D    0xBAU   /* primary   */
#define GT911_ADDR_R_5D    0xBBU
#define GT911_ADDR_W_14    0x28U   /* fallback  */
#define GT911_ADDR_R_14    0x29U

static uint8_t s_addr_w = GT911_ADDR_W_5D;
static uint8_t s_addr_r = GT911_ADDR_R_5D;

/* ---------- GT911 registers ---------- */
#define GT_PRODUCT_ID   0x8140U
#define GT_STATUS       0x814EU
#define GT_PT1_XL       0x8150U

/* ---------- soft I2C primitives ---------- */

/* Half-bit delay. The GT911 soft-I2C must stay <=400 kHz; 10 NOPs @600 MHz was
 * multi-MHz (the old value was for the never-connected 8080-panel touch and was
 * never validated on hardware) and the GT911 NACKed everything. ~200 volatile
 * iterations ≈ 2 us half-period ≈ ~250 kHz — comfortably within spec. */
static void _i2c_delay(void)
{
    volatile uint32_t i = 200U;
    while (i--) { __NOP(); }
}

static void _sda_out(void)
{
    gpio_mode_set(SDA_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_PULLUP, SDA_PIN);
    gpio_output_options_set(SDA_PORT, GPIO_OTYPE_OD, GPIO_OSPEED_12MHZ, SDA_PIN);
}

static void _sda_in(void)
{
    gpio_mode_set(SDA_PORT, GPIO_MODE_INPUT, GPIO_PUPD_PULLUP, SDA_PIN);
}

static void _scl(uint8_t v)
{
    if (v) gpio_bit_set(SCL_PORT, SCL_PIN);
    else   gpio_bit_reset(SCL_PORT, SCL_PIN);
    _i2c_delay();
}

static void _sda(uint8_t v)
{
    if (v) gpio_bit_set(SDA_PORT, SDA_PIN);
    else   gpio_bit_reset(SDA_PORT, SDA_PIN);
    _i2c_delay();
}

static uint8_t _sda_read(void)
{
    return (gpio_input_bit_get(SDA_PORT, SDA_PIN) != RESET) ? 1U : 0U;
}

static void _i2c_start(void)
{
    _sda_out();
    _sda(1); _scl(1);
    _sda(0); _scl(0);
}

static void _i2c_stop(void)
{
    _sda_out();
    _sda(0); _scl(1);
    _sda(1);
}

static uint8_t _i2c_wait_ack(void)
{
    uint8_t t = 50U;
    _sda_in();
    _scl(1);
    while (_sda_read()) {
        if (--t == 0U) { _i2c_stop(); return 1U; } /* timeout = NAK */
    }
    _scl(0);
    return 0U; /* ACK */
}

static void _i2c_ack(void)
{
    _sda_out(); _sda(0); _scl(1); _scl(0); _sda(1);
}

static void _i2c_nack(void)
{
    _sda_out(); _sda(1); _scl(1); _scl(0);
}

static void _i2c_write_byte(uint8_t dat)
{
    uint8_t i;
    _sda_out();
    for (i = 0U; i < 8U; i++) {
        _sda((dat & 0x80U) ? 1U : 0U);
        dat <<= 1;
        _scl(1); _scl(0);
    }
}

static uint8_t _i2c_read_byte(uint8_t ack)
{
    uint8_t i, dat = 0U;
    _sda_in();
    for (i = 0U; i < 8U; i++) {
        _scl(1);
        dat <<= 1;
        if (_sda_read()) dat |= 0x01U;
        _scl(0);
    }
    if (ack) _i2c_ack();
    else     _i2c_nack();
    return dat;
}

/* ---------- GT911 register read/write (uses s_addr_w / s_addr_r) ---------- */

static uint8_t _gt911_write_reg(uint16_t reg, const uint8_t *buf, uint8_t len)
{
    uint8_t i;
    _i2c_start();
    _i2c_write_byte(s_addr_w);
    if (_i2c_wait_ack()) return 1U;
    _i2c_write_byte((uint8_t)(reg >> 8));
    if (_i2c_wait_ack()) return 1U;
    _i2c_write_byte((uint8_t)(reg));
    if (_i2c_wait_ack()) return 1U;
    for (i = 0U; i < len; i++) {
        _i2c_write_byte(buf[i]);
        if (_i2c_wait_ack()) return 1U;
    }
    _i2c_stop();
    return 0U;
}

static uint8_t _gt911_read_reg(uint16_t reg, uint8_t *buf, uint8_t len)
{
    uint8_t i;
    _i2c_start();
    _i2c_write_byte(s_addr_w);
    if (_i2c_wait_ack()) return 1U;
    _i2c_write_byte((uint8_t)(reg >> 8));
    if (_i2c_wait_ack()) return 1U;
    _i2c_write_byte((uint8_t)(reg));
    if (_i2c_wait_ack()) return 1U;
    _i2c_start();
    _i2c_write_byte(s_addr_r);
    if (_i2c_wait_ack()) return 1U;
    for (i = 0U; i < len; i++) {
        buf[i] = _i2c_read_byte((i < (len - 1U)) ? 1U : 0U);
    }
    _i2c_stop();
    return 0U;
}

/* Raw product-ID bytes read at each candidate address, kept for the boot
 * diagnostic ([touch] PID@5D / @14): all-00 = NACK/no answer, all-FF = bus
 * stuck high (no pull-up / not present), "911" = GT911 alive at that address. */
static uint8_t s_id5d[4] = {0};
static uint8_t s_id14[4] = {0};

/* A real GT911 product ID is ASCII "911\0"; accept any printable-digit lead. */
static uint8_t _id_valid(const uint8_t *id)
{
    return (id[0] == '9' && id[1] == '1' && id[2] == '1') ? 1U : 0U;
}

void gt911_debug_ids(uint8_t out5d[4], uint8_t out14[4])
{
    uint8_t i;
    for (i = 0U; i < 4U; i++) { out5d[i] = s_id5d[i]; out14[i] = s_id14[i]; }
}

/* ---------- public API ---------- */

uint8_t gt911_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOD);   /* SCL/SDA: PD5/PD7 */
    rcu_periph_clock_enable(RCU_GPIOH);   /* INT/RST: PH15/PH13 */

    /* Bring the I2C lines up FIRST and leave them idle-high (matches the proven
     * official bsp_touch_cap.c ordering — lines high before RST is released). */
    gpio_mode_set(SCL_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_PULLUP, SCL_PIN);
    gpio_output_options_set(SCL_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_12MHZ, SCL_PIN);
    _sda_out();
    _scl(1); _sda(1);

    /* ---- reset pulse ----
     * The RGB panel has no controller, so GT911 owns RST (PH13). Hold INT (PH15)
     * low across the RST rising edge (per the reference), then release INT to
     * input. The actual I2C address is decided below by reading the product ID
     * at both candidates, so the exact INT level isn't relied upon. */
    gpio_mode_set(RST_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, RST_PIN);
    gpio_output_options_set(RST_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_12MHZ, RST_PIN);
    gpio_mode_set(INT_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, INT_PIN);
    gpio_output_options_set(INT_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_12MHZ, INT_PIN);

    gpio_bit_reset(RST_PORT, RST_PIN);    /* assert reset (active low) */
    gpio_bit_reset(INT_PORT, INT_PIN);    /* INT low across the edge */
    delay_1ms(12U);
    gpio_bit_set(RST_PORT, RST_PIN);      /* release reset */
    delay_1ms(100U);                      /* GT911 firmware boot (INT still low) */

    /* INT now becomes the touch-ready interrupt-out input. */
    gpio_mode_set(INT_PORT, GPIO_MODE_INPUT, GPIO_PUPD_PULLUP, INT_PIN);
    delay_1ms(10U);

    /* Read product ID at both candidate addresses (zero first so a NACK shows
     * as 00s in the diagnostic), then bind to whichever answers like a GT911. */
    s_id5d[0] = s_id5d[1] = s_id5d[2] = s_id5d[3] = 0U;
    s_id14[0] = s_id14[1] = s_id14[2] = s_id14[3] = 0U;

    s_addr_w = GT911_ADDR_W_5D; s_addr_r = GT911_ADDR_R_5D;
    (void)_gt911_read_reg(GT_PRODUCT_ID, s_id5d, 4U);
    if (_id_valid(s_id5d)) return 0U;     /* bound to 0x5D */

    s_addr_w = GT911_ADDR_W_14; s_addr_r = GT911_ADDR_R_14;
    (void)_gt911_read_reg(GT_PRODUCT_ID, s_id14, 4U);
    if (_id_valid(s_id14)) return 0U;     /* bound to 0x14 */

    /* Neither answered. Leave address at 0x5D for any later retry; report fail. */
    s_addr_w = GT911_ADDR_W_5D; s_addr_r = GT911_ADDR_R_5D;
    return 1U;
}

uint8_t gt911_scan(gt911_point_t *pts, uint8_t max_pts)
{
    uint8_t status = 0U, count = 0U, clr = 0U;
    uint8_t raw[6];
    uint8_t i;

    if (_gt911_read_reg(GT_STATUS, &status, 1U) != 0U) return 0U;

    /* bit7 = buffer-status, bits[3:0] = touch count */
    if ((status & 0x80U) == 0U) return 0U;  /* buffer not ready */

    count = status & 0x0FU;
    if (count > GT911_MAX_TOUCH) count = GT911_MAX_TOUCH;
    if (count > max_pts)         count = max_pts;

    for (i = 0U; i < count; i++) {
        uint16_t reg = (uint16_t)(GT_PT1_XL + (uint16_t)(i * 8U));
        if (_gt911_read_reg(reg, raw, 6U) != 0U) { count = 0U; break; }
        pts[i].x     = (uint16_t)raw[0] | ((uint16_t)raw[1] << 8);
        pts[i].y     = (uint16_t)raw[2] | ((uint16_t)raw[3] << 8);
        pts[i].valid = 1U;
    }

    /* clear buffer-status flag */
    _gt911_write_reg(GT_STATUS, &clr, 1U);

    return count;
}
