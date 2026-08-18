/******************************************************************************
 * sccb.c — software SCCB on PB4 (SCL) / PB7 (SDA).
 *
 * 2026-05-29: SCL 改回 PB4 以匹配 CIMC 官方板摄像头排针表 (DCMI_SCL=PB4 排针2,
 * DCMI_SDA=PB7 排针4). 旧注释"PB4 是 smoke AO"已过时 — 烟雾传感器现在 AO=PA9 /
 * DO=PC3, PB4 空闲 (NJTRST, SWD 调试下可用作 GPIO). PA5 让回给 INMP441 备用.
 *
 * Bit-bang topology (open-drain emulation):
 *   To drive LOW  → set GPIO mode OUTPUT, write 0
 *   To release    → set GPIO mode INPUT (pull-up keeps line high)
 ******************************************************************************/

#include "sccb.h"

#define SCCB_SCL_PORT   GPIOB
#define SCCB_SCL_PIN    GPIO_PIN_4   /* PB4 — DCMI_SCL 排针2 (官方板) */
#define SCCB_SDA_PORT   GPIOB
#define SCCB_SDA_PIN    GPIO_PIN_7   /* PB7 — DCMI_SDA 排针4 */

/* OV5640 7-bit I²C addr 0x3C → 8-bit write 0x78, read 0x79. */
#define OV5640_ADDR_W   0x78U
#define OV5640_ADDR_R   0x79U

/* ~5 µs per delay tick → ~50 kHz SCL. CPU is 600 MHz, the existing
 * sensors_i2c uses the same idiom and works at ~50 kHz, so reuse the
 * pattern instead of fiddling with cycle-accurate timing. */
static void _sccb_delay(void)
{
    volatile uint32_t i;
    for (i = 0U; i < 600U; i++) { __NOP(); }
}

static inline void _sda_high(void)
{
    gpio_mode_set(SCCB_SDA_PORT, GPIO_MODE_INPUT, GPIO_PUPD_PULLUP, SCCB_SDA_PIN);
}

static inline void _sda_low(void)
{
    gpio_mode_set(SCCB_SDA_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, SCCB_SDA_PIN);
    gpio_bit_reset(SCCB_SDA_PORT, SCCB_SDA_PIN);
}

static inline void _scl_high(void)
{
    gpio_bit_set(SCCB_SCL_PORT, SCCB_SCL_PIN);
}

static inline void _scl_low(void)
{
    gpio_bit_reset(SCCB_SCL_PORT, SCCB_SCL_PIN);
}

static inline uint8_t _sda_read(void)
{
    return (uint8_t)gpio_input_bit_get(SCCB_SDA_PORT, SCCB_SDA_PIN);
}

void sccb_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOB);   /* SCL: PB4 + SDA: PB7 (同 bank) */

    /* SCL: push-pull output (we never read SCL). */
    gpio_mode_set(SCCB_SCL_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_PULLUP, SCCB_SCL_PIN);
    gpio_output_options_set(SCCB_SCL_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ,
                            SCCB_SCL_PIN);

    /* SDA: idle high via input + pull-up. */
    _sda_high();

    _scl_high();
    _sda_high();
    _sccb_delay();
}

static void _start(void)
{
    _sda_high(); _sccb_delay();
    _scl_high(); _sccb_delay();
    _sda_low();  _sccb_delay();
    _scl_low();  _sccb_delay();
}

static void _stop(void)
{
    _sda_low();  _sccb_delay();
    _scl_high(); _sccb_delay();
    _sda_high(); _sccb_delay();
}

static uint8_t _wait_ack(void)
{
    uint8_t ack;
    _sda_high(); _sccb_delay();
    _scl_high(); _sccb_delay();
    ack = _sda_read();
    _scl_low();  _sccb_delay();
    return ack;        /* 0 = ACK, 1 = NACK */
}

static void _send_nack(void)
{
    _sda_high(); _sccb_delay();
    _scl_high(); _sccb_delay();
    _scl_low();  _sccb_delay();
}

static uint8_t _write_byte(uint8_t b)
{
    uint8_t i;
    for (i = 0U; i < 8U; i++) {
        if (b & 0x80U) _sda_high(); else _sda_low();
        _sccb_delay();
        _scl_high(); _sccb_delay();
        _scl_low();  _sccb_delay();
        b <<= 1;
    }
    return _wait_ack();
}

static uint8_t _read_byte(void)
{
    uint8_t i, v = 0U;
    _sda_high();
    for (i = 0U; i < 8U; i++) {
        v <<= 1;
        _scl_high(); _sccb_delay();
        if (_sda_read()) v |= 0x01U;
        _scl_low();  _sccb_delay();
    }
    return v;
}

uint8_t sccb_write_reg(uint16_t reg, uint8_t val)
{
    _start();
    if (_write_byte(OV5640_ADDR_W) != 0U)         { _stop(); return 1U; }
    if (_write_byte((uint8_t)(reg >> 8)) != 0U)   { _stop(); return 1U; }
    if (_write_byte((uint8_t)(reg & 0xFFU)) != 0U){ _stop(); return 1U; }
    if (_write_byte(val) != 0U)                   { _stop(); return 1U; }
    _stop();
    return 0U;
}

uint8_t sccb_read_reg(uint16_t reg, uint8_t *val)
{
    if (val == NULL) return 1U;

    _start();
    if (_write_byte(OV5640_ADDR_W) != 0U)         { _stop(); return 1U; }
    if (_write_byte((uint8_t)(reg >> 8)) != 0U)   { _stop(); return 1U; }
    if (_write_byte((uint8_t)(reg & 0xFFU)) != 0U){ _stop(); return 1U; }
    _stop();

    _start();
    if (_write_byte(OV5640_ADDR_R) != 0U)         { _stop(); return 1U; }
    *val = _read_byte();
    _send_nack();           /* SCCB requires NACK on the last (here only) byte. */
    _stop();
    return 0U;
}

/****************************End*****************************/
