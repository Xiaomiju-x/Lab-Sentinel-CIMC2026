/******************************************************************************
 * sensors_i2c.c — software I²C on PH7 (SCL) / PH8 (SDA).
 *
 * 2026-05-16: moved from PC10/PC11 back to PB10/PB11. PC10/PC11 have no I²C
 *             hardware AF (only SPI/USART/SDIO/DCI). PB10/PB11 are natural I²C1
 *             pins and proved more reliable for SHT30.
 * 2026-05-28a: SDA 从 PB11 挪到 PB15 (PB11 被新 RGB LCD 占作 G5 AF14).
 * 2026-05-28b: ★ 按新接线 "模块(2).docx" 整体挪到 PH7 (SCL) / PH8 (SDA).
 *              PH7/PH8 在 GD32H759 上有 I2C2 硬件 AF, 但本驱动仍走 bit-bang
 *              GPIO 软件 I²C — 兼容任何 GPIO. PH7 旧 LED3 (已弃用) /
 *              PH8 was used by a retired 8080 LCD path; the final LCD is RGB.
 *              新 RGB LCD 只占 PH4(G4)+PH10(R4), PH7/PH8 完全空闲.
 *              ★ 硬件: 把 SHT30+ADXL345 模块 SDA 接到 PH8, SCL 接到 PH7.
 *              模块自带 4.7kΩ 上拉到 3V3, 电源 3V3+GND.
 ******************************************************************************/

#include "sensors_i2c.h"
#include "FreeRTOS.h"
#include "semphr.h"

#define SCL_PORT   GPIOH
#define SCL_PIN    GPIO_PIN_7
#define SDA_PORT   GPIOH
#define SDA_PIN    GPIO_PIN_8

/* Bus mutex — sensor_task (PRIO_HIGH, 200 Hz ADXL345) and env_task (PRIO_LOW,
 * 1 Hz SHT30) share this bit-bang bus. Without a mutex, sensor_task preempts
 * env_task mid-SHT30-transaction and shreds the START/STOP/ACK timing.
 * xSemaphoreCreateMutex() provides priority inheritance to avoid inversion. */
static SemaphoreHandle_t s_bus_mutex = NULL;

#define BUS_LOCK_TIMEOUT_TICKS   pdMS_TO_TICKS(100U)

static inline uint8_t _lock(void)
{
    if (s_bus_mutex == NULL) return 0U;   /* not initialised yet — allow */
    return (xSemaphoreTake(s_bus_mutex, BUS_LOCK_TIMEOUT_TICKS) == pdTRUE) ? 0U : 1U;
}

static inline void _unlock(void)
{
    if (s_bus_mutex != NULL) xSemaphoreGive(s_bus_mutex);
}

/* ~10 µs delay tuned for ~50 kHz SCL on a 600 MHz M7. Sensors are happy here. */
static inline void _delay(void)
{
    volatile uint16_t i = 600U;
    while (i--) { __NOP(); }
}

static void _sda_out(void)
{
    gpio_mode_set(SDA_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_PULLUP, SDA_PIN);
    gpio_output_options_set(SDA_PORT, GPIO_OTYPE_OD, GPIO_OSPEED_60MHZ, SDA_PIN);
}

static void _sda_in(void)
{
    gpio_mode_set(SDA_PORT, GPIO_MODE_INPUT, GPIO_PUPD_PULLUP, SDA_PIN);
}

static inline void _scl(uint8_t v)
{
    if (v) gpio_bit_set(SCL_PORT, SCL_PIN);
    else   gpio_bit_reset(SCL_PORT, SCL_PIN);
    _delay();
}

static inline void _sda(uint8_t v)
{
    if (v) gpio_bit_set(SDA_PORT, SDA_PIN);
    else   gpio_bit_reset(SDA_PORT, SDA_PIN);
    _delay();
}

static inline uint8_t _sda_read(void)
{
    return (gpio_input_bit_get(SDA_PORT, SDA_PIN) != RESET) ? 1U : 0U;
}

/* ---------- low level ---------- */

static void _start(void)
{
    _sda_out();
    _sda(1); _scl(1);
    _sda(0); _scl(0);
}

static void _stop(void)
{
    _sda_out();
    _sda(0); _scl(1);
    _sda(1);
}

static uint8_t _wait_ack(void)
{
    uint8_t t = 100U;
    _sda_in();
    _scl(1);
    while (_sda_read()) {
        if (--t == 0U) { _scl(0); _stop(); return 1U; }
    }
    _scl(0);
    return 0U;
}

static void _send_ack(void)  { _sda_out(); _sda(0); _scl(1); _scl(0); _sda(1); }
static void _send_nack(void) { _sda_out(); _sda(1); _scl(1); _scl(0); }

static void _write_byte(uint8_t b)
{
    uint8_t i;
    _sda_out();
    for (i = 0U; i < 8U; i++) {
        _sda((b & 0x80U) ? 1U : 0U);
        b <<= 1;
        _scl(1); _scl(0);
    }
}

static uint8_t _read_byte(uint8_t ack)
{
    uint8_t i, dat = 0U;
    _sda_in();
    for (i = 0U; i < 8U; i++) {
        _scl(1);
        dat <<= 1;
        if (_sda_read()) dat |= 0x01U;
        _scl(0);
    }
    if (ack) _send_ack();
    else     _send_nack();
    return dat;
}

/* ---------- public ---------- */

void sensors_i2c_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOH);   /* 2026-05-28b: 传感器 I²C 挪到 PH7/PH8 */

    /* Bus mutex: must exist before any task calls a public function. */
    if (s_bus_mutex == NULL) {
        s_bus_mutex = xSemaphoreCreateMutex();
    }

    /* SCL: open-drain output with pull-up */
    gpio_mode_set(SCL_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_PULLUP, SCL_PIN);
    gpio_output_options_set(SCL_PORT, GPIO_OTYPE_OD, GPIO_OSPEED_60MHZ, SCL_PIN);

    _sda_out();
    _scl(1); _sda(1);
}

uint8_t sensors_i2c_write(uint8_t addr_w, uint8_t reg,
                          const uint8_t *buf, uint16_t len)
{
    uint16_t i;
    uint8_t ret = 1U;

    if (_lock()) return 1U;

    _start();
    _write_byte(addr_w);   if (_wait_ack()) { _stop(); goto done; }
    _write_byte(reg);      if (_wait_ack()) { _stop(); goto done; }
    for (i = 0U; i < len; i++) {
        _write_byte(buf[i]);
        if (_wait_ack()) { _stop(); goto done; }
    }
    _stop();
    ret = 0U;
done:
    _unlock();
    return ret;
}

uint8_t sensors_i2c_read(uint8_t addr_w, uint8_t addr_r, uint8_t reg,
                         uint8_t *buf, uint16_t len)
{
    uint16_t i;
    uint8_t ret = 1U;

    if (_lock()) return 1U;

    _start();
    _write_byte(addr_w);   if (_wait_ack()) { _stop(); goto done; }
    _write_byte(reg);      if (_wait_ack()) { _stop(); goto done; }
    _start();              /* RESTART */
    _write_byte(addr_r);   if (_wait_ack()) { _stop(); goto done; }
    for (i = 0U; i < len; i++) {
        buf[i] = _read_byte((i < (len - 1U)) ? 1U : 0U);
    }
    _stop();
    ret = 0U;
done:
    _unlock();
    return ret;
}

uint8_t sensors_i2c_write_cmd16(uint8_t addr_w, uint16_t cmd)
{
    uint8_t ret = 1U;

    if (_lock()) return 1U;

    _start();
    _write_byte(addr_w);                if (_wait_ack()) { _stop(); goto done; }
    _write_byte((uint8_t)(cmd >> 8));   if (_wait_ack()) { _stop(); goto done; }
    _write_byte((uint8_t)(cmd & 0xFFU));if (_wait_ack()) { _stop(); goto done; }
    _stop();
    ret = 0U;
done:
    _unlock();
    return ret;
}

uint8_t sensors_i2c_read_n(uint8_t addr_r, uint8_t *buf, uint16_t len)
{
    uint16_t i;
    uint8_t ret = 1U;

    if (_lock()) return 1U;

    _start();
    _write_byte(addr_r);   if (_wait_ack()) { _stop(); goto done; }
    for (i = 0U; i < len; i++) {
        buf[i] = _read_byte((i < (len - 1U)) ? 1U : 0U);
    }
    _stop();
    ret = 0U;
done:
    _unlock();
    return ret;
}

/* -------------------------------------------------------------------------- */
/* ESP32-C3 IIoT bridge (I2C slave 0x42 on this same PH7/PH8 bus).            */
/* Reuses the bus-mutexed register read/write so it interleaves safely with   */
/* the SHT30 / ADXL345 traffic that sensor_task + env_task already drive.     */
/*   REG_TELEM 0x10 (write) : GD32 -> ESP32 telemetry block                   */
/*   REG_STATUS 0x20 (read) : [link, rssi_mag, diag_len, uplink_lo, uplink_hi]*/
/*   REG_DIAG  0x21 (read)  : the last R1 diagnosis string (<= 63 B)          */
/* -------------------------------------------------------------------------- */
#define ESP32_ADDR_W     (uint8_t)(ESP32_I2C_ADDR << 1)         /* 0x84 */
#define ESP32_ADDR_R     (uint8_t)((ESP32_I2C_ADDR << 1) | 1U)  /* 0x85 */
#define ESP32_REG_TELEM  0x10U
#define ESP32_REG_STATUS 0x20U
#define ESP32_REG_DIAG   0x21U

uint8_t esp32_push_telemetry(const uint8_t *buf, uint16_t len)
{
    return sensors_i2c_write(ESP32_ADDR_W, ESP32_REG_TELEM, buf, len);
}

uint8_t esp32_read_status(uint8_t *st)
{
    return sensors_i2c_read(ESP32_ADDR_W, ESP32_ADDR_R, ESP32_REG_STATUS,
                            st, ESP32_STATUS_LEN);
}

uint8_t esp32_read_diag(uint8_t *buf, uint16_t maxlen)
{
    if (maxlen == 0U) return 1U;
    if (maxlen > (ESP32_DIAG_MAX + 1U)) maxlen = ESP32_DIAG_MAX + 1U;
    return sensors_i2c_read(ESP32_ADDR_W, ESP32_ADDR_R, ESP32_REG_DIAG,
                            buf, maxlen);
}

/****************************End*****************************/
