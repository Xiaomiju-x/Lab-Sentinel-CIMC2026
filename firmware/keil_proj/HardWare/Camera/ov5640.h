/******************************************************************************
 * ov5640.h
 *
 * OV5640 5MP camera (ALIENTEK 18-pin breakout) → DCI parallel + SCCB control.
 *
 *   Output mode  : QVGA  320 × 240 RGB565   (8-bit DCI bus)
 *   Frame rate   : ~30 fps with 24 MHz XCLK + sensor PLL
 *   Buffer       : SDRAM_CAMERA_FB (153,600 B), DMA-filled by DCI
 *
 * Pin map (CIMC 官方板摄像头 30-pin 排针, 2026-05-29 按实物排针表):
 *   VCC  → 3V3                    SCL  → PB4   (SCCB, 排针2)
 *   GND  → GND (排针6)            SDA  → PB7   (SCCB, 排针4)
 *   XCLK → PG7  (TIMER30_CH2 PWM @ ~24 MHz, AF4, 排针14)
 *   PCLK → PE3  (DCI_PIXCLK,  AF13, 排针12)   ← 旧驱动错放 PA6 (现 RGB G2)
 *   HSYNC→ PA4  (DCI_HSYNC,   AF13, 排针10)
 *   VSYNC→ PG9  (DCI_VSYNC,   AF13, 排针8)
 *   D0   → PC6  (DCI_D0,      AF13, 排针16)
 *   D1   → PA10 (DCI_D1,      AF13, 排针18)
 *   D2   → PC8  (DCI_D2,      AF13, 排针20)
 *   D3   → PC9  (DCI_D3,      AF13, 排针22)
 *   D4   → PE4  (DCI_D4,      AF13, 排针24)   ← 旧驱动错放 PC11 (CI1302 RX)
 *   D5   → PB6  (DCI_D5,      AF13, 排针26)   ← 旧驱动错放 PD3 (RGB G7)
 *   D6   → PB8  (DCI_D6,      AF13, 排针28)   ★ 需禁 GT911 触摸 (INT 原占 PB8)
 *   D7   → PE6  (DCI_D7,      AF13, 排针30)   ← 旧驱动错放 PB9 (RGB B7)
 *   RESET/PWDN → 排针未引出, 模组自管理. MCU 不再碰 PD0 (=SDRAM D2, 旧驱动配
 *                GPIO 直接毁 SDRAM 总线) 也不碰 PD2. 复位走 SCCB 软复位 0x3008=0x82.
 *
 * NOTE: D0..D7 here are the SENSOR side. The DCI peripheral is little-endian
 * about byte order; combined with OV5640's RGB565 output you get a uint16_t
 * pixel where the MSB is R5..R0 G7..G5 (high byte) and LSB G4..G2 B7..B0.
 ******************************************************************************/

#ifndef __OV5640_H__
#define __OV5640_H__

#include "HeaderFiles.h"

#define OV5640_QVGA_WIDTH        320U
#define OV5640_QVGA_HEIGHT       240U
#define OV5640_QVGA_PIXELS       (OV5640_QVGA_WIDTH * OV5640_QVGA_HEIGHT)
#define OV5640_QVGA_BYTES        (OV5640_QVGA_PIXELS * 2U)

/* Where the DMA writes the frame. Cast to (uint16_t *) to read pixels. */
extern volatile uint16_t * const ov5640_framebuf;

/* Returns 0 on full success, non-zero on probe / SCCB / DCI failure. The
 * specific code helps narrow which sub-step failed:
 *   1 = SCCB chip-id mismatch (0x5640 expected)
 *   2 = SCCB sensor init wrote a register and got NACK
 *   3 = DCI / DMA bring-up failure (currently never returned, reserved) */
uint8_t ov5640_init(void);

/* Read sensor's product ID register pair (0x300A=0x56, 0x300B=0x40 expected). */
uint8_t ov5640_read_chip_id(uint16_t *id);

/* True after at least one full frame has landed in `ov5640_framebuf`.
 * Vision_task should poll this before reading the buffer. */
uint8_t ov5640_frame_ready(void);

/* Clears the frame_ready latch (call before waiting for the next snapshot). */
void    ov5640_frame_consume(void);

/* Trigger one snapshot (single-frame capture mode). */
void    ov5640_capture_one(void);

/* Diagnostic: enable(1) / tristate(0) the sensor's DVP output pads
 * (D[9:0], VSYNC, HREF, PCLK) via SCCB 0x3017/0x3018. Used to tell whether the
 * sensor is the source holding a sync line at a level vs an external board short. */
void    ov5640_dvp_pads(uint8_t on);

#endif /* __OV5640_H__ */
