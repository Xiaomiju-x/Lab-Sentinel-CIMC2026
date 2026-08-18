/******************************************************************************
 * rgb_lcd.h
 *
 * GD32H759 TLI RGB-565 800×480 LCD driver (CIMC 官方主板配套屏, FPC).
 *
 * 2026-05-28: 从 8080 bit-bang NT35510 (杜邦线 30+ 根) 换到官方主板 +
 * 官方 4.3" 800×480 RGB LCD (FPC 接到主板). 接口变了:
 *
 *   旧屏: NT35510 GRAM + 8080 16-bit GPIO bit-bang. MCU push 像素到 controller
 *         GRAM, controller 自己刷面板. ~45ms/全屏, CPU 100% 占用.
 *
 *   新屏: 纯 RGB 面板 (无 controller / 无 GRAM). MCU 必须用 TLI 硬件
 *         (TFT-LCD Interface) 输出像素流 + HSYNC/VSYNC/DE. TLI 通过 AHB
 *         master 直接 DMA SDRAM framebuffer, 60Hz 硬件刷新, **CPU 0 占用**.
 *
 * 信号 (CIMC 官方主板, 见 LCD屏幕接线.docx):
 *   控制:
 *     HSYNC  → PE15  (AF10)
 *     VSYNC  → PA7   (AF14)
 *     PCLK   → PB3   (AF2, as required by the final-board pin contract)
 *     DE     → PE13  (AF14)
 *     BL     → PD11  (GPIO out, active HIGH; PWM 用 TIMER40_CH1 备选)
 *
 *   RGB565 数据 (高位):
 *     R7=PC4  R6=PB1  R5=PA9  R4=PH10 R3=PB0    (5 bit, AF14/9/14/14/9)
 *     G7=PD3  G6=PC7  G5=PB11 G4=PH4  G3=PG10 G2=PA6  (6 bit, AF14)
 *     B7=PB9  B6=PA15 B5=PA3  B4=PG12 B3=PA8    (5 bit, AF14)
 *
 * Framebuffer: SDRAM 起始 0xC0000000 (SDRAM_LVGL_FB1), 800×480×2 = 768,000 B.
 *
 * PLL2 配置 (像素时钟):
 *   HXTAL=25MHz / PSC=25 → 1MHz VCO_in
 *   × PLLN=240          → 240MHz VCO_out
 *   / PLLR=2             → 120MHz PLL2R
 *   / TLI_DIV4           → 30MHz pixel clock
 *
 * 像素时序 (800+10+150+15) × (480+10+140+40) / 30M = 975×670/30M ≈ 21.8ms = 45.7 FPS
 *
 * LVGL flush_cb: 因为 TLI 异步刷, LVGL 写完 fb 直接 lv_disp_flush_ready,
 *                不需要 push / blit. D-cache 需要 clean (TLI bypass D-cache).
 ******************************************************************************/

#ifndef __RGB_LCD_H__
#define __RGB_LCD_H__

#include "HeaderFiles.h"

/* ---------- panel geometry (800×480 landscape, 物理硬件不可改) ---------- */
#define LCD_WIDTH      800U
#define LCD_HEIGHT     480U
#define LCD_BPP        2U   /* RGB565 */
#define LCD_FB_BYTES   ((uint32_t)LCD_WIDTH * (uint32_t)LCD_HEIGHT * LCD_BPP)

/* ---------- TLI display timing (官方 800×480 4.3") ---------- */
#define LCD_HSYNC_PULSE     10U
#define LCD_HBACK_PORCH     150U
#define LCD_HFRONT_PORCH    15U
#define LCD_VSYNC_PULSE     10U
#define LCD_VBACK_PORCH     140U
#define LCD_VFRONT_PORCH    40U

/* ---------- RGB565 colour helpers (跟旧 st7796.h 一致) ---------- */
#define RGB565(r,g,b) ((uint16_t)(((r)&0x1FU)<<11|(((g)&0x3FU)<<5)|((b)&0x1FU)))
#ifndef COLOR_BLACK
#define COLOR_BLACK     0x0000U
#define COLOR_WHITE     0xFFFFU
#define COLOR_GREEN     RGB565( 0, 52,  0)
#define COLOR_YELLOW    RGB565(31, 63,  0)
#define COLOR_ORANGE    RGB565(31, 38,  0)
#define COLOR_RED       RGB565(28,  0,  0)
#define COLOR_DARKGRAY  RGB565( 8, 16,  8)
#define COLOR_PANEL_BG  RGB565( 3,  6,  3)
#endif

/* ---------- public API ---------- */
void        rgb_lcd_init(void);
void        rgb_lcd_set_backlight(uint8_t on);
uint16_t   *rgb_lcd_framebuffer(void);    /* fb1 (= 0xC0000000), 首帧 TLI scan base */
uint16_t   *rgb_lcd_framebuffer2(void);   /* fb2 (= 0xC0300000), 双缓冲第二块 */

/* 把 TLI layer0 scan base 切到 fb_addr, 在下一个 VBlank 生效, 然后 busy-wait
 * 直到 reload 完成 (TLI_INT_FLAG_LCR 置位). 调用者保证 fb_addr 已 D-cache clean.
 * 最坏耗时 = 1 帧 (~22ms). 给 LVGL flush_cb 做 swap. */
void        rgb_lcd_swap_to(uint32_t fb_addr);

#endif /* __RGB_LCD_H__ */
