/******************************************************************************
 * st7796.h
 *
 * ★ 2026-05-28: 8080 bit-bang NT35510 已废弃, 切换到 CIMC 官方主板 +
 *   FPC 配套 RGB 4.3" 800×480 LCD (TLI 硬件刷新, 见 rgb_lcd.h).
 *
 *   此头文件保留只为不破坏现有 #include / 函数调用 — 把原 8080 API 简化成
 *   inline forward 到新 TLI driver:
 *
 *     st7796_init()         → rgb_lcd_init()
 *     st7796_set_backlight()→ rgb_lcd_set_backlight()
 *
 *   下列旧 API 已**删除**, 不再支持 (LVGL 不再需要 push 像素到 GRAM, TLI
 *   异步扫 framebuffer):
 *     st7796_set_window / st7796_write_pixel / st7796_fill_rect /
 *     st7796_fill_buf
 *
 *   宏 ST7796_WIDTH/HEIGHT 改成 800/480 (横屏物理像素), 调用方 (lv_port_disp,
 *   ui_screen) 拿到的尺寸现在跟物理面板一致. 现有 UI 是 480×800 portrait 设计,
 *   切到 800×480 landscape 后控件会错位/部分越界 — 这是 LCD 验证阶段的预期,
 *   先点亮再重排 UI.
 *
 *   The retired 8080 implementation is deliberately excluded from the public
 *   source set; this header keeps only the compatibility names used by the UI.
 ******************************************************************************/

#ifndef __ST7796_H__
#define __ST7796_H__

#include "HeaderFiles.h"
#include "rgb_lcd.h"

/* ---------- panel geometry (现在是 RGB 800×480 landscape) ---------- */
#define ST7796_WIDTH    LCD_WIDTH    /* 800 */
#define ST7796_HEIGHT   LCD_HEIGHT   /* 480 */

/* RGB565 colour helpers 已在 rgb_lcd.h 定义 (含 #ifndef 保护). */

/* ---------- public API (compat shim → rgb_lcd_*) ---------- */
static inline void st7796_init(void)                 { rgb_lcd_init(); }
static inline void st7796_set_backlight(uint8_t on)  { rgb_lcd_set_backlight(on); }

#endif /* __ST7796_H__ */
