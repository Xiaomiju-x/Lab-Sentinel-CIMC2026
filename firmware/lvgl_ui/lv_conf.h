/**
 * lv_conf.h — LVGL 8.3 LTS configuration for CIMC Lab-Sentinel
 *
 * Place this file at firmware/lvgl_ui/lv_conf.h  (sibling of the lvgl/ folder).
 * LVGL 8.3 auto-includes "../lv_conf.h" relative to its source root.
 *
 * Tuned for GD32H759IMK6: 600 MHz M7, 1 MB SRAM, 32 MB SDRAM,
 * 320×480 ST7796 8080 LCD, FreeRTOS 1 kHz tick.
 */

/* clang-format off */
#if 1   /* Set to 0 to disable — LVGL reads this guard. */

#ifndef LV_CONF_H
#define LV_CONF_H

#include <stdint.h>

/*==============================================================================
 * COLOR
 *============================================================================*/
#define LV_COLOR_DEPTH          16      /* RGB565 */
#define LV_COLOR_16_SWAP         0      /* 16-bit 8080 parallel: no byte-swap */
#define LV_COLOR_SCREEN_TRANSP   0
#define LV_COLOR_MIX_ROUND_OFS   0

/*==============================================================================
 * MEMORY MANAGER — static pool in SDRAM
 *
 * Pool MUST be placed at a known address: Keil ARM system heap is only 2 KB
 * (Heap_Size in startup_gd32h7xx.s), so the LV_MEM_POOL_ALLOC=malloc path
 * silently returns NULL and LVGL panics in LV_ASSERT_HANDLER.
 *
 * Address 0xC00C0000 corresponds to SDRAM_LVGL_POOL in sdram.h (after 768KB
 * fb1 for 480×800×2). Keep the two values in sync if either side changes.
 * 2026-05-22: 升级 LCD 320×480 → 480×800 (NT35510 4.3"), pool 后移 768→256KB.
 *============================================================================*/
#define LV_MEM_CUSTOM            0
/* 2026-05-23: 256KB 装不下 480×800 widget tree (4 tile + log list + 28pt font
 * glyph cache), 全白屏. 扩到 1MB. SDRAM 富裕: pool 0xC00C0000-0xC01C0000,
 * camera FB 0xC0200000 留 256KB 安全 margin. */
#define LV_MEM_SIZE        (1024U * 1024U)           /* 1 MB pool */
#define LV_MEM_ADR         (0xC00C0000U)             /* SDRAM, after FB1 (480×800×2 = 768KB) */
#define LV_MEM_POOL_INCLUDE      <stdlib.h>          /* unused when LV_MEM_ADR != 0 */
#define LV_MEM_POOL_ALLOC        malloc
#define LV_MEM_POOL_FREE         free

/*==============================================================================
 * HARDWARE ABSTRACTION — tick via FreeRTOS (no tick-hook required)
 *============================================================================*/
#define LV_TICK_CUSTOM              1
#define LV_TICK_CUSTOM_INCLUDE      "lv_tick_port.h"
/* configTICK_RATE_HZ = 1000 → each tick = 1 ms */
#define LV_TICK_CUSTOM_SYS_TIME_EXPR  ((uint32_t)xTaskGetTickCount())

#define LV_DPI_DEF              130     /* ~5" diagonal 320×480 ≈ 130 dpi */

/*==============================================================================
 * FEATURE FLAGS
 *============================================================================*/
#define LV_DRAW_COMPLEX          1
#define LV_SHADOW_CACHE_SIZE     0
#define LV_CIRCLE_CACHE_SIZE     4
#define LV_IMG_CACHE_DEF_SIZE    1
#define LV_GRADIENT_MAX_STOPS    2
#define LV_GRAD_CACHE_DEF_SIZE   0
#define LV_DITHER_GRADIENT       0
#define LV_DISP_ROT_MAX_BUF     (10 * 1024)

/*==============================================================================
 * LOGGING — disabled to save flash
 *============================================================================*/
#define LV_USE_LOG               0
#define LV_LOG_LEVEL             LV_LOG_LEVEL_NONE
#define LV_LOG_PRINTF            0
#define LV_LOG_TRACE_MEM         0
#define LV_LOG_TRACE_TIMER       0
#define LV_LOG_TRACE_INDEV       0
#define LV_LOG_TRACE_DISP_REFR   0
#define LV_LOG_TRACE_EVENT       0
#define LV_LOG_TRACE_OBJ_CREATE  0
#define LV_LOG_TRACE_LAYOUT      0
#define LV_LOG_TRACE_ANIM        0

/*==============================================================================
 * ASSERT
 *============================================================================*/
#define LV_USE_ASSERT_NULL       1
#define LV_USE_ASSERT_MALLOC     1
#define LV_USE_ASSERT_STYLE      0
#define LV_USE_ASSERT_MEM_INTEGRITY 0
#define LV_USE_ASSERT_OBJ        0
#define LV_ASSERT_HANDLER_INCLUDE <stdint.h>
#define LV_ASSERT_HANDLER        while(1);

/*==============================================================================
 * DEBUG — disabled
 *============================================================================*/
#define LV_USE_PERF_MONITOR      0
#define LV_USE_MEM_MONITOR       0
#define LV_USE_REFR_DEBUG        0

/*==============================================================================
 * FONTS
 *============================================================================*/
#define LV_FONT_MONTSERRAT_8     0
#define LV_FONT_MONTSERRAT_10    0
#define LV_FONT_MONTSERRAT_12    0
#define LV_FONT_MONTSERRAT_14    1   /* default text */
#define LV_FONT_MONTSERRAT_16    0   /* 字体源文件未加入 Keil 工程, 退回 _14 */
#define LV_FONT_MONTSERRAT_18    1   /* tile titles + values on 480x800 */
#define LV_FONT_MONTSERRAT_20    1   /* (kept; may be unused after 480x800 resize) */
#define LV_FONT_MONTSERRAT_22    0
#define LV_FONT_MONTSERRAT_24    0
#define LV_FONT_MONTSERRAT_26    0
#define LV_FONT_MONTSERRAT_28    1   /* risk banner on 480x800 */
#define LV_FONT_MONTSERRAT_30    0
#define LV_FONT_MONTSERRAT_32    0
#define LV_FONT_MONTSERRAT_34    0
#define LV_FONT_MONTSERRAT_36    0
#define LV_FONT_MONTSERRAT_38    0
#define LV_FONT_MONTSERRAT_40    0
#define LV_FONT_MONTSERRAT_42    0
#define LV_FONT_MONTSERRAT_44    0
#define LV_FONT_MONTSERRAT_46    0
#define LV_FONT_MONTSERRAT_48    0

#define LV_FONT_MONTSERRAT_12_SUBPX      0
#define LV_FONT_MONTSERRAT_28_COMPRESSED 0
#define LV_FONT_DEJAVU_16_PERSIAN_HEBREW 0
#define LV_FONT_SIMSUN_16_CJK            0
#define LV_FONT_UNSCII_8                 0
#define LV_FONT_UNSCII_16                0

#define LV_FONT_CUSTOM_DECLARE

#define LV_FONT_DEFAULT  &lv_font_montserrat_14

#define LV_FONT_FMT_TXT_LARGE   0
#define LV_USE_FONT_SUBPX       0
#define LV_FONT_SUBPX_BGR       0
#define LV_USE_USER_DATA        1

/*==============================================================================
 * THEME
 *============================================================================*/
#define LV_USE_THEME_DEFAULT     1
#define LV_THEME_DEFAULT_DARK    1   /* start in dark mode */
#define LV_THEME_DEFAULT_GROW    1
#define LV_THEME_DEFAULT_TRANSITION_TIME 80
#define LV_USE_THEME_SIMPLE      0
#define LV_USE_THEME_MONO        0

/*==============================================================================
 * LAYOUT
 *============================================================================*/
#define LV_USE_FLEX              1
#define LV_USE_GRID              0

/*==============================================================================
 * WIDGETS (enable only what the UI uses)
 *============================================================================*/
#define LV_USE_ARC           1
#define LV_USE_BAR           1
#define LV_USE_BTN           1
#define LV_USE_BTNMATRIX     0
#define LV_USE_CALENDAR      0
#define LV_USE_CANVAS        0
#define LV_USE_CHECKBOX      0
#define LV_USE_CHART         0
#define LV_USE_COLORWHEEL    0
#define LV_USE_IMGBTN        0
#define LV_USE_KEYBOARD      0
#define LV_USE_LABEL         1
    #define LV_LABEL_TEXT_SELECTION  0
    #define LV_LABEL_LONG_TXT_HINT   0
#define LV_USE_LED           1
#define LV_USE_LINE          1
#define LV_USE_LIST          1
#define LV_USE_MENU          0
#define LV_USE_METER         0
#define LV_USE_MSGBOX        0
#define LV_USE_ROLLER        0
#define LV_USE_SLIDER        0
#define LV_USE_SPAN          0
#define LV_USE_SPINBOX       0
#define LV_USE_SPINNER       0
#define LV_USE_TABLE         0
#define LV_USE_TABVIEW       0
#define LV_USE_TEXTAREA      0
#define LV_USE_TILEVIEW      0
#define LV_USE_WIN           0
#define LV_USE_IMG           1      /* Wave C: Camera LIVE real-frame view (RGB565 lv_img) */
#define LV_USE_ANIMIMG       0

/*==============================================================================
 * EXTRA COMPONENTS
 *============================================================================*/
#define LV_USE_SNAPSHOT      0
#define LV_USE_MONKEY        0
#define LV_USE_GRIDNAV       0
#define LV_USE_FRAGMENT      0
#define LV_USE_IMGFONT       0
#define LV_USE_MSG           0
#define LV_USE_IME_PINYIN    0

#endif  /* LV_CONF_H */
#endif  /* End of guard */
