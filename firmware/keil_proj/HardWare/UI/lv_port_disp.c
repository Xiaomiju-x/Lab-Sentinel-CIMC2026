/******************************************************************************
 * lv_port_disp.c
 *
 * LVGL 8.3 display driver for the CIMC 官方 4.3" RGB 800×480 TLI LCD.
 *
 * 2026-05-28a 单缓冲版本: framebuffer = TLI scan base, flush_cb 只 D-cache clean.
 *             已工作但有 tearing/闪烁 (LVGL 写 + TLI 异步扫同一块 fb).
 * 2026-05-28b ★ 切到双缓冲 + VBlank-synced swap, 消除 tearing/闪烁.
 *             fb1 (0xC0000000) + fb2 (0xC0300000), full_refresh=1 direct mode.
 *
 * 工作流:
 *   t=0:   LVGL 把整屏渲染到 buf_act (假设 = fb2)
 *   t=R:   渲染完, flush_cb fires color_p = fb2
 *          → D-cache clean fb2
 *          → 写 TLI scan addr = fb2, 请求 VBlank reload
 *          → busy-wait 直到 hardware 在 VBlank 真正 reload (≤ 22ms)
 *          → lv_disp_flush_ready → LVGL 把 buf_act 切到 fb1
 *   返回: LVGL 开始下一帧渲染到 fb1, 此时 TLI 正在扫 fb2 (没冲突)
 *
 * 关键: rgb_lcd_swap_to() 内部 busy-wait LCR flag, 保证返回时 TLI 已经在
 * 扫新 buf, 老 buf (= LVGL 下帧要写的那块) 不再被 TLI 读, 没有 R/W 冲突.
 *
 * D-cache: M7 D-cache 16-32KB 装不下 768KB fb, LVGL 写完后有 dirty line 在
 * cache 里没回写 SDRAM. TLI 是独立 master 不走 D-cache, 必须 clean 后才能看到
 * 最新数据.
 ******************************************************************************/

#include "lv_port_disp.h"
#include "rgb_lcd.h"
#include "sdram.h"

static lv_disp_draw_buf_t  s_draw_buf;
static lv_disp_drv_t       s_disp_drv;

static void _flush_cb(lv_disp_drv_t *drv, const lv_area_t *area,
                      lv_color_t *color_p)
{
    (void)area;

    /* LVGL 刚渲染完整屏到 color_p (= fb1 或 fb2). 把 dirty cache line 回写
     * SDRAM 让 TLI 看得到. 768KB clean 在 M7 上 ~1-2ms (cache miss path). */
    SCB_CleanInvalidateDCache_by_Addr((uint32_t *)(uintptr_t)color_p,
                                      (int32_t)LCD_FB_BYTES);

    /* 切 TLI scan base 到刚画完的 buf, VBlank 同步生效 (内部 busy-wait ≤ 22ms).
     * 返回后 TLI 正在扫这一块, LVGL 下帧渲染的另一块 buf 完全空闲. */
    rgb_lcd_swap_to((uint32_t)(uintptr_t)color_p);

    lv_disp_flush_ready(drv);
}

void lv_port_disp_init(void)
{
    /* 双缓冲 direct mode: 两块全屏 buf, LVGL 自动 ping-pong.
     * full_refresh=1 强制每帧全屏重画 (不做 dirty rect join), 配双 buf 完美
     * 消除 tearing. */
    lv_color_t *fb1 = (lv_color_t *)rgb_lcd_framebuffer();
    lv_color_t *fb2 = (lv_color_t *)rgb_lcd_framebuffer2();

    lv_disp_draw_buf_init(&s_draw_buf, fb1, fb2,
                          (uint32_t)LCD_WIDTH * (uint32_t)LCD_HEIGHT);

    lv_disp_drv_init(&s_disp_drv);
    s_disp_drv.hor_res       = (lv_coord_t)LCD_WIDTH;   /* 800 */
    s_disp_drv.ver_res       = (lv_coord_t)LCD_HEIGHT;  /* 480 */
    s_disp_drv.flush_cb      = _flush_cb;
    s_disp_drv.draw_buf      = &s_draw_buf;
    s_disp_drv.full_refresh  = 1;

    lv_disp_drv_register(&s_disp_drv);
}
