/******************************************************************************
 * rgb_lcd.c
 *
 * GD32H759 TLI RGB-565 800×480 LCD driver (CIMC 官方主板 + FPC 配套屏).
 *
 * 关键架构:
 *   - PLL2 → 30 MHz TLI pixel clock
 *   - GPIO 全 AF mode (R3-R7/G2-G7/B3-B7 + HSYNC/VSYNC/PCLK/DE)
 *   - Framebuffer 在 SDRAM 0xC0000000, TLI 通过 AHB master 自动 DMA 扫描
 *   - 全程零 CPU 占用 (跟 8080 bit-bang 45ms/全屏 100% CPU 形成对比)
 *
 * 详细 pin 表和时序参数见 rgb_lcd.h.
 ******************************************************************************/

#include "rgb_lcd.h"
#include "sdram.h"

#include <string.h>

/* ============================================================================
 * Backlight (PD11, GPIO out, active HIGH)
 * ============================================================================ */
#define BL_PORT        GPIOD
#define BL_PIN         GPIO_PIN_11

static void _backlight_gpio_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOD);
    gpio_mode_set(BL_PORT, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, BL_PIN);
    gpio_output_options_set(BL_PORT, GPIO_OTYPE_PP, GPIO_OSPEED_12MHZ, BL_PIN);
    gpio_bit_reset(BL_PORT, BL_PIN);   /* OFF initially */
}

void rgb_lcd_set_backlight(uint8_t on)
{
    if (on) {
        gpio_bit_set(BL_PORT, BL_PIN);
    } else {
        gpio_bit_reset(BL_PORT, BL_PIN);
    }
}

/* ============================================================================
 * GPIO AF config — 21 pins across 7 ports
 *
 * 注: AF mapping 跟 CIMC 官方 bsp_rgb_lcd.c 精确一致 (官方板已验证),
 *     不要自己改; AF 值是 GD32H7 silicon-level 固定的 TLI 输出 mux.
 * ============================================================================ */
static void _tli_gpio_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOA);
    rcu_periph_clock_enable(RCU_GPIOB);
    rcu_periph_clock_enable(RCU_GPIOC);
    rcu_periph_clock_enable(RCU_GPIOD);
    rcu_periph_clock_enable(RCU_GPIOE);
    rcu_periph_clock_enable(RCU_GPIOG);
    rcu_periph_clock_enable(RCU_GPIOH);

    /* 控制信号: HSYNC(PE15), VSYNC(PA7), PCLK(PB3), DE(PE13) */
    gpio_af_set(GPIOE, GPIO_AF_10, GPIO_PIN_15);              /* HSYNC */
    gpio_af_set(GPIOA, GPIO_AF_14, GPIO_PIN_7);               /* VSYNC */
    gpio_af_set(GPIOB, GPIO_AF_2,  GPIO_PIN_3);               /* PCLK  */
    gpio_af_set(GPIOE, GPIO_AF_14, GPIO_PIN_13);              /* DE    */

    /* RGB565 data: 16 pins */
    /* Red */
    gpio_af_set(GPIOC, GPIO_AF_14, GPIO_PIN_4);   /* R7 */
    gpio_af_set(GPIOB, GPIO_AF_9,  GPIO_PIN_1);   /* R6 */
    gpio_af_set(GPIOA, GPIO_AF_14, GPIO_PIN_9);   /* R5 */
    gpio_af_set(GPIOH, GPIO_AF_14, GPIO_PIN_10);  /* R4 */
    gpio_af_set(GPIOB, GPIO_AF_9,  GPIO_PIN_0);   /* R3 */
    /* Green */
    gpio_af_set(GPIOD, GPIO_AF_14, GPIO_PIN_3);   /* G7 */
    gpio_af_set(GPIOC, GPIO_AF_14, GPIO_PIN_7);   /* G6 */
    gpio_af_set(GPIOB, GPIO_AF_14, GPIO_PIN_11);  /* G5 */
    gpio_af_set(GPIOH, GPIO_AF_14, GPIO_PIN_4);   /* G4 */
    gpio_af_set(GPIOG, GPIO_AF_9,  GPIO_PIN_10);  /* G3 */
    gpio_af_set(GPIOA, GPIO_AF_14, GPIO_PIN_6);   /* G2 */
    /* Blue */
    gpio_af_set(GPIOB, GPIO_AF_14, GPIO_PIN_9);   /* B7 */
    gpio_af_set(GPIOA, GPIO_AF_14, GPIO_PIN_15);  /* B6 */
    gpio_af_set(GPIOA, GPIO_AF_14, GPIO_PIN_3);   /* B5 */
    gpio_af_set(GPIOG, GPIO_AF_9,  GPIO_PIN_12);  /* B4 */
    gpio_af_set(GPIOA, GPIO_AF_13, GPIO_PIN_8);   /* B3 */

    /* 全部设为 AF push-pull, 60MHz drive */
    #define _CFG_AF(port, pin) do {                                          \
        gpio_mode_set(port, GPIO_MODE_AF, GPIO_PUPD_NONE, pin);              \
        gpio_output_options_set(port, GPIO_OTYPE_PP, GPIO_OSPEED_60MHZ, pin);\
    } while (0)

    _CFG_AF(GPIOA, GPIO_PIN_3  | GPIO_PIN_6  | GPIO_PIN_7  | GPIO_PIN_8  |
                   GPIO_PIN_9  | GPIO_PIN_15);
    _CFG_AF(GPIOB, GPIO_PIN_0  | GPIO_PIN_1  | GPIO_PIN_3  | GPIO_PIN_9  |
                   GPIO_PIN_11);
    _CFG_AF(GPIOC, GPIO_PIN_4  | GPIO_PIN_7);
    _CFG_AF(GPIOD, GPIO_PIN_3);
    _CFG_AF(GPIOE, GPIO_PIN_13 | GPIO_PIN_15);
    _CFG_AF(GPIOG, GPIO_PIN_10 | GPIO_PIN_12);
    _CFG_AF(GPIOH, GPIO_PIN_4  | GPIO_PIN_10);

    #undef _CFG_AF
}

/* ============================================================================
 * PLL2 → TLI clock
 *   25MHz HXTAL / 25 = 1MHz · ×240 = 240MHz · /2 = 120MHz PLL2R · /4 = 30MHz pix
 * ============================================================================ */
static void _tli_clock_init(void)
{
    rcu_pll_input_output_clock_range_config(IDX_PLL2,
                                            RCU_PLL2RNG_1M_2M,
                                            RCU_PLL2VCO_150M_420M);

    /* PSC=25, PLLN=240, PLLP=3, PLLQ=3, PLLR=2 */
    if (rcu_pll2_config(25U, 240U, 3U, 3U, 2U) == ERROR) {
        while (1) { /* PLL2 config failed — clock 异常, halt */ }
    }
    rcu_pll_clock_output_enable(RCU_PLL2R);
    rcu_tli_clock_div_config(RCU_PLL2R_DIV4);

    rcu_osci_on(RCU_PLL2_CK);
    if (rcu_osci_stab_wait(RCU_PLL2_CK) == ERROR) {
        while (1) { /* PLL2 不稳定, halt */ }
    }
}

/* ============================================================================
 * TLI parameter + layer 0 config
 *
 * 时序约定 (datasheet TLI register):
 *   synpsz   = HSP - 1
 *   backpsz  = HSP + HBP - 1
 *   activesz = HSP + HBP + ACTIVE - 1
 *   totalsz  = HSP + HBP + ACTIVE + HFP - 1
 *   (V 同理)
 * ============================================================================ */
static void _tli_engine_init(uint32_t fb_addr)
{
    tli_parameter_struct        tli_p;
    tli_layer_parameter_struct  tli_l;

    rcu_periph_clock_enable(RCU_TLI);

    /* signal polarity */
    tli_p.signalpolarity_hs      = TLI_HSYN_ACTLIVE_LOW;
    tli_p.signalpolarity_vs      = TLI_VSYN_ACTLIVE_LOW;
    tli_p.signalpolarity_de      = TLI_DE_ACTLIVE_LOW;
    tli_p.signalpolarity_pixelck = TLI_PIXEL_CLOCK_TLI;

    /* horizontal/vertical timing */
    tli_p.synpsz_hpsz   = LCD_HSYNC_PULSE - 1U;
    tli_p.synpsz_vpsz   = LCD_VSYNC_PULSE - 1U;
    tli_p.backpsz_hbpsz = LCD_HSYNC_PULSE + LCD_HBACK_PORCH - 1U;
    tli_p.backpsz_vbpsz = LCD_VSYNC_PULSE + LCD_VBACK_PORCH - 1U;
    tli_p.activesz_hasz = LCD_HSYNC_PULSE + LCD_HBACK_PORCH + LCD_WIDTH  - 1U;
    tli_p.activesz_vasz = LCD_VSYNC_PULSE + LCD_VBACK_PORCH + LCD_HEIGHT - 1U;
    tli_p.totalsz_htsz  = LCD_HSYNC_PULSE + LCD_HBACK_PORCH + LCD_WIDTH  + LCD_HFRONT_PORCH - 1U;
    tli_p.totalsz_vtsz  = LCD_VSYNC_PULSE + LCD_VBACK_PORCH + LCD_HEIGHT + LCD_VFRONT_PORCH - 1U;

    /* background colour 黑 (空白处) */
    tli_p.backcolor_red   = 0x00U;
    tli_p.backcolor_green = 0x00U;
    tli_p.backcolor_blue  = 0x00U;
    tli_init(&tli_p);

    /* layer 0: 全屏 active area */
    tli_l.layer_window_leftpos   = LCD_HSYNC_PULSE + LCD_HBACK_PORCH;
    tli_l.layer_window_rightpos  = LCD_HSYNC_PULSE + LCD_HBACK_PORCH + LCD_WIDTH  - 1U;
    tli_l.layer_window_toppos    = LCD_VSYNC_PULSE + LCD_VBACK_PORCH;
    tli_l.layer_window_bottompos = LCD_VSYNC_PULSE + LCD_VBACK_PORCH + LCD_HEIGHT - 1U;
    tli_l.layer_ppf              = LAYER_PPF_RGB565;
    tli_l.layer_sa               = 255U;
    tli_l.layer_default_red      = 0x00U;
    tli_l.layer_default_green    = 0x00U;
    tli_l.layer_default_blue     = 0x00U;
    tli_l.layer_default_alpha    = 0x00U;
    tli_l.layer_acf1             = LAYER_ACF1_SA;
    tli_l.layer_acf2             = LAYER_ACF2_SA;

    tli_l.layer_frame_bufaddr           = fb_addr;
    tli_l.layer_frame_line_length       = (LCD_WIDTH * LCD_BPP) + 3U;
    tli_l.layer_frame_buf_stride_offset = (LCD_WIDTH * LCD_BPP);
    tli_l.layer_frame_total_line_number = LCD_HEIGHT;
    tli_layer_init(LAYER0, &tli_l);

    tli_dither_config(TLI_DITHER_ENABLE);

    tli_layer_enable(LAYER0);
    tli_reload_config(TLI_FRAME_BLANK_RELOAD_EN);
    tli_enable();
}

/* ============================================================================
 * Public init: clear fb + start TLI + light up backlight
 *
 * 调用前 sdram_init() 必须已完成 (TLI 通过 EXMC bus 读 SDRAM, SDRAM 没起就花屏).
 * ============================================================================ */
uint16_t *rgb_lcd_framebuffer(void)
{
    return (uint16_t *)(uintptr_t)SDRAM_LVGL_FB1;
}

uint16_t *rgb_lcd_framebuffer2(void)
{
    return (uint16_t *)(uintptr_t)SDRAM_LVGL_FB2;
}

void rgb_lcd_swap_to(uint32_t fb_addr)
{
    /* 写 layer0 framebuffer 寄存器 (shadow). 下一个 VBlank 由 hardware 真正 reload. */
    TLI_LXFBADDR(LAYER0) = fb_addr;

    /* 清掉上一次的 LCR (layer-config-reloaded) 中断 flag, 然后请求 VBlank reload. */
    tli_interrupt_flag_clear(TLI_INT_FLAG_LCR);
    tli_reload_config(TLI_FRAME_BLANK_RELOAD_EN);

    /* 用 tli_flag_get(TLI_FLAG_LCR) 而不是 tli_interrupt_flag_get(): 后者会顺带
     * 查 TLI_INTEN, 只在中断已使能时才返回 SET. 我们没 enable LCR 中断
     * (不需要 ISR) → tli_interrupt_flag_get 永远 RESET → busy-wait 永远超时.
     * tli_flag_get 直接读 TLI_INTF, 不管中断是否使能, 这才对.
     *
     * 超时 = 20M 周期 ≈ 33ms @ 600MHz, 比 1 帧 (22ms) 多 ~1.5×, 兜底防 hardware
     * 死锁; 正常路径 LCR 会在 VBlank 到来时 (≤22ms) 立即置位, 不会跑满超时. */
    uint32_t timeout = 20000000U;
    while ((tli_flag_get(TLI_FLAG_LCR) == RESET) && (timeout != 0U)) {
        timeout--;
    }
    tli_interrupt_flag_clear(TLI_INT_FLAG_LCR);
}

void rgb_lcd_init(void)
{
    uint32_t fb1 = (uint32_t)SDRAM_LVGL_FB1;
    uint32_t fb2 = (uint32_t)SDRAM_LVGL_FB2;

    _backlight_gpio_init();
    rgb_lcd_set_backlight(0U);   /* 留黑屏直到 LVGL 渲染第一帧 */

    /* zero 两块 framebuffer (避免 SDRAM 残留显示出花斑) */
    memset((void *)(uintptr_t)fb1, 0x00, LCD_FB_BYTES);
    memset((void *)(uintptr_t)fb2, 0x00, LCD_FB_BYTES);
    /* clean D-cache: TLI 不经 D-cache, 写完后必须 writeback 才能被 TLI 看到 */
    SCB_CleanInvalidateDCache_by_Addr((uint32_t *)(uintptr_t)fb1,
                                      (int32_t)LCD_FB_BYTES);
    SCB_CleanInvalidateDCache_by_Addr((uint32_t *)(uintptr_t)fb2,
                                      (int32_t)LCD_FB_BYTES);

    _tli_clock_init();
    _tli_gpio_init();
    _tli_engine_init(fb1);   /* 首帧扫 fb1, LVGL 第一帧渲染到 fb2 */

    /* TLI 跑起来后再开背光, 避免开机闪一帧花屏 */
    rgb_lcd_set_backlight(1U);
}
