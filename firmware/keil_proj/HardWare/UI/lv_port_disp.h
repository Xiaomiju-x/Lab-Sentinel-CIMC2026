/******************************************************************************
 * lv_port_disp.h
 *
 * LVGL 8.3 display port — ST7796 320×480 via EXMC 8080.
 * Single full-screen framebuffer in SDRAM at SDRAM_LVGL_FB1 (0xC0000000).
 ******************************************************************************/

#ifndef __LV_PORT_DISP_H__
#define __LV_PORT_DISP_H__

#include "lvgl.h"

void lv_port_disp_init(void);

#endif /* __LV_PORT_DISP_H__ */
