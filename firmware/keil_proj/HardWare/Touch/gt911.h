/******************************************************************************
 * gt911.h
 *
 * GT911 capacitive touch controller driver — soft I2C (bit-bang).
 *
 * Hardware connections (CIMC official board + FPC RGB 800x480 panel, 2026-05-30):
 *   SCL  → PD5   GPIO   (was PD12 on the retired 8080 NT35510 panel)
 *   SDA  → PD7   GPIO   (was PD13)
 *   INT  → PH15  GPIO   (was PB8 — that conflicted with OV5640 D6; the official
 *                        panel routes INT to PH15, so camera + touch coexist)
 *   RST  → PH13  GPIO   (the RGB panel has no controller/RST; GT911 RST is its
 *                        own pin — freed when CI1302 moved to PC10/PC11)
 *
 * Pinout per official LCD接线 docx + 36_Touch_driver/bsp_touch_cap.c. None of
 * PD5/PD7/PH13/PH15 touch SDRAM / RGB-TLI / OV5640 lines (verified safe).
 *
 * gt911_init() drives a RST pulse with INT held to latch the I2C address, then
 * auto-probes 0x5D first, falls back to 0x14. Max touch points: 5.
 ******************************************************************************/

#ifndef __GT911_H__
#define __GT911_H__

#include "HeaderFiles.h"

/* ---------- max touch points ---------- */
#define GT911_MAX_TOUCH     5U

/* ---------- one touch point ---------- */
typedef struct {
    uint16_t x;
    uint16_t y;
    uint8_t  valid;   /* 1 = this slot is active */
} gt911_point_t;

/* ---------- public API ---------- */
uint8_t gt911_init(void);   /* 0 = GT911 present (probe OK), 1 = not found */
uint8_t gt911_scan(gt911_point_t *pts, uint8_t max_pts);  /* returns point count */

/* Diagnostic: raw product-ID bytes read at each candidate address during init.
 * 00=NACK/no answer, FF=bus stuck high (no pull-up/not present), "911"=alive. */
void    gt911_debug_ids(uint8_t out5d[4], uint8_t out14[4]);

#endif /* __GT911_H__ */
