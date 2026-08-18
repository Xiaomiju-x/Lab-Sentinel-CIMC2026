// SPDX-License-Identifier: Apache-2.0
/* Copyright 2026 Lab-Sentinel contributors */

#ifndef LAB_SENTINEL_SDRAM_H
#define LAB_SENTINEL_SDRAM_H

#include "HeaderFiles.h"

/* Final-board 32 MiB, 16-bit SDRAM on EXMC SDRAM device 0. */
#define SDRAM_BASE_ADDR     ((uint32_t)0xC0000000U)

/* Two RGB565 800 x 480 framebuffers. */
#define SDRAM_LVGL_FB1      ((uint32_t)SDRAM_BASE_ADDR)
#define SDRAM_LVGL_FB1_SZ   ((uint32_t)(800U * 480U * 2U))
#define SDRAM_LVGL_FB2      ((uint32_t)(SDRAM_BASE_ADDR + 0x00300000U))
#define SDRAM_LVGL_FB2_SZ   SDRAM_LVGL_FB1_SZ

/* LVGL heap and fixed scratch regions. Addresses are part of the firmware ABI. */
#define SDRAM_LVGL_POOL     ((uint32_t)(SDRAM_BASE_ADDR + 0x000C0000U))
#define SDRAM_LVGL_POOL_SZ  ((uint32_t)(1024U * 1024U))
#define SDRAM_CAMERA_FB     ((uint32_t)(SDRAM_BASE_ADDR + 0x00200000U))
#define SDRAM_CAMERA_FB_SZ  ((uint32_t)(320U * 240U * 2U))
#define SDRAM_AI_SCRATCH    ((uint32_t)(SDRAM_BASE_ADDR + 0x00400000U))
#define SDRAM_AI_SCRATCH_SZ ((uint32_t)(256U * 1024U))
#define SDRAM_CAM_VIEW      ((uint32_t)(SDRAM_BASE_ADDR + 0x00500000U))
#define SDRAM_CAM_VIEW_SZ   ((uint32_t)(320U * 240U * 2U))

/* Configure the EXMC and initialise the SDRAM command sequence. */
void sdram_init(void);

#endif /* LAB_SENTINEL_SDRAM_H */
