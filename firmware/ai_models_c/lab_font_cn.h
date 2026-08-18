/* SPDX-License-Identifier: Apache-2.0
 * Public-build fallback for the nano-LM diagnosis font.
 * Generate an OFL CJK font with tools/generate_cjk_font.py and replace this
 * macro when Chinese glyph rendering is required. */
#ifndef LAB_FONT_CN_H
#define LAB_FONT_CN_H
#include "lvgl.h"
#define LAB_PUBLIC_CJK_FONT_FALLBACK 1
#define lab_font_cn16 lv_font_montserrat_14
#endif
