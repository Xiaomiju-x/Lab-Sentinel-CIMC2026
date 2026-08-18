/******************************************************************************
 * inmp441.h
 *
 * INMP441 MEMS I²S microphone driver — SPI0 (I²S0) master receive + DMA.
 *
 * Pin map (2026-05-15, moved off PD6 conflict with CI1302 USART1_RX):
 *   WS  → PA0   TIMER1_CH0 AF1 — 16 kHz LRCLK square wave (50% duty)
 *   CK  → PA5   SPI0_SCK   AF5 — I²S bit clock (~512 kHz)
 *   SD  → PA7   SPI0_MOSI  AF5 — data FROM mic (input in I²S receive mode)
 *   L/R → PD5   GPIO output LOW — left channel permanently selected
 *
 * NOTE: PA7 also overlaps Ethernet RMII CRS_DV (AF11). Ethernet is downgraded
 * to a Phase 6 optional add-on (eth_task not spawned) so PA7 is free here. If
 * Ethernet is ever re-enabled, move SD to PB5 (SPI0_MOSI AF5) or PD7 first.
 *
 * Sampling: 16 kHz, 16-bit data in 32-bit I²S frame (MSB first, Philips).
 * DMA fills a 512-sample (1024-byte) ping-pong buffer; inmp441_get_sample()
 * returns the latest 16-bit left-channel value.
 *
 * Usage:
 *   inmp441_init();          // once at boot
 *   int16_t s = inmp441_get_sample();  // from any task, non-blocking
 *   uint16_t rms = inmp441_rms_256();  // average RMS over last 256 samples
 ******************************************************************************/

#ifndef __INMP441_H__
#define __INMP441_H__

#include "HeaderFiles.h"

#define INMP441_BUF_SAMPLES   512U   /* circular DMA buffer depth (16-bit words) */

void     inmp441_init(void);
int16_t  inmp441_get_sample(void);   /* latest left-channel sample */
uint16_t inmp441_rms_256(void);      /* RMS of last 256 samples (0..32767) */

#endif /* __INMP441_H__ */
