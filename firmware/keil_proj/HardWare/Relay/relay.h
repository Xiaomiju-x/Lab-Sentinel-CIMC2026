/******************************************************************************
 * relay.h
 *
 * Single-channel relay driver.
 *   IN → PA1   (MCU output, active HIGH — relay closes on HIGH)
 *                Moved from PE3 on 2026-05-16 to free PE3 for DCMI_PCLK.
 *   VCC → 5V, GND → GND  (relay module power separate from MCU)
 *
 * Used by fusion_task to activate ventilation/exhaust when smoke is detected.
 ******************************************************************************/

#ifndef __RELAY_H__
#define __RELAY_H__

#include "HeaderFiles.h"

void relay_init(void);
void relay_on(void);    /* close relay (activate load) */
void relay_off(void);   /* open  relay */
uint8_t relay_state(void);  /* returns 1 if currently ON */

/* ---- Second relay channel: the PTC HEATING PLATE (real "furnace" for the
 *      bench closed-loop demo). IN -> PD12, active HIGH. A SEPARATE relay/module
 *      from the PA1 ventilation fan above. The heater is energised while a sinter
 *      batch runs; the safety supervisor cuts it the instant the real MAX31855
 *      thermocouple crosses the over-temp limit (see ctrl_task / TC_PROBE_TRIP_C)
 *      — a genuine sensor -> actuator -> safety closed loop on real hardware,
 *      no 1500C furnace needed. Default OFF (safe).
 *      ⚠ confirm PD12 is free of the RGB-TLI pin map on your carrier board
 *        (PD12 is NOT an SDRAM pin; GT911 touch moved off PD12 to PD5/PD7).      */
void heater_init(void);
void heater_on(void);   /* PD12 HIGH: energise the PTC plate  */
void heater_off(void);  /* PD12 LOW : cut the heater (safe)   */
uint8_t heater_state(void);

#endif /* __RELAY_H__ */
