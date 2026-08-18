// SPDX-License-Identifier: Apache-2.0
/* Copyright 2026 Lab-Sentinel contributors */

#ifndef LAB_SENTINEL_LED_H
#define LAB_SENTINEL_LED_H

#include "HeaderFiles.h"

/*
 * The development-board LED pins were reassigned on the final prototype
 * (notably PE2 is the 12 V fan enable). The compatibility API is intentionally
 * inert so legacy status calls cannot drive an actuator. Status is presented
 * through the LCD, alarm path and UART instead.
 */
#define LED1_OFF() ((void)0)
#define LED1_ON()  ((void)0)
#define LED2_OFF() ((void)0)
#define LED2_ON()  ((void)0)
#define LED3_OFF() ((void)0)
#define LED3_ON()  ((void)0)

void LED_Init(void);

#endif /* LAB_SENTINEL_LED_H */
