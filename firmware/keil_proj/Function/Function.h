// SPDX-License-Identifier: Apache-2.0
/* Copyright 2026 Lab-Sentinel contributors */

#ifndef LAB_SENTINEL_FUNCTION_H
#define LAB_SENTINEL_FUNCTION_H

#include "HeaderFiles.h"

/* Initialise the Cortex-M7 cache and the 1 ms system tick. */
void System_Init(void);

/* Enter the application task bootstrap. This function does not return. */
void UsrFunction(void);

#endif /* LAB_SENTINEL_FUNCTION_H */
