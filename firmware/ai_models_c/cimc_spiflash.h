// SPDX-License-Identifier: Apache-2.0
/* Copyright 2026 Lab-Sentinel contributors */

#ifndef LAB_SENTINEL_SPI_FLASH_H
#define LAB_SENTINEL_SPI_FLASH_H

#include <stdint.h>

/* Final-board 8 MiB SPI NOR: SPI4 PH6/PF8/PF9, chip-select PF10. */
#define CL_SPIFLASH_CAPACITY ((uint32_t)(8U * 1024U * 1024U))
#define CL_SPIFLASH_SECTOR   ((uint32_t)4096U)

void cl_spiflash_init(void);
uint32_t cl_spiflash_id(void);

int cl_spiflash_read(uint8_t *destination, uint32_t address, uint32_t length);
int cl_spiflash_erase_range(uint32_t address, uint32_t length);
int cl_spiflash_write(const uint8_t *source,
                      uint32_t address,
                      uint32_t length);

#endif /* LAB_SENTINEL_SPI_FLASH_H */
