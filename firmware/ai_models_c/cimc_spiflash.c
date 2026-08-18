// SPDX-License-Identifier: Apache-2.0
/* Copyright 2026 Lab-Sentinel contributors */

/*
 * Minimal SPI-NOR driver for the flash fitted to the Lab-Sentinel final board.
 * The command bytes and status semantics are the standard device data-sheet
 * contract; GPIO and alternate-function values come from the board schematic.
 */

#include "cimc_spiflash.h"
#include "gd32h7xx.h"

#define FLASH_CMD_WRITE_ENABLE  ((uint8_t)0x06U)
#define FLASH_CMD_READ_STATUS   ((uint8_t)0x05U)
#define FLASH_CMD_READ_ID       ((uint8_t)0x9FU)
#define FLASH_CMD_READ          ((uint8_t)0x03U)
#define FLASH_CMD_PAGE_PROGRAM  ((uint8_t)0x02U)
#define FLASH_CMD_SECTOR_ERASE  ((uint8_t)0x20U)
#define FLASH_STATUS_BUSY       ((uint8_t)0x01U)
#define FLASH_PAGE_BYTES        ((uint32_t)256U)
#define FLASH_READY_POLLS       ((uint32_t)12000000U)

#define FLASH_CS_LOW()  gpio_bit_reset(GPIOF, GPIO_PIN_10)
#define FLASH_CS_HIGH() gpio_bit_set(GPIOF, GPIO_PIN_10)

static uint8_t transfer_byte(uint8_t outgoing)
{
    while (spi_i2s_flag_get(SPI4, SPI_FLAG_TP) == RESET) { }
    spi_i2s_data_transmit(SPI4, outgoing);
    spi_master_transfer_start(SPI4, SPI_TRANS_START);
    while (spi_i2s_flag_get(SPI4, SPI_FLAG_RP) == RESET) { }
    return (uint8_t)spi_i2s_data_receive(SPI4);
}

static void send_address(uint32_t address)
{
    (void)transfer_byte((uint8_t)(address >> 16));
    (void)transfer_byte((uint8_t)(address >> 8));
    (void)transfer_byte((uint8_t)address);
}

static uint8_t range_is_valid(uint32_t address, uint32_t length)
{
    return (uint8_t)(address <= CL_SPIFLASH_CAPACITY &&
                     length <= (CL_SPIFLASH_CAPACITY - address));
}

static void write_enable(void)
{
    FLASH_CS_LOW();
    (void)transfer_byte(FLASH_CMD_WRITE_ENABLE);
    FLASH_CS_HIGH();
}

static int wait_until_ready(void)
{
    uint32_t polls = FLASH_READY_POLLS;
    uint8_t status;

    FLASH_CS_LOW();
    (void)transfer_byte(FLASH_CMD_READ_STATUS);
    do {
        status = transfer_byte(0xFFU);
        if ((status & FLASH_STATUS_BUSY) == 0U) {
            FLASH_CS_HIGH();
            return 0;
        }
        --polls;
    } while (polls != 0U);
    FLASH_CS_HIGH();
    return -2;
}

void cl_spiflash_init(void)
{
    spi_parameter_struct configuration;

    rcu_periph_clock_enable(RCU_GPIOH);
    rcu_periph_clock_enable(RCU_GPIOF);
    rcu_periph_clock_enable(RCU_SPI4);
    rcu_spi_clock_config(IDX_SPI4, RCU_SPISRC_APB2);

    gpio_mode_set(GPIOF, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO_PIN_10);
    gpio_output_options_set(GPIOF, GPIO_OTYPE_PP,
                            GPIO_OSPEED_60MHZ, GPIO_PIN_10);
    FLASH_CS_HIGH();

    gpio_af_set(GPIOH, GPIO_AF_5, GPIO_PIN_6);
    gpio_mode_set(GPIOH, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO_PIN_6);
    gpio_output_options_set(GPIOH, GPIO_OTYPE_PP,
                            GPIO_OSPEED_60MHZ, GPIO_PIN_6);

    gpio_af_set(GPIOF, GPIO_AF_5, GPIO_PIN_8 | GPIO_PIN_9);
    gpio_mode_set(GPIOF, GPIO_MODE_AF, GPIO_PUPD_NONE,
                  GPIO_PIN_8 | GPIO_PIN_9);
    gpio_output_options_set(GPIOF, GPIO_OTYPE_PP,
                            GPIO_OSPEED_60MHZ, GPIO_PIN_8 | GPIO_PIN_9);

    spi_i2s_deinit(SPI4);
    spi_struct_para_init(&configuration);
    configuration.trans_mode = SPI_TRANSMODE_FULLDUPLEX;
    configuration.device_mode = SPI_MASTER;
    configuration.data_size = SPI_DATASIZE_8BIT;
    configuration.clock_polarity_phase = SPI_CK_PL_LOW_PH_1EDGE;
    configuration.nss = SPI_NSS_SOFT;
    configuration.prescale = SPI_PSC_16;
    configuration.endian = SPI_ENDIAN_MSB;
    spi_init(SPI4, &configuration);
    spi_byte_access_enable(SPI4);
    spi_nss_internal_high(SPI4);
    spi_current_data_num_config(SPI4, 1U);
    spi_enable(SPI4);
}

uint32_t cl_spiflash_id(void)
{
    uint32_t identifier;

    FLASH_CS_LOW();
    (void)transfer_byte(FLASH_CMD_READ_ID);
    identifier = ((uint32_t)transfer_byte(0xFFU) << 16);
    identifier |= ((uint32_t)transfer_byte(0xFFU) << 8);
    identifier |= (uint32_t)transfer_byte(0xFFU);
    FLASH_CS_HIGH();
    return identifier;
}

int cl_spiflash_read(uint8_t *destination, uint32_t address, uint32_t length)
{
    uint32_t index;

    if (length == 0U) {
        return 0;
    }
    if (destination == (uint8_t *)0 || range_is_valid(address, length) == 0U) {
        return -1;
    }

    FLASH_CS_LOW();
    (void)transfer_byte(FLASH_CMD_READ);
    send_address(address);
    for (index = 0U; index < length; ++index) {
        destination[index] = transfer_byte(0xFFU);
    }
    FLASH_CS_HIGH();
    return 0;
}

int cl_spiflash_erase_range(uint32_t address, uint32_t length)
{
    uint32_t sector;
    uint32_t end;
    int status;

    if (length == 0U) {
        return 0;
    }
    if (range_is_valid(address, length) == 0U) {
        return -1;
    }

    sector = address & ~(CL_SPIFLASH_SECTOR - 1U);
    end = address + length;
    while (sector < end) {
        write_enable();
        FLASH_CS_LOW();
        (void)transfer_byte(FLASH_CMD_SECTOR_ERASE);
        send_address(sector);
        FLASH_CS_HIGH();
        status = wait_until_ready();
        if (status != 0) {
            return status;
        }
        sector += CL_SPIFLASH_SECTOR;
    }
    return 0;
}

int cl_spiflash_write(const uint8_t *source,
                      uint32_t address,
                      uint32_t length)
{
    uint32_t completed = 0U;

    if (length == 0U) {
        return 0;
    }
    if (source == (const uint8_t *)0 || range_is_valid(address, length) == 0U) {
        return -1;
    }

    while (completed < length) {
        uint32_t current_address = address + completed;
        uint32_t chunk = FLASH_PAGE_BYTES -
                         (current_address & (FLASH_PAGE_BYTES - 1U));
        uint32_t index;
        int status;

        if (chunk > (length - completed)) {
            chunk = length - completed;
        }

        write_enable();
        FLASH_CS_LOW();
        (void)transfer_byte(FLASH_CMD_PAGE_PROGRAM);
        send_address(current_address);
        for (index = 0U; index < chunk; ++index) {
            (void)transfer_byte(source[completed + index]);
        }
        FLASH_CS_HIGH();

        status = wait_until_ready();
        if (status != 0) {
            return status;
        }
        completed += chunk;
    }
    return 0;
}
