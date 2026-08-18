// SPDX-License-Identifier: Apache-2.0
/* Copyright 2026 Lab-Sentinel contributors */

/*
 * SDRAM bring-up for the Lab-Sentinel GD32H759 board. Pin multiplexing and
 * timing values are derived from the board schematic, the SDRAM data sheet and
 * the GD32H7 EXMC peripheral reference manual. They are board-specific facts;
 * change them only with a matching hardware revision and timing analysis.
 */

#include "sdram.h"
#include "gd32h7xx_exmc.h"

#define SDRAM_READY_TIMEOUT ((uint32_t)0x0000FFFFU)
#define SDRAM_MODE_REGISTER ((uint16_t)0x0230U) /* BL=1, sequential, CAS=3 */

typedef struct {
    rcu_periph_enum clock;
    uint32_t port;
    uint32_t pins;
    uint32_t alternate_function;
} sdram_gpio_group_t;

static const sdram_gpio_group_t s_gpio_groups[] = {
    { RCU_GPIOB, GPIOB, GPIO_PIN_2, GPIO_AF_3 },
    { RCU_GPIOC, GPIOC, GPIO_PIN_0, GPIO_AF_1 },
    { RCU_GPIOD, GPIOD,
      GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_8 | GPIO_PIN_9 |
      GPIO_PIN_10 | GPIO_PIN_14 | GPIO_PIN_15, GPIO_AF_12 },
    { RCU_GPIOE, GPIOE,
      GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_7 | GPIO_PIN_8 | GPIO_PIN_9 |
      GPIO_PIN_10 | GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_14, GPIO_AF_12 },
    { RCU_GPIOF, GPIOF,
      GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3 | GPIO_PIN_4 |
      GPIO_PIN_5 | GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_13 |
      GPIO_PIN_14 | GPIO_PIN_15, GPIO_AF_12 },
    { RCU_GPIOG, GPIOG,
      GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_4 | GPIO_PIN_5 |
      GPIO_PIN_8 | GPIO_PIN_15, GPIO_AF_12 },
    { RCU_GPIOH, GPIOH, GPIO_PIN_2 | GPIO_PIN_3 | GPIO_PIN_5, GPIO_AF_12 }
};

static void configure_gpio(void)
{
    uint32_t index;
    for (index = 0U;
         index < (uint32_t)(sizeof(s_gpio_groups) / sizeof(s_gpio_groups[0]));
         ++index) {
        const sdram_gpio_group_t *group = &s_gpio_groups[index];
        rcu_periph_clock_enable(group->clock);
        gpio_af_set(group->port, group->alternate_function, group->pins);
        gpio_mode_set(group->port, GPIO_MODE_AF, GPIO_PUPD_PULLUP, group->pins);
        gpio_output_options_set(group->port, GPIO_OTYPE_PP,
                                GPIO_OSPEED_85MHZ, group->pins);
    }
}

static uint8_t wait_until_ready(void)
{
    uint32_t remaining = SDRAM_READY_TIMEOUT;
    while (exmc_flag_get(EXMC_SDRAM_DEVICE0,
                         EXMC_SDRAM_FLAG_NREADY) != RESET) {
        if (remaining == 0U) {
            return 0U;
        }
        --remaining;
    }
    return 1U;
}

static uint8_t issue_command(uint32_t command,
                             uint32_t refresh_count,
                             uint16_t mode_register)
{
    exmc_sdram_command_parameter_struct request;

    if (wait_until_ready() == 0U) {
        return 0U;
    }
    request.command = command;
    request.bank_select = EXMC_SDRAM_DEVICE0_SELECT;
    request.auto_refresh_number = refresh_count;
    request.mode_register_content = mode_register;
    exmc_sdram_command_config(&request);
    return 1U;
}

static void fail_stop(void)
{
    for (;;) {
        __NOP();
    }
}

void sdram_init(void)
{
    exmc_sdram_parameter_struct device;
    exmc_sdram_timing_parameter_struct timing;

    rcu_periph_clock_enable(RCU_EXMC);
    configure_gpio();

    exmc_sdram_struct_para_init(&device);

    /* Validated final-board timing at EXMC clock SYSCLK/3. */
    timing.load_mode_register_delay = 2U;
    timing.exit_selfrefresh_delay = 12U;
    timing.row_address_select_delay = 8U;
    timing.auto_refresh_delay = 11U;
    timing.write_recovery_delay = 2U;
    timing.row_precharge_delay = 4U;
    timing.row_to_column_delay = 4U;

    device.sdram_device = EXMC_SDRAM_DEVICE0;
    device.column_address_width = EXMC_SDRAM_COW_ADDRESS_9;
    device.row_address_width = EXMC_SDRAM_ROW_ADDRESS_13;
    device.data_width = EXMC_SDRAM_DATABUS_WIDTH_16B;
    device.internal_bank_number = EXMC_SDRAM_4_INTER_BANK;
    device.cas_latency = EXMC_CAS_LATENCY_3_SDCLK;
    device.write_protection = DISABLE;
    device.sdclock_config = EXMC_SDCLK_PERIODS_3_CK_EXMC;
    device.burst_read_switch = ENABLE;
    device.pipeline_read_delay = EXMC_PIPELINE_DELAY_1_CK_EXMC;
    device.timing = &timing;
    exmc_sdram_init(&device);

    if (issue_command(EXMC_SDRAM_CLOCK_ENABLE,
                      EXMC_SDRAM_AUTO_REFLESH_1_SDCLK, 0U) == 0U) {
        fail_stop();
    }

    /* The SDRAM data sheet requires at least 100 us after clock enable. */
    delay_1ms(1U);

    if (issue_command(EXMC_SDRAM_PRECHARGE_ALL,
                      EXMC_SDRAM_AUTO_REFLESH_1_SDCLK, 0U) == 0U ||
        issue_command(EXMC_SDRAM_AUTO_REFRESH,
                      EXMC_SDRAM_AUTO_REFLESH_8_SDCLK, 0U) == 0U ||
        issue_command(EXMC_SDRAM_LOAD_MODE_REGISTER,
                      EXMC_SDRAM_AUTO_REFLESH_1_SDCLK,
                      SDRAM_MODE_REGISTER) == 0U) {
        fail_stop();
    }

    /* 64 ms / 8192 rows at 200 MHz, minus the controller margin. */
    exmc_sdram_refresh_count_set(1542U);
    if (wait_until_ready() == 0U) {
        fail_stop();
    }
}
