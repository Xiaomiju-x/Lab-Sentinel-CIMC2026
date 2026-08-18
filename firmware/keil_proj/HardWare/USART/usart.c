// SPDX-License-Identifier: Apache-2.0
/* Copyright 2026 Lab-Sentinel contributors */

#include "usart.h"

void my_usart_init(void)
{
    const uint32_t pins = GPIO_PIN_13 | GPIO_PIN_5;

    rcu_periph_clock_enable(RCU_GPIOB);
    rcu_periph_clock_enable(RCU_UART4);

    /* Board console contract: PB13=UART4_TX and PB5=UART4_RX, AF14. */
    gpio_af_set(GPIOB, GPIO_AF_14, pins);
    gpio_mode_set(GPIOB, GPIO_MODE_AF, GPIO_PUPD_PULLUP, pins);
    gpio_output_options_set(GPIOB, GPIO_OTYPE_PP,
                            GPIO_OSPEED_100_220MHZ, pins);

    usart_deinit(UART4);
    usart_baudrate_set(UART4, 115200U);
    usart_word_length_set(UART4, USART_WL_8BIT);
    usart_stop_bit_set(UART4, USART_STB_1BIT);
    usart_parity_config(UART4, USART_PM_NONE);
    usart_receive_config(UART4, USART_RECEIVE_ENABLE);
    usart_transmit_config(UART4, USART_TRANSMIT_ENABLE);
    usart_enable(UART4);

    /* Normal operation is transmit-only. Provisioning masks this IRQ and polls
     * RBNE directly; otherwise the handler drains unsolicited console bytes. */
    nvic_priority_group_set(NVIC_PRIGROUP_PRE4_SUB0);
    nvic_irq_enable(UART4_IRQn, 3U, 0U);
    usart_interrupt_flag_clear(UART4, USART_INT_FLAG_RBNE);
    usart_interrupt_flag_clear(UART4, USART_INT_FLAG_IDLE);
    usart_interrupt_enable(UART4, USART_INT_RBNE);
    usart_interrupt_enable(UART4, USART_INT_IDLE);
}

void UART4_IRQHandler(void)
{
    if (usart_interrupt_flag_get(UART4, USART_INT_FLAG_RBNE) != RESET) {
        (void)usart_data_receive(UART4);
    }
    if (usart_interrupt_flag_get(UART4, USART_INT_FLAG_IDLE) != RESET) {
        usart_interrupt_flag_clear(UART4, USART_INT_FLAG_IDLE);
    }
}

int fputc(int ch, FILE *stream)
{
    (void)stream;
    while (usart_flag_get(UART4, USART_FLAG_TBE) == RESET) { }
    usart_data_transmit(UART4, (uint8_t)ch);
    return ch;
}
