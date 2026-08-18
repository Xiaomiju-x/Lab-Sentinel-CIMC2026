/******************************************************************************
 * ci1302.c
 *
 * CI1302 AI Voice Module Driver — UART Interface
 *
 * UART3 (PC10=TX, PC11=RX, AF8, 115200 8N1) — 2026-05-28 迁到 PC10/PC11.
 *
 * 2026-05-28 ★ 重烧自定义固件 (Chipintelli ICS 平台 8 字节协议):
 *   [0xA5][0xFA][SEQ=00][TYPE][CMD][DATA=00][CHK][0xFB]
 *
 *   - ISR enqueues each received byte into xQueue_CI1302Rx (depth=32).
 *   - voice_task in lab_sentinel.c dequeues bytes, runs 8-byte parser
 *     (validates HDR + TYPE + CHK + TAIL, rejects malformed frames).
 *   - ci1302_play(cmd_id) sends an 8-byte downlink frame (TYPE=0x82) to
 *     trigger CI1302 to play the TTS associated with cmd_id.
 *
 ******************************************************************************/

#include "ci1302.h"
#include "semphr.h"

/* -------------------------------------------------------------------------- */
/* Globals                                                                    */
/* -------------------------------------------------------------------------- */
QueueHandle_t        xQueue_CI1302Rx  = NULL;
static SemaphoreHandle_t xMutex_CI1302Tx = NULL;

/* -------------------------------------------------------------------------- */
/* Internal helpers — checksum 算法在 ci1302.h 里作为 static inline 提供.     */
/* -------------------------------------------------------------------------- */

/* -------------------------------------------------------------------------- */
/* ci1302_init                                                                */
/* -------------------------------------------------------------------------- */
void ci1302_init(void)
{
    /* IPC primitives — must be created before NVIC is enabled. */
    xQueue_CI1302Rx  = xQueueCreate(32U, sizeof(uint8_t));
    xMutex_CI1302Tx  = xSemaphoreCreateMutex();

    /* Clocks. */
    rcu_periph_clock_enable(CI1302_RCU_USART);
    rcu_periph_clock_enable(CI1302_RCU_TX_GPIO);   /* GPIOC for PC10/PC11 */

    /* TX pin: PC10, AF8 (UART3_TX), push-pull output.
     * 2026-05-28: GPIOC bank 不跟 SDRAM 控制脚共 bank (SDRAM 控制在 GPIOH PH2/3/5),
     * 不再需要 OSPEED_12MHZ 抑制 bounce. 用标准 OSPEED_60MHZ 即可. */
    gpio_af_set(CI1302_TX_PORT, CI1302_GPIO_AF, CI1302_TX_PIN);
    gpio_mode_set(CI1302_TX_PORT, GPIO_MODE_AF, GPIO_PUPD_NONE, CI1302_TX_PIN);
    gpio_output_options_set(CI1302_TX_PORT, GPIO_OTYPE_PP,
                            GPIO_OSPEED_60MHZ, CI1302_TX_PIN);

    /* RX pin: PC11, AF8 (UART3_RX), pull-up input. */
    gpio_af_set(CI1302_RX_PORT, CI1302_GPIO_AF, CI1302_RX_PIN);
    gpio_mode_set(CI1302_RX_PORT, GPIO_MODE_AF, GPIO_PUPD_PULLUP, CI1302_RX_PIN);

    /* UART3 configuration: 115200 8N1. */
    usart_deinit(CI1302_USART);
    usart_word_length_set(CI1302_USART, USART_WL_8BIT);
    usart_stop_bit_set(CI1302_USART, USART_STB_1BIT);
    usart_parity_config(CI1302_USART, USART_PM_NONE);
    usart_baudrate_set(CI1302_USART, CI1302_BAUD);
    usart_receive_config(CI1302_USART, USART_RECEIVE_ENABLE);
    usart_transmit_config(CI1302_USART, USART_TRANSMIT_ENABLE);
    usart_enable(CI1302_USART);

    /* NVIC — 优先级 14 (最低): 减小对 EXMC SDRAM 时序敏感操作的抢占影响.
     * (priority group PRE4_SUB0 下, 0=最高, 15=最低, FreeRTOS API 限 ≥5).  */
    nvic_irq_enable(CI1302_USART_IRQn, 14U, 0U);
    usart_interrupt_enable(CI1302_USART, USART_INT_RBNE);
}

/* -------------------------------------------------------------------------- */
/* UART3 ISR — called on every received byte                                  */
/* -------------------------------------------------------------------------- */
void UART3_IRQHandler(void)
{
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;

    if (usart_interrupt_flag_get(CI1302_USART, USART_INT_FLAG_RBNE) != RESET) {
        uint8_t b = (uint8_t)usart_data_receive(CI1302_USART);
        if (xQueue_CI1302Rx != NULL) {
            xQueueSendFromISR(xQueue_CI1302Rx, &b, &xHigherPriorityTaskWoken);
        }
    }

    portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
}

/* -------------------------------------------------------------------------- */
/* ci1302_play — 主动让 CI1302 播报 cmd_id 对应的 TTS                          */
/*                                                                            */
/* Chipintelli 8-byte 下行协议 (MCU → Module):                                */
/*   A5 FA 00 82 <cmd_id> 00 <chk> FB                                         */
/*   chk = (0x21 + cmd_id) & 0xFF                                             */
/*                                                                            */
/* CI1302 收到此帧 → 自动播放 ICS 平台里 cmd_id 对应的"播报语句" TTS.         */
/* 例: ci1302_play(CI1302_CMD_EMERGENCY) 让模块播 "已紧急停止".                */
/* -------------------------------------------------------------------------- */
void ci1302_play(uint8_t cmd_id)
{
    uint8_t frame[CI1302_FRAME_LEN] = {
        CI1302_HDR0,
        CI1302_HDR1,
        CI1302_SEQ,
        CI1302_TYPE_SEND,
        cmd_id,
        CI1302_DATA,
        ci1302_checksum_send(cmd_id),
        CI1302_TAIL
    };

    xSemaphoreTake(xMutex_CI1302Tx, portMAX_DELAY);
    for (uint8_t i = 0U; i < CI1302_FRAME_LEN; i++) {
        usart_data_transmit(CI1302_USART, frame[i]);
        while (RESET == usart_flag_get(CI1302_USART, USART_FLAG_TBE)) { }
    }
    xSemaphoreGive(xMutex_CI1302Tx);
}

/******************************* End of File *********************************/
