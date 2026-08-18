/******************************************************************************
 * ci1302.h
 *
 * CI1302 AI Voice Module Driver — UART Interface
 *
 * Hardware: Yahboom YB-MA851 AI voice interaction module (亚博智能 CI1302)
 *   2026-05-28 走 B 路重烧自定义固件 (Chipintelli ICS 智能语音工坊),
 *   17 个语义标签 (1 唤醒 + 16 命令), Pro2 声学模型 + 自然说.
 *   完整工件清单见 CIMC/docs/ci1302_custom_firmware.md.
 *
 * GD32H759 wiring (UART3, PC10/PC11, AF8) — 2026-05-28 迁到 PC10/PC11 per
 * 模块(2).docx 新接线.
 *   CI1302 RX ← PC10 (UART3_TX, AF8)    — MCU drives, CI1302 listens
 *   CI1302 TX → PC11 (UART3_RX, AF8)    — CI1302 drives, MCU listens
 *
 * Frame protocol (Chipintelli 默认 8-byte, 含 SEQ + TYPE + CHK):
 *   [0xA5][0xFA][SEQ][TYPE][CMD_ID][DATA][CHK][0xFB]
 *
 *   byte 0  = 0xA5     固定帧头 1
 *   byte 1  = 0xFA     固定帧头 2
 *   byte 2  = 0x00     序号 (固定 0, 当前不用)
 *   byte 3  = TYPE     0x81 = 上行 (CI1302→MCU 命令识别/唤醒事件)
 *                      0x82 = 下行 (MCU→CI1302 主动触发播报)
 *   byte 4  = CMD_ID   1-17 整数 (语义标签, 跟 ICS 平台编号一致)
 *   byte 5  = 0x00     数据字节 (当前不用)
 *   byte 6  = CHK      校验:
 *                      上行 (TYPE=81): CHK = (0x20 + CMD_ID) & 0xFF
 *                      下行 (TYPE=82): CHK = (0x21 + CMD_ID) & 0xFF
 *   byte 7  = 0xFB     固定帧尾
 *
 * Module → MCU 事件 (CI1302 识别到唤醒或命令后自动发):
 *   cmd=01: 唤醒 "你好小亚" (CI1302 内部自动播 "我在")
 *   cmd=02-17: 命令识别 (CI1302 自动播报对应 TTS, MCU 只 dispatch action)
 *
 * MCU → Module 主动触发播报:
 *   ci1302_play(cmd_id) 让 CI1302 播报命令 cmd_id 对应的 TTS.
 *   用于 fusion_task 自动检测异常后主动报警, 不必等用户语音命令.
 ******************************************************************************/

#ifndef __CI1302_H__
#define __CI1302_H__

#include "HeaderFiles.h"
#include "FreeRTOS.h"
#include "queue.h"

/* -------------------------------------------------------------------------- */
/* UART hardware                                                              */
/* -------------------------------------------------------------------------- */
#define CI1302_USART            UART3
#define CI1302_USART_IRQn       UART3_IRQn
#define CI1302_RCU_USART        RCU_UART3
#define CI1302_RCU_TX_GPIO      RCU_GPIOC
#define CI1302_RCU_RX_GPIO      RCU_GPIOC
#define CI1302_TX_PORT          GPIOC
#define CI1302_TX_PIN           GPIO_PIN_10   /* PC10, UART3_TX, AF8 */
#define CI1302_RX_PORT          GPIOC
#define CI1302_RX_PIN           GPIO_PIN_11   /* PC11, UART3_RX, AF8 */
#define CI1302_GPIO_AF          GPIO_AF_8
#define CI1302_BAUD             115200U

/* -------------------------------------------------------------------------- */
/* 8-byte frame protocol constants (Chipintelli ICS 平台默认协议)              */
/* -------------------------------------------------------------------------- */
#define CI1302_FRAME_LEN        8U
#define CI1302_HDR0             0xA5U   /* byte 0: 帧头 1                     */
#define CI1302_HDR1             0xFAU   /* byte 1: 帧头 2                     */
#define CI1302_SEQ              0x00U   /* byte 2: 序号                       */
#define CI1302_TYPE_RECOG       0x81U   /* byte 3: 上行 CI1302→MCU            */
#define CI1302_TYPE_SEND        0x82U   /* byte 3: 下行 MCU→CI1302            */
#define CI1302_DATA             0x00U   /* byte 5: 数据                       */
#define CI1302_TAIL             0xFBU   /* byte 7: 帧尾                       */

/* -------------------------------------------------------------------------- */
/* Voice command IDs (整数语义标签, 跟 ICS 平台 1-17 编号完全对齐)            */
/* -------------------------------------------------------------------------- */
#define CI1302_CMD_WAKE          0x01U  /* 你好小亚                            */

#define CI1302_CMD_START         0x02U  /* 开始烧结      → motor FORWARD       */
#define CI1302_CMD_STOP          0x03U  /* 结束烧结      → motor STOP          */
#define CI1302_CMD_PAUSE         0x04U  /* 暂停监测                            */
#define CI1302_CMD_RESUME        0x05U  /* 继续监测                            */
#define CI1302_CMD_EMERGENCY     0x06U  /* 紧急停止      → motor BRAKE+relay   */
#define CI1302_CMD_ACK_ALARM     0x07U  /* 复位报警      → RISK_NORMAL         */

#define CI1302_CMD_QUERY_TEMP    0x08U  /* 查询温度                            */
#define CI1302_CMD_QUERY_HUMI    0x09U  /* 查询湿度                            */
#define CI1302_CMD_QUERY_GAS     0x0AU  /* 查询气体                            */
#define CI1302_CMD_QUERY_STATUS  0x0BU  /* 查询状态                            */

#define CI1302_CMD_FAN_ON        0x0CU  /* 打开风扇      → motor FORWARD       */
#define CI1302_CMD_FAN_OFF       0x0DU  /* 关闭风扇      → motor STOP          */
#define CI1302_CMD_VENT_ON       0x0EU  /* 打开通风      → relay ON            */
#define CI1302_CMD_VENT_OFF      0x0FU  /* 关闭通风      → relay OFF           */

#define CI1302_CMD_TEST_LED      0x10U  /* 测试灯光                            */
#define CI1302_CMD_TEST_ALARM    0x11U  /* 测试报警                            */

/* -------------------------------------------------------------------------- */
/* IPC: raw-byte queue fed from UART3 ISR, consumed by voice_task             */
/* -------------------------------------------------------------------------- */
extern QueueHandle_t xQueue_CI1302Rx;

/* -------------------------------------------------------------------------- */
/* Public API                                                                 */
/* -------------------------------------------------------------------------- */

/* ci1302_init — configure UART3 GPIO/clock/NVIC, create xQueue_CI1302Rx. */
void ci1302_init(void);

/* ci1302_play — send 8-byte downlink frame (TYPE=0x82) to make CI1302 play
 * the TTS associated with cmd_id. Thread-safe (mutex-protected TX).
 * Use for MCU-side proactive broadcast (e.g. fusion_task 检测到 SEVERE 时
 * 主动让 CI1302 播报 "已紧急停止" 不必等用户语音命令). */
void ci1302_play(uint8_t cmd_id);

/* ci1302_checksum_recog — compute byte 6 of an uplink frame (TYPE=0x81). */
static inline uint8_t ci1302_checksum_recog(uint8_t cmd_id) {
    return (uint8_t)((0x20U + cmd_id) & 0xFFU);
}

/* ci1302_checksum_send — compute byte 6 of a downlink frame (TYPE=0x82). */
static inline uint8_t ci1302_checksum_send(uint8_t cmd_id) {
    return (uint8_t)((0x21U + cmd_id) & 0xFFU);
}

#endif /* __CI1302_H__ */

/******************************* End of File *********************************/
