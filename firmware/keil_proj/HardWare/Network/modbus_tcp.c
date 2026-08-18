/******************************************************************************
 * modbus_tcp.c — Modbus TCP server (port 502)
 *
 * Supports FC=0x03 (Read Holding Registers) only.
 * 6 holding registers, updated by sensor tasks via modbus_tcp_update_regs().
 *
 * One FreeRTOS task accepts connections; each connection handled inline
 * (sequential, not parallel) — suitable for one Modbus master (SCADA/HMI).
 ******************************************************************************/

#include "modbus_tcp.h"
#include "lwip/api.h"
#include "lwip/sys.h"
#include "FreeRTOS.h"
#include "semphr.h"
#include <string.h>

/* ---------- shared register bank ---------- */
static uint16_t         s_regs[MB_HR_COUNT];
static SemaphoreHandle_t s_mutex;

void modbus_tcp_update_regs(uint16_t temp_q8, uint16_t humidity_q8,
                             uint16_t mq135_raw, uint16_t vib_rms_mg,
                             uint16_t risk, uint16_t smoke_alarm)
{
    if (s_mutex == NULL) return;
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    s_regs[MB_HR_TEMP]     = temp_q8;
    s_regs[MB_HR_HUMIDITY] = humidity_q8;
    s_regs[MB_HR_MQ135]    = mq135_raw;
    s_regs[MB_HR_VIB_RMS]  = vib_rms_mg;
    s_regs[MB_HR_RISK]     = risk;
    s_regs[MB_HR_SMOKE]    = smoke_alarm;
    xSemaphoreGive(s_mutex);
}

/* ---------- Modbus TCP frame format ----------
 *  Bytes  Field
 *  0-1    Transaction ID   (echoed back)
 *  2-3    Protocol ID      (always 0x0000)
 *  4-5    Length           (byte count following)
 *  6      Unit ID          (echoed back)
 *  7      Function code    (0x03 = Read Holding Registers)
 *  8-9    Start address
 *  10-11  Quantity
 * ------------------------------------------- */

#define MBAP_HDR_LEN   6U   /* Transaction + Protocol + Length fields */
#define PDU_MIN_LEN    2U   /* FC + at least 1 byte of data */
#define BUF_SIZE       260U

/* Build a FC=0x03 response into buf; return total frame length. */
static uint16_t _build_fc03_response(const uint8_t *req, uint16_t tid,
                                     uint8_t uid, uint8_t *buf)
{
    uint16_t start_addr = ((uint16_t)req[0] << 8) | req[1];
    uint16_t quantity   = ((uint16_t)req[2] << 8) | req[3];
    uint8_t  byte_count;
    uint16_t i;

    /* Validate */
    if (quantity == 0U || quantity > MB_HR_COUNT ||
        start_addr + quantity > MB_HR_COUNT) {
        /* Exception response: FC | 0x80, exception code 0x02 */
        byte_count = 2U;
        buf[0]  = (uint8_t)(tid >> 8);
        buf[1]  = (uint8_t)(tid & 0xFFU);
        buf[2]  = 0x00U;
        buf[3]  = 0x00U;
        buf[4]  = 0x00U;
        buf[5]  = (uint8_t)(byte_count + 1U);
        buf[6]  = uid;
        buf[7]  = 0x03U | 0x80U;  /* FC + error flag */
        buf[8]  = 0x02U;           /* Illegal Data Address */
        return 9U;
    }

    byte_count = (uint8_t)(quantity * 2U);

    /* MBAP header */
    buf[0] = (uint8_t)(tid >> 8);
    buf[1] = (uint8_t)(tid & 0xFFU);
    buf[2] = 0x00U;
    buf[3] = 0x00U;
    buf[4] = 0x00U;
    buf[5] = (uint8_t)(byte_count + 2U);   /* uid + fc + data */
    buf[6] = uid;
    buf[7] = 0x03U;
    buf[8] = byte_count;

    /* Register data — big-endian */
    xSemaphoreTake(s_mutex, portMAX_DELAY);
    for (i = 0U; i < quantity; i++) {
        buf[9U + i * 2U]      = (uint8_t)(s_regs[start_addr + i] >> 8);
        buf[9U + i * 2U + 1U] = (uint8_t)(s_regs[start_addr + i] & 0xFFU);
    }
    xSemaphoreGive(s_mutex);

    return (uint16_t)(9U + byte_count);
}

/* Handle one connected client until it closes or errors. */
static void _handle_client(struct netconn *conn)
{
    struct netbuf *inbuf;
    uint8_t       *data;
    uint16_t       len;
    uint8_t        rxbuf[BUF_SIZE];
    uint8_t        txbuf[BUF_SIZE];
    uint16_t       rxlen = 0U;
    uint16_t       resp_len;

    netconn_set_recvtimeout(conn, 5000);   /* 5 s idle timeout */

    for (;;) {
        err_t err = netconn_recv(conn, &inbuf);
        if (err != ERR_OK) break;

        /* Accumulate fragments into rxbuf */
        do {
            netbuf_data(inbuf, (void **)&data, &len);
            if (rxlen + len < BUF_SIZE) {
                memcpy(rxbuf + rxlen, data, len);
                rxlen = (uint16_t)(rxlen + len);
            }
        } while (netbuf_next(inbuf) >= 0);
        netbuf_delete(inbuf);

        /* Need at least MBAP (6) + UID (1) + FC (1) + 4 bytes PDU = 12 */
        if (rxlen < 12U) { rxlen = 0U; continue; }

        uint16_t tid = ((uint16_t)rxbuf[0] << 8) | rxbuf[1];
        uint8_t  uid = rxbuf[6];
        uint8_t  fc  = rxbuf[7];

        if (fc == 0x03U) {
            resp_len = _build_fc03_response(&rxbuf[8], tid, uid, txbuf);
            netconn_write(conn, txbuf, resp_len, NETCONN_COPY);
        }
        rxlen = 0U;
    }
}

/* ---------- server task ---------- */
#define MODBUS_TASK_STACK  256U   /* words */
#define MODBUS_TASK_PRIO   3U

static void _modbus_server_task(void *pv)
{
    struct netconn *listener, *client;
    err_t err;

    (void)pv;

    listener = netconn_new(NETCONN_TCP);
    if (listener == NULL) { vTaskDelete(NULL); return; }

    netconn_bind(listener, IP_ADDR_ANY, MODBUS_TCP_PORT);
    netconn_listen(listener);

    for (;;) {
        err = netconn_accept(listener, &client);
        if (err == ERR_OK) {
            _handle_client(client);
            netconn_close(client);
            netconn_delete(client);
        }
    }
}

void modbus_tcp_server_start(void)
{
    s_mutex = xSemaphoreCreateMutex();
    sys_thread_new("modbus", _modbus_server_task, NULL,
                   MODBUS_TASK_STACK, MODBUS_TASK_PRIO);
}

/****************************End*****************************/
