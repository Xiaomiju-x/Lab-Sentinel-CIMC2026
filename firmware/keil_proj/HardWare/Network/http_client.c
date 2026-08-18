/******************************************************************************
 * http_client.c — HTTP POST to a remote telemetry host (<telemetry-host>:<port>)
 *
 * Uses lwIP netconn API (blocking, no socket dependency).
 * A new TCP connection is opened per POST and closed after receiving
 * the response status line. This keeps code simple and avoids keepalive
 * complexity for a low-frequency (every 5 s) report.
 ******************************************************************************/

#include "http_client.h"
#include "lwipopts.h"
#include "lwip/api.h"
#include "lwip/ip_addr.h"
#include <stdio.h>
#include <string.h>

/* Maximum length of assembled request */
#define HTTP_BUF_SIZE   512U

/* ------------------------------------------------------------------ */

uint8_t http_client_post(uint16_t temp_q8, uint16_t humidity_q8,
                         uint16_t mq135_raw, uint16_t vib_rms_mg,
                         uint16_t risk, uint16_t smoke_alarm)
{
    struct netconn *conn;
    struct netbuf  *inbuf;
    char           *recv_data;
    uint16_t        recv_len;
    char            buf[HTTP_BUF_SIZE];
    char            body[200];
    ip_addr_t       server_ip;
    int             body_len, req_len;
    uint8_t         result = 1U;

    /* Build JSON body — Q8.8 → float string */
    body_len = snprintf(body, sizeof(body),
        "{\"temp_c\":%.2f,\"humidity_pct\":%.2f,"
        "\"mq135_adc\":%u,\"vib_rms_mg\":%u,"
        "\"risk\":%u,\"smoke\":%u}",
        (float)(int16_t)temp_q8 / 256.0f,   /* Q8.8 signed → °C */
        (float)humidity_q8      / 256.0f,
        (unsigned)mq135_raw,
        (unsigned)vib_rms_mg,
        (unsigned)risk,
        (unsigned)smoke_alarm);

    if (body_len <= 0) return 1U;

    req_len = snprintf(buf, sizeof(buf),
        "POST /api/cimc_report HTTP/1.1\r\n"
        "Host: " XRD_AI_BRAIN_IP ":%u\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n"
        "%s",
        (unsigned)XRD_AI_BRAIN_PORT, body_len, body);

    if (req_len <= 0 || req_len >= (int)HTTP_BUF_SIZE) return 1U;

    /* Connect */
    conn = netconn_new(NETCONN_TCP);
    if (conn == NULL) return 1U;

    netconn_set_recvtimeout(conn, 3000);  /* 3 s receive timeout */
    netconn_set_sendtimeout(conn, 3000);

    IP4_ADDR(&server_ip,
             XRD_AI_BRAIN_IP[0], XRD_AI_BRAIN_IP[1],
             XRD_AI_BRAIN_IP[2], XRD_AI_BRAIN_IP[3]);

    /* IP4_ADDR macro expects individual octets — use ipaddr_aton instead */
    if (!ipaddr_aton(XRD_AI_BRAIN_IP, &server_ip)) goto cleanup;

    if (netconn_connect(conn, &server_ip, XRD_AI_BRAIN_PORT) != ERR_OK) {
        goto cleanup;
    }

    /* Send request */
    if (netconn_write(conn, buf, (size_t)req_len, NETCONN_COPY) != ERR_OK) {
        goto cleanup;
    }

    /* Read response status line — just check first 12 bytes "HTTP/1.x 2xx" */
    if (netconn_recv(conn, &inbuf) == ERR_OK) {
        netbuf_data(inbuf, (void **)&recv_data, &recv_len);
        /* "HTTP/1.1 200" → bytes 9-11 are "200" */
        if (recv_len >= 12U && recv_data[9] == '2') {
            result = 0U;   /* HTTP 2xx */
        }
        netbuf_delete(inbuf);
    }

cleanup:
    netconn_close(conn);
    netconn_delete(conn);
    return result;
}

/****************************End*****************************/
