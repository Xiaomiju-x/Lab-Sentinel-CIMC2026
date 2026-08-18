/******************************************************************************
 * http_client.h — HTTP POST sensor data to a remote telemetry host
 *
 * Target: http://<telemetry-host>:<port>/api/cimc_report
 * Method: POST, Content-Type: application/json
 * Body:   {"temp_c":25.5,"humidity_pct":60.0,"mq135_adc":1234,
 *          "vib_rms_mg":12,"risk":1,"smoke":0}
 *
 * Call http_client_post() from eth_task after link is up.
 * Returns 0 on success (HTTP 2xx), non-zero on error.
 ******************************************************************************/
#ifndef __HTTP_CLIENT_H__
#define __HTTP_CLIENT_H__

#include <stdint.h>

/* Post one JSON reading to XRD AI brain.
 * temp_q8 / humidity_q8: Q8.8 fixed-point (value = raw / 256.0)
 * Returns 0 on HTTP 200-299, 1 on any error. */
uint8_t http_client_post(uint16_t temp_q8, uint16_t humidity_q8,
                         uint16_t mq135_raw, uint16_t vib_rms_mg,
                         uint16_t risk, uint16_t smoke_alarm);

#endif /* __HTTP_CLIENT_H__ */
