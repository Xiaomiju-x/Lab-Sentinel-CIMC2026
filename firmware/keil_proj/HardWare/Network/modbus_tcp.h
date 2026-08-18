/******************************************************************************
 * modbus_tcp.h — Modbus TCP server (port 502) for CIMC Lab-Sentinel
 *
 * Exposes 6 read-only holding registers (FC=0x03):
 *   HR0  temperature     Q8.8  (°C × 256, e.g. 25.5°C → 0x1980)
 *   HR1  humidity        Q8.8  (% × 256)
 *   HR2  MQ-135 ADC      raw 12-bit (0–4095)
 *   HR3  vibration RMS   mg (10-bit, ~0–1000)
 *   HR4  risk level      0–4  (R1 verdict grade)
 *   HR5  smoke alarm     0 or 1
 *
 * Call modbus_tcp_server_start() from a FreeRTOS task after lwIP is up.
 * Update registers anytime with modbus_tcp_update_regs().
 ******************************************************************************/
#ifndef __MODBUS_TCP_H__
#define __MODBUS_TCP_H__

#include <stdint.h>

/* Index constants for the 6 holding registers */
#define MB_HR_TEMP      0U
#define MB_HR_HUMIDITY  1U
#define MB_HR_MQ135     2U
#define MB_HR_VIB_RMS   3U
#define MB_HR_RISK      4U
#define MB_HR_SMOKE     5U
#define MB_HR_COUNT     6U

/* Update the shared register bank (safe to call from any task). */
void modbus_tcp_update_regs(uint16_t temp_q8, uint16_t humidity_q8,
                             uint16_t mq135_raw, uint16_t vib_rms_mg,
                             uint16_t risk, uint16_t smoke_alarm);

/* Start the Modbus TCP listener task (call once after tcpip_init). */
void modbus_tcp_server_start(void);

#endif /* __MODBUS_TCP_H__ */
