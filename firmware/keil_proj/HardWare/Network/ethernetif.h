/******************************************************************************
 * ethernetif.h — GD32H759 ENET0 + LAN8720A RMII lwIP netif driver
 *
 * Pin map (all ENET0 RMII, AF11):
 *   PA1  → ENET0_REF_CLK  (50 MHz in from LAN8720A REFCLKO)
 *   PA2  → ENET0_MDIO
 *   PA7  → ENET0_CRS_DV
 *   PC1  → ENET0_MDC
 *   PC4  → ENET0_RXD0
 *   PC5  → ENET0_RXD1
 *   PB11 → ENET0_TX_EN
 *   PB12 → ENET0_TXD0
 *   PB13 → ENET0_TXD1
 *   PC0  → LAN8720A nRST (GPIO, active-low reset)
 *
 * NOTE: PA1 was previously smoke sensor DO. Move DO to PB0 (ADC1_CH8).
 *       PB11 was previously soft I2C SDA. Move I2C to PC10/PC11.
 *
 * LAN8720A PHY address: 0x01 (PHY_ADDRESS in gd32h7xx_enet.h)
 ******************************************************************************/
#ifndef __ETHERNETIF_H__
#define __ETHERNETIF_H__

#include "HeaderFiles.h"
#include "lwip/netif.h"
#include "lwip/err.h"

/* Initialise the ENET0 netif. Pass to netif_add(). */
err_t ethernetif_init(struct netif *netif);

/* Poll for received frames and pass to lwIP. Call from eth_task loop. */
void  ethernetif_input(struct netif *netif);

/* Returns 1 if PHY reports link up, 0 otherwise. */
uint8_t ethernetif_link_up(void);

#endif /* __ETHERNETIF_H__ */
