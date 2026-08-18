/******************************************************************************
 * ethernetif.c — GD32H759 ENET0 + LAN8720A RMII netif for lwIP 2.2.0
 *
 * RMII GPIO (all AF11):
 *   PA1 REF_CLK, PA2 MDIO, PA7 CRS_DV
 *   PC0 nRST(GPIO), PC1 MDC, PC4 RXD0, PC5 RXD1
 *   PB11 TX_EN, PB12 TXD0, PB13 TXD1
 *
 * Cache strategy: SCB_CleanDCache before TX, SCB_InvalidateDCache after RX.
 *
 * PHY: LAN8720A (PHY_ADDRESS=1 as set in gd32h7xx_enet.h).
 ******************************************************************************/

#include "ethernetif.h"
#include "lwip/pbuf.h"
#include "lwip/netif.h"
#include "lwip/etharp.h"
#include "netif/ethernet.h"

#include <string.h>

/* PC0 = SDRAM D12 (EXMC AF1) — MUST NOT be reconfigured as GPIO.
 * LAN8720A nRST is NOT driven by MCU; rely on PHY power-on reset + BMCR SW reset. */

/* LAN8720 PHY register indices (BCR/BSR standard IEEE 802.3) */
#define PHY_BCR         0x00U   /* basic control */
#define PHY_BSR         0x01U   /* basic status */
#define PHY_BMCR_RESET  0x8000U
#define PHY_BMCR_ANEG   0x1000U
#define PHY_BMCR_RANEG  0x0200U
#define PHY_BSR_ANEGCPLT 0x0020U
#define PHY_BSR_LINKUP   0x0004U
#define LAN8720_SCR     0x1FU   /* special control/status */
#define LAN8720_SPEED_FD100  0x0018U

/* ---------- DMA descriptor arrays (ENET SPL expects globals) ---------- */
enet_descriptors_struct s_rx_desc[ENET_RXBUF_NUM];
enet_descriptors_struct s_tx_desc[ENET_TXBUF_NUM];

/* Buffers must be 4-byte aligned; placed in D1 AXI SRAM (DMA-accessible).
 * Cache maintenance (SCB_Clean/Invalidate) is used instead of MPU config. */
static uint8_t s_rx_buf[ENET_RXBUF_NUM][ENET_RXBUF_SIZE] __attribute__((aligned(32)));
static uint8_t s_tx_buf[ENET_TXBUF_NUM][ENET_TXBUF_SIZE] __attribute__((aligned(32)));

/* lwIP requires each netif to have a unique MAC. Use locally-administered range. */
static uint8_t s_mac[6] = { 0x02U, 0xCBU, 0x17U, 0x59U, 0x00U, 0x01U };

/* Tracks which TX buffer slot is in use (round-robin, 1 frame at a time). */
static uint8_t s_tx_idx = 0U;

/* ── forward declarations ── */
extern enet_descriptors_struct *dma_current_rxdesc;   /* from gd32h7xx_enet.c */
extern enet_descriptors_struct *dma_current_txdesc;

/* ============================================================================
 * GPIO + RCU init (RMII, AF11)
 * ============================================================================ */
static void _enet_gpio_init(void)
{
    rcu_periph_clock_enable(RCU_GPIOA);
    rcu_periph_clock_enable(RCU_GPIOB);
    rcu_periph_clock_enable(RCU_GPIOC);

    /* PA1=REF_CLK, PA2=MDIO, PA7=CRS_DV — all AF11 */
    gpio_af_set(GPIOA, GPIO_AF_11, GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_7);
    gpio_mode_set(GPIOA, GPIO_MODE_AF, GPIO_PUPD_NONE,
                  GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_7);
    gpio_output_options_set(GPIOA, GPIO_OTYPE_PP, GPIO_OSPEED_100_220MHZ,
                            GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_7);

    /* PC1=MDC, PC4=RXD0, PC5=RXD1 — all AF11 */
    gpio_af_set(GPIOC, GPIO_AF_11, GPIO_PIN_1 | GPIO_PIN_4 | GPIO_PIN_5);
    gpio_mode_set(GPIOC, GPIO_MODE_AF, GPIO_PUPD_NONE,
                  GPIO_PIN_1 | GPIO_PIN_4 | GPIO_PIN_5);
    gpio_output_options_set(GPIOC, GPIO_OTYPE_PP, GPIO_OSPEED_100_220MHZ,
                            GPIO_PIN_1 | GPIO_PIN_4 | GPIO_PIN_5);

    /* PB11=TX_EN, PB12=TXD0, PB13=TXD1 — all AF11 */
    gpio_af_set(GPIOB, GPIO_AF_11, GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_13);
    gpio_mode_set(GPIOB, GPIO_MODE_AF, GPIO_PUPD_NONE,
                  GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_13);
    gpio_output_options_set(GPIOB, GPIO_OTYPE_PP, GPIO_OSPEED_100_220MHZ,
                            GPIO_PIN_11 | GPIO_PIN_12 | GPIO_PIN_13);

    /* PC0 is SDRAM D12 — do NOT touch it here. PHY nRST left floating/high. */
}

/* ============================================================================
 * LAN8720A PHY init and link detection
 * ============================================================================ */
static uint8_t _phy_init(void)
{
    uint16_t val;
    uint32_t timeout;

    /* PHY nRST not driven by MCU (PC0 = SDRAM D12). LAN8720A relies on its
     * own power-on reset. Wait 300 ms for PHY PLL to stabilise before MDIO. */
    {
        volatile uint32_t d = 18000000U; /* ~300 ms @ 600 MHz */
        while (d--) { __NOP(); }
    }

    /* Software reset */
    val = (uint16_t)PHY_BMCR_RESET;
    enet_phy_write_read(ENET0, ENET_PHY_WRITE, PHY_ADDRESS, PHY_BCR, &val);
    timeout = 0x08FFFFU;
    do {
        enet_phy_write_read(ENET0, ENET_PHY_READ, PHY_ADDRESS, PHY_BCR, &val);
        if (--timeout == 0U) return 0U;
    } while (val & PHY_BMCR_RESET);

    /* Enable auto-negotiation + restart */
    val = PHY_BMCR_ANEG | PHY_BMCR_RANEG;
    enet_phy_write_read(ENET0, ENET_PHY_WRITE, PHY_ADDRESS, PHY_BCR, &val);

    /* Wait for auto-negotiation complete (up to 3 s) */
    timeout = 0xFFFFFU;
    do {
        enet_phy_write_read(ENET0, ENET_PHY_READ, PHY_ADDRESS, PHY_BSR, &val);
        if (--timeout == 0U) return 0U;
    } while (!(val & PHY_BSR_ANEGCPLT));

    return 1U;   /* success */
}

uint8_t ethernetif_link_up(void)
{
    uint16_t bsr;
    enet_phy_write_read(ENET0, ENET_PHY_READ, PHY_ADDRESS, PHY_BSR, &bsr);
    return (bsr & PHY_BSR_LINKUP) ? 1U : 0U;
}

/* ============================================================================
 * low_level_init — called from ethernetif_init()
 * ============================================================================ */
static err_t _low_level_init(struct netif *netif)
{
    netif->hwaddr_len = 6U;
    netif->mtu        = 1500U;
    netif->flags      = NETIF_FLAG_BROADCAST | NETIF_FLAG_ETHARP |
                        NETIF_FLAG_LINK_UP;
    memcpy(netif->hwaddr, s_mac, 6U);

    /* RCU */
    rcu_periph_clock_enable(RCU_SYSCFG);
    rcu_periph_clock_enable(RCU_ENET0);
    rcu_periph_clock_enable(RCU_ENET0TX);
    rcu_periph_clock_enable(RCU_ENET0RX);

    _enet_gpio_init();

    /* Select RMII interface */
    syscfg_enet_phy_interface_config(ENET0, SYSCFG_ENET_PHY_RMII);

    /* Peripheral soft-reset */
    if (enet_software_reset(ENET0) != SUCCESS) {
        return ERR_IF;
    }

    /* PHY init (auto-negotiate) */
    if (_phy_init() == 0U) {
        /* Link not up — continue anyway; ethernetif_link_up() can be polled */
    }

    /* ENET MAC init: let SPL handle speed/duplex after auto-negotiate */
    enet_initpara_reset();
    enet_initpara_config(STORE_OPTION,
                         ENET_RX_MODE_STOREFORWARD | ENET_TX_MODE_STOREFORWARD);
    if (enet_init(ENET0, ENET_AUTO_NEGOTIATION,
                  ENET_NO_AUTOCHECKSUM,
                  ENET_BROADCAST_FRAMES_PASS) != SUCCESS) {
        return ERR_IF;
    }

    /* Set MAC address in hardware */
    enet_mac_address_set(ENET0, ENET_MAC_ADDRESS0, s_mac);

    /* DMA descriptor chain init — SPL sets buffer1_addr from s_rx/tx_buf */
    enet_descriptors_chain_init(ENET0, ENET_DMA_TX);
    enet_descriptors_chain_init(ENET0, ENET_DMA_RX);

    /* After chain init, patch buffer1_addr to our static arrays
     * (the SPL default may be NULL; we assign here for safety).            */
    {
        uint32_t i;
        enet_descriptors_struct *d = s_tx_desc;
        for (i = 0U; i < ENET_TXBUF_NUM; i++, d++) {
            d->buffer1_addr = (uint32_t)s_tx_buf[i];
        }
        d = s_rx_desc;
        for (i = 0U; i < ENET_RXBUF_NUM; i++, d++) {
            d->buffer1_addr = (uint32_t)s_rx_buf[i];
            d->status       = ENET_RDES0_DAV;      /* give to DMA */
        }
    }

    enet_enable(ENET0);
    return ERR_OK;
}

/* ============================================================================
 * low_level_output — called by lwIP to send a packet
 * ============================================================================ */
static err_t _low_level_output(struct netif *netif, struct pbuf *p)
{
    struct pbuf *q;
    uint8_t    *buf = s_tx_buf[s_tx_idx];
    uint32_t    len = 0U;

    (void)netif;

    /* Wait until TX descriptor is owned by CPU (DMA finished with it) */
    {
        uint32_t timeout = 0x8FFFFU;
        while ((dma_current_txdesc->status & ENET_TDES0_DAV) &&
               (--timeout != 0U)) { __NOP(); }
        if (timeout == 0U) return ERR_TIMEOUT;
    }

    /* Flatten pbuf chain into single TX buffer */
    for (q = p; q != NULL; q = q->next) {
        if (len + q->len > ENET_TXBUF_SIZE) break;
        memcpy(buf + len, q->payload, q->len);
        len += q->len;
    }

    /* Flush CPU D-cache so DMA reads fresh data */
    SCB_CleanDCache_by_Addr((uint32_t *)buf,
                            (int32_t)((len + 31U) & ~31U));

    /* Hand to ENET DMA */
    if (enet_frame_transmit(ENET0, buf, len) != SUCCESS) {
        return ERR_BUF;
    }

    s_tx_idx = (uint8_t)((s_tx_idx + 1U) % ENET_TXBUF_NUM);
    return ERR_OK;
}

/* ============================================================================
 * ethernetif_input — poll for received frames and deliver to lwIP
 * ============================================================================ */
void ethernetif_input(struct netif *netif)
{
    struct pbuf *p;
    uint32_t framelength;
    struct pbuf *q;
    uint8_t *src;
    uint32_t offset;

    /* Check if CPU owns the descriptor (DAV=0) */
    if (dma_current_rxdesc->status & ENET_RDES0_DAV) {
        return;  /* DMA still owns it — no frame yet */
    }

    /* Error check: drop if error or not a complete single-descriptor frame */
    if ((dma_current_rxdesc->status & ENET_RDES0_ERRS) ||
        !(dma_current_rxdesc->status & ENET_RDES0_LDES) ||
        !(dma_current_rxdesc->status & ENET_RDES0_FDES)) {
        enet_rxframe_drop(ENET0);
        return;
    }

    /* Extract frame length (bits [29:16]) minus 4-byte CRC */
    framelength = (dma_current_rxdesc->status & ENET_RDES0_FRML) >> 16;
    if (framelength > 4U) framelength -= 4U;

    /* Invalidate D-cache for RX buffer before CPU reads */
    SCB_InvalidateDCache_by_Addr(
        (uint32_t *)(dma_current_rxdesc->buffer1_addr),
        (int32_t)((framelength + 31U) & ~31U));

    /* Allocate pbuf from pool */
    p = pbuf_alloc(PBUF_RAW, (u16_t)framelength, PBUF_POOL);
    if (p == NULL) {
        enet_rxframe_drop(ENET0);
        return;
    }

    /* Copy from DMA buffer to pbuf chain */
    src    = (uint8_t *)(dma_current_rxdesc->buffer1_addr);
    offset = 0U;
    for (q = p; q != NULL; q = q->next) {
        memcpy(q->payload, src + offset, q->len);
        offset += q->len;
    }

    /* Release descriptor back to DMA */
    dma_current_rxdesc->status = ENET_RDES0_DAV;
    /* Advance internal DMA pointer — write to RPEN to resume */
    ENET_DMA_RPEN(ENET0) = 0U;

    /* Advance dma_current_rxdesc to next descriptor (chained) */
    if (dma_current_rxdesc->control_buffer_size & ENET_RDES1_RCHM) {
        dma_current_rxdesc = (enet_descriptors_struct *)
                             (dma_current_rxdesc->buffer2_next_desc_addr);
    }

    /* Deliver to lwIP */
    if (netif->input(p, netif) != ERR_OK) {
        pbuf_free(p);
    }
}

/* ============================================================================
 * ethernetif_init — registered with netif_add()
 * ============================================================================ */
err_t ethernetif_init(struct netif *netif)
{
    netif->output     = etharp_output;
    netif->linkoutput = _low_level_output;

#if LWIP_NETIF_HOSTNAME
    netif->hostname = "cimc-sentinel";
#endif

    return _low_level_init(netif);
}

/****************************End*****************************/
