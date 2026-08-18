/******************************************************************************
 * flash_provision.c  —  one-time UART provisioning of SPI-flash images. Two
 * independent regions, each with its own magic so either can be (re)provisioned
 * on its own and later boots skip whatever is already present:
 *   - cluster image  @ NLM_CL_SPI_BASE,  magic 'CLM1' @ 0x7F0000
 *   - LM size bank   @ BANK_PROV_BASE,    magic 'LMB1' @ 0x7E0000
 * Protocol (PC = model/nanolm/provision_cluster.py --target cluster|bank):
 *   MCU prints "[prov] READY <total_bytes>"; PC sends the image in 4096-byte
 *   sectors; after each the MCU erases+writes+reads-back-verifies the SPI sector
 *   and replies one byte 'K' (or 'E'); after the last it stamps the magic.
 * MCU-only. Verified on hardware by the user (no host build).
 ******************************************************************************/
#include "flash_provision.h"
#include "cimc_spiflash.h"
#include "nlm_cluster_vocab.h"     /* NLM_CL_SPI_BASE / NLM_CL_STRIDE / NLM_CL_NEXPERT */
#include "nlm_bank_prov.h"         /* BANK_PROV_BASE / BANK_PROV_BYTES (standalone, no vocab) */
#include "gd32h7xx.h"
#include <stdint.h>
#include <string.h>

extern void lab_log(const char *s);     /* lab_sentinel UART4 console */

#define CL_TOTAL     ((uint32_t)NLM_CL_STRIDE * NLM_CL_NEXPERT)
#define CL_MAGIC     0x7F0000u            /* near end of 8MB, clear of both images */
#define BANK_MAGIC   0x7E0000u
#define SEC          4096u

static uint8_t s_sec[SEC];

static int prov_getc(uint32_t spins)
{
    while (spins--) {
        if (usart_flag_get(UART4, USART_FLAG_RBNE) != RESET)
            return (int)(usart_data_receive(UART4) & 0xFF);
    }
    return -1;
}

static void prov_putc(uint8_t c)
{
    while (usart_flag_get(UART4, USART_FLAG_TBE) == RESET) {}
    usart_data_transmit(UART4, c);
}

static int magic_present(uint32_t magic_addr, char a, char b, char c, char d)
{
    uint8_t m[4];
    cl_spiflash_read(m, magic_addr, 4);
    return (m[0] == (uint8_t)a && m[1] == (uint8_t)b && m[2] == (uint8_t)c && m[3] == (uint8_t)d) ? 1 : 0;
}

/* Receive `total` bytes (must be a multiple of SEC) over UART, writing each
 * 4096B sector into SPI flash at spi_base+offset with read-back verify, then
 * stamp the 4-byte magic. Returns 0 ok, <0 timeout/verify error. */
static int prov_image(uint32_t spi_base, uint32_t total, uint32_t magic_addr,
                      const char m4[4], const char *label)
{
    char msg[80];
    uint32_t addr = spi_base, got = 0;
    int first, i, p = 0;

    /* announce: "[prov:<label>] READY <total>" */
    memcpy(msg, "[prov:", 6); p = 6;
    for (i = 0; label[i] && p < 20; i++) msg[p++] = label[i];
    memcpy(msg + p, "] READY ", 8); p += 8;
    { uint32_t t = total; char tmp[12]; int k = 0;
      if (!t) tmp[k++] = '0'; while (t) { tmp[k++] = (char)('0' + t % 10); t /= 10; }
      while (k) msg[p++] = tmp[--k]; }
    msg[p++] = '\r'; msg[p++] = '\n'; msg[p] = 0; lab_log(msg);

    /* Polled UART4 has no FIFO: mask IRQs from the very first byte. The first byte
     * used to be read with IRQs on (for the no-sender check), but any interrupt
     * (e.g. CI1302's UART3 RX, enabled earlier in boot) firing between byte 0 and
     * the receive loop stalls us >1 byte-time and overruns byte 1 of sector 0 ->
     * RX timeout. Mask before byte 0; the no-sender path re-enables and returns,
     * the sender path keeps IRQs masked straight into the sector-0 burst below. */
    __disable_irq();
    first = prov_getc(60000000u);
    if (first < 0) { __enable_irq(); lab_log("[prov] no sender, skip\r\n"); return -1; }

    while (got < total) {
        uint32_t n = 0;
        /* re-assert mask each sector (PRIMASK is absolute; balanced by __enable_irq
         * after the SPI write). For sector 0 it is already masked from above. */
        __disable_irq();
        if (got == 0) { s_sec[0] = (uint8_t)first; n = 1; }
        while (n < SEC) {
            int ch = prov_getc(20000000u);
            if (ch < 0) { __enable_irq(); lab_log("[prov] RX timeout\r\n"); return -2; }
            s_sec[n++] = (uint8_t)ch;
        }
        __enable_irq();
        cl_spiflash_erase_range(addr, SEC);
        cl_spiflash_write(s_sec, addr, SEC);
        { uint8_t v[16]; int bad = 0;
          cl_spiflash_read(v, addr, 16);
          for (i = 0; i < 16; i++) if (v[i] != s_sec[i]) { bad = 1; break; }
          prov_putc(bad ? 'E' : 'K');
          if (bad) { lab_log("[prov] verify FAIL\r\n"); return -3; } }
        addr += SEC; got += SEC;
    }

    { uint8_t m[4]; m[0] = (uint8_t)m4[0]; m[1] = (uint8_t)m4[1]; m[2] = (uint8_t)m4[2]; m[3] = (uint8_t)m4[3];
      cl_spiflash_erase_range(magic_addr, SEC); cl_spiflash_write(m, magic_addr, 4); }
    lab_log("[prov] DONE image written\r\n");
    return 0;
}

/* Magic strings carry a layout VERSION digit. Bumped 1->2 when the 5-expert
 * cluster grew to 7 experts (new stride) and the LM bank moved up to make room:
 * the old 'CLM1'/'LMB1' stamps from the previous provisioning no longer match,
 * so new firmware re-provisions both regions cleanly instead of reading the old
 * 5-expert layout with 7-expert offsets (which would be garbage). */
int flash_cluster_present(void)  { return magic_present(CL_MAGIC, 'C', 'L', 'M', '2'); }
int flash_bank_present(void)     { return magic_present(BANK_MAGIC, 'L', 'M', 'B', '2'); }

int flash_cluster_provision(void)
{
    static const char m[4] = {'C', 'L', 'M', '2'};
    return prov_image(NLM_CL_SPI_BASE, CL_TOTAL, CL_MAGIC, m, "cluster");
}

int flash_bank_provision(void)
{
    static const char m[4] = {'L', 'M', 'B', '2'};
    return prov_image(BANK_PROV_BASE, BANK_PROV_BYTES, BANK_MAGIC, m, "bank");
}
