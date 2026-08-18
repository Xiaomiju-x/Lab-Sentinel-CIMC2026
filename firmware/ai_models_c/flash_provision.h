/******************************************************************************
 * flash_provision.h  —  one-time provisioning of the edge-LLM-cluster image into
 * the 8MB SPI flash, over the UART4 console (CH340). Run ONCE after flashing the
 * firmware: the PC sends cluster_image.bin (model/nanolm/provision_cluster.py),
 * the MCU writes it to SPI flash and stamps a magic so later boots skip this.
 *
 * Boot flow (lab_sentinel): if cluster image absent AND a PC is sending within a
 * short window -> provision; else skip (cluster disabled, board boots normally).
 ******************************************************************************/
#ifndef FLASH_PROVISION_H
#define FLASH_PROVISION_H

/* 1 if a valid cluster image (magic) is already in SPI flash. */
int flash_cluster_present(void);

/* Blocking UART provisioning. Returns 0 on success (image written + magic set),
 * <0 on timeout/error. Prints progress via lab_log (the public UART console). */
int flash_cluster_provision(void);

/* Same, for the LM size bank (s0p6/m1p35) at BANK_PROV_BASE, magic 'LMB1'. */
int flash_bank_present(void);
int flash_bank_provision(void);

#endif
