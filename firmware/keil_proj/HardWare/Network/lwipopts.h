/******************************************************************************
 * lwipopts.h — lwIP 2.2.0 configuration for CIMC Lab-Sentinel
 *
 * Platform: GD32H759IMK6 @ 600 MHz, FreeRTOS 10.x
 * Stack:    ENET0 + LAN8720A RMII
 * IP mode:  Static (no DHCP)
 ******************************************************************************/
#ifndef __LWIPOPTS_H__
#define __LWIPOPTS_H__

/* ---- RTOS integration ---- */
#define NO_SYS                      0       /* FreeRTOS sys_arch */
#define SYS_LIGHTWEIGHT_PROT        1       /* use sys_arch_protect */
#define LWIP_NETCONN                1
#define LWIP_SOCKET                 0

/* ---- Memory ---- */
#define MEM_ALIGNMENT               4
#define MEM_SIZE                    (12 * 1024)   /* 12 KB lwIP heap */
#define MEMP_NUM_PBUF               16
#define MEMP_NUM_RAW_PCB            2
#define MEMP_NUM_UDP_PCB            2
#define MEMP_NUM_TCP_PCB            4
#define MEMP_NUM_TCP_PCB_LISTEN     2
#define MEMP_NUM_TCP_SEG            16
#define MEMP_NUM_SYS_TIMEOUT        8
#define MEMP_NUM_NETBUF             4
#define MEMP_NUM_NETCONN            6
#define MEMP_NUM_TCPIP_MSG_API      8
#define MEMP_NUM_TCPIP_MSG_INPKT    8
#define PBUF_POOL_SIZE              8
#define PBUF_POOL_BUFSIZE           1524

/* ---- Protocols ---- */
#define LWIP_ARP                    1
#define LWIP_ETHERNET               1
#define LWIP_IPV4                   1
#define LWIP_ICMP                   1
#define LWIP_RAW                    0
#define LWIP_UDP                    0
#define LWIP_TCP                    1
#define LWIP_DHCP                   0       /* static IP */
#define LWIP_AUTOIP                 0
#define LWIP_IGMP                   0
#define LWIP_DNS                    0
#define LWIP_SNMP                   0
#define LWIP_IPV6                   0
#define LWIP_STATS                  0
#define LWIP_LOOPIF_MULTICAST       0
#define LWIP_NETIF_LOOPBACK         0

/* ---- TCP ---- */
#define TCP_WND                     (4 * TCP_MSS)
#define TCP_MSS                     1460
#define TCP_SND_BUF                 (4 * TCP_MSS)
#define TCP_SND_QUEUELEN            ((4 * TCP_SND_BUF) / TCP_MSS)
#define TCP_QUEUE_OOSEQ             0
#define LWIP_TCP_KEEPALIVE          0

/* ---- ARP ---- */
#define ARP_QUEUEING                1
#define ARP_TABLE_SIZE              4

/* ---- Netif ---- */
#define LWIP_NETIF_STATUS_CALLBACK  1
#define LWIP_NETIF_LINK_CALLBACK    1
#define LWIP_NETIF_HOSTNAME         0

/* ---- tcpip thread ---- */
#define TCPIP_THREAD_NAME           "tcpip"
#define TCPIP_THREAD_STACKSIZE      512     /* words */
#define TCPIP_MBOX_SIZE             8
#define DEFAULT_ACCEPTMBOX_SIZE     4
#define DEFAULT_RAW_RECVMBOX_SIZE   4
#define DEFAULT_UDP_RECVMBOX_SIZE   4
#define DEFAULT_TCP_RECVMBOX_SIZE   4
#define DEFAULT_THREAD_STACKSIZE    128     /* words */
#define DEFAULT_THREAD_PRIO         2

/* ---- Socket/netconn timeouts (enables netconn_set_recvtimeout/sendtimeout macros) ---- */
#define LWIP_SO_RCVTIMEO            1
#define LWIP_SO_SNDTIMEO            1

/* ---- Misc ---- */
#define LWIP_PROVIDE_ERRNO          1
#define LWIP_RAND()                 ((u32_t)xTaskGetTickCount())

/* ---- Static IP configuration (edit to match your subnet) ---- */
#define CIMC_IP_ADDR0       192
#define CIMC_IP_ADDR1       168
#define CIMC_IP_ADDR2       1
#define CIMC_IP_ADDR3       100

#define CIMC_NETMASK0       255
#define CIMC_NETMASK1       255
#define CIMC_NETMASK2       255
#define CIMC_NETMASK3       0

#define CIMC_GW_ADDR0       192
#define CIMC_GW_ADDR1       168
#define CIMC_GW_ADDR2       1
#define CIMC_GW_ADDR3       1

/* Remote telemetry host address (fill in your own host:port) */
#define XRD_AI_BRAIN_IP     "YOUR_TELEMETRY_HOST"
#define XRD_AI_BRAIN_PORT   0

/* Modbus TCP port */
#define MODBUS_TCP_PORT     502

#endif /* __LWIPOPTS_H__ */
