/* arch/cc.h — lwIP 2.2.0 platform types for GD32H759 Cortex-M7 (ARMCC/Keil) */
#ifndef LWIP_ARCH_CC_H
#define LWIP_ARCH_CC_H

#include <stdint.h>
#include <stddef.h>

/* Basic type definitions required by lwIP */
typedef uint8_t   u8_t;
typedef int8_t    s8_t;
typedef uint16_t  u16_t;
typedef int16_t   s16_t;
typedef uint32_t  u32_t;
typedef int32_t   s32_t;
typedef uintptr_t mem_ptr_t;

/* Byte order — ARM Cortex-M7 is little-endian */
#ifndef BYTE_ORDER
#define BYTE_ORDER  LITTLE_ENDIAN
#endif

/* Packed struct support for ARMCC */
#define PACK_STRUCT_BEGIN
#define PACK_STRUCT_STRUCT  __attribute__((packed))
#define PACK_STRUCT_END
#define PACK_STRUCT_FIELD(x)  x

/* Platform output — routed to nothing; add UART call here if debug needed */
#define LWIP_PLATFORM_DIAG(x)    do { } while(0)
#define LWIP_PLATFORM_ASSERT(x)  do { } while(0)

/* Use standard headers */
#define LWIP_NO_STDINT_H   0
#define LWIP_NO_INTTYPES_H 0
#define LWIP_NO_STDDEF_H   0

#endif /* LWIP_ARCH_CC_H */
