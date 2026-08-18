#ifndef FORGE200_SHARED_SPI_H
#define FORGE200_SHARED_SPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FORGE200_SPI_OWNER_NONE = 0,
    FORGE200_SPI_OWNER_SD = 1,
    FORGE200_SPI_OWNER_MAX31856 = 2
} forge200_spi_owner_t;

typedef enum {
    FORGE200_SPI_MODE_SD_MODE0 = 0,
    FORGE200_SPI_MODE_MAX31856_MODE1 = 1
} forge200_spi_mode_t;

typedef struct {
    void *context;
    void (*critical_enter)(void *context);
    void (*critical_exit)(void *context);
    void (*deassert_sd_cs_pc5)(void *context);
    void (*deassert_max31856_cs_pg3)(void *context);
    void (*set_mode)(void *context, forge200_spi_mode_t mode);
} forge200_spi_hooks_t;

typedef struct {
    volatile forge200_spi_owner_t owner;
    volatile uint32_t collision_refusals;
    forge200_spi_hooks_t hooks;
} forge200_shared_spi_t;

int forge200_shared_spi_init(forge200_shared_spi_t *bus,
                             const forge200_spi_hooks_t *hooks);
int forge200_shared_spi_try_acquire(forge200_shared_spi_t *bus,
                                    forge200_spi_owner_t owner);
int forge200_shared_spi_release(forge200_shared_spi_t *bus,
                                forge200_spi_owner_t owner);

#ifdef __cplusplus
}
#endif

#endif
