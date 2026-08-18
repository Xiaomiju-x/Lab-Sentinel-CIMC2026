#include "forge200_shared_spi.h"

#include <stddef.h>
#include <string.h>

static void safe_idle(forge200_shared_spi_t *bus)
{
    bus->hooks.deassert_sd_cs_pc5(bus->hooks.context);
    bus->hooks.deassert_max31856_cs_pg3(bus->hooks.context);
    bus->hooks.set_mode(bus->hooks.context, FORGE200_SPI_MODE_SD_MODE0);
}

int forge200_shared_spi_init(forge200_shared_spi_t *bus,
                             const forge200_spi_hooks_t *hooks)
{
    if (bus == NULL || hooks == NULL || hooks->critical_enter == NULL ||
        hooks->critical_exit == NULL || hooks->deassert_sd_cs_pc5 == NULL ||
        hooks->deassert_max31856_cs_pg3 == NULL || hooks->set_mode == NULL) {
        return -1;
    }
    memset(bus, 0, sizeof(*bus));
    bus->hooks = *hooks;
    safe_idle(bus);
    return 0;
}

int forge200_shared_spi_try_acquire(forge200_shared_spi_t *bus,
                                    forge200_spi_owner_t owner)
{
    int result = -1;
    if (bus == NULL || (owner != FORGE200_SPI_OWNER_SD &&
                        owner != FORGE200_SPI_OWNER_MAX31856)) {
        return -1;
    }
    bus->hooks.critical_enter(bus->hooks.context);
    if (bus->owner == FORGE200_SPI_OWNER_NONE) {
        safe_idle(bus);
        bus->hooks.set_mode(bus->hooks.context,
                            owner == FORGE200_SPI_OWNER_SD
                                ? FORGE200_SPI_MODE_SD_MODE0
                                : FORGE200_SPI_MODE_MAX31856_MODE1);
        bus->owner = owner;
        result = 0;
    } else {
        bus->collision_refusals += 1U;
    }
    bus->hooks.critical_exit(bus->hooks.context);
    return result;
}

int forge200_shared_spi_release(forge200_shared_spi_t *bus,
                                forge200_spi_owner_t owner)
{
    int result = -1;
    if (bus == NULL) {
        return -1;
    }
    bus->hooks.critical_enter(bus->hooks.context);
    if (bus->owner == owner && owner != FORGE200_SPI_OWNER_NONE) {
        safe_idle(bus);
        bus->owner = FORGE200_SPI_OWNER_NONE;
        result = 0;
    }
    bus->hooks.critical_exit(bus->hooks.context);
    return result;
}
