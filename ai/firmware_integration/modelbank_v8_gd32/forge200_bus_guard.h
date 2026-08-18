#ifndef FORGE200_BUS_GUARD_H
#define FORGE200_BUS_GUARD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    FORGE200_BUS_OWNER_SD = 1,
    FORGE200_BUS_OWNER_MAX31856 = 2
} forge200_bus_owner_t;

typedef struct {
    uint32_t sd_acquisitions;
    uint32_t max31856_acquisitions;
    uint32_t timeout_refusals;
    uint32_t collision_refusals;
    uint32_t max_wait_ticks;
    uint32_t max_hold_ticks;
    uint32_t current_owner;
} forge200_bus_metrics_t;

int forge200_bus_guard_init(void);
int forge200_bus_guard_acquire(forge200_bus_owner_t owner, uint32_t timeout_ms);
int forge200_bus_guard_release(forge200_bus_owner_t owner);
void forge200_bus_guard_snapshot(forge200_bus_metrics_t *metrics);
int forge200_inference_guard_acquire(uint32_t timeout_ms);
int forge200_inference_guard_release(void);

#ifdef __cplusplus
}
#endif

#endif
