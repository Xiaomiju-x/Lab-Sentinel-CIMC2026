/* gas_safety.h -- formula-aware furnace gas-evolution safety supervisor
 * ---------------------------------------------------------------------
 * CIMC Lab-Sentinel. Fuses domain chemistry knowledge (distilled from the
 * XRD project raw-material library, see gas_safety_table.h) with the live
 * MQ-135 gas sensor:
 *   given the current batch bill-of-materials + furnace temperature, predict
 *   which hazardous gas should be evolving, then cross-check the sensor.
 *
 * Integer-only, no float, no heap, no libc beyond strcmp -> armcc/M7 safe.
 */
#ifndef GAS_SAFETY_H
#define GAS_SAFETY_H

#include <stdint.h>
#include "gas_safety_table.h"

typedef struct {
    uint8_t     expected_mask; /* bitmask (1<<GAS_*) expected at current temp */
    uint8_t     max_sev;       /* highest severity among active rules (0..3)  */
    uint8_t     n_active;      /* number of active rules                      */
    uint8_t     top_gas;       /* GAS_* of the highest-severity active rule   */
    const char *top_reason;    /* its ASCII rationale (or NULL)               */
} gas_status_t;

/* cross-check outcome codes */
enum {
    GAS_XC_QUIET     = 0, /* nothing expected, sensor flat -> ok               */
    GAS_XC_CONFIRMED = 1, /* expected gas + sensor rise -> ventilate as needed */
    GAS_XC_UNEXPECTED= 2, /* sensor rise with NO chemistry reason -> leak/charge*/
    GAS_XC_SENSORFLAT= 3  /* dangerous gas expected but sensor flat -> sensor? */
};

/* Evaluate the active gas hazards for a batch at a given temperature.
 * batch     : array of ASCII material formula strings present in the charge
 * n_batch   : number of materials
 * temp_c    : current furnace temperature in C
 * out       : result (must be non-NULL) */
void gas_safety_eval(const char *const *batch, int n_batch,
                     int temp_c, gas_status_t *out);

/* Fuse the prediction with the MQ-135 reading.
 * mq_rise : 1 if the gas sensor is elevated above its baseline, else 0.
 * returns one of GAS_XC_*. */
int gas_safety_crosscheck(const gas_status_t *st, int mq_rise);

/* On-chip self-test: returns 1 on PASS, 0 on FAIL (mirrors ai_selftest style). */
int gas_safety_selftest(void);

#endif /* GAS_SAFETY_H */
