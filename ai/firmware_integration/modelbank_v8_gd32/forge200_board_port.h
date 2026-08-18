#ifndef FORGE200_BOARD_PORT_H
#define FORGE200_BOARD_PORT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runs the read-only authority=0 unified acceptance workload, then returns. */
int forge200_board_acceptance_run(void);
void forge200_board_control_tick(uint32_t tick);

#ifdef __cplusplus
}
#endif

#endif
