#ifndef VERIPROCESS_BOARD_V9_H
#define VERIPROCESS_BOARD_V9_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t ledger_generation;
    uint32_t ledger_records;
    uint32_t chrono_events;
    uint32_t independent_families;
    uint8_t ds3231_valid;
    uint8_t wal_recovered;
    uint8_t sintergraph_frozen;
    uint8_t authority;
} veriprocess_board_receipt_v9_t;

#define VERIPROCESS_BOARD_OK 0
#define VERIPROCESS_BOARD_POWER_CUT_ARMED 1

int veriprocess_board_selftest_v9(veriprocess_board_receipt_v9_t *receipt);

#ifdef __cplusplus
}
#endif

#endif
