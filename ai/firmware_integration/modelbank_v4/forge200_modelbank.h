#ifndef FORGE200_MODELBANK_H
#define FORGE200_MODELBANK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FORGE200_HEADER_BYTES                 256U
#define FORGE200_PACKAGE_BYTES_MAX            7340032UL
#define FORGE200_NANOLM_PACKAGE_BYTES_MAX     2097152UL
#define FORGE200_ARENA_PLUS_KV_BYTES_MAX      2621440UL
#define FORGE200_MODEL_ID_BYTES               32U
#define FORGE200_SHA256_BYTES                 32U

typedef enum {
    FORGE200_OK = 0,
    FORGE200_ERR_ARGUMENT = 1,
    FORGE200_ERR_IO = 2,
    FORGE200_ERR_SCHEMA = 3,
    FORGE200_ERR_AUTHORITY = 4,
    FORGE200_ERR_ENGINE = 5,
    FORGE200_ERR_LIMIT = 6,
    FORGE200_ERR_MODEL_ID = 7,
    FORGE200_ERR_PAYLOAD_SHA = 8,
    FORGE200_ERR_ROLLBACK = 9,
    FORGE200_ERR_GOLDEN = 10,
    FORGE200_ERR_BUSY = 11,
    FORGE200_ERR_COMMIT = 12
} forge200_status_t;

typedef enum {
    FORGE200_EVENT_LOAD_BEGIN = 1024,
    FORGE200_EVENT_SCHEMA_VERIFIED = 1025,
    FORGE200_EVENT_SHA256_VERIFIED = 1026,
    FORGE200_EVENT_GENERATION_VERIFIED = 1027,
    FORGE200_EVENT_GOLDEN_VERIFIED = 1028,
    FORGE200_EVENT_COMMIT = 1029,
    FORGE200_EVENT_ROLLBACK_REFUSE = 1030
} forge200_event_t;

typedef struct {
    uint16_t engine_id;
    uint16_t opset;
    uint8_t flags;
    uint16_t tensor_count;
    uint64_t package_generation;
    uint64_t payload_bytes;
    uint32_t scratch_bytes;
    uint32_t arena_bytes;
    uint32_t kv_bytes;
    char model_id[FORGE200_MODEL_ID_BYTES + 1U];
    uint8_t payload_sha256[FORGE200_SHA256_BYTES];
    uint8_t golden_sha256[FORGE200_SHA256_BYTES];
    uint8_t release_root[FORGE200_SHA256_BYTES];
    uint8_t output_schema_sha256[FORGE200_SHA256_BYTES];
} forge200_package_info_t;

typedef struct {
    void *context;
    uint64_t package_bytes;
    int (*read)(void *context, uint64_t offset, void *destination, uint32_t bytes);
    int (*sha256)(void *context, uint64_t offset, uint64_t bytes,
                  uint8_t output[FORGE200_SHA256_BYTES]);
} forge200_reader_t;

typedef struct {
    void *context;
    int (*try_lock)(void *context);
    void (*unlock)(void *context);
    int (*engine_supported)(void *context, uint16_t engine_id, uint16_t opset);
    int (*golden_check)(void *context, const forge200_package_info_t *package,
                        const uint8_t *payload, uint64_t payload_bytes);
    int (*activate)(void *context, const forge200_package_info_t *package,
                    const uint8_t *payload, uint64_t payload_bytes);
    void (*cache_clean)(void *context, const void *address, uint64_t bytes);
    void (*event)(void *context, forge200_event_t event, forge200_status_t status);
} forge200_runtime_t;

typedef struct {
    uint8_t *slots[2];
    uint64_t slot_capacity[2];
    uint8_t active_slot;
    uint8_t has_active;
    uint8_t load_in_progress;
    uint64_t accepted_catalog_generation;
    uint64_t minimum_package_generation;
    uint64_t successful_commits;
    forge200_package_info_t active_package;
} forge200_modelbank_t;

void forge200_modelbank_init(forge200_modelbank_t *bank,
                             uint8_t *slot_a, uint64_t slot_a_bytes,
                             uint8_t *slot_b, uint64_t slot_b_bytes,
                             uint64_t minimum_package_generation);

forge200_status_t forge200_modelbank_load(
    forge200_modelbank_t *bank,
    const forge200_reader_t *reader,
    const forge200_runtime_t *runtime,
    const char *expected_model_id,
    uint64_t catalog_generation);

#ifdef __cplusplus
}
#endif

#endif
