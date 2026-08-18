#include "forge200_modelbank.h"

#include <string.h>

static uint16_t read_u16_le(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t read_u32_le(const uint8_t *p)
{
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t read_u64_le(const uint8_t *p)
{
    return (uint64_t)read_u32_le(p) | ((uint64_t)read_u32_le(p + 4) << 32);
}

static void emit(const forge200_runtime_t *runtime, forge200_event_t event,
                 forge200_status_t status)
{
    if (runtime->event != NULL) {
        runtime->event(runtime->context, event, status);
    }
}

static forge200_status_t parse_header(const uint8_t raw[FORGE200_HEADER_BYTES],
                                      forge200_package_info_t *package)
{
    uint32_t i;
    uint32_t model_length = 0U;
    if (memcmp(raw, "ICMF", 4U) != 0 || read_u16_le(raw + 4U) != 1U ||
        read_u16_le(raw + 6U) != FORGE200_HEADER_BYTES) {
        return FORGE200_ERR_SCHEMA;
    }
    if (raw[12] != 0U) {
        return FORGE200_ERR_AUTHORITY;
    }
    for (i = 204U; i < FORGE200_HEADER_BYTES; ++i) {
        if (raw[i] != 0U) {
            return FORGE200_ERR_SCHEMA;
        }
    }
    while (model_length < FORGE200_MODEL_ID_BYTES && raw[44U + model_length] != 0U) {
        ++model_length;
    }
    if (model_length == 0U || model_length == FORGE200_MODEL_ID_BYTES) {
        return FORGE200_ERR_MODEL_ID;
    }
    memset(package, 0, sizeof(*package));
    package->engine_id = read_u16_le(raw + 8U);
    package->opset = read_u16_le(raw + 10U);
    package->flags = raw[13U];
    package->tensor_count = read_u16_le(raw + 14U);
    package->package_generation = read_u64_le(raw + 16U);
    package->payload_bytes = read_u64_le(raw + 24U);
    package->scratch_bytes = read_u32_le(raw + 32U);
    package->arena_bytes = read_u32_le(raw + 36U);
    package->kv_bytes = read_u32_le(raw + 40U);
    memcpy(package->model_id, raw + 44U, model_length);
    package->model_id[model_length] = '\0';
    memcpy(package->payload_sha256, raw + 76U, FORGE200_SHA256_BYTES);
    memcpy(package->golden_sha256, raw + 108U, FORGE200_SHA256_BYTES);
    memcpy(package->release_root, raw + 140U, FORGE200_SHA256_BYTES);
    memcpy(package->output_schema_sha256, raw + 172U, FORGE200_SHA256_BYTES);
    return FORGE200_OK;
}

static forge200_status_t validate_limits(const forge200_reader_t *reader,
                                         const forge200_package_info_t *package,
                                         const forge200_modelbank_t *bank,
                                         uint8_t target_slot)
{
    uint64_t expected_bytes = (uint64_t)FORGE200_HEADER_BYTES + package->payload_bytes;
    uint64_t working_bytes = (uint64_t)package->arena_bytes + package->kv_bytes;
    if (reader->package_bytes != expected_bytes ||
        reader->package_bytes > FORGE200_PACKAGE_BYTES_MAX ||
        package->payload_bytes > bank->slot_capacity[target_slot] ||
        working_bytes > FORGE200_ARENA_PLUS_KV_BYTES_MAX) {
        return FORGE200_ERR_LIMIT;
    }
    if (package->engine_id == 5U &&
        reader->package_bytes > FORGE200_NANOLM_PACKAGE_BYTES_MAX) {
        return FORGE200_ERR_LIMIT;
    }
    return FORGE200_OK;
}

void forge200_modelbank_init(forge200_modelbank_t *bank,
                             uint8_t *slot_a, uint64_t slot_a_bytes,
                             uint8_t *slot_b, uint64_t slot_b_bytes,
                             uint64_t minimum_package_generation)
{
    if (bank == NULL) {
        return;
    }
    memset(bank, 0, sizeof(*bank));
    bank->slots[0] = slot_a;
    bank->slots[1] = slot_b;
    bank->slot_capacity[0] = slot_a_bytes;
    bank->slot_capacity[1] = slot_b_bytes;
    bank->minimum_package_generation = minimum_package_generation;
}

forge200_status_t forge200_modelbank_load(
    forge200_modelbank_t *bank,
    const forge200_reader_t *reader,
    const forge200_runtime_t *runtime,
    const char *expected_model_id,
    uint64_t catalog_generation)
{
    uint8_t header[FORGE200_HEADER_BYTES];
    uint8_t actual_sha[FORGE200_SHA256_BYTES];
    uint8_t target_slot;
    uint8_t locked = 0U;
    uint8_t owns_load_flag = 0U;
    forge200_package_info_t package;
    forge200_status_t status = FORGE200_ERR_ARGUMENT;

    if (bank == NULL || reader == NULL || runtime == NULL ||
        expected_model_id == NULL || reader->read == NULL || reader->sha256 == NULL ||
        runtime->engine_supported == NULL || runtime->golden_check == NULL ||
        runtime->activate == NULL || bank->slots[0] == NULL || bank->slots[1] == NULL) {
        return FORGE200_ERR_ARGUMENT;
    }
    if (runtime->try_lock != NULL) {
        if (runtime->try_lock(runtime->context) != 0) {
            return FORGE200_ERR_BUSY;
        }
        locked = 1U;
    }
    if (bank->load_in_progress != 0U) {
        status = FORGE200_ERR_BUSY;
        goto done;
    }
    bank->load_in_progress = 1U;
    owns_load_flag = 1U;
    emit(runtime, FORGE200_EVENT_LOAD_BEGIN, FORGE200_OK);
    target_slot = bank->has_active != 0U ? (uint8_t)(bank->active_slot ^ 1U) : 0U;

    if (reader->package_bytes < FORGE200_HEADER_BYTES ||
        reader->read(reader->context, 0U, header, FORGE200_HEADER_BYTES) != 0) {
        status = FORGE200_ERR_IO;
        goto refuse;
    }
    status = parse_header(header, &package);
    if (status != FORGE200_OK) {
        goto refuse;
    }
    if (strcmp(package.model_id, expected_model_id) != 0) {
        status = FORGE200_ERR_MODEL_ID;
        goto refuse;
    }
    if (runtime->engine_supported(runtime->context, package.engine_id,
                                  package.opset) == 0) {
        status = FORGE200_ERR_ENGINE;
        goto refuse;
    }
    status = validate_limits(reader, &package, bank, target_slot);
    if (status != FORGE200_OK) {
        goto refuse;
    }
    emit(runtime, FORGE200_EVENT_SCHEMA_VERIFIED, FORGE200_OK);

    if (reader->sha256(reader->context, FORGE200_HEADER_BYTES,
                       package.payload_bytes, actual_sha) != 0) {
        status = FORGE200_ERR_IO;
        goto refuse;
    }
    if (memcmp(actual_sha, package.payload_sha256, FORGE200_SHA256_BYTES) != 0) {
        status = FORGE200_ERR_PAYLOAD_SHA;
        goto refuse;
    }
    emit(runtime, FORGE200_EVENT_SHA256_VERIFIED, FORGE200_OK);

    if (catalog_generation < bank->accepted_catalog_generation ||
        package.package_generation < bank->minimum_package_generation) {
        status = FORGE200_ERR_ROLLBACK;
        goto refuse;
    }
    emit(runtime, FORGE200_EVENT_GENERATION_VERIFIED, FORGE200_OK);

    if (reader->read(reader->context, FORGE200_HEADER_BYTES,
                     bank->slots[target_slot], (uint32_t)package.payload_bytes) != 0) {
        status = FORGE200_ERR_IO;
        goto refuse;
    }
    if (runtime->cache_clean != NULL) {
        runtime->cache_clean(runtime->context, bank->slots[target_slot],
                             package.payload_bytes);
    }
    if (runtime->golden_check(runtime->context, &package,
                              bank->slots[target_slot], package.payload_bytes) == 0) {
        status = FORGE200_ERR_GOLDEN;
        goto refuse;
    }
    emit(runtime, FORGE200_EVENT_GOLDEN_VERIFIED, FORGE200_OK);

    if (runtime->activate(runtime->context, &package,
                          bank->slots[target_slot], package.payload_bytes) == 0) {
        status = FORGE200_ERR_COMMIT;
        goto refuse;
    }
    bank->active_slot = target_slot;
    bank->has_active = 1U;
    bank->accepted_catalog_generation = catalog_generation;
    bank->successful_commits += 1U;
    bank->active_package = package;
    status = FORGE200_OK;
    emit(runtime, FORGE200_EVENT_COMMIT, status);
    goto done;

refuse:
    emit(runtime, FORGE200_EVENT_ROLLBACK_REFUSE, status);
done:
    if (owns_load_flag != 0U) {
        bank->load_in_progress = 0U;
    }
    if (locked != 0U && runtime->unlock != NULL) {
        runtime->unlock(runtime->context);
    }
    return status;
}
