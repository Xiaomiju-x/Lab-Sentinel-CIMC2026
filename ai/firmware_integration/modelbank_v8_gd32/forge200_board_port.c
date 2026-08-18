#include "forge200_board_port.h"

#include "HeaderFiles.h"
#include "FreeRTOS.h"
#include "task.h"
#include "FatFs/ff.h"
#include "lab_sentinel.h"
#include "max31856.h"
#include "sd_spi.h"
#include "sha256.h"
#include "forge200_bus_guard.h"
#include "forge200_modelbank.h"
#include "forge200_runtime_v8.h"
#include "forge200_rag_board_v9.h"
#include "veriprocess_board_v9.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define F2_CATALOG_HEADER_BYTES 128U
#define F2_CATALOG_ENTRY_BYTES 160U
#define F2_MODEL_COUNT 170U
#define F2_CATALOG_PATH_A "0:/F200/CATALOGA.BIN"
#define F2_CATALOG_PATH_B "0:/F200/CATALOGB.BIN"
#define F2_SLOT_BYTES (0x740000UL)
#define F2_SLOT_A ((uint8_t *)0xC0F80000UL)
#define F2_SLOT_B ((uint8_t *)0xC16C0000UL)
#define F2_WORKSPACE ((float *)0xC0C00000UL)
#define F2_WORKSPACE_ELEMS (0x280000UL / 4UL)
#define F2_IO_BUFFER ((uint8_t *)0xC0F00000UL)
#define F2_IO_BUFFER_BYTES 0x00020000UL
#define F2_OUTPUT_BUFFER ((uint8_t *)0xC0F20000UL)
#define F2_OUTPUT_BUFFER_BYTES 0x00020000UL
#define F2_GOLDEN_MAX_BYTES F2_IO_BUFFER_BYTES
#define F2_SEQUENTIAL_BENCH_BYTES (64UL * 1024UL * 1024UL)
#define F2_RANDOM_PAGE_BYTES 4096U
#define F2_RANDOM_PAGE_COUNT 256U
#define F2_MIN_SD_KIB_PER_S 512U
#define F2_TOTAL_SWAP_LOADS 1000U
#define F2_SOAK_HOURS 24U
#define F2_SOAK_PERIOD_MS (5UL * 60UL * 1000UL)
#define F2_SOAK_FAULT_PERIOD_CYCLES 24U
#define F2_TIMING_BUCKET_MS 5U
#define F2_TIMING_BUCKETS 201U
#define F2_HEAP_MIN_BYTES (16UL * 1024UL)
#define F2_STACK_MIN_BYTES 1536UL
#define F2_WORKSPACE_CANARY_ELEMS 8U
#define F2_LOAD_BUCKET_MS 100U
#define F2_LOAD_BUCKETS 601U
#define F2_RAG_QUERY_COUNT 120U

typedef struct {
    char model_id[33];
    char package_path[25];
    char golden_path[25];
    uint8_t category;
    uint8_t tier;
    uint16_t engine_id;
    uint16_t opset;
    uint64_t package_bytes;
    uint8_t package_sha256[32];
    uint8_t golden_sha256[32];
} f2_catalog_entry_t;

typedef struct {
    char path[24];
    uint64_t generation;
    uint32_t entry_count;
    uint8_t valid;
} f2_catalog_t;

typedef struct {
    FIL *package_file;
    char golden_path[25];
    uint8_t expected_golden_sha256[32];
    uint32_t last_golden_cycles;
    uint32_t last_output_elems;
    f2rt_model_t active_model;
} f2_runtime_context_t;

static FATFS s_fatfs;
static FIL s_catalog_file;
static FIL s_package_file;
static FIL s_golden_file;
static forge200_modelbank_t s_bank;
static f2_runtime_context_t s_runtime_context;
static uint16_t s_load_counts[F2_MODEL_COUNT];
static uint16_t s_swap_load_counts[F2_MODEL_COUNT];
static uint16_t s_swap_load_ms_bins[F2_LOAD_BUCKETS];
static uint32_t s_total_loads;
static uint32_t s_total_failures;
static uint32_t s_last_load_ms;
static uint32_t s_swap_load_max_ms;
static volatile uint8_t s_timing_phase;
static volatile uint32_t s_timing_last_tick;
static volatile uint32_t s_timing_samples;
static volatile uint32_t s_timing_bins[F2_TIMING_BUCKETS];
static TaskStatus_t s_task_status[24];
static uint32_t s_rag_stage_ms[F2RAG_STAGE_COUNT][F2_RAG_QUERY_COUNT];
static uint32_t s_rag_total_ms[F2_RAG_QUERY_COUNT];
static FRESULT s_file_sha_last_fr;
static UINT s_file_sha_last_got;
static uint32_t s_file_sha_last_chunk;
static uint64_t s_file_sha_last_remaining;

static const uint8_t s_release_content_root[32] = {
    0x0cU, 0x95U, 0xb4U, 0x32U, 0x64U, 0x36U, 0xa7U, 0xddU,
    0xcdU, 0x91U, 0x8aU, 0xc5U, 0xecU, 0xaeU, 0xfeU, 0x5eU,
    0x2cU, 0x16U, 0x56U, 0xf6U, 0x2bU, 0xbdU, 0x97U, 0x22U,
    0x20U, 0x29U, 0x73U, 0xa6U, 0xddU, 0x57U, 0x59U, 0x2eU
};

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
    return (uint64_t)read_u32_le(p) |
           ((uint64_t)read_u32_le(p + 4U) << 32);
}

void forge200_board_control_tick(uint32_t tick)
{
    uint32_t elapsed_ms;
    uint32_t bucket;
    if (s_timing_phase == 0U) {
        return;
    }
    taskENTER_CRITICAL();
    if (s_timing_last_tick != 0U) {
        elapsed_ms = (tick - s_timing_last_tick) * portTICK_PERIOD_MS;
        bucket = elapsed_ms / F2_TIMING_BUCKET_MS;
        if (bucket >= F2_TIMING_BUCKETS) {
            bucket = F2_TIMING_BUCKETS - 1U;
        }
        s_timing_bins[bucket]++;
        s_timing_samples++;
    }
    s_timing_last_tick = tick;
    taskEXIT_CRITICAL();
}

static void timing_begin(uint8_t phase)
{
    taskENTER_CRITICAL();
    memset((void *)s_timing_bins, 0, sizeof(s_timing_bins));
    s_timing_samples = 0U;
    s_timing_last_tick = 0U;
    s_timing_phase = phase;
    taskEXIT_CRITICAL();
}

static uint32_t timing_p99_ms(uint32_t *samples)
{
    uint32_t local_samples;
    uint32_t threshold;
    uint32_t cumulative = 0U;
    uint32_t bucket;
    taskENTER_CRITICAL();
    local_samples = s_timing_samples;
    threshold = (local_samples * 99U + 99U) / 100U;
    for (bucket = 0U; bucket < F2_TIMING_BUCKETS; ++bucket) {
        cumulative += s_timing_bins[bucket];
        if (cumulative >= threshold && threshold != 0U) {
            break;
        }
    }
    taskEXIT_CRITICAL();
    if (samples != NULL) *samples = local_samples;
    if (bucket >= F2_TIMING_BUCKETS) bucket = F2_TIMING_BUCKETS - 1U;
    return bucket * F2_TIMING_BUCKET_MS;
}

static int resource_snapshot(uint32_t *heap_min_bytes,
                             uint32_t *critical_stack_min_bytes)
{
    UBaseType_t count;
    UBaseType_t i;
    uint32_t minimum = 0xFFFFFFFFUL;
    uint32_t found = 0U;
    static const char *const critical[] = {
        "init", "sensor", "fusion", "env", "ctrl", "wdg"
    };
    uint32_t name_index;
    *heap_min_bytes = (uint32_t)xPortGetMinimumEverFreeHeapSize();
    count = uxTaskGetSystemState(
        s_task_status,
        (UBaseType_t)(sizeof(s_task_status) / sizeof(s_task_status[0])),
        NULL);
    for (i = 0U; i < count; ++i) {
        for (name_index = 0U;
             name_index < sizeof(critical) / sizeof(critical[0]);
             ++name_index) {
            if (strcmp(s_task_status[i].pcTaskName, critical[name_index]) == 0) {
                uint32_t bytes =
                    (uint32_t)s_task_status[i].usStackHighWaterMark *
                    sizeof(StackType_t);
                if (bytes < minimum) minimum = bytes;
                found++;
                break;
            }
        }
    }
    *critical_stack_min_bytes = minimum == 0xFFFFFFFFUL ? 0U : minimum;
    return *heap_min_bytes >= F2_HEAP_MIN_BYTES &&
           *critical_stack_min_bytes >= F2_STACK_MIN_BYTES &&
           found == sizeof(critical) / sizeof(critical[0]);
}

static void f2_log(const char *event, const char *detail)
{
    char line[512];
    int n = snprintf(
        line, sizeof(line),
        "@F2BOARD|v=1|event=%s|%s|authority=0|control=unchanged\r\n",
        event, detail != NULL ? detail : "detail=none");
    if (n > 0 && n < (int)sizeof(line)) {
        lab_log(line);
    }
}

static int file_read_exact(FIL *file, uint64_t offset, void *destination,
                           uint32_t bytes)
{
    UINT got = 0U;
    if (file == NULL || destination == NULL ||
        f_lseek(file, (FSIZE_t)offset) != FR_OK ||
        f_read(file, destination, bytes, &got) != FR_OK ||
        got != bytes) {
        return -1;
    }
    return 0;
}

static int file_sha256(FIL *file, uint64_t offset, uint64_t bytes,
                       uint8_t output[32])
{
    sha256_ctx sha;
    FRESULT fr;
    UINT got;
    uint32_t chunk;
    if (file == NULL || output == NULL ||
        f_lseek(file, (FSIZE_t)offset) != FR_OK) {
        return -1;
    }
    sha256_init(&sha);
    while (bytes != 0U) {
        chunk = bytes > 8192U ? 8192U : (uint32_t)bytes;
        got = 0U;
        fr = f_read(file, F2_OUTPUT_BUFFER, chunk, &got);
        if (fr != FR_OK || got != chunk) {
            s_file_sha_last_fr = fr;
            s_file_sha_last_got = got;
            s_file_sha_last_chunk = chunk;
            s_file_sha_last_remaining = bytes;
            return -2;
        }
        sha256_update(&sha, F2_OUTPUT_BUFFER, chunk);
        bytes -= chunk;
    }
    sha256_final(&sha, output);
    return 0;
}

static int reader_read(void *context, uint64_t offset, void *destination,
                       uint32_t bytes)
{
    return file_read_exact((FIL *)context, offset, destination, bytes);
}

static int reader_sha256(void *context, uint64_t offset, uint64_t bytes,
                         uint8_t output[32])
{
    return file_sha256((FIL *)context, offset, bytes, output);
}

static int runtime_try_lock(void *context)
{
    (void)context;
    return forge200_inference_guard_acquire(30000U);
}

static void runtime_unlock(void *context)
{
    (void)context;
    (void)forge200_inference_guard_release();
}

static int runtime_engine_supported(void *context, uint16_t engine_id,
                                    uint16_t opset)
{
    (void)context;
    return (engine_id == 1U || engine_id == 2U || engine_id == 5U) &&
           opset == 1U;
}

static int runtime_golden_check(void *context,
                                const forge200_package_info_t *package,
                                const uint8_t *payload,
                                uint64_t payload_bytes)
{
    f2_runtime_context_t *runtime = (f2_runtime_context_t *)context;
    uint8_t actual_sha[32];
    uint8_t *golden = F2_IO_BUFFER;
    uint64_t golden_bytes;
    uint32_t input_dtype;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t prompt_length;
    uint32_t tolerance_bits;
    uint32_t started_cycles;
    uint32_t i;
    uint32_t *workspace_canary = (uint32_t *)(void *)(
        F2_WORKSPACE + F2_WORKSPACE_ELEMS - F2_WORKSPACE_CANARY_ELEMS);
    f2rt_model_t model;
    f2rt_status_t status;
    if (runtime == NULL || package == NULL || payload == NULL ||
        payload_bytes > 0xFFFFFFFFULL ||
        f_open(&s_golden_file, runtime->golden_path, FA_READ) != FR_OK) {
        return 0;
    }
    golden_bytes = (uint64_t)f_size(&s_golden_file);
    if (golden_bytes < 64U || golden_bytes > F2_GOLDEN_MAX_BYTES ||
        file_sha256(&s_golden_file, 0U, golden_bytes, actual_sha) != 0 ||
        memcmp(actual_sha, package->golden_sha256, 32U) != 0 ||
        memcmp(actual_sha, runtime->expected_golden_sha256, 32U) != 0 ||
        file_read_exact(&s_golden_file, 0U, golden,
                        (uint32_t)golden_bytes) != 0) {
        (void)f_close(&s_golden_file);
        return 0;
    }
    (void)f_close(&s_golden_file);
    if (memcmp(golden, "F2GV", 4U) != 0 ||
        read_u32_le(golden + 4U) != 1U ||
        read_u32_le(golden + 8U) != 64U ||
        read_u32_le(golden + 12U) != package->engine_id) {
        return 0;
    }
    status = f2rt_bind(payload, (uint32_t)payload_bytes, &model);
    if (status != F2RT_OK || model.workspace_elems > F2_WORKSPACE_ELEMS) {
        return 0;
    }
    input_dtype = read_u32_le(golden + 20U);
    input_count = read_u32_le(golden + 28U);
    output_count = read_u32_le(golden + 32U);
    prompt_length = read_u32_le(golden + 36U);
    tolerance_bits = read_u32_le(golden + 40U);
    if (output_count == 0U) {
        return 0;
    }
    for (i = 0U; i < F2_WORKSPACE_CANARY_ELEMS; ++i) {
        workspace_canary[i] = 0xA55A0000UL + i;
    }
    started_cycles = DWT->CYCCNT;
    if (input_dtype == 4U) {
        const uint16_t *input;
        const uint16_t *expected;
        uint16_t *actual = (uint16_t *)(void *)F2_OUTPUT_BUFFER;
        if (output_count > F2_OUTPUT_BUFFER_BYTES / sizeof(uint16_t) ||
            !f2rt_golden_layout_ok(golden_bytes, input_count, output_count,
                                   sizeof(uint16_t)) ||
            prompt_length == 0U || prompt_length > input_count) {
            return 0;
        }
        input = (const uint16_t *)(const void *)(golden + 64U);
        expected = input + input_count;
        status = f2rt_generate_u16(
            &model, input, prompt_length, actual, output_count,
            F2_WORKSPACE, F2_WORKSPACE_ELEMS);
        if (status != F2RT_OK) {
            return 0;
        }
        for (i = 0U; i < output_count; ++i) {
            if (actual[i] != expected[i]) {
                return 0;
            }
        }
    } else if (input_dtype == 2U) {
        const float *input;
        const float *expected;
        float *actual = (float *)(void *)F2_OUTPUT_BUFFER;
        float tolerance;
        if (output_count > F2_OUTPUT_BUFFER_BYTES / sizeof(float) ||
            !f2rt_golden_layout_ok(golden_bytes, input_count, output_count,
                                   sizeof(float))) {
            return 0;
        }
        memcpy(&tolerance, &tolerance_bits, sizeof(tolerance));
        input = (const float *)(const void *)(golden + 64U);
        expected = input + input_count;
        status = f2rt_infer_f32(
            &model, input, input_count, actual, output_count,
            F2_WORKSPACE, F2_WORKSPACE_ELEMS);
        if (status != F2RT_OK) {
            return 0;
        }
        for (i = 0U; i < output_count; ++i) {
            float difference = fabsf(actual[i] - expected[i]);
            if (!isfinite(actual[i]) || difference > tolerance) {
                return 0;
            }
        }
    } else {
        return 0;
    }
    for (i = 0U; i < F2_WORKSPACE_CANARY_ELEMS; ++i) {
        if (workspace_canary[i] != 0xA55A0000UL + i) {
            return 0;
        }
    }
    runtime->last_golden_cycles = DWT->CYCCNT - started_cycles;
    runtime->last_output_elems = output_count;
    return 1;
}

static int runtime_activate(void *context,
                            const forge200_package_info_t *package,
                            const uint8_t *payload, uint64_t payload_bytes)
{
    f2_runtime_context_t *runtime = (f2_runtime_context_t *)context;
    (void)package;
    if (runtime == NULL || payload_bytes > 0xFFFFFFFFULL ||
        f2rt_bind(payload, (uint32_t)payload_bytes,
                  &runtime->active_model) != F2RT_OK) {
        return 0;
    }
    return 1;
}

static void runtime_cache_clean(void *context, const void *address,
                                uint64_t bytes)
{
    (void)context;
    (void)address;
    (void)bytes;
    SCB_CleanInvalidateDCache();
}

static void runtime_event(void *context, forge200_event_t event,
                          forge200_status_t status)
{
    (void)context;
    (void)event;
    (void)status;
}

static int parse_catalog_entry(const uint8_t raw[F2_CATALOG_ENTRY_BYTES],
                               f2_catalog_entry_t *entry)
{
    if (entry == NULL || raw[31] != 0U || raw[55] != 0U || raw[79] != 0U) {
        return -1;
    }
    memset(entry, 0, sizeof(*entry));
    memcpy(entry->model_id, raw, 32U);
    memcpy(entry->package_path, raw + 32U, 24U);
    memcpy(entry->golden_path, raw + 56U, 24U);
    entry->category = raw[80];
    entry->tier = raw[81];
    entry->engine_id = read_u16_le(raw + 82U);
    entry->opset = read_u16_le(raw + 84U);
    entry->package_bytes = read_u64_le(raw + 88U);
    memcpy(entry->package_sha256, raw + 96U, 32U);
    memcpy(entry->golden_sha256, raw + 128U, 32U);
    if ((entry->category != (uint8_t)'P' &&
         entry->category != (uint8_t)'G' &&
         entry->category != (uint8_t)'S') ||
        (entry->tier != 1U && entry->tier != 2U) ||
        entry->model_id[0] == '\0' ||
        entry->package_path[0] == '\0' ||
        entry->golden_path[0] == '\0') {
        return -2;
    }
    return 0;
}

static int validate_catalog(const char *path, f2_catalog_t *catalog)
{
    uint8_t header[F2_CATALOG_HEADER_BYTES];
    uint8_t body_sha[32];
    uint64_t body_bytes;
    FRESULT fr;
    memset(catalog, 0, sizeof(*catalog));
    (void)snprintf(catalog->path, sizeof(catalog->path), "%s", path);
    fr = f_open(&s_catalog_file, path, FA_READ);
    if (fr != FR_OK) return -10 - (int)fr;
    if (file_read_exact(&s_catalog_file, 0U, header, sizeof(header)) != 0) {
        (void)f_close(&s_catalog_file);
        return -20;
    }
    if (memcmp(header, "F2CT", 4U) != 0) {
        (void)f_close(&s_catalog_file);
        return -21;
    }
    if (read_u16_le(header + 4U) != 1U) {
        (void)f_close(&s_catalog_file);
        return -22;
    }
    if (read_u16_le(header + 6U) != F2_CATALOG_HEADER_BYTES) {
        (void)f_close(&s_catalog_file);
        return -23;
    }
    if (read_u32_le(header + 16U) != F2_MODEL_COUNT) {
        (void)f_close(&s_catalog_file);
        return -24;
    }
    if (read_u32_le(header + 20U) != F2_CATALOG_ENTRY_BYTES) {
        (void)f_close(&s_catalog_file);
        return -25;
    }
    if (memcmp(header + 64U, s_release_content_root, 32U) != 0) {
        (void)f_close(&s_catalog_file);
        return -26;
    }
    body_bytes = read_u64_le(header + 24U);
    if (body_bytes != (uint64_t)F2_MODEL_COUNT * F2_CATALOG_ENTRY_BYTES) {
        (void)f_close(&s_catalog_file);
        return -30;
    }
    if ((uint64_t)f_size(&s_catalog_file) !=
        F2_CATALOG_HEADER_BYTES + body_bytes) {
        (void)f_close(&s_catalog_file);
        return -31;
    }
    if (file_sha256(&s_catalog_file, F2_CATALOG_HEADER_BYTES,
                    body_bytes, body_sha) != 0) {
        (void)f_close(&s_catalog_file);
        return -32;
    }
    if (memcmp(body_sha, header + 32U, 32U) != 0) {
        (void)f_close(&s_catalog_file);
        return -33;
    }
    catalog->generation = read_u64_le(header + 8U);
    catalog->entry_count = read_u32_le(header + 16U);
    catalog->valid = 1U;
    (void)f_close(&s_catalog_file);
    return 0;
}

static int read_catalog_entry(const f2_catalog_t *catalog, uint32_t index,
                              f2_catalog_entry_t *entry)
{
    uint8_t raw[F2_CATALOG_ENTRY_BYTES];
    uint64_t offset;
    if (catalog == NULL || entry == NULL || catalog->valid == 0U ||
        index >= catalog->entry_count) {
        return -1;
    }
    if (f_open(&s_catalog_file, catalog->path, FA_READ) != FR_OK) {
        return -2;
    }
    offset = F2_CATALOG_HEADER_BYTES +
             (uint64_t)index * F2_CATALOG_ENTRY_BYTES;
    if (file_read_exact(&s_catalog_file, offset, raw, sizeof(raw)) != 0) {
        (void)f_close(&s_catalog_file);
        return -3;
    }
    (void)f_close(&s_catalog_file);
    return parse_catalog_entry(raw, entry);
}

static forge200_runtime_t make_runtime(void)
{
    forge200_runtime_t runtime;
    memset(&runtime, 0, sizeof(runtime));
    runtime.context = &s_runtime_context;
    runtime.try_lock = runtime_try_lock;
    runtime.unlock = runtime_unlock;
    runtime.engine_supported = runtime_engine_supported;
    runtime.golden_check = runtime_golden_check;
    runtime.activate = runtime_activate;
    runtime.cache_clean = runtime_cache_clean;
    runtime.event = runtime_event;
    return runtime;
}

static forge200_status_t load_entry(const f2_catalog_t *catalog,
                                    uint32_t index, uint8_t verbose)
{
    f2_catalog_entry_t entry;
    forge200_reader_t reader;
    forge200_runtime_t runtime;
    forge200_status_t status;
    uint8_t package_sha[32];
    TickType_t started;
    TickType_t elapsed;
    char detail[192];
    if (read_catalog_entry(catalog, index, &entry) != 0) {
        return FORGE200_ERR_IO;
    }
    if (f_open(&s_package_file, entry.package_path, FA_READ) != FR_OK ||
        (uint64_t)f_size(&s_package_file) != entry.package_bytes ||
        file_sha256(&s_package_file, 0U, entry.package_bytes,
                    package_sha) != 0 ||
        memcmp(package_sha, entry.package_sha256, 32U) != 0) {
        (void)f_close(&s_package_file);
        return FORGE200_ERR_IO;
    }
    memset(&s_runtime_context, 0, sizeof(s_runtime_context));
    s_runtime_context.package_file = &s_package_file;
    (void)snprintf(s_runtime_context.golden_path,
                   sizeof(s_runtime_context.golden_path), "%s",
                   entry.golden_path);
    memcpy(s_runtime_context.expected_golden_sha256,
           entry.golden_sha256, 32U);
    memset(&reader, 0, sizeof(reader));
    reader.context = &s_package_file;
    reader.package_bytes = entry.package_bytes;
    reader.read = reader_read;
    reader.sha256 = reader_sha256;
    runtime = make_runtime();
    started = xTaskGetTickCount();
    status = forge200_modelbank_load(
        &s_bank, &reader, &runtime, entry.model_id, catalog->generation);
    elapsed = xTaskGetTickCount() - started;
    s_last_load_ms = (uint32_t)(elapsed * portTICK_PERIOD_MS);
    (void)f_close(&s_package_file);
    if (status == FORGE200_OK) {
        s_total_loads++;
        if (index < F2_MODEL_COUNT) s_load_counts[index]++;
    } else {
        s_total_failures++;
    }
    if (verbose != 0U || status != FORGE200_OK) {
        (void)snprintf(
            detail, sizeof(detail),
            "model=%s|cat=%c|tier=%s|status=%u|load_ms=%lu|dwt=%lu|slot=%u|bytes=%llu",
            entry.model_id, (char)entry.category,
            entry.tier == 1U ? "EXACT" : "SIM_ONLY",
            (unsigned)status, (unsigned long)(elapsed * portTICK_PERIOD_MS),
            (unsigned long)s_runtime_context.last_golden_cycles,
            (unsigned)s_bank.active_slot,
            (unsigned long long)entry.package_bytes);
        f2_log("MODEL", detail);
    }
    return status;
}

static int sequential_benchmark(const f2_catalog_t *catalog,
                                uint32_t *kib_per_s)
{
    uint64_t remaining = F2_SEQUENTIAL_BENCH_BYTES;
    uint64_t completed = 0U;
    uint32_t index = 0U;
    TickType_t started = xTaskGetTickCount();
    TickType_t elapsed;
    while (remaining != 0U) {
        f2_catalog_entry_t entry;
        uint64_t file_remaining;
        if (read_catalog_entry(catalog, index, &entry) != 0 ||
            f_open(&s_package_file, entry.package_path, FA_READ) != FR_OK) {
            return -1;
        }
        file_remaining = entry.package_bytes;
        while (file_remaining != 0U && remaining != 0U) {
            UINT got = 0U;
            uint32_t chunk = file_remaining > 8192U ? 8192U :
                             (uint32_t)file_remaining;
            if (chunk > remaining) chunk = (uint32_t)remaining;
            if (f_read(&s_package_file, F2_OUTPUT_BUFFER, chunk, &got) != FR_OK ||
                got != chunk) {
                (void)f_close(&s_package_file);
                return -2;
            }
            file_remaining -= chunk;
            remaining -= chunk;
            completed += chunk;
        }
        (void)f_close(&s_package_file);
        index = (index + 1U) % F2_MODEL_COUNT;
    }
    elapsed = xTaskGetTickCount() - started;
    if (elapsed == 0U) elapsed = 1U;
    *kib_per_s = (uint32_t)(
        (completed * 1000ULL) /
        ((uint64_t)elapsed * portTICK_PERIOD_MS * 1024ULL));
    return 0;
}

static int random_page_benchmark(const f2_catalog_t *catalog,
                                 uint32_t *pages_per_s)
{
    uint32_t seed = 0x46524732UL;
    uint32_t count;
    TickType_t started = xTaskGetTickCount();
    TickType_t elapsed;
    for (count = 0U; count < F2_RANDOM_PAGE_COUNT; ++count) {
        f2_catalog_entry_t entry;
        uint32_t index;
        uint64_t page_count;
        uint64_t page_index;
        UINT got = 0U;
        seed = seed * 1664525UL + 1013904223UL;
        index = seed % F2_MODEL_COUNT;
        if (read_catalog_entry(catalog, index, &entry) != 0 ||
            f_open(&s_package_file, entry.package_path, FA_READ) != FR_OK) {
            return -1;
        }
        page_count = (entry.package_bytes + F2_RANDOM_PAGE_BYTES - 1U) /
                     F2_RANDOM_PAGE_BYTES;
        seed = seed * 1664525UL + 1013904223UL;
        page_index = page_count == 0U ? 0U : seed % page_count;
        if (f_lseek(&s_package_file,
                    (FSIZE_t)(page_index * F2_RANDOM_PAGE_BYTES)) != FR_OK ||
            f_read(&s_package_file, F2_OUTPUT_BUFFER,
                   F2_RANDOM_PAGE_BYTES, &got) != FR_OK ||
            got == 0U) {
            (void)f_close(&s_package_file);
            return -2;
        }
        (void)f_close(&s_package_file);
    }
    elapsed = xTaskGetTickCount() - started;
    if (elapsed == 0U) elapsed = 1U;
    *pages_per_s = (uint32_t)(
        ((uint64_t)F2_RANDOM_PAGE_COUNT * 1000ULL) /
        ((uint64_t)elapsed * portTICK_PERIOD_MS));
    return 0;
}

static int fault_refusal_checks(const f2_catalog_t *catalog)
{
    f2_catalog_entry_t entry;
    forge200_reader_t reader;
    forge200_runtime_t runtime;
    forge200_status_t status;
    static const struct {
        const char *path;
        const char *golden;
        const char *expected_model;
        forge200_status_t expected_status;
    } faults[] = {
        {"0:/F200/FAULT/BADMAG.ICM", "0:/F200/P001.GLD",
         "CAND-P-001", FORGE200_ERR_SCHEMA},
        {"0:/F200/FAULT/BADAUT.ICM", "0:/F200/P001.GLD",
         "CAND-P-001", FORGE200_ERR_AUTHORITY},
        {"0:/F200/FAULT/BADPAY.ICM", "0:/F200/P001.GLD",
         "CAND-P-001", FORGE200_ERR_PAYLOAD_SHA},
        {"0:/F200/FAULT/BADGEN.ICM", "0:/F200/P001.GLD",
         "CAND-P-001", FORGE200_ERR_ROLLBACK},
        {"0:/F200/FAULT/BADENG.ICM", "0:/F200/P001.GLD",
         "CAND-P-001", FORGE200_ERR_ENGINE},
        {"0:/F200/FAULT/BADGLD.ICM", "0:/F200/FAULT/BADGLD.GLD",
         "CAND-P-001", FORGE200_ERR_GOLDEN}
    };
    uint32_t i;
    if (read_catalog_entry(catalog, 30U, &entry) != 0 ||
        strcmp(entry.model_id, "CAND-P-001") != 0) {
        return -1;
    }
    for (i = 0U; i < sizeof(faults) / sizeof(faults[0]); ++i) {
        if (f_open(&s_package_file, faults[i].path, FA_READ) != FR_OK) {
            return -2;
        }
        memset(&s_runtime_context, 0, sizeof(s_runtime_context));
        (void)snprintf(s_runtime_context.golden_path,
                       sizeof(s_runtime_context.golden_path), "%s",
                       faults[i].golden);
        memcpy(s_runtime_context.expected_golden_sha256,
               entry.golden_sha256, 32U);
        memset(&reader, 0, sizeof(reader));
        reader.context = &s_package_file;
        reader.package_bytes = (uint64_t)f_size(&s_package_file);
        reader.read = reader_read;
        reader.sha256 = reader_sha256;
        runtime = make_runtime();
        status = forge200_modelbank_load(
            &s_bank, &reader, &runtime, faults[i].expected_model,
            catalog->generation);
        (void)f_close(&s_package_file);
        if (status != faults[i].expected_status) {
            return (int)(10U + i);
        }
    }
    return 0;
}

static int minimum_load_count(void)
{
    uint32_t i;
    uint16_t minimum = 0xFFFFU;
    for (i = 0U; i < F2_MODEL_COUNT; ++i) {
        if (s_load_counts[i] < minimum) minimum = s_load_counts[i];
    }
    return (int)minimum;
}

static int minimum_swap_load_count(void)
{
    uint32_t i;
    uint16_t minimum = 0xFFFFU;
    for (i = 0U; i < F2_MODEL_COUNT; ++i) {
        if (s_swap_load_counts[i] < minimum) minimum = s_swap_load_counts[i];
    }
    return (int)minimum;
}

static void record_swap_latency(uint32_t latency_ms)
{
    uint32_t bucket = latency_ms / F2_LOAD_BUCKET_MS;
    if (bucket >= F2_LOAD_BUCKETS) bucket = F2_LOAD_BUCKETS - 1U;
    if (s_swap_load_ms_bins[bucket] != 0xFFFFU) {
        s_swap_load_ms_bins[bucket]++;
    }
    if (latency_ms > s_swap_load_max_ms) s_swap_load_max_ms = latency_ms;
}

static uint32_t swap_load_percentile_ms(uint32_t percentile)
{
    uint32_t threshold = (F2_TOTAL_SWAP_LOADS * percentile + 99U) / 100U;
    uint32_t cumulative = 0U;
    uint32_t bucket;
    for (bucket = 0U; bucket < F2_LOAD_BUCKETS; ++bucket) {
        cumulative += s_swap_load_ms_bins[bucket];
        if (cumulative >= threshold) break;
    }
    if (bucket >= F2_LOAD_BUCKETS) bucket = F2_LOAD_BUCKETS - 1U;
    return bucket * F2_LOAD_BUCKET_MS;
}

static int cancelled_load_probe(const f2_catalog_t *catalog, uint32_t index)
{
    f2_catalog_entry_t entry;
    uint8_t active_slot = s_bank.active_slot;
    uint64_t commits = s_bank.successful_commits;
    UINT got = 0U;
    uint32_t bytes;
    if (read_catalog_entry(catalog, index, &entry) != 0 ||
        f_open(&s_package_file, entry.package_path, FA_READ) != FR_OK) {
        return -1;
    }
    bytes = entry.package_bytes > 4096ULL ? 4096U : (uint32_t)entry.package_bytes;
    if (f_read(&s_package_file, F2_IO_BUFFER, bytes, &got) != FR_OK || got != bytes ||
        f_close(&s_package_file) != FR_OK) {
        return -2;
    }
    memset(F2_IO_BUFFER, 0, bytes);
    return s_bank.active_slot == active_slot &&
           s_bank.successful_commits == commits ? 0 : -3;
}

static uint32_t rag_percentile(uint32_t *values, uint32_t count,
                               uint32_t percentile)
{
    uint32_t i;
    uint32_t j;
    uint32_t rank;
    if (values == NULL || count == 0U || count > F2_RAG_QUERY_COUNT) {
        return 0xFFFFFFFFUL;
    }
    for (i = 1U; i < count; ++i) {
        uint32_t value = values[i];
        j = i;
        while (j != 0U && values[j - 1U] > value) {
            values[j] = values[j - 1U];
            --j;
        }
        values[j] = value;
    }
    rank = (count * percentile + 99U) / 100U;
    if (rank == 0U) rank = 1U;
    if (rank > count) rank = count;
    return values[rank - 1U];
}

int forge200_board_acceptance_run(void)
{
    sd_spi_diag_t sd;
    f2_catalog_t catalog_a;
    f2_catalog_t catalog_b;
    const f2_catalog_t *catalog;
    forge200_bus_metrics_t bus_before;
    forge200_bus_metrics_t bus_after;
    veriprocess_board_receipt_v9_t vp_receipt;
    uint32_t sequential_kib = 0U;
    uint32_t random_pages = 0U;
    uint32_t baseline_p99_ms;
    uint32_t active_p99_ms;
    uint32_t timing_samples;
    uint32_t heap_min_bytes;
    uint32_t critical_stack_min_bytes;
    uint32_t index;
    int catalog_a_rc;
    int catalog_b_rc;
    uint32_t swap_loads = 0U;
    uint32_t exact_pass = 0U;
    uint32_t sim_pass = 0U;
    uint32_t rag_safe = 0U;
    uint32_t rag_source_bound = 0U;
    uint32_t rag_refused = 0U;
    uint32_t rag_negative_refused = 0U;
    uint32_t rag_cold_ms[F2RAG_DOMAIN_COUNT];
    uint32_t rag_warm_ms[F2_RAG_QUERY_COUNT - F2RAG_DOMAIN_COUNT];
    uint32_t rag_warm_count = 0U;
    uint32_t soak_rag_queries = 0U;
    uint32_t same_model_reload_cases = 0U;
    uint32_t mid_cancel_cases = 0U;
    uint32_t previous_swap_index = F2_MODEL_COUNT;
    uint32_t soak_cycle;
    uint32_t soak_cycles = (F2_SOAK_HOURS * 60U * 60U * 1000U) /
                           F2_SOAK_PERIOD_MS;
    char detail[384];

    memset(&sd, 0, sizeof(sd));
    memset(s_load_counts, 0, sizeof(s_load_counts));
    memset(s_swap_load_counts, 0, sizeof(s_swap_load_counts));
    memset(s_swap_load_ms_bins, 0, sizeof(s_swap_load_ms_bins));
    memset(&s_runtime_context, 0, sizeof(s_runtime_context));
    s_total_loads = 0U;
    s_total_failures = 0U;
    s_last_load_ms = 0U;
    s_swap_load_max_ms = 0U;
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    f2_log("BEGIN",
           "models=170|exact=78|sim_only=92|target_swaps=1000|soak_hours=24");

    timing_begin(1U);
    vTaskDelay(pdMS_TO_TICKS(10000U));
    baseline_p99_ms = timing_p99_ms(&timing_samples);
    (void)snprintf(
        detail, sizeof(detail),
        "phase=baseline|samples=%lu|control_p99_ms=%lu",
        (unsigned long)timing_samples, (unsigned long)baseline_p99_ms);
    f2_log("CONTROL_TIMING", detail);
    if (timing_samples < 50U || baseline_p99_ms == 0U ||
        !resource_snapshot(&heap_min_bytes, &critical_stack_min_bytes)) {
        (void)snprintf(
            detail, sizeof(detail),
            "reason=RESOURCE_BASELINE|heap_min=%lu|critical_stack_min=%lu|timing_samples=%lu",
            (unsigned long)heap_min_bytes,
            (unsigned long)critical_stack_min_bytes,
            (unsigned long)timing_samples);
        f2_log("STOP", detail);
        timing_begin(0U);
        return -11;
    }
    (void)snprintf(
        detail, sizeof(detail),
        "heap_min=%lu|heap_gate=%u|critical_stack_min=%lu|stack_gate=%u",
        (unsigned long)heap_min_bytes, (unsigned)F2_HEAP_MIN_BYTES,
        (unsigned long)critical_stack_min_bytes, (unsigned)F2_STACK_MIN_BYTES);
    f2_log("RESOURCE", detail);
    timing_begin(2U);

    if (sd_spi_boot_probe(&sd) != 0U || sd.init_ok == 0U ||
        f_mount(&s_fatfs, "0:", 1U) != FR_OK) {
        f2_log("STOP", "reason=SD_INIT_OR_MOUNT");
        timing_begin(0U);
        return -1;
    }
    (void)snprintf(
        detail, sizeof(detail),
        "capacity_mb=%lu|fs=%s|cid=%u|csd=%u|sd_delay=%lu|sd_peak_delay=%lu|sd_retries=%lu|shared=PB10_PC1_PC2|sd_cs=PC5|max_cs=PG3",
        (unsigned long)sd.capacity_mb, sd_spi_fs_name(sd.fs_kind),
        (unsigned)sd.cid_ok, (unsigned)sd.csd_ok,
        (unsigned long)sd_spi_active_delay_cycles(),
        (unsigned long)sd_spi_peak_retry_delay_cycles(),
        (unsigned long)sd_spi_read_retry_count());
    f2_log("SD_READY", detail);

    {
        int vp_status = veriprocess_board_selftest_v9(&vp_receipt);
        if (vp_status == VERIPROCESS_BOARD_POWER_CUT_ARMED) {
            f2_log("POWER_CUT_ARMED",
                   "component=VERIPROCESS|wal_synced=1|header_flipped=0|instruction=REMOVE_BOARD_POWER_NOW");
            (void)f_mount(NULL, "0:", 0U);
            timing_begin(0U);
            return 1;
        }
        if (vp_status != VERIPROCESS_BOARD_OK) {
        f2_log("STOP", "reason=VERIPROCESS_TRACELEDGER_SELFTEST");
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -18;
        }
    }
    (void)snprintf(
        detail, sizeof(detail),
        "ledger_generation=%llu|ledger_records=%lu|chrono_events=%lu|independent_families=%lu|ds3231=%u|wal_recovered=%u|sintergraph_frozen=%u|authority=0",
        (unsigned long long)vp_receipt.ledger_generation,
        (unsigned long)vp_receipt.ledger_records,
        (unsigned long)vp_receipt.chrono_events,
        (unsigned long)vp_receipt.independent_families,
        (unsigned)vp_receipt.ds3231_valid,
        (unsigned)vp_receipt.wal_recovered,
        (unsigned)vp_receipt.sintergraph_frozen);
    f2_log("VERIPROCESS", detail);

    catalog_a_rc = validate_catalog(F2_CATALOG_PATH_A, &catalog_a);
    catalog_b_rc = validate_catalog(F2_CATALOG_PATH_B, &catalog_b);
    if (catalog_a.valid == 0U && catalog_b.valid == 0U) {
        (void)snprintf(
            detail, sizeof(detail),
            "reason=BOTH_CATALOGS_INVALID|a_rc=%d|b_rc=%d|fatfs_fr=%u|got=%u|chunk=%lu|remaining=%llu|sd_delay=%lu|sd_peak_delay=%lu|sd_retries=%lu|failed_lba=%lu|sd_read_rc=%u",
            catalog_a_rc, catalog_b_rc, (unsigned)s_file_sha_last_fr,
            (unsigned)s_file_sha_last_got,
            (unsigned long)s_file_sha_last_chunk,
            (unsigned long long)s_file_sha_last_remaining,
            (unsigned long)sd_spi_active_delay_cycles(),
            (unsigned long)sd_spi_peak_retry_delay_cycles(),
            (unsigned long)sd_spi_read_retry_count(),
            (unsigned long)sd_spi_last_failed_sector(),
            (unsigned)sd_spi_last_read_rc());
        f2_log("STOP", detail);
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -2;
    }
    if (catalog_a.valid != 0U &&
        (catalog_b.valid == 0U ||
         catalog_a.generation >= catalog_b.generation)) {
        catalog = &catalog_a;
    } else {
        catalog = &catalog_b;
    }
    (void)snprintf(
        detail, sizeof(detail),
        "a=%u|b=%u|selected=%c|generation=%llu|entries=%lu",
        (unsigned)catalog_a.valid, (unsigned)catalog_b.valid,
        catalog == &catalog_a ? 'A' : 'B',
        (unsigned long long)catalog->generation,
        (unsigned long)catalog->entry_count);
    f2_log("CATALOG", detail);

    forge200_bus_guard_snapshot(&bus_before);
    if (sequential_benchmark(catalog, &sequential_kib) != 0 ||
        random_page_benchmark(catalog, &random_pages) != 0) {
        f2_log("STOP", "reason=SD_BENCH_IO");
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -3;
    }
    (void)snprintf(
        detail, sizeof(detail),
        "sequential_kib_s=%lu|random_4k_pages_s=%lu|min_kib_s=%u|sd_delay=%lu|sd_peak_delay=%lu|sd_retries=%lu",
        (unsigned long)sequential_kib, (unsigned long)random_pages,
        (unsigned)F2_MIN_SD_KIB_PER_S,
        (unsigned long)sd_spi_active_delay_cycles(),
        (unsigned long)sd_spi_peak_retry_delay_cycles(),
        (unsigned long)sd_spi_read_retry_count());
    f2_log("SD_BENCH", detail);
    if (sequential_kib < F2_MIN_SD_KIB_PER_S) {
        f2_log("STOP", "reason=SD_THROUGHPUT_BELOW_512_KIB_S");
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -4;
    }

    forge200_modelbank_init(
        &s_bank, F2_SLOT_A, F2_SLOT_BYTES, F2_SLOT_B, F2_SLOT_BYTES, 1U);
    for (index = 0U; index < F2_MODEL_COUNT; ++index) {
        f2_catalog_entry_t entry;
        if (read_catalog_entry(catalog, index, &entry) != 0 ||
            load_entry(catalog, index, 1U) != FORGE200_OK) {
            f2_log("STOP", "reason=INITIAL_170_GOLDEN");
            (void)f_mount(NULL, "0:", 0U);
            timing_begin(0U);
            return -5;
        }
        if (entry.tier == 1U) exact_pass++;
        else sim_pass++;
    }
    (void)snprintf(
        detail, sizeof(detail),
        "passed=%lu|exact=%lu|sim_only=%lu|package_commits=%llu",
        (unsigned long)(exact_pass + sim_pass), (unsigned long)exact_pass,
        (unsigned long)sim_pass,
        (unsigned long long)s_bank.successful_commits);
    f2_log("BATCH170", detail);

    if (fault_refusal_checks(catalog) != 0) {
        f2_log("STOP", "reason=FAULT_REFUSAL");
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -6;
    }
    f2_log("FAULTS",
           "bad_magic=REFUSED|bad_authority=REFUSED|payload=REFUSED|rollback=REFUSED|engine=REFUSED|golden=REFUSED");

    memset(s_rag_stage_ms, 0, sizeof(s_rag_stage_ms));
    memset(s_rag_total_ms, 0, sizeof(s_rag_total_ms));
    forge200_rag_board_reset_cache();
    for (index = 0U; index < F2_RAG_QUERY_COUNT; ++index) {
        f2rag_metrics_t metrics;
        f2rag_result_t result;
        uint32_t domain_id = index / F2RAG_WORKLOAD_PER_DOMAIN;
        uint32_t local_query = index % F2RAG_WORKLOAD_PER_DOMAIN;
        uint32_t stage;
        uint32_t total_ms = 0U;
        uint32_t cycles_per_ms = SystemCoreClock / 1000U;
        if (cycles_per_ms == 0U || forge200_rag_board_run(
                domain_id, local_query, local_query == 0U ? 1U : 0U,
                &metrics, &result) != 0) {
            f2_log("STOP", "reason=RAG120_EXECUTION");
            (void)f_mount(NULL, "0:", 0U);
            timing_begin(0U);
            return -14;
        }
        rag_safe += result.safe_outcome;
        rag_source_bound += result.source_bound;
        rag_refused += result.refused;
        if (result.expected_refusal != 0U && result.refused != 0U) {
            rag_negative_refused++;
        }
        for (stage = 0U; stage < F2RAG_STAGE_COUNT; ++stage) {
            uint32_t stage_ms = metrics.stage_ticks[stage] / cycles_per_ms;
            s_rag_stage_ms[stage][index] = stage_ms;
            total_ms += stage_ms;
        }
        s_rag_total_ms[index] = total_ms;
        if (local_query == 0U) {
            rag_cold_ms[domain_id] = total_ms;
        } else {
            rag_warm_ms[rag_warm_count++] = total_ms;
        }
    }
    (void)snprintf(
        detail, sizeof(detail),
        "queries=120|safe=%lu|source_bound=%lu|refused=%lu|negative_refused=%lu|cold_p95_ms=%lu|cold_p99_ms=%lu|warm_p95_ms=%lu|warm_p99_ms=%lu|warm_hits=%lu|warm_hit_percent=%lu",
        (unsigned long)rag_safe, (unsigned long)rag_source_bound,
        (unsigned long)rag_refused, (unsigned long)rag_negative_refused,
        (unsigned long)rag_percentile(rag_cold_ms, F2RAG_DOMAIN_COUNT, 95U),
        (unsigned long)rag_percentile(rag_cold_ms, F2RAG_DOMAIN_COUNT, 99U),
        (unsigned long)rag_percentile(rag_warm_ms, rag_warm_count, 95U),
        (unsigned long)rag_percentile(rag_warm_ms, rag_warm_count, 99U),
        (unsigned long)rag_warm_count,
        (unsigned long)(rag_warm_count * 100U / F2_RAG_QUERY_COUNT));
    f2_log("RAG120", detail);
    (void)snprintf(
        detail, sizeof(detail),
        "load_support_p95_ms=%lu|route_retrieve_p95_ms=%lu|load_lm_p95_ms=%lu|generate_p95_ms=%lu|unload_p95_ms=%lu|nli_p95_ms=%lu|commit_p95_ms=%lu|zeroize_p95_ms=%lu",
        (unsigned long)rag_percentile(s_rag_stage_ms[0], F2_RAG_QUERY_COUNT, 95U),
        (unsigned long)rag_percentile(s_rag_stage_ms[1], F2_RAG_QUERY_COUNT, 95U),
        (unsigned long)rag_percentile(s_rag_stage_ms[2], F2_RAG_QUERY_COUNT, 95U),
        (unsigned long)rag_percentile(s_rag_stage_ms[3], F2_RAG_QUERY_COUNT, 95U),
        (unsigned long)rag_percentile(s_rag_stage_ms[4], F2_RAG_QUERY_COUNT, 95U),
        (unsigned long)rag_percentile(s_rag_stage_ms[5], F2_RAG_QUERY_COUNT, 95U),
        (unsigned long)rag_percentile(s_rag_stage_ms[6], F2_RAG_QUERY_COUNT, 95U),
        (unsigned long)rag_percentile(s_rag_stage_ms[7], F2_RAG_QUERY_COUNT, 95U));
    f2_log("RAG_STAGE", detail);
    if (rag_safe != F2_RAG_QUERY_COUNT || rag_negative_refused != 60U ||
        rag_source_bound == 0U || rag_warm_count * 100U < 80U * F2_RAG_QUERY_COUNT ||
        rag_percentile(rag_cold_ms, F2RAG_DOMAIN_COUNT, 95U) > 20000U ||
        rag_percentile(rag_cold_ms, F2RAG_DOMAIN_COUNT, 99U) > 30000U ||
        rag_percentile(rag_warm_ms, rag_warm_count, 95U) > 8000U ||
        rag_percentile(rag_warm_ms, rag_warm_count, 99U) > 12000U) {
        f2_log("STOP", "reason=RAG120_GATE");
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -15;
    }
    forge200_rag_board_reset_cache();

    index = 0U;
    while (swap_loads < F2_TOTAL_SWAP_LOADS) {
        if (load_entry(catalog, index, 0U) != FORGE200_OK) {
            f2_log("STOP", "reason=SWAP1000_LOAD");
            (void)f_mount(NULL, "0:", 0U);
            timing_begin(0U);
            return -7;
        }
        s_swap_load_counts[index]++;
        swap_loads++;
        record_swap_latency(s_last_load_ms);
        if (index == previous_swap_index) {
            same_model_reload_cases++;
        }
        previous_swap_index = index;
        if ((swap_loads % 250U) == 0U) {
            if (cancelled_load_probe(catalog, index) != 0) {
                f2_log("STOP", "reason=MID_LOAD_CANCEL_PRESERVATION");
                (void)f_mount(NULL, "0:", 0U);
                timing_begin(0U);
                return -17;
            }
            mid_cancel_cases++;
        }
        /* Hold the index after loads 99, 199, ... 999 so that loads
         * 100, 200, ... 1000 are ten actual same-model reloads. */
        if ((swap_loads % 100U) != 99U) {
            index = (index + 1U) % F2_MODEL_COUNT;
        }
        if ((swap_loads % 100U) == 0U) {
            (void)snprintf(
                detail, sizeof(detail),
                "swap_loads=%lu|total_loads=%lu|min_swap_per_model=%d|active_slot=%u|failures=%lu",
                (unsigned long)swap_loads, (unsigned long)s_total_loads,
                minimum_swap_load_count(),
                (unsigned)s_bank.active_slot,
                (unsigned long)s_total_failures);
            f2_log("SWAP_PROGRESS", detail);
        }
    }
    if (minimum_swap_load_count() < 4) {
        f2_log("STOP", "reason=PER_MODEL_LOAD_LT4");
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -8;
    }
    forge200_bus_guard_snapshot(&bus_after);
    active_p99_ms = timing_p99_ms(&timing_samples);
    (void)snprintf(
        detail, sizeof(detail),
        "swap_loads=%lu|same_model_reload=%lu|mid_cancel=%lu|total_loads=%lu|min_swap_per_model=%d|load_p95_ms=%lu|load_p99_ms=%lu|load_max_ms=%lu|canary=PASS|generation=%llu|sd_acq=%lu|max31856_acq_delta=%lu|max_wait_ticks=%lu|max_hold_ticks=%lu|timeouts=%lu|collisions=%lu|control_p99_baseline_ms=%lu|control_p99_active_ms=%lu",
        (unsigned long)swap_loads,
        (unsigned long)same_model_reload_cases,
        (unsigned long)mid_cancel_cases,
        (unsigned long)s_total_loads,
        minimum_swap_load_count(),
        (unsigned long)swap_load_percentile_ms(95U),
        (unsigned long)swap_load_percentile_ms(99U),
        (unsigned long)s_swap_load_max_ms,
        (unsigned long long)s_bank.accepted_catalog_generation,
        (unsigned long)bus_after.sd_acquisitions,
        (unsigned long)(bus_after.max31856_acquisitions -
                        bus_before.max31856_acquisitions),
        (unsigned long)bus_after.max_wait_ticks,
        (unsigned long)bus_after.max_hold_ticks,
        (unsigned long)bus_after.timeout_refusals,
        (unsigned long)bus_after.collision_refusals,
        (unsigned long)baseline_p99_ms, (unsigned long)active_p99_ms);
    f2_log("SWAP1000", detail);
    if (bus_after.timeout_refusals != 0U ||
        bus_after.collision_refusals != 0U ||
        bus_after.max31856_acquisitions ==
            bus_before.max31856_acquisitions ||
        active_p99_ms * 100U > baseline_p99_ms * 105U) {
        f2_log("STOP", "reason=SHARED_SPI_COEXISTENCE");
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -9;
    }

    f2_log("SOAK_BEGIN",
           "hours=24|period_s=300|loads_per_period=2|rag_golden_per_hour=12");
    for (soak_cycle = 0U; soak_cycle < soak_cycles; ++soak_cycle) {
        uint32_t general_index = soak_cycle % F2_MODEL_COUNT;
        uint32_t generative_index = soak_cycle % 30U;
        f2rag_metrics_t soak_rag_metrics;
        f2rag_result_t soak_rag_result;
        if (load_entry(catalog, general_index, 0U) != FORGE200_OK ||
            load_entry(catalog, generative_index, 0U) != FORGE200_OK) {
            f2_log("STOP", "reason=SOAK_LOAD");
            (void)f_mount(NULL, "0:", 0U);
            timing_begin(0U);
            return -10;
        }
        if (forge200_rag_board_run(
                (soak_cycle / F2RAG_WORKLOAD_PER_DOMAIN) % F2RAG_DOMAIN_COUNT,
                soak_cycle % F2RAG_WORKLOAD_PER_DOMAIN, 1U,
                &soak_rag_metrics, &soak_rag_result) != 0 ||
            soak_rag_result.safe_outcome == 0U) {
            f2_log("STOP", "reason=SOAK_RAG_QUERY");
            (void)f_mount(NULL, "0:", 0U);
            timing_begin(0U);
            return -16;
        }
        soak_rag_queries++;
        vTaskDelay(pdMS_TO_TICKS(F2_SOAK_PERIOD_MS));
        if (((soak_cycle + 1U) % F2_SOAK_FAULT_PERIOD_CYCLES) == 0U) {
            if (fault_refusal_checks(catalog) != 0) {
                f2_log("STOP", "reason=SOAK_FAULT_REFUSAL");
                (void)f_mount(NULL, "0:", 0U);
                timing_begin(0U);
                return -13;
            }
            (void)snprintf(
                detail, sizeof(detail),
                "hour=%lu|interval_hours=2|six_refusal_classes=PASS",
                (unsigned long)((soak_cycle + 1U) / 12U));
            f2_log("SOAK_FAULT", detail);
        }
        if (((soak_cycle + 1U) % 12U) == 0U) {
            forge200_bus_guard_snapshot(&bus_after);
            (void)snprintf(
                detail, sizeof(detail),
                "hour=%lu|loads=%lu|rag_queries=%lu|max_acq=%lu|timeouts=%lu|collisions=%lu",
                (unsigned long)((soak_cycle + 1U) / 12U),
                (unsigned long)s_total_loads,
                (unsigned long)soak_rag_queries,
                (unsigned long)bus_after.max31856_acquisitions,
                (unsigned long)bus_after.timeout_refusals,
                (unsigned long)bus_after.collision_refusals);
            f2_log("SOAK_HOUR", detail);
        }
    }
    forge200_bus_guard_snapshot(&bus_after);
    active_p99_ms = timing_p99_ms(&timing_samples);
    if (!resource_snapshot(&heap_min_bytes, &critical_stack_min_bytes) ||
        bus_after.timeout_refusals != 0U ||
        bus_after.collision_refusals != 0U ||
        active_p99_ms * 100U > baseline_p99_ms * 105U) {
        f2_log("STOP", "reason=FINAL_RESOURCE_OR_CONTROL_P99");
        (void)f_mount(NULL, "0:", 0U);
        timing_begin(0U);
        return -12;
    }
    (void)snprintf(
        detail, sizeof(detail),
        "models=170|initial_exact=78|initial_sim_only=92|rag120_safe=%lu|rag120_source_bound=%lu|soak_rag_queries=%lu|swap_loads=%lu|loads=%lu|min_swap_per_model=%d|min_total_per_model=%d|failures=%lu|sd_kib_s=%lu|timeouts=%lu|collisions=%lu|control_p99_ms=%lu|heap_min=%lu|critical_stack_min=%lu",
        (unsigned long)rag_safe, (unsigned long)rag_source_bound,
        (unsigned long)soak_rag_queries,
        (unsigned long)swap_loads, (unsigned long)s_total_loads,
        minimum_swap_load_count(), minimum_load_count(),
        (unsigned long)s_total_failures, (unsigned long)sequential_kib,
        (unsigned long)bus_after.timeout_refusals,
        (unsigned long)bus_after.collision_refusals,
        (unsigned long)active_p99_ms, (unsigned long)heap_min_bytes,
        (unsigned long)critical_stack_min_bytes);
    f2_log("PASS", detail);
    (void)f_mount(NULL, "0:", 0U);
    timing_begin(0U);
    return 0;
}
