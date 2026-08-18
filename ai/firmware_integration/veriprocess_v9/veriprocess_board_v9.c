#include "veriprocess_board_v9.h"

#include "FreeRTOS.h"
#include "task.h"
#include "FatFs/ff.h"
#include "ds3231.h"
#include "sha256.h"
#include "veriprocess_v9.h"

#include <string.h>

#define VP_TRACE_A "0:/F200/TRACE/VPA.BIN"
#define VP_TRACE_B "0:/F200/TRACE/VPB.BIN"
#define VP_TRACE_WAL "0:/F200/TRACE/VPWAL.BIN"
#define VP_TRACE_DRILL_DONE "0:/F200/TRACE/VPDRILL.OK"

static vp_trace_ledger_t s_ledger;
static vp_trace_ledger_t s_candidate_a;
static vp_trace_ledger_t s_candidate_b;
static vp_chrono_t s_chrono;
static vp_evidence_card_t s_cards[2];
static const uint8_t s_drill_done[16] = "VPDRILL-V9-DONE";

static int write_sync(const char *path, const void *value, uint32_t bytes)
{
    FIL file;
    UINT written = 0U;
    if (f_open(&file, path, FA_CREATE_ALWAYS | FA_WRITE) != FR_OK) {
        return -1;
    }
    if (f_write(&file, value, bytes, &written) != FR_OK || written != bytes ||
        f_sync(&file) != FR_OK || f_close(&file) != FR_OK) {
        return -2;
    }
    return 0;
}

static int read_exact(const char *path, void *value, uint32_t bytes)
{
    FIL file;
    UINT read = 0U;
    if (f_open(&file, path, FA_READ) != FR_OK) {
        return -1;
    }
    if (f_size(&file) != bytes || f_read(&file, value, bytes, &read) != FR_OK ||
        read != bytes || f_close(&file) != FR_OK) {
        return -2;
    }
    return 0;
}

static uint64_t ledger_generation(const vp_trace_ledger_t *ledger)
{
    return ledger->headers[ledger->active_header].generation;
}

static int load_best_ledger(vp_trace_ledger_t *ledger)
{
    int valid_a = read_exact(VP_TRACE_A, &s_candidate_a, sizeof(s_candidate_a)) == 0 &&
                  vp_ledger_verify(&s_candidate_a) == VP_OK;
    int valid_b = read_exact(VP_TRACE_B, &s_candidate_b, sizeof(s_candidate_b)) == 0 &&
                  vp_ledger_verify(&s_candidate_b) == VP_OK;
    if (!valid_a && !valid_b) {
        vp_ledger_init(ledger);
        return 0;
    }
    if (valid_b && (!valid_a || ledger_generation(&s_candidate_b) >
                              ledger_generation(&s_candidate_a))) {
        *ledger = s_candidate_b;
    } else {
        *ledger = s_candidate_a;
    }
    return 0;
}

static int persist_active(const vp_trace_ledger_t *ledger)
{
    return write_sync(ledger->active_header == 0U ? VP_TRACE_A : VP_TRACE_B,
                      ledger, sizeof(*ledger));
}

static void fill_id(uint8_t value[VP_ID_BYTES], const char *text)
{
    uint32_t bytes = (uint32_t)strlen(text);
    memset(value, 0, VP_ID_BYTES);
    if (bytes > VP_ID_BYTES) bytes = VP_ID_BYTES;
    memcpy(value, text, bytes);
}

static vp_evidence_card_t make_fixture_card(const char *source,
                                            const char *family,
                                            uint64_t seq,
                                            uint64_t monotonic_ms,
                                            uint8_t source_kind)
{
    vp_evidence_card_t card;
    memset(&card, 0, sizeof(card));
    fill_id(card.session_id, "BOARD-ACCEPT-V9");
    fill_id(card.run_id, "FIXTURE-RUN-V9");
    fill_id(card.source_id, source);
    fill_id(card.independence_family, family);
    card.seq = seq;
    card.monotonic_ms = monotonic_ms;
    card.rtc_seconds = 20260804000000ULL;
    card.quality = 0.95f;
    card.interval_lower = 0.4f;
    card.interval_upper = 0.6f;
    card.interval_coverage = 0.95f;
    card.stage = VP_STAGE_FIXTURE;
    card.source_kind = source_kind;
    card.truth_class = 7U;
    card.has_interval = 1U;
    vp_evidence_seal(&card);
    return card;
}

static uint64_t rtc_order_value(const ds3231_time_t *time)
{
    uint64_t value = time->year;
    value = value * 100ULL + time->month;
    value = value * 100ULL + time->day;
    value = value * 100ULL + time->hour;
    value = value * 100ULL + time->minute;
    return value * 100ULL + time->second;
}

int veriprocess_board_selftest_v9(veriprocess_board_receipt_v9_t *receipt)
{
    ds3231_snapshot_t rtc;
    vp_ledger_wal_t recovered_wal;
    vp_sintergraph_request_t request;
    vp_sintergraph_output_t prediction;
    uint8_t payload[VP_SHA_BYTES];
    uint8_t payload_wire[32];
    uint8_t drill_marker[sizeof(s_drill_done)];
    uint64_t monotonic_ms;
    uint64_t rtc_value;
    uint64_t next_seq;
    uint32_t independent;
    uint32_t i;
    static const uint16_t chrono_ids[] = {
        1024U, 1025U, 1026U, 1027U, 1028U, 1029U,
        1060U, 1061U, 1070U, 1071U, 1072U
    };
    static const uint32_t chrono_elapsed[] = {
        1U, 20U, 100U, 2U, 100U, 2U, 1U, 0U, 10U, 100U, 100U
    };
    if (receipt == NULL) {
        return -1;
    }
    memset(receipt, 0, sizeof(*receipt));
    /* FF_FS_MINIMIZE=1 deliberately excludes f_mkdir in the frozen FatFs
     * configuration.  The immutable SD staging release creates F200/TRACE;
     * fail closed through the first write if the directory is absent. */
    (void)load_best_ledger(&s_ledger);
    receipt->wal_recovered = 0U;
    memset(&recovered_wal, 0, sizeof(recovered_wal));
    if (read_exact(VP_TRACE_WAL, &recovered_wal, sizeof(recovered_wal)) == 0 &&
        recovered_wal.pending != 0U) {
        s_ledger.wal = recovered_wal;
        if (vp_ledger_recover(&s_ledger) != VP_OK || persist_active(&s_ledger) != 0) {
            return -3;
        }
        receipt->wal_recovered = 1U;
        if (write_sync(VP_TRACE_DRILL_DONE, s_drill_done,
                       sizeof(s_drill_done)) != 0) {
            return -10;
        }
    }
    memset(&rtc, 0, sizeof(rtc));
    if (ds3231_read_snapshot(&rtc) != 0U || rtc.time.valid == 0U || rtc.osf != 0U) {
        return -4;
    }
    receipt->ds3231_valid = 1U;
    monotonic_ms = (uint64_t)xTaskGetTickCount() * portTICK_PERIOD_MS;
    if (s_ledger.headers[s_ledger.active_header].record_count != 0U) {
        monotonic_ms += s_ledger.records[
            s_ledger.headers[s_ledger.active_header].record_count - 1U].monotonic_ms + 1U;
    }
    rtc_value = rtc_order_value(&rtc.time);
    next_seq = s_ledger.headers[s_ledger.active_header].record_count == 0U ? 1U :
        s_ledger.records[s_ledger.headers[s_ledger.active_header].record_count - 1U].seq + 1U;
    memset(payload_wire, 0, sizeof(payload_wire));
    memcpy(payload_wire, "VERIPROCESS-BOARD-V9", 20U);
    memcpy(payload_wire + 20U, &monotonic_ms, sizeof(monotonic_ms));
    sha256(payload_wire, sizeof(payload_wire), payload);

    /* On a fresh staging card, stop after a durable WAL prepare and require a
     * real power removal.  The next boot must recover this exact record before
     * VPDRILL.OK is sealed, after which normal and 24 h boots do not re-arm. */
    memset(drill_marker, 0, sizeof(drill_marker));
    if (receipt->wal_recovered == 0U &&
        (read_exact(VP_TRACE_DRILL_DONE, drill_marker,
                    sizeof(drill_marker)) != 0 ||
         memcmp(drill_marker, s_drill_done, sizeof(drill_marker)) != 0)) {
        if (vp_ledger_prepare(&s_ledger, payload, next_seq, monotonic_ms,
                              rtc_value) != VP_OK ||
            write_sync(VP_TRACE_WAL, &s_ledger.wal, sizeof(s_ledger.wal)) != 0) {
            return -11;
        }
        receipt->ledger_generation = ledger_generation(&s_ledger);
        receipt->ledger_records =
            s_ledger.headers[s_ledger.active_header].record_count;
        receipt->authority = 0U;
        return VERIPROCESS_BOARD_POWER_CUT_ARMED;
    }
    if (vp_ledger_prepare(&s_ledger, payload, next_seq, monotonic_ms, rtc_value) != VP_OK ||
        write_sync(VP_TRACE_WAL, &s_ledger.wal, sizeof(s_ledger.wal)) != 0 ||
        vp_ledger_commit(&s_ledger) != VP_OK || persist_active(&s_ledger) != 0) {
        return -5;
    }
    memset(&recovered_wal, 0, sizeof(recovered_wal));
    if (write_sync(VP_TRACE_WAL, &recovered_wal, sizeof(recovered_wal)) != 0 ||
        vp_ledger_verify(&s_ledger) != VP_OK) {
        return -6;
    }

    s_cards[0] = make_fixture_card("MAX31856", "THERMOCOUPLE-K", 1U, 1000U,
                                    VP_SOURCE_MAX31856);
    s_cards[1] = make_fixture_card("MLX90640", "THERMAL-IMAGER", 2U, 1100U,
                                    VP_SOURCE_MLX90640);
    if (vp_root_admit(s_cards, 2U, 2U, &independent) != VP_OK) {
        return -7;
    }
    memset(&request, 0, sizeof(request));
    fill_id(request.run_id, "FIXTURE-RUN-V9");
    memset(request.recipe_sha, 0x42, VP_SHA_BYTES);
    request.as_of_seq = 2U;
    request.as_of_monotonic_ms = 1200U;
    request.curve_count = 2U;
    request.curve[0].t_s = 0.0f;
    request.curve[0].temperature_c = 25.0f;
    request.curve[1].t_s = 900.0f;
    request.curve[1].temperature_c = 800.0f;
    request.cards = s_cards;
    request.card_count = 2U;
    if (vp_sintergraph_freeze(&request, .5f, .8f, .08f, &prediction) != VP_OK) {
        return -8;
    }
    vp_chrono_init(&s_chrono);
    for (i = 0U; i < sizeof(chrono_ids) / sizeof(chrono_ids[0]); ++i) {
        if (vp_chrono_append(&s_chrono, chrono_ids[i], i + 1U,
                             chrono_elapsed[i], 0U) != VP_OK) {
            return -9;
        }
    }
    receipt->ledger_generation = ledger_generation(&s_ledger);
    receipt->ledger_records = s_ledger.headers[s_ledger.active_header].record_count;
    receipt->chrono_events = s_chrono.count;
    receipt->independent_families = independent;
    receipt->sintergraph_frozen = prediction.frozen;
    receipt->authority = 0U;
    return 0;
}
