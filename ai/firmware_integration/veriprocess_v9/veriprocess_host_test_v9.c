#include "veriprocess_v9.h"

#include "sha256.h"

#include <stdio.h>
#include <string.h>

static uint32_t passed;
static uint32_t failed;

static void check_case(const char *name, int condition)
{
    if (condition) {
        ++passed;
    } else {
        ++failed;
        fprintf(stderr, "case_failed:%s\n", name);
    }
}

static void fill_id(uint8_t value[VP_ID_BYTES], const char *text)
{
    size_t bytes = strlen(text);
    memset(value, 0, VP_ID_BYTES);
    if (bytes > VP_ID_BYTES) {
        bytes = VP_ID_BYTES;
    }
    memcpy(value, text, bytes);
}

static vp_evidence_card_t make_card(const char *run, const char *source,
                                    const char *family, uint64_t seq,
                                    uint64_t monotonic_ms, uint8_t stage,
                                    uint8_t source_kind)
{
    vp_evidence_card_t card;
    memset(&card, 0, sizeof(card));
    fill_id(card.session_id, "SESSION-20260804");
    fill_id(card.run_id, run);
    fill_id(card.source_id, source);
    fill_id(card.independence_family, family);
    card.seq = seq;
    card.monotonic_ms = monotonic_ms;
    card.rtc_seconds = 1785780000ULL + monotonic_ms / 1000ULL;
    card.age_ms = 10U;
    card.quality = 0.98f;
    card.interval_lower = 0.40f;
    card.interval_upper = 0.60f;
    card.interval_coverage = 0.95f;
    card.stage = stage;
    card.source_kind = source_kind;
    card.truth_class = 0U;
    card.has_interval = 1U;
    card.authority = 0U;
    vp_evidence_seal(&card);
    return card;
}

static void payload_digest(uint32_t index, uint8_t output[VP_SHA_BYTES])
{
    uint8_t wire[8];
    uint32_t i;
    for (i = 0U; i < sizeof(wire); ++i) {
        wire[i] = (uint8_t)(index + 17U * i);
    }
    sha256(wire, sizeof(wire), output);
}

int main(void)
{
    vp_evidence_card_t cards[2];
    vp_evidence_card_t bad;
    vp_evidence_card_t post;
    vp_sintergraph_request_t request;
    vp_sintergraph_output_t prediction;
    vp_fulfillment_t fulfillment;
    vp_chrono_t chrono;
    vp_trace_ledger_t ledger;
    vp_trace_ledger_t copy;
    uint8_t payload[VP_SHA_BYTES];
    uint32_t independent = 0U;
    uint32_t i;
    uint8_t active_before;

    cards[0] = make_card("RUN-01", "MAX31856", "THERMOCOUPLE-K", 7U, 7000U,
                         VP_STAGE_SOAK, VP_SOURCE_MAX31856);
    cards[1] = make_card("RUN-01", "MLX90640", "THERMAL-IMAGER", 8U, 7200U,
                         VP_STAGE_SOAK, VP_SOURCE_MLX90640);
    check_case("evidence_valid", vp_evidence_validate(&cards[0]) == VP_OK);
    bad = cards[0]; bad.quality = 0.50f;
    check_case("evidence_tamper_hash", vp_evidence_validate(&bad) == VP_ERR_HASH);
    bad = cards[0]; bad.authority = 1U; vp_evidence_seal(&bad);
    check_case("evidence_authority", vp_evidence_validate(&bad) == VP_ERR_AUTHORITY);
    bad = cards[0]; bad.parent_count = 2U; memset(bad.parent_sha[0], 1, VP_SHA_BYTES);
    memcpy(bad.parent_sha[1], bad.parent_sha[0], VP_SHA_BYTES); vp_evidence_seal(&bad);
    check_case("evidence_duplicate_parent", vp_evidence_validate(&bad) == VP_ERR_SCHEMA);
    bad = cards[0]; bad.interval_lower = 2.0f; bad.interval_upper = 1.0f; vp_evidence_seal(&bad);
    check_case("evidence_interval_order", vp_evidence_validate(&bad) == VP_ERR_SCHEMA);

    memset(&request, 0, sizeof(request));
    fill_id(request.run_id, "RUN-01");
    memset(request.recipe_sha, 0x31, VP_SHA_BYTES);
    request.as_of_seq = 10U;
    request.as_of_monotonic_ms = 10000U;
    request.curve_count = 3U;
    request.curve[0].t_s = 0.0f; request.curve[0].temperature_c = 25.0f;
    request.curve[1].t_s = 900.0f; request.curve[1].temperature_c = 800.0f;
    request.curve[2].t_s = 1800.0f; request.curve[2].temperature_c = 800.0f;
    request.cards = cards;
    request.card_count = 2U;
    check_case("sintergraph_valid", vp_sintergraph_freeze(
        &request, 0.50f, 0.80f, 0.08f, &prediction) == VP_OK);
    check_case("sintergraph_output_frozen", prediction.frozen == 1U &&
        prediction.authority == 0U && prediction.psp_edges == 3U);
    bad = cards[0]; bad.seq = 11U; vp_evidence_seal(&bad); request.cards = &bad; request.card_count = 1U;
    check_case("sintergraph_future_seq", vp_sintergraph_freeze(
        &request, .5f, .8f, .08f, &prediction) == VP_ERR_FUTURE_EVIDENCE);
    bad = cards[0]; bad.monotonic_ms = 10001U; vp_evidence_seal(&bad); request.cards = &bad;
    check_case("sintergraph_future_time", vp_sintergraph_freeze(
        &request, .5f, .8f, .08f, &prediction) == VP_ERR_FUTURE_EVIDENCE);
    bad = cards[0]; bad.source_kind = VP_SOURCE_XRD; vp_evidence_seal(&bad); request.cards = &bad;
    check_case("sintergraph_same_run_xrd", vp_sintergraph_freeze(
        &request, .5f, .8f, .08f, &prediction) == VP_ERR_POSTBURN_LEAKAGE);
    bad = cards[0]; bad.stage = VP_STAGE_POST_RUN; vp_evidence_seal(&bad); request.cards = &bad;
    check_case("sintergraph_same_run_post", vp_sintergraph_freeze(
        &request, .5f, .8f, .08f, &prediction) == VP_ERR_POSTBURN_LEAKAGE);
    request.cards = cards; request.card_count = 2U;
    request.curve[1].t_s = 0.0f;
    check_case("sintergraph_curve_order", vp_sintergraph_freeze(
        &request, .5f, .8f, .08f, &prediction) == VP_ERR_SCHEMA);
    request.curve[1].t_s = 900.0f;
    check_case("sintergraph_refreeze", vp_sintergraph_freeze(
        &request, .5f, .8f, .08f, &prediction) == VP_OK);

    post = make_card("RUN-01", "XRD", "DIFFRACTION", 12U, 20000U,
                     VP_STAGE_METROLOGY, VP_SOURCE_XRD);
    check_case("fulfillment_valid", vp_sintergraph_fulfill(
        &prediction, &post, .52f, .78f, &fulfillment) == VP_OK);
    check_case("fulfillment_authority_zero", fulfillment.authority == 0U);
    bad = post; bad.monotonic_ms = 9000U; vp_evidence_seal(&bad);
    check_case("fulfillment_before_freeze", vp_sintergraph_fulfill(
        &prediction, &bad, .52f, .78f, &fulfillment) == VP_ERR_TIME);
    bad = post; bad.source_kind = VP_SOURCE_MAX31856; vp_evidence_seal(&bad);
    check_case("fulfillment_wrong_source", vp_sintergraph_fulfill(
        &prediction, &bad, .52f, .78f, &fulfillment) == VP_ERR_SCHEMA);

    check_case("root_two_independent", vp_root_admit(cards, 2U, 2U, &independent) == VP_OK &&
               independent == 2U);
    bad = cards[1]; memcpy(bad.independence_family, cards[0].independence_family, VP_ID_BYTES);
    vp_evidence_seal(&bad); cards[1] = bad;
    check_case("root_same_family_reject", vp_root_admit(cards, 2U, 2U, &independent) == VP_ERR_INDEPENDENCE);
    cards[1] = make_card("RUN-01", "MLX90640", "THERMAL-IMAGER", 8U, 7200U,
                         VP_STAGE_SOAK, VP_SOURCE_MLX90640);

    vp_chrono_init(&chrono);
    check_case("chrono_commit_without_load", vp_chrono_append(&chrono, 1029U, 1U, 1U, 0U) == VP_ERR_STATE);
    check_case("chrono_load_begin", vp_chrono_append(&chrono, 1024U, 2U, 1U, 0U) == VP_OK);
    check_case("chrono_schema", vp_chrono_append(&chrono, 1025U, 3U, 20U, 0U) == VP_OK);
    check_case("chrono_sha", vp_chrono_append(&chrono, 1026U, 4U, 7000U, 0U) == VP_OK);
    check_case("chrono_generation", vp_chrono_append(&chrono, 1027U, 5U, 2U, 0U) == VP_OK);
    check_case("chrono_golden", vp_chrono_append(&chrono, 1028U, 6U, 1000U, 0U) == VP_OK);
    check_case("chrono_commit", vp_chrono_append(&chrono, 1029U, 7U, 2U, 0U) == VP_OK);
    check_case("chrono_time_regress", vp_chrono_append(&chrono, 1040U, 6U, 2U, 0U) == VP_ERR_TIME);
    check_case("chrono_deadline", vp_chrono_append(&chrono, 1040U, 8U, 101U, 0U) == VP_ERR_TIME);
    vp_chrono_init(&chrono);
    check_case("chrono_fulfill_before_freeze", vp_chrono_append(&chrono, 1061U, 1U, 0U, 0U) == VP_ERR_STATE);
    check_case("chrono_sinter_freeze", vp_chrono_append(&chrono, 1060U, 2U, 1U, 0U) == VP_OK);
    check_case("chrono_sinter_fulfill", vp_chrono_append(&chrono, 1061U, 3U, 0U, 0U) == VP_OK);
    check_case("chrono_sync_without_wal", vp_chrono_append(&chrono, 1071U, 4U, 1U, 0U) == VP_ERR_STATE);
    check_case("chrono_wal", vp_chrono_append(&chrono, 1070U, 5U, 10U, 0U) == VP_OK);
    check_case("chrono_sync", vp_chrono_append(&chrono, 1071U, 6U, 100U, 0U) == VP_OK);

    vp_ledger_init(&ledger);
    check_case("ledger_init_verify", vp_ledger_verify(&ledger) == VP_OK);
    for (i = 0U; i < 10U; ++i) {
        payload_digest(i, payload);
        check_case("ledger_prepare_loop", vp_ledger_prepare(
            &ledger, payload, i + 1U, 1000U + i * 10U, 2000U + i * 10U) == VP_OK);
        check_case("ledger_commit_loop", vp_ledger_commit(&ledger) == VP_OK);
    }
    check_case("ledger_ab_generation", ledger.headers[ledger.active_header].record_count == 10U &&
               ledger.headers[ledger.active_header].generation == 11U);
    check_case("ledger_chain_verify", vp_ledger_verify(&ledger) == VP_OK);
    payload_digest(99U, payload);
    check_case("ledger_crash_prepare", vp_ledger_prepare(
        &ledger, payload, 11U, 1200U, 2200U) == VP_OK && ledger.wal.pending == 1U);
    check_case("ledger_recover_wal", vp_ledger_recover(&ledger) == VP_OK &&
               ledger.headers[ledger.active_header].record_count == 11U);
    check_case("ledger_recovered_verify", vp_ledger_verify(&ledger) == VP_OK);
    payload_digest(100U, payload);
    check_case("ledger_prepare_corrupt_wal", vp_ledger_prepare(
        &ledger, payload, 12U, 1300U, 2300U) == VP_OK);
    ledger.wal.wal_sha[0] ^= 1U;
    check_case("ledger_corrupt_wal_reject", vp_ledger_recover(&ledger) == VP_ERR_HASH);
    ledger = (copy = ledger); /* keep compiler-visible explicit snapshot */
    (void)copy;
    vp_ledger_init(&ledger);
    for (i = 0U; i < 3U; ++i) {
        payload_digest(200U + i, payload);
        (void)vp_ledger_prepare(&ledger, payload, i + 1U, 100U + i, 200U + i);
        (void)vp_ledger_commit(&ledger);
    }
    active_before = ledger.active_header;
    ledger.headers[active_before].checksum[0] ^= 1U;
    check_case("ledger_ab_header_fallback", vp_ledger_recover(&ledger) == VP_OK &&
               ledger.active_header != active_before);
    copy = ledger;
    memset(copy.headers[0].checksum, 0, VP_SHA_BYTES);
    memset(copy.headers[1].checksum, 0, VP_SHA_BYTES);
    check_case("ledger_both_headers_bad", vp_ledger_recover(&copy) == VP_ERR_STORAGE);
    vp_ledger_init(&ledger);
    payload_digest(1U, payload);
    (void)vp_ledger_prepare(&ledger, payload, 5U, 500U, 500U);
    (void)vp_ledger_commit(&ledger);
    check_case("ledger_seq_regress", vp_ledger_prepare(&ledger, payload, 5U, 501U, 501U) == VP_ERR_TIME);
    check_case("ledger_monotonic_regress", vp_ledger_prepare(&ledger, payload, 6U, 499U, 501U) == VP_ERR_TIME);
    check_case("ledger_rtc_regress", vp_ledger_prepare(&ledger, payload, 6U, 501U, 499U) == VP_ERR_TIME);
    copy = ledger; copy.records[0].payload_sha[0] ^= 1U;
    check_case("ledger_record_tamper", vp_ledger_verify(&copy) == VP_ERR_HASH);
    copy = ledger; copy.authority = 1U;
    check_case("ledger_authority", vp_ledger_verify(&copy) == VP_ERR_AUTHORITY);

    printf("{\"status\":\"%s\",\"passed\":%u,\"failed\":%u,"
           "\"authority\":0,\"board_accepted\":false}\n",
           failed == 0U ? "PASS" : "FAIL", passed, failed);
    return failed == 0U ? 0 : 1;
}
