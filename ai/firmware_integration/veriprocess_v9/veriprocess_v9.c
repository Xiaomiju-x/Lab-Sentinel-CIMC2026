#include "veriprocess_v9.h"

#include "sha256.h"

#include <math.h>
#include <string.h>

static int bytes_zero(const uint8_t *value, uint32_t bytes)
{
    uint32_t i;
    for (i = 0U; i < bytes; ++i) {
        if (value[i] != 0U) {
            return 0;
        }
    }
    return 1;
}

static void sha_u32(sha256_ctx *sha, uint32_t value)
{
    uint8_t wire[4];
    wire[0] = (uint8_t)value;
    wire[1] = (uint8_t)(value >> 8);
    wire[2] = (uint8_t)(value >> 16);
    wire[3] = (uint8_t)(value >> 24);
    sha256_update(sha, wire, sizeof(wire));
}

static void sha_u64(sha256_ctx *sha, uint64_t value)
{
    uint8_t wire[8];
    uint32_t i;
    for (i = 0U; i < sizeof(wire); ++i) {
        wire[i] = (uint8_t)(value >> (8U * i));
    }
    sha256_update(sha, wire, sizeof(wire));
}

static void sha_float(sha256_ctx *sha, float value)
{
    uint32_t wire;
    memcpy(&wire, &value, sizeof(wire));
    sha_u32(sha, wire);
}

static void evidence_digest(const vp_evidence_card_t *card, uint8_t output[VP_SHA_BYTES])
{
    static const uint8_t domain[] = "CIMC-EVIDENCE-CARD-V2";
    sha256_ctx sha;
    uint32_t i;
    sha256_init(&sha);
    sha256_update(&sha, domain, sizeof(domain) - 1U);
    sha256_update(&sha, card->session_id, VP_ID_BYTES);
    sha256_update(&sha, card->run_id, VP_ID_BYTES);
    sha256_update(&sha, card->source_id, VP_ID_BYTES);
    sha256_update(&sha, card->independence_family, VP_ID_BYTES);
    sha256_update(&sha, card->model_release_root, VP_SHA_BYTES);
    sha_u64(&sha, card->seq);
    sha_u64(&sha, card->monotonic_ms);
    sha_u64(&sha, card->rtc_seconds);
    sha_u64(&sha, card->age_ms);
    sha_float(&sha, card->quality);
    sha_float(&sha, card->interval_lower);
    sha_float(&sha, card->interval_upper);
    sha_float(&sha, card->interval_coverage);
    sha256_update(&sha, &card->stage, 1U);
    sha256_update(&sha, &card->source_kind, 1U);
    sha256_update(&sha, &card->truth_class, 1U);
    sha256_update(&sha, &card->parent_count, 1U);
    sha256_update(&sha, &card->has_interval, 1U);
    sha256_update(&sha, &card->has_model, 1U);
    sha256_update(&sha, &card->authority, 1U);
    for (i = 0U; i < card->parent_count && i < VP_EVIDENCE_MAX; ++i) {
        sha256_update(&sha, card->parent_sha[i], VP_SHA_BYTES);
    }
    sha256_final(&sha, output);
}

void vp_evidence_seal(vp_evidence_card_t *card)
{
    if (card != NULL) {
        evidence_digest(card, card->summary_sha);
    }
}

vp_status_t vp_evidence_validate(const vp_evidence_card_t *card)
{
    uint8_t digest[VP_SHA_BYTES];
    uint32_t i;
    uint32_t j;
    if (card == NULL) {
        return VP_ERR_ARGUMENT;
    }
    if (card->authority != 0U) {
        return VP_ERR_AUTHORITY;
    }
    if (bytes_zero(card->session_id, VP_ID_BYTES) ||
        bytes_zero(card->run_id, VP_ID_BYTES) ||
        bytes_zero(card->source_id, VP_ID_BYTES) ||
        bytes_zero(card->independence_family, VP_ID_BYTES) ||
        card->stage > VP_STAGE_FIXTURE || card->parent_count > VP_EVIDENCE_MAX ||
        card->has_interval > 1U || card->has_model > 1U ||
        !isfinite(card->quality) || card->quality < 0.0f || card->quality > 1.0f) {
        return VP_ERR_SCHEMA;
    }
    if (card->has_model == 0U && !bytes_zero(card->model_release_root, VP_SHA_BYTES)) {
        return VP_ERR_SCHEMA;
    }
    if (card->has_interval != 0U &&
        (!isfinite(card->interval_lower) || !isfinite(card->interval_upper) ||
         !isfinite(card->interval_coverage) ||
         card->interval_lower > card->interval_upper ||
         card->interval_coverage <= 0.0f || card->interval_coverage >= 1.0f)) {
        return VP_ERR_SCHEMA;
    }
    for (i = 0U; i < card->parent_count; ++i) {
        if (bytes_zero(card->parent_sha[i], VP_SHA_BYTES)) {
            return VP_ERR_SCHEMA;
        }
        for (j = i + 1U; j < card->parent_count; ++j) {
            if (memcmp(card->parent_sha[i], card->parent_sha[j], VP_SHA_BYTES) == 0) {
                return VP_ERR_SCHEMA;
            }
        }
    }
    evidence_digest(card, digest);
    return memcmp(digest, card->summary_sha, VP_SHA_BYTES) == 0 ? VP_OK : VP_ERR_HASH;
}

static int source_is_postburn(uint8_t source_kind)
{
    return source_kind == VP_SOURCE_XRD || source_kind == VP_SOURCE_PL ||
           source_kind == VP_SOURCE_SEM || source_kind == VP_SOURCE_EDS ||
           source_kind == VP_SOURCE_POST_RUN_QUALITY;
}

vp_status_t vp_sintergraph_freeze(
    const vp_sintergraph_request_t *request,
    float model_structure_mean,
    float model_performance_mean,
    float model_uncertainty,
    vp_sintergraph_output_t *output)
{
    static const uint8_t domain[] = "CIMC-SINTERGRAPH-PSP-R1";
    sha256_ctx sha;
    uint32_t i;
    float peak = -1000.0f;
    float thermal_dose = 0.0f;
    if (request == NULL || output == NULL || request->cards == NULL) {
        return VP_ERR_ARGUMENT;
    }
    if (request->authority != 0U) {
        return VP_ERR_AUTHORITY;
    }
    if (bytes_zero(request->run_id, VP_ID_BYTES) ||
        bytes_zero(request->recipe_sha, VP_SHA_BYTES) ||
        request->curve_count < 2U || request->curve_count > VP_CURVE_MAX ||
        request->card_count == 0U || request->card_count > VP_EVIDENCE_MAX ||
        !isfinite(model_structure_mean) || !isfinite(model_performance_mean) ||
        !isfinite(model_uncertainty) || model_uncertainty < 0.0f) {
        return VP_ERR_SCHEMA;
    }
    for (i = 0U; i < request->curve_count; ++i) {
        if (!isfinite(request->curve[i].t_s) ||
            !isfinite(request->curve[i].temperature_c) ||
            request->curve[i].t_s < 0.0f ||
            (i != 0U && request->curve[i].t_s <= request->curve[i - 1U].t_s)) {
            return VP_ERR_SCHEMA;
        }
        if (request->curve[i].temperature_c > peak) {
            peak = request->curve[i].temperature_c;
        }
        if (i != 0U) {
            float dt = request->curve[i].t_s - request->curve[i - 1U].t_s;
            float average = 0.5f * (request->curve[i].temperature_c +
                                    request->curve[i - 1U].temperature_c);
            thermal_dose += dt * fmaxf(average - 20.0f, 0.0f);
        }
    }
    for (i = 0U; i < request->card_count; ++i) {
        const vp_evidence_card_t *card = &request->cards[i];
        vp_status_t status = vp_evidence_validate(card);
        if (status != VP_OK) {
            return status;
        }
        if (card->seq > request->as_of_seq ||
            card->monotonic_ms > request->as_of_monotonic_ms) {
            return VP_ERR_FUTURE_EVIDENCE;
        }
        if (memcmp(card->run_id, request->run_id, VP_ID_BYTES) == 0 &&
            (source_is_postburn(card->source_kind) ||
             card->stage == VP_STAGE_POST_RUN || card->stage == VP_STAGE_METROLOGY)) {
            return VP_ERR_POSTBURN_LEAKAGE;
        }
    }
    memset(output, 0, sizeof(*output));
    memcpy(output->run_id, request->run_id, VP_ID_BYTES);
    output->as_of_seq = request->as_of_seq;
    output->as_of_monotonic_ms = request->as_of_monotonic_ms;
    output->structure_mean = model_structure_mean;
    output->performance_mean = model_performance_mean;
    output->uncertainty = fminf(model_uncertainty + 0.02f * (float)(request->card_count - 1U), 1.0f);
    output->structure_std = fmaxf(0.01f, output->uncertainty * (1.0f + peak / 1000.0f));
    output->performance_std = fmaxf(0.01f, output->uncertainty * 1.5f);
    output->transfer_confidence = fmaxf(0.0f, fminf(1.0f,
        1.0f - output->uncertainty - 1.0f / (1.0f + thermal_dose / 100000.0f)));
    output->psp_edges = 3U;
    output->authority = 0U;
    output->frozen = 1U;
    sha256_init(&sha);
    sha256_update(&sha, domain, sizeof(domain) - 1U);
    sha256_update(&sha, request->run_id, VP_ID_BYTES);
    sha256_update(&sha, request->recipe_sha, VP_SHA_BYTES);
    sha_u64(&sha, request->as_of_seq);
    sha_u64(&sha, request->as_of_monotonic_ms);
    for (i = 0U; i < request->curve_count; ++i) {
        sha_float(&sha, request->curve[i].t_s);
        sha_float(&sha, request->curve[i].temperature_c);
    }
    for (i = 0U; i < request->card_count; ++i) {
        sha256_update(&sha, request->cards[i].summary_sha, VP_SHA_BYTES);
    }
    sha_float(&sha, output->structure_mean);
    sha_float(&sha, output->performance_mean);
    sha_float(&sha, output->uncertainty);
    sha256_final(&sha, output->release_root);
    return VP_OK;
}

vp_status_t vp_sintergraph_fulfill(
    const vp_sintergraph_output_t *prediction,
    const vp_evidence_card_t *postburn_card,
    float observed_structure,
    float observed_performance,
    vp_fulfillment_t *fulfillment)
{
    uint64_t elapsed;
    vp_status_t status;
    if (prediction == NULL || postburn_card == NULL || fulfillment == NULL) {
        return VP_ERR_ARGUMENT;
    }
    if (prediction->authority != 0U || postburn_card->authority != 0U) {
        return VP_ERR_AUTHORITY;
    }
    status = vp_evidence_validate(postburn_card);
    if (status != VP_OK) {
        return status;
    }
    if (prediction->frozen == 0U ||
        memcmp(prediction->run_id, postburn_card->run_id, VP_ID_BYTES) != 0) {
        return VP_ERR_STATE;
    }
    if (postburn_card->seq <= prediction->as_of_seq ||
        postburn_card->monotonic_ms <= prediction->as_of_monotonic_ms) {
        return VP_ERR_TIME;
    }
    if (!source_is_postburn(postburn_card->source_kind) ||
        (postburn_card->stage != VP_STAGE_POST_RUN &&
         postburn_card->stage != VP_STAGE_METROLOGY)) {
        return VP_ERR_SCHEMA;
    }
    if (!isfinite(observed_structure) || !isfinite(observed_performance)) {
        return VP_ERR_SCHEMA;
    }
    memset(fulfillment, 0, sizeof(*fulfillment));
    fulfillment->structure_residual = observed_structure - prediction->structure_mean;
    fulfillment->performance_residual = observed_performance - prediction->performance_mean;
    fulfillment->interval_consistent = (uint8_t)(
        fabsf(fulfillment->structure_residual) <= 2.0f * prediction->structure_std &&
        fabsf(fulfillment->performance_residual) <= 2.0f * prediction->performance_std);
    elapsed = postburn_card->monotonic_ms - prediction->as_of_monotonic_ms;
    fulfillment->horizon_short_mature = (uint8_t)(elapsed >= 1000ULL);
    fulfillment->horizon_medium_mature = (uint8_t)(elapsed >= 60000ULL);
    fulfillment->horizon_long_mature = (uint8_t)(elapsed >= 3600000ULL);
    fulfillment->authority = 0U;
    return VP_OK;
}

vp_status_t vp_root_admit(const vp_evidence_card_t *cards, uint32_t card_count,
                          uint32_t minimum_independent_families,
                          uint32_t *independent_families)
{
    uint32_t unique = 0U;
    uint32_t i;
    uint32_t j;
    if (cards == NULL || independent_families == NULL || card_count == 0U ||
        card_count > VP_EVIDENCE_MAX || minimum_independent_families < 2U) {
        return VP_ERR_ARGUMENT;
    }
    for (i = 0U; i < card_count; ++i) {
        int seen = 0;
        vp_status_t status = vp_evidence_validate(&cards[i]);
        if (status != VP_OK) {
            return status;
        }
        for (j = 0U; j < i; ++j) {
            if (memcmp(cards[i].independence_family,
                       cards[j].independence_family, VP_ID_BYTES) == 0) {
                seen = 1;
            }
        }
        if (!seen) {
            ++unique;
        }
    }
    *independent_families = unique;
    return unique >= minimum_independent_families ? VP_OK : VP_ERR_INDEPENDENCE;
}

void vp_chrono_init(vp_chrono_t *chrono)
{
    if (chrono != NULL) {
        memset(chrono, 0, sizeof(*chrono));
    }
}

static uint32_t event_deadline(uint16_t event_id)
{
    switch (event_id) {
    case 1024: return 1U;
    case 1025: return 50U;
    case 1026: return 8000U;
    case 1027: return 10U;
    case 1028: return 2000U;
    case 1029: return 10U;
    case 1030: return 10U;
    case 1040: return 100U;
    case 1041: return 30000U;
    case 1050: return 1U;
    case 1051: return 2000U;
    case 1052: return 25000U;
    case 1060: return 1U;
    case 1061: return 0U;
    case 1070: return 100U;
    case 1071: return 2000U;
    case 1072: return 5000U;
    default: return 0xFFFFFFFFUL;
    }
}

vp_status_t vp_chrono_append(vp_chrono_t *chrono, uint16_t event_id,
                             uint64_t monotonic_ms, uint32_t elapsed_ms,
                             uint16_t status)
{
    uint32_t deadline;
    if (chrono == NULL) {
        return VP_ERR_ARGUMENT;
    }
    if (chrono->authority != 0U) {
        return VP_ERR_AUTHORITY;
    }
    deadline = event_deadline(event_id);
    if (deadline == 0xFFFFFFFFUL || chrono->count >= VP_CHRONO_MAX ||
        monotonic_ms < chrono->last_monotonic_ms) {
        return deadline == 0xFFFFFFFFUL ? VP_ERR_SCHEMA : VP_ERR_TIME;
    }
    if (deadline != 0U && event_id != 1041U && elapsed_ms > deadline) {
        return VP_ERR_TIME;
    }
    switch (event_id) {
    case 1024: chrono->model_mask = 1U; break;
    case 1025: if (chrono->model_mask != 1U) return VP_ERR_STATE; chrono->model_mask |= 2U; break;
    case 1026: if (chrono->model_mask != 3U) return VP_ERR_STATE; chrono->model_mask |= 4U; break;
    case 1027: if (chrono->model_mask != 7U) return VP_ERR_STATE; chrono->model_mask |= 8U; break;
    case 1028: if (chrono->model_mask != 15U) return VP_ERR_STATE; chrono->model_mask |= 16U; break;
    case 1029: if (chrono->model_mask != 31U) return VP_ERR_STATE; break;
    case 1030: chrono->model_mask = 0U; break;
    case 1060: chrono->sinter_frozen = 1U; break;
    case 1061: if (chrono->sinter_frozen == 0U) return VP_ERR_STATE; break;
    case 1070: chrono->wal_prepared = 1U; break;
    case 1071: if (chrono->wal_prepared == 0U) return VP_ERR_STATE; chrono->wal_prepared = 0U; break;
    default: break;
    }
    chrono->events[chrono->count].id = event_id;
    chrono->events[chrono->count].status = status;
    chrono->events[chrono->count].elapsed_ms = elapsed_ms;
    chrono->events[chrono->count].monotonic_ms = monotonic_ms;
    ++chrono->count;
    chrono->last_monotonic_ms = monotonic_ms;
    return VP_OK;
}

static void ledger_record_hash(const vp_ledger_record_t *record,
                               uint8_t output[VP_SHA_BYTES])
{
    static const uint8_t domain[] = "CIMC-PROOFPASS-R3-RECORD";
    sha256_ctx sha;
    sha256_init(&sha);
    sha256_update(&sha, domain, sizeof(domain) - 1U);
    sha256_update(&sha, record->payload_sha, VP_SHA_BYTES);
    sha256_update(&sha, record->previous_root, VP_SHA_BYTES);
    sha_u64(&sha, record->seq);
    sha_u64(&sha, record->monotonic_ms);
    sha_u64(&sha, record->rtc_seconds);
    sha256_final(&sha, output);
}

static void wal_hash(const vp_ledger_wal_t *wal, uint8_t output[VP_SHA_BYTES])
{
    static const uint8_t domain[] = "CIMC-TRACE-WAL-V1";
    sha256_ctx sha;
    sha256_init(&sha);
    sha256_update(&sha, domain, sizeof(domain) - 1U);
    sha256_update(&sha, wal->record.record_sha, VP_SHA_BYTES);
    sha256_update(&sha, wal->record.payload_sha, VP_SHA_BYTES);
    sha_u64(&sha, wal->record.seq);
    sha_u64(&sha, wal->record.monotonic_ms);
    sha_u64(&sha, wal->record.rtc_seconds);
    sha256_final(&sha, output);
}

static void ledger_root(const uint8_t previous[VP_SHA_BYTES],
                        const uint8_t record[VP_SHA_BYTES], uint64_t seq,
                        uint8_t output[VP_SHA_BYTES])
{
    static const uint8_t domain[] = "CIMC-TRACE-MERKLE-SEGMENT-R1";
    sha256_ctx sha;
    sha256_init(&sha);
    sha256_update(&sha, domain, sizeof(domain) - 1U);
    sha256_update(&sha, previous, VP_SHA_BYTES);
    sha256_update(&sha, record, VP_SHA_BYTES);
    sha_u64(&sha, seq);
    sha256_final(&sha, output);
}

static void header_checksum(const vp_ledger_header_t *header,
                            uint8_t output[VP_SHA_BYTES])
{
    static const uint8_t domain[] = "CIMC-TRACE-AB-HEADER-R1";
    sha256_ctx sha;
    sha256_init(&sha);
    sha256_update(&sha, domain, sizeof(domain) - 1U);
    sha256_update(&sha, header->magic, sizeof(header->magic));
    sha256_update(&sha, header->root, VP_SHA_BYTES);
    sha_u64(&sha, header->generation);
    sha_u32(&sha, header->record_count);
    sha256_update(&sha, &header->valid, 1U);
    sha256_final(&sha, output);
}

static int header_valid(const vp_ledger_header_t *header)
{
    uint8_t digest[VP_SHA_BYTES];
    if (memcmp(header->magic, "VPLH", 4U) != 0 || header->valid != 1U ||
        header->record_count > VP_LEDGER_RECORD_MAX || header->generation == 0U) {
        return 0;
    }
    header_checksum(header, digest);
    return memcmp(digest, header->checksum, VP_SHA_BYTES) == 0;
}

void vp_ledger_init(vp_trace_ledger_t *ledger)
{
    if (ledger != NULL) {
        memset(ledger, 0, sizeof(*ledger));
        memcpy(ledger->headers[0].magic, "VPLH", 4U);
        ledger->headers[0].generation = 1U;
        ledger->headers[0].valid = 1U;
        header_checksum(&ledger->headers[0], ledger->headers[0].checksum);
        ledger->active_header = 0U;
    }
}

vp_status_t vp_ledger_prepare(vp_trace_ledger_t *ledger,
                              const uint8_t payload_sha[VP_SHA_BYTES],
                              uint64_t seq, uint64_t monotonic_ms,
                              uint64_t rtc_seconds)
{
    const vp_ledger_header_t *active;
    const vp_ledger_record_t *last;
    if (ledger == NULL || payload_sha == NULL) {
        return VP_ERR_ARGUMENT;
    }
    if (ledger->authority != 0U) {
        return VP_ERR_AUTHORITY;
    }
    if (ledger->wal.pending != 0U || bytes_zero(payload_sha, VP_SHA_BYTES) ||
        ledger->active_header > 1U || !header_valid(&ledger->headers[ledger->active_header])) {
        return VP_ERR_STATE;
    }
    active = &ledger->headers[ledger->active_header];
    if (active->record_count >= VP_LEDGER_RECORD_MAX) {
        return VP_ERR_CAPACITY;
    }
    if (active->record_count != 0U) {
        last = &ledger->records[active->record_count - 1U];
        if (seq <= last->seq || monotonic_ms < last->monotonic_ms ||
            rtc_seconds < last->rtc_seconds) {
            return VP_ERR_TIME;
        }
    }
    memset(&ledger->wal, 0, sizeof(ledger->wal));
    memcpy(ledger->wal.record.payload_sha, payload_sha, VP_SHA_BYTES);
    memcpy(ledger->wal.record.previous_root, active->root, VP_SHA_BYTES);
    ledger->wal.record.seq = seq;
    ledger->wal.record.monotonic_ms = monotonic_ms;
    ledger->wal.record.rtc_seconds = rtc_seconds;
    ledger_record_hash(&ledger->wal.record, ledger->wal.record.record_sha);
    ledger->wal.pending = 1U;
    wal_hash(&ledger->wal, ledger->wal.wal_sha);
    return VP_OK;
}

vp_status_t vp_ledger_commit(vp_trace_ledger_t *ledger)
{
    uint8_t digest[VP_SHA_BYTES];
    uint8_t inactive;
    vp_ledger_header_t *next;
    const vp_ledger_header_t *active;
    if (ledger == NULL) {
        return VP_ERR_ARGUMENT;
    }
    if (ledger->authority != 0U) {
        return VP_ERR_AUTHORITY;
    }
    if (ledger->wal.pending != 1U || ledger->active_header > 1U ||
        !header_valid(&ledger->headers[ledger->active_header])) {
        return VP_ERR_STATE;
    }
    wal_hash(&ledger->wal, digest);
    if (memcmp(digest, ledger->wal.wal_sha, VP_SHA_BYTES) != 0) {
        return VP_ERR_HASH;
    }
    ledger_record_hash(&ledger->wal.record, digest);
    if (memcmp(digest, ledger->wal.record.record_sha, VP_SHA_BYTES) != 0) {
        return VP_ERR_HASH;
    }
    active = &ledger->headers[ledger->active_header];
    if (active->record_count >= VP_LEDGER_RECORD_MAX ||
        memcmp(active->root, ledger->wal.record.previous_root, VP_SHA_BYTES) != 0) {
        return VP_ERR_STATE;
    }
    ledger->records[active->record_count] = ledger->wal.record;
    inactive = (uint8_t)(ledger->active_header ^ 1U);
    next = &ledger->headers[inactive];
    memset(next, 0, sizeof(*next));
    memcpy(next->magic, "VPLH", 4U);
    ledger_root(active->root, ledger->wal.record.record_sha,
                ledger->wal.record.seq, next->root);
    next->generation = active->generation + 1U;
    next->record_count = active->record_count + 1U;
    next->valid = 1U;
    header_checksum(next, next->checksum);
    ledger->active_header = inactive;
    memset(&ledger->wal, 0, sizeof(ledger->wal));
    return VP_OK;
}

vp_status_t vp_ledger_recover(vp_trace_ledger_t *ledger)
{
    uint8_t best;
    if (ledger == NULL) {
        return VP_ERR_ARGUMENT;
    }
    if (ledger->authority != 0U) {
        return VP_ERR_AUTHORITY;
    }
    if (!header_valid(&ledger->headers[ledger->active_header])) {
        int valid0 = header_valid(&ledger->headers[0]);
        int valid1 = header_valid(&ledger->headers[1]);
        if (!valid0 && !valid1) {
            return VP_ERR_STORAGE;
        }
        best = valid1 && (!valid0 || ledger->headers[1].generation > ledger->headers[0].generation)
            ? 1U : 0U;
        ledger->active_header = best;
    }
    return ledger->wal.pending != 0U ? vp_ledger_commit(ledger) : vp_ledger_verify(ledger);
}

vp_status_t vp_ledger_verify(const vp_trace_ledger_t *ledger)
{
    uint8_t root[VP_SHA_BYTES] = {0U};
    uint8_t digest[VP_SHA_BYTES];
    const vp_ledger_header_t *active;
    uint32_t i;
    if (ledger == NULL || ledger->authority != 0U || ledger->active_header > 1U) {
        return ledger != NULL && ledger->authority != 0U ? VP_ERR_AUTHORITY : VP_ERR_ARGUMENT;
    }
    active = &ledger->headers[ledger->active_header];
    if (!header_valid(active)) {
        return VP_ERR_STORAGE;
    }
    for (i = 0U; i < active->record_count; ++i) {
        const vp_ledger_record_t *record = &ledger->records[i];
        if (memcmp(root, record->previous_root, VP_SHA_BYTES) != 0 ||
            (i != 0U && (record->seq <= ledger->records[i - 1U].seq ||
             record->monotonic_ms < ledger->records[i - 1U].monotonic_ms ||
             record->rtc_seconds < ledger->records[i - 1U].rtc_seconds))) {
            return VP_ERR_STORAGE;
        }
        ledger_record_hash(record, digest);
        if (memcmp(digest, record->record_sha, VP_SHA_BYTES) != 0) {
            return VP_ERR_HASH;
        }
        ledger_root(root, record->record_sha, record->seq, root);
    }
    return memcmp(root, active->root, VP_SHA_BYTES) == 0 ? VP_OK : VP_ERR_HASH;
}
