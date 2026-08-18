#ifndef VERIPROCESS_V9_H
#define VERIPROCESS_V9_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VP_SHA_BYTES 32U
#define VP_ID_BYTES 16U
#define VP_EVIDENCE_MAX 8U
#define VP_CURVE_MAX 16U
#define VP_CHRONO_MAX 64U
#define VP_LEDGER_RECORD_MAX 32U

typedef enum {
    VP_OK = 0,
    VP_ERR_ARGUMENT = -1,
    VP_ERR_SCHEMA = -2,
    VP_ERR_AUTHORITY = -3,
    VP_ERR_HASH = -4,
    VP_ERR_TIME = -5,
    VP_ERR_FUTURE_EVIDENCE = -6,
    VP_ERR_POSTBURN_LEAKAGE = -7,
    VP_ERR_INDEPENDENCE = -8,
    VP_ERR_STATE = -9,
    VP_ERR_CAPACITY = -10,
    VP_ERR_STORAGE = -11
} vp_status_t;

typedef enum {
    VP_STAGE_PRE_FLIGHT = 0,
    VP_STAGE_RAMP = 1,
    VP_STAGE_SOAK = 2,
    VP_STAGE_COOL = 3,
    VP_STAGE_POST_RUN = 4,
    VP_STAGE_METROLOGY = 5,
    VP_STAGE_REPLAY = 6,
    VP_STAGE_FIXTURE = 7
} vp_stage_t;

typedef enum {
    VP_SOURCE_PROCESS = 0,
    VP_SOURCE_MAX31856 = 1,
    VP_SOURCE_MLX90640 = 2,
    VP_SOURCE_RECIPE = 3,
    VP_SOURCE_XRD = 16,
    VP_SOURCE_PL = 17,
    VP_SOURCE_SEM = 18,
    VP_SOURCE_EDS = 19,
    VP_SOURCE_POST_RUN_QUALITY = 20
} vp_source_kind_t;

typedef struct {
    uint8_t session_id[VP_ID_BYTES];
    uint8_t run_id[VP_ID_BYTES];
    uint8_t source_id[VP_ID_BYTES];
    uint8_t independence_family[VP_ID_BYTES];
    uint8_t model_release_root[VP_SHA_BYTES];
    uint8_t parent_sha[VP_EVIDENCE_MAX][VP_SHA_BYTES];
    uint8_t summary_sha[VP_SHA_BYTES];
    uint64_t seq;
    uint64_t monotonic_ms;
    uint64_t rtc_seconds;
    uint64_t age_ms;
    float quality;
    float interval_lower;
    float interval_upper;
    float interval_coverage;
    uint8_t stage;
    uint8_t source_kind;
    uint8_t truth_class;
    uint8_t parent_count;
    uint8_t has_interval;
    uint8_t has_model;
    uint8_t authority;
    uint8_t reserved;
} vp_evidence_card_t;

typedef struct {
    float t_s;
    float temperature_c;
} vp_curve_point_t;

typedef struct {
    uint8_t run_id[VP_ID_BYTES];
    uint8_t recipe_sha[VP_SHA_BYTES];
    uint64_t as_of_seq;
    uint64_t as_of_monotonic_ms;
    vp_curve_point_t curve[VP_CURVE_MAX];
    const vp_evidence_card_t *cards;
    uint8_t curve_count;
    uint8_t card_count;
    uint8_t authority;
    uint8_t reserved;
} vp_sintergraph_request_t;

typedef struct {
    uint8_t run_id[VP_ID_BYTES];
    uint8_t release_root[VP_SHA_BYTES];
    uint64_t as_of_seq;
    uint64_t as_of_monotonic_ms;
    float structure_mean;
    float structure_std;
    float performance_mean;
    float performance_std;
    float uncertainty;
    float transfer_confidence;
    uint8_t psp_edges;
    uint8_t authority;
    uint8_t frozen;
    uint8_t reserved;
} vp_sintergraph_output_t;

typedef struct {
    float structure_residual;
    float performance_residual;
    uint8_t interval_consistent;
    uint8_t horizon_short_mature;
    uint8_t horizon_medium_mature;
    uint8_t horizon_long_mature;
    uint8_t authority;
} vp_fulfillment_t;

typedef struct {
    uint16_t id;
    uint16_t status;
    uint32_t elapsed_ms;
    uint64_t monotonic_ms;
} vp_chrono_event_t;

typedef struct {
    vp_chrono_event_t events[VP_CHRONO_MAX];
    uint32_t model_mask;
    uint8_t count;
    uint8_t sinter_frozen;
    uint8_t wal_prepared;
    uint8_t authority;
    uint64_t last_monotonic_ms;
} vp_chrono_t;

typedef struct {
    uint8_t magic[4];
    uint8_t root[VP_SHA_BYTES];
    uint8_t checksum[VP_SHA_BYTES];
    uint64_t generation;
    uint32_t record_count;
    uint8_t valid;
    uint8_t reserved[3];
} vp_ledger_header_t;

typedef struct {
    uint8_t payload_sha[VP_SHA_BYTES];
    uint8_t previous_root[VP_SHA_BYTES];
    uint8_t record_sha[VP_SHA_BYTES];
    uint64_t seq;
    uint64_t monotonic_ms;
    uint64_t rtc_seconds;
} vp_ledger_record_t;

typedef struct {
    vp_ledger_record_t record;
    uint8_t wal_sha[VP_SHA_BYTES];
    uint8_t pending;
    uint8_t reserved[7];
} vp_ledger_wal_t;

typedef struct {
    vp_ledger_header_t headers[2];
    vp_ledger_record_t records[VP_LEDGER_RECORD_MAX];
    vp_ledger_wal_t wal;
    uint8_t active_header;
    uint8_t authority;
    uint8_t reserved[6];
} vp_trace_ledger_t;

void vp_evidence_seal(vp_evidence_card_t *card);
vp_status_t vp_evidence_validate(const vp_evidence_card_t *card);

vp_status_t vp_sintergraph_freeze(
    const vp_sintergraph_request_t *request,
    float model_structure_mean,
    float model_performance_mean,
    float model_uncertainty,
    vp_sintergraph_output_t *output);

vp_status_t vp_sintergraph_fulfill(
    const vp_sintergraph_output_t *prediction,
    const vp_evidence_card_t *postburn_card,
    float observed_structure,
    float observed_performance,
    vp_fulfillment_t *fulfillment);

vp_status_t vp_root_admit(
    const vp_evidence_card_t *cards,
    uint32_t card_count,
    uint32_t minimum_independent_families,
    uint32_t *independent_families);

void vp_chrono_init(vp_chrono_t *chrono);
vp_status_t vp_chrono_append(vp_chrono_t *chrono, uint16_t event_id,
                             uint64_t monotonic_ms, uint32_t elapsed_ms,
                             uint16_t status);

void vp_ledger_init(vp_trace_ledger_t *ledger);
vp_status_t vp_ledger_prepare(vp_trace_ledger_t *ledger,
                              const uint8_t payload_sha[VP_SHA_BYTES],
                              uint64_t seq, uint64_t monotonic_ms,
                              uint64_t rtc_seconds);
vp_status_t vp_ledger_commit(vp_trace_ledger_t *ledger);
vp_status_t vp_ledger_recover(vp_trace_ledger_t *ledger);
vp_status_t vp_ledger_verify(const vp_trace_ledger_t *ledger);

#ifdef __cplusplus
}
#endif

#endif
