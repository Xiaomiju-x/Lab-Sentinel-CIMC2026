/******************************************************************************
 * batch_record.h — tamper-evident electronic batch record (traceability)
 *
 * Each completed sintering batch produces a fixed record stamped with the
 * recipe, the SPC verdict (Cpk + control), element health, AI-alarm count and
 * the final controller state. Records are SHA-256 hash-chained (each record's
 * hash folds in the previous record's hash) so the batch history is an
 * append-only, tamper-evident ledger — the on-chip equivalent of a 21 CFR
 * Part 11 electronic record / AMS2750 traceability log. This mirrors the XRD
 * project's predictions.jsonl SHA-256 chain, now embedded on the GD32.
 *
 * Serialization is explicit & big-endian (no struct-padding dependence) so the
 * hash is reproducible across host and target. Host-verified by record_test.c.
 ******************************************************************************/
#ifndef BATCH_RECORD_H
#define BATCH_RECORD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BR_RECIPE_LEN   24
#define BR_OPERATOR_LEN 16

typedef struct {
    uint32_t batch_id;
    uint32_t unix_time;                 /* batch start (epoch s)              */
    char     recipe[BR_RECIPE_LEN];     /* NUL-padded                         */
    char     operator_id[BR_OPERATOR_LEN];
    float    peak_C;                    /* peak temperature reached           */
    float    soak_cpk;                  /* SPC capability of the soak         */
    float    elem_remaining_pct;        /* heating-element remaining life     */
    uint16_t n_ai_alarms;               /* AI safety events during the batch  */
    uint8_t  in_control;                /* SPC control verdict                */
    uint8_t  capable;                   /* Cpk >= 1.33                        */
    uint8_t  final_state;               /* ctrl_state_t                       */
    uint8_t  fault;                     /* ctrl_fault_t                        */
    uint8_t  prev_hash[32];             /* hash of the previous record        */
    uint8_t  this_hash[32];             /* SHA-256(serialize(record)||prev)   */
} batch_record_t;

/* genesis predecessor hash = 32 zero bytes */
extern const uint8_t BR_GENESIS_PREV[32];

/* Seal a record: copy prev_hash in, then compute this_hash. Call once the batch
 * is complete and all quality fields are filled. */
void batch_record_seal(batch_record_t *r, const uint8_t prev_hash[32]);

/* Verify a single record against an expected predecessor hash. Returns 1 if the
 * stored prev_hash matches AND this_hash recomputes correctly. */
int  batch_record_verify(const batch_record_t *r, const uint8_t prev_hash[32]);

/* Verify a whole chain (recs[0] links to genesis). Returns -1 if intact, else
 * the index of the first broken/tampered record. */
int  batch_chain_verify(const batch_record_t *recs, int n);

#ifdef __cplusplus
}
#endif
#endif /* BATCH_RECORD_H */
