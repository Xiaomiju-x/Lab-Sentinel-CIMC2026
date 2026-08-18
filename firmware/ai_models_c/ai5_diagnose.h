/******************************************************************************
 * ai5_diagnose.h  --  AI-5 Root-Cause Diagnoser (on-chip)
 *
 * Role: AI-1/2/3/4 detect THAT a sintering run is anomalous; AI-5 diagnoses
 *       WHY + WHAT-TO-DO. It maps the 27-D upstream-AI diagnostic vector
 *       (AI-1/2/3 outputs + raw furnace/gas/humidity context) to one of 9
 *       NAMED process root causes, each with a corrective action.
 *
 * Arch : Linear(27->32)-ReLU-Linear(32->16)-ReLU-Linear(16->9)
 *        (BatchNorm folded into the Linears at export). ~1.6 K params.
 *        Synthetic process-root-cause test acc 99.9% (taxonomy.json, gated).
 *
 * Scope: PROCESS-stage faults the GD32 can observe. Composition faults
 *        (dopant %, radius mismatch, host choice) stay with an off-device model.
 ******************************************************************************/
#ifndef AI5_DIAGNOSE_H
#define AI5_DIAGNOSE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AI5_IN_DIM   27
#define AI5_N_CLASS  9

/* root-cause class ids (match taxonomy.json / ai5_config.json order) */
typedef enum {
    AI5_RC_NORMAL        = 0,
    AI5_RC_RAMP_TOO_FAST = 1,
    AI5_RC_UNDER_TEMP    = 2,
    AI5_RC_OXIDIZE_CR6   = 3,
    AI5_RC_DECOMP_GAS    = 4,
    AI5_RC_VOLATIL_LOSS  = 5,
    AI5_RC_MOISTURE      = 6,
    AI5_RC_GRIND_INHOMO  = 7,
    AI5_RC_SOAK_DRIFT    = 8
} ai5_rootcause_t;

/* Pure forward -> raw logits (used by self-test). */
void ai5_forward(const float *feat27, float *logits9);

/* Forward + softmax into probs9; returns argmax root-cause id (0..8).
 * out_conf (NULL ok) <- top probability. */
int ai5_diagnose(const float *feat27, float *probs9, float *out_conf);

/* Short ASCII corrective-action string for a root-cause id (auto-generated
 * from the gas_safety reasons + sintering_profiles + literature; see
 * CIMC/docs/ai5_rootcause_labels.md). Never NULL. */
const char *ai5_action(int cls);

/* Short ASCII root-cause name (e.g. "OXIDIZE_CR6"). Never NULL. */
const char *ai5_name(int cls);

/* On-chip self-test vs golden vector: returns 1 PASS / 0 FAIL (ai_selftest style). */
int ai5_selftest(void);

#ifdef __cplusplus
}
#endif

#endif /* AI5_DIAGNOSE_H */
