/******************************************************************************
 * ai2_ae.h  —  AI-2 Sintering Auto-Encoder anomaly detector (on-chip)
 *
 * 32-16-8-16-32 symmetric AE. Reconstruction MSE compared to a Split-Conformal
 * q_hat (90% CI) trained on 13-host normal sintering profiles. Per-feature
 * residual gives anomaly attribution (which sensor group deviates).
 *
 * 32-D feature layout matches CIMC/model/ai2_env_ae/synth_data.py exactly
 * (built at runtime by feature_build.c):
 *   [0:8]   temperature group     [8:14]  gas/atmosphere group
 *   [14:18] room environment      [18:26] vibration group
 *   [26:30] vision proxy          [30:32] progress
 ******************************************************************************/
#ifndef AI2_AE_H
#define AI2_AE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AI2_FEAT_DIM 32

/* Attribution bits (matches env_result_t.attribution_mask). */
#define AI2_ATTR_TEMP   0x01u
#define AI2_ATTR_HUMID  0x02u
#define AI2_ATTR_GAS    0x04u
#define AI2_ATTR_VIB    0x08u

/* Pure reconstruction on a NORMALISED 32-vector (used by the self-test). */
void ai2_ae_reconstruct(const float *xn, float *recon);

/* Full anomaly score on a RAW 32-vector (normalises internally with mu/std).
 *   out_resid3[0]=temp, [1]=vib, [2]=gas   group residuals (0-1, for AI-4)
 *   out_attr   = bitmask of deviating sensor groups (NULL ok)
 *   out_ratio  = mse / q_hat, clipped [0,6] (anomaly when >=1, for AI-4) (NULL ok)
 *               uses the ADAPTIVE q_hat (see ai2_ae_adapt).
 * Returns the reconstruction MSE. */
float ai2_ae_score(const float *feat_raw, float out_resid3[3],
                   uint8_t *out_attr, float *out_ratio);

/* ---- Adaptive Conformal Inference (upgrade H, Gibbs & Candes 2021) ----
 * The trained split-conformal q_hat (AI2_QHAT_90) is only a prior; on-site
 * sensor noise differs. ai2_ae_adapt() nudges q_hat online to hold the target
 * 90% coverage: q <- q + gamma*( 1[mse>q] - alpha ).  Call it ONCE PER NORMAL
 * sample only (furnace running + AI-3 normal + no injected fault) so genuine
 * anomalies are scored against the normal-calibrated threshold, not absorbed. */
void  ai2_ae_adapt(float mse);
float ai2_ae_qhat(void);        /* current adaptive q_hat */
float ai2_ae_qhat_trained(void);/* the frozen training prior, for comparison */

#ifdef __cplusplus
}
#endif

#endif /* AI2_AE_H */
