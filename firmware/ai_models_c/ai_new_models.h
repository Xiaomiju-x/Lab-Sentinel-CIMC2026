/* ai_new_models.h — AI-6/7/8/9/10 edge inference (CIMC Lab-Sentinel)
 *
 * Five new on-chip models (project: 6 -> 11 AI models). All hand-written float32,
 * reuse nn_ops.c. Weights AUTO-GENERATED (export_new_models.py); honest provenance
 * documented in each weights header.
 *
 *   AI-6  optical surrogate   24-D formula descriptor -> lambda_em(nm), FWHM(nm)
 *                             (distilled from the validated ts_torch TSPredictor)
 *   AI-7  thermal-quench      24-D descriptor -> thermal_stability%@150C -> 3-band
 *   AI-8  energy/carbon       5 recipe feats  -> kWh, CO2(kg)
 *   AI-9  recipe retrieval    18-D recipe vec -> nearest historical recipe (kNN)
 *   AI-10 vibration PdM       64-sample ADXL window -> class (0..3) + probs
 *
 * The 24-D / 18-D / 5-D feature vectors are PC-precomputed per recipe preset
 * (recipe_presets.h) — no on-device formula parser. AI-10 extracts its 8 features
 * from a live vibration window on-device.
 */
#ifndef AI_NEW_MODELS_H
#define AI_NEW_MODELS_H

/* AI-6: desc24 -> emission peak + bandwidth (nm). */
void  ai6_optical(const float *desc24, float *lambda_em_nm, float *fwhm_nm);

/* AI-7: desc24 -> thermal stability % @150C; returns band 0=poor/1=marginal/2=good
 * (the % itself is coarse ~17% real, so the band is the honest read-out). */
int   ai7_thermal(const float *desc24, float *thermal_pct);

/* AI-8: e5 recipe feats -> energy (kWh) + carbon (kg CO2). */
void  ai8_energy(const float *e5, float *kwh, float *co2_kg);

/* AI-9: q18 recipe vector -> nearest historical recipe index (0..AI9_N-1);
 * out_dist = squared normalised distance. Caller reads ai9_name/outcome via the DB. */
int   ai9_retrieve(const float *q18, float *out_dist);

/* AI-10: 64-sample vibration window -> class (0 normal/1 imbalance/2 bearing/
 * 3 looseness); probs4 (may be NULL) gets the softmax. */
int   ai10_vibration(const float *window64, float *probs4);

/* AI-10 feature extraction exposed for the host golden test. */
void  ai10_features(const float *window64, float *feat8);

/* AI-9 recipe-DB accessors (the 67-row DB is encapsulated in ai_new_models.c
 * so callers don't need to include the bulky weights header). */
int         ai9_count(void);
const char *ai9_recipe_name(int idx);

#endif /* AI_NEW_MODELS_H */
