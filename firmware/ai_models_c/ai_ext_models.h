/* ai_ext_models.h — AI-11/12/13 edge inference (CIMC Lab-Sentinel)
 *
 * Three NEW on-chip models (project: 11 -> 14 AI models). All hand-written float32,
 * reuse nn_ops.c. Weights AUTO-GENERATED (export_ext_models.py) from REAL XRD data;
 * honest provenance documented in each weights header. Host-verified vs PyTorch.
 *
 *   AI-11 phase-purity prior   24-D formula descriptor -> pure/impure (+P(pure))
 *                              37 real observed_pl phase labels; LOO 70% vs 60% base.
 *                              Same desc24 input as AI-6/7 (precomputed per preset).
 *                              Edge TRIAGE only; deep compositional call -> off-device.
 *   AI-12 PL dopant classifier 64-pt normalised emission spectrum -> Cr/Ni/Cr+Ni
 *                              281 real Fluoromax spectra; 5-fold CV 98.2%.
 *   AI-13 PL-QC autoencoder    64-pt emission spectrum -> recon + MSE; anomaly if >q_hat
 *                              281 real spectra, AE+conformal (AI-2 recipe in the
 *                              spectral domain). NB: the planned "Dq/B regressor" was
 *                              rejected by measure-first (Dq = algebra from AI-6's
 *                              lambda_em; B ~ constant) — the crystal-field read-out is
 *                              shipped as a DERIVED display, not a model.
 *
 * AI-12/13 input is a PL emission spectrum from the lab Fluoromax spectrometer (the
 * sentinel has no on-board spectrometer) — replayed on-device from stored real spectra
 * (ai12_demo_spectra.h), the same honest approach as furnace_sim. Decision-support.
 */
#ifndef AI_EXT_MODELS_H
#define AI_EXT_MODELS_H

#include <stdint.h>   /* uint32_t (robustness perturbation seed) */

/* ---- B4 Adaptive Conformal (online recalibration) ----
 * Self-correcting anomaly threshold that holds the target false-positive rate alpha
 * under in-distribution DRIFT. Pairs with AI-2 / AI-13 (their offline q_hat = init). */
typedef struct { float qhat; float alpha; float gamma; float qmin; } aconf_t;
void aconf_init(aconf_t *c, float q0, float alpha, float gamma);
int  aconf_step(aconf_t *c, float score);   /* returns 1 if score>qhat; updates qhat */

/* AI-11: desc24 -> class (0 impure / 1 pure). p_pure (may be NULL) = softmax P(pure). */
int   ai11_purity(const float *desc24, float *p_pure);

/* AI-12: spec64 (normalised) -> class (0 Cr / 1 Ni / 2 Cr+Ni). probs3 may be NULL. */
int   ai12_plclass(const float *spec64, float *probs3);

/* AI-12 INT8 twin (B1 quantisation): same I/O as ai12_plclass, weight-only INT8.
 * Same 98.9% accuracy as fp32 on 281 spectra, weights 3.7x smaller. */
int   ai12_plclass_int8(const float *spec64, float *probs3);

/* AI-13: spec64 -> reconstruction (recon may be NULL) + MSE (out_mse may be NULL).
 * Returns 1 if anomalous (MSE > q_hat), else 0. */
int   ai13_plqc(const float *spec64, float *recon, float *out_mse);
float ai13_plqc_qhat(void);

/* AI-14 furnace-temperature multi-step forecaster (a NEW task vs AI-3's anomaly
 * CLASSIFIER — this is multi-step REGRESSION). win_norm[24] = last 24 temps / 1600,
 * out_norm[12] = next 12 predicted temps / 1600 (x1600 for degrees C). Beats linear
 * extrapolation at stage knees (measure-first gated). See ai14_forecast_weights.h. */
void  ai14_forecast(const float *win_norm, float *out_norm);

/* AI-15 PL host-ID: spec64 -> host class (0 NaY2Ga2InGe2O12 / 1 Y3ZnGa3GeO12).
 * probs2 may be NULL. 281 real spectra, 5-fold CV 97.1% vs 63% majority. */
int   ai15_hostid(const float *spec64, float *probs2);

/* AI-16 PL lambda_em regressor: spec64 -> emission peak (nm), read from the MEASURED
 * spectrum. 5-fold MAE 18.9 nm where 64-bin argmax is off 59.7 nm. Distinct from AI-6
 * (lambda from RECIPE, forward design) — this is from the measured spectrum (QC). */
float ai16_lambda(const float *spec64);

/* ---- AI-17 PL Few-Shot NCM (spectral twin of AI-1b vision few-shot) ----
 * On-device registration of a NEW phosphor sample type from a few measured spectra,
 * over AI-12's 16-D embedding. 5-shot ~87% on dopant classes (majority 58%). */
#define AI17_EMB_DIM   16
#define AI17_MAX_CLASS 6
void  ai12_embed(const float *spec64, float *emb16);   /* AI-12 16-D fingerprint */
void  ai17_pl_reset(void);
int   ai17_pl_num_classes(void);
int   ai17_pl_add_sample(int cid, const float *emb16); /* few-shot register */
int   ai17_pl_classify(const float *emb16, float *out_d2);

/* AI-19 sintering RUL/ETA: feat[26] = 24 temps/1600 + hold_cum/600 + stage/5 ->
 * MINUTES remaining to firing-complete (>=0). New quantity; robust to ramp anomalies
 * (76 min MAE vs 200 min nominal-schedule baseline). See ai19_rul_weights.h. */
float ai19_rul(const float *feat26);

/* AI-20 thermocouple-integrity: win2L = [measured/1600 x L, setpoint/1600 x L] ->
 * 0 healthy / 1 open-circuit / 2 erratic. probs3 may be NULL. Analytical-redundancy
 * SENSOR monitor (acc 97.7%, healthy-recall 99.7%). See ai20_tcfault_weights.h. */
int   ai20_tcfault(const float *win2L, float *probs3);
void  ai20_tcfault_logits(const float *win2L, float *logit3);  /* host golden */

/* ---- On-device ONLINE-LEARNING risk head (TinyML continual learning) ----
 * Linear softmax OL_F(16)->OL_K(4) {good,warn,bad,crit}, seeded on PC
 * (online_head_weights.h), adapted on-chip by 1 SGD step per operator correction.
 * Real forward+backward+SGD on the M7 — the head learns this furnace/operator. */
void online_reset(void);                              /* restore PC-seed weights      */
int  online_predict(const float *f16, float *probs4); /* argmax risk; probs4 may NULL */
void online_update(const float *f16, int label);      /* one SGD step from a label    */
int  online_selftest(float *w_err);                   /* golden: reproduce numpy run  */

/* ---- Robustness perturbations (reliability demo + host regression) ----
 * Deterministic input perturbations applied to a PRIVATE copy of a model input,
 * shared by the firmware live-injection demo and the host robustness regression.
 *   spec mode: 0 clean / 1 noise / 2 occlusion / 3 baseline-drift
 *   img  mode: 0 clean / 1 noise / 2 dark / 3 bright / 4 occlusion  (CHW, square) */
void rob_perturb_spec(float *spec, int n, int mode, uint32_t *seed);
void rob_perturb_img(float *chw, int n, int mode, uint32_t *seed);

/* ---- AI-4 fusion DEPLOYED from the GD32 Embedded AI Tool's TFLite output ----
 * Re-executes on-chip the FullyConnected graph the tool produced (weights EXTRACTED
 * from the .tflite flatbuffer -> ai4_tflite_deploy.h, NOT re-derived from PyTorch).
 * Demonstrates the official toolchain end-to-end: PyTorch -> GD32 AI Tool TFLite ->
 * on-chip, host+chip byte-verified.  ai4_tflite_deploy(x16, logits4): forward;
 * ai4_tflite_selftest(&err): rerun the tool's golden, returns 1 if match. */
void ai4_tflite_deploy(const float *x16, float *logits4);
int  ai4_tflite_selftest(float *err_out);

#endif /* AI_EXT_MODELS_H */
