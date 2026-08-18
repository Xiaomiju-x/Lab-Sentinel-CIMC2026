/* ai_ext_models.c — AI-11/12/13 edge inference (CIMC Lab-Sentinel)
 * Hand-written float32, reuses nn_ops.c. See ai_ext_models.h for the contract and
 * the per-model weights headers for honest provenance. Host-verified vs PyTorch.
 */
#include "ai_ext_models.h"
#include "nn_ops.h"
#include "ai11_purity_weights.h"
#include "ai12_plspec_weights.h"
#include "ai13_plqc_weights.h"
#include "ai14_forecast_weights.h"
#include "ai15_hostid_weights.h"
#include "ai16_lambda_weights.h"
#include "ai12_int8_weights.h"      /* B1: weight-only INT8 twin of AI-12 */
#include "ai19_rul_weights.h"       /* AI-19 sintering RUL/ETA regressor */
#include "ai20_tcfault_weights.h"   /* AI-20 thermocouple-integrity classifier */
#include "online_head_weights.h"    /* on-device online-learning risk head (seed W/b) */
#include "online_golden.h"          /* golden update stream for online_selftest */
#include "ai4_tflite_deploy.h"      /* AI-4 weights EXTRACTED from the GD32-AI-Tool TFLite */
#include <math.h>                   /* sqrtf/fabsf for AI-20; expf for online softmax */

/* ------------------------------------------------------------ AI-11 phase purity */
int ai11_purity(const float *desc24, float *p_pure)
{
    float h[AI11_H], o[AI11_OUT];
    int i;
    /* layer-0 has the (x-mu)/sd standardisation folded in (see weights header) */
    nn_linear(desc24, ai11_w0, ai11_b0, h, AI11_IN, AI11_H); nn_relu(h, AI11_H);
    nn_linear(h, ai11_w1, ai11_b1, o, AI11_H, AI11_OUT);
    if (p_pure) {
        float p[AI11_OUT];
        for (i = 0; i < AI11_OUT; i++) p[i] = o[i];
        nn_softmax(p, AI11_OUT);
        *p_pure = p[1];                          /* class 1 = pure */
    }
    return nn_argmax(o, AI11_OUT);
}

/* ------------------------------------------------------------ AI-12 PL classifier */
int ai12_plclass(const float *spec64, float *probs3)
{
    float h1[AI12_H1], h2[AI12_H2], o[AI12_OUT];
    int i;
    nn_linear(spec64, ai12_w0, ai12_b0, h1, AI12_IN, AI12_H1); nn_relu(h1, AI12_H1);
    nn_linear(h1,     ai12_w1, ai12_b1, h2, AI12_H1, AI12_H2); nn_relu(h2, AI12_H2);
    nn_linear(h2,     ai12_w2, ai12_b2, o,  AI12_H2, AI12_OUT);
    if (probs3) {
        for (i = 0; i < AI12_OUT; i++) probs3[i] = o[i];
        nn_softmax(probs3, AI12_OUT);
    }
    return nn_argmax(o, AI12_OUT);
}

/* ----------------------------------------------------- AI-12 INT8 (B1 quantisation)
 * Weight-only per-output INT8 twin of ai12_plclass (biases stay fp32). On 281 real
 * spectra: same 98.9% accuracy as fp32, 100% class agreement, weights 3.7x smaller.
 * Demonstrates the deployable on-chip INT8 path (nn_linear_int8). */
int ai12_plclass_int8(const float *spec64, float *probs3)
{
    float h1[AI12_H1], h2[AI12_H2], o[AI12_OUT];
    int i;
    nn_linear_int8(spec64, ai12_q_w0, ai12_q_s0, ai12_b0, h1, AI12_IN, AI12_H1); nn_relu(h1, AI12_H1);
    nn_linear_int8(h1,     ai12_q_w1, ai12_q_s1, ai12_b1, h2, AI12_H1, AI12_H2); nn_relu(h2, AI12_H2);
    nn_linear_int8(h2,     ai12_q_w2, ai12_q_s2, ai12_b2, o,  AI12_H2, AI12_OUT);
    if (probs3) {
        for (i = 0; i < AI12_OUT; i++) probs3[i] = o[i];
        nn_softmax(probs3, AI12_OUT);
    }
    return nn_argmax(o, AI12_OUT);
}

/* ------------------------------------------------------------ AI-13 PL-QC autoencoder */
int ai13_plqc(const float *spec64, float *recon, float *out_mse)
{
    float h[AI13_H], z[AI13_Z], r[AI13_IN];
    float mse = 0.0f;
    int i;
    /* encoder 64 -> 32 -> 8 (ReLU after each) */
    nn_linear(spec64, ai13_e0w, ai13_e0b, h, AI13_IN, AI13_H); nn_relu(h, AI13_H);
    nn_linear(h,      ai13_e1w, ai13_e1b, z, AI13_H,  AI13_Z); nn_relu(z, AI13_Z);
    /* decoder 8 -> 32 -> 64 (ReLU after first; linear output) */
    nn_linear(z, ai13_d0w, ai13_d0b, h, AI13_Z, AI13_H); nn_relu(h, AI13_H);
    nn_linear(h, ai13_d1w, ai13_d1b, r, AI13_H, AI13_IN);
    for (i = 0; i < AI13_IN; i++) {
        float d = r[i] - spec64[i];
        mse += d * d;
        if (recon) recon[i] = r[i];
    }
    mse /= (float)AI13_IN;
    if (out_mse) *out_mse = mse;
    return (mse > ai13_q_hat) ? 1 : 0;
}

float ai13_plqc_qhat(void) { return ai13_q_hat; }

/* ============================================================ B4 Adaptive Conformal
 * Online recalibration (Gibbs & Candes-style Adaptive Conformal Inference). Holds a
 * running anomaly threshold q_hat that self-corrects to keep the long-run exceedance
 * (false-positive) rate at the target alpha EVEN IF the in-distribution baseline
 * DRIFTS (furnace ageing / sensor drift). Update on each conformity score s:
 *     q_hat += gamma * (I[s > q_hat] - alpha)
 * One multiply-add per sample, no runtime recalibration set. The offline AI-2 / AI-13
 * q_hat is the init; this keeps it honest under drift instead of a frozen threshold. */
void aconf_init(aconf_t *c, float q0, float alpha, float gamma)
{
    c->qhat = q0; c->alpha = alpha; c->gamma = gamma; c->qmin = 0.0f;
}

int aconf_step(aconf_t *c, float score)
{
    int exceed = (score > c->qhat) ? 1 : 0;
    c->qhat += c->gamma * ((float)exceed - c->alpha);
    if (c->qhat < c->qmin) c->qhat = c->qmin;
    return exceed;
}

/* ------------------------------------------------------------ AI-14 temp forecaster
 * win_norm[AI14_WIN] = last 24 furnace temps / AI14_TNORM  ->  out_norm[AI14_HOR] =
 * next 12 predicted temps / AI14_TNORM. Caller multiplies by AI14_TNORM for degrees C.
 * MLP 24 -> 32 -> 32 -> 12 (ReLU on the two hidden layers, linear output). */
void ai14_forecast(const float *win_norm, float *out_norm)
{
    float h0[AI14_HID], h1[AI14_HID];
    nn_linear(win_norm, ai14_w0, ai14_b0, h0, AI14_WIN, AI14_HID); nn_relu(h0, AI14_HID);
    nn_linear(h0,       ai14_w1, ai14_b1, h1, AI14_HID, AI14_HID); nn_relu(h1, AI14_HID);
    nn_linear(h1,       ai14_w2, ai14_b2, out_norm, AI14_HID, AI14_HOR);
}

/* ------------------------------------------------------------ AI-15 PL host-ID
 * spec64 (normalised emission) -> host class (0 NaY2Ga2InGe2O12 / 1 Y3ZnGa3GeO12).
 * probs2 (may be NULL) = softmax. Two garnet hosts -> separable band shape (CV 97%). */
int ai15_hostid(const float *spec64, float *probs2)
{
    float h0[AI15_H], h1[AI15_H], o[AI15_OUT];
    int i;
    nn_linear(spec64, ai15_w0, ai15_b0, h0, AI15_IN, AI15_H); nn_relu(h0, AI15_H);
    nn_linear(h0,     ai15_w1, ai15_b1, h1, AI15_H,  AI15_H); nn_relu(h1, AI15_H);
    nn_linear(h1,     ai15_w2, ai15_b2, o,  AI15_H,  AI15_OUT);
    if (probs2) {
        for (i = 0; i < AI15_OUT; i++) probs2[i] = o[i];
        nn_softmax(probs2, AI15_OUT);
    }
    return nn_argmax(o, AI15_OUT);
}

/* ------------------------------------------------------------ AI-16 PL lambda_em
 * spec64 -> emission peak wavelength (nm), read from the MEASURED spectrum.
 * Recovers the physical peak where 64-bin argmax is off ~60 nm (MAE 8-19 nm). */
float ai16_lambda(const float *spec64)
{
    float h0[AI16_H], h1[AI16_H], o[1];
    nn_linear(spec64, ai16_w0, ai16_b0, h0, AI16_IN, AI16_H); nn_relu(h0, AI16_H);
    nn_linear(h0,     ai16_w1, ai16_b1, h1, AI16_H,  AI16_H); nn_relu(h1, AI16_H);
    nn_linear(h1,     ai16_w2, ai16_b2, o,  AI16_H,  1);
    return o[0] * AI16_SPAN + AI16_LO;
}

/* ===================================================================== AI-17
 * PL Few-Shot NCM — the spectral-modality twin of AI-1b (vision few-shot).
 * Reuses AI-12's discriminative 16-D penultimate activation as a general PL
 * "fingerprint", then nearest-class-mean over it lets the lab register a BRAND-NEW
 * phosphor sample type from 3-5 measured spectra WITHOUT retraining (5-shot ~87%
 * on dopant classes; majority 58%). Same incremental-mean NCM math as ai1b_ncm.c.
 */

/* ai12_embed: spec64 -> 16-D embedding (AI-12 layers 0-1, no logits/softmax). */
void ai12_embed(const float *spec64, float *emb16)
{
    float h1[AI12_H1];
    nn_linear(spec64, ai12_w0, ai12_b0, h1,    AI12_IN, AI12_H1); nn_relu(h1, AI12_H1);
    nn_linear(h1,     ai12_w1, ai12_b1, emb16, AI12_H1, AI12_H2); nn_relu(emb16, AI12_H2);
}

static float    s_ai17_mean[AI17_MAX_CLASS][AI17_EMB_DIM];
static uint32_t s_ai17_count[AI17_MAX_CLASS];
static int      s_ai17_nclass;

void ai17_pl_reset(void)
{
    int c, d;
    for (c = 0; c < AI17_MAX_CLASS; c++) {
        s_ai17_count[c] = 0u;
        for (d = 0; d < AI17_EMB_DIM; d++) s_ai17_mean[c][d] = 0.0f;
    }
    s_ai17_nclass = 0;
}

int ai17_pl_num_classes(void) { return s_ai17_nclass; }

/* register one few-shot sample (its AI-12 embedding) into class cid. */
int ai17_pl_add_sample(int cid, const float *emb16)
{
    int d; float n1;
    if (cid < 0 || cid >= AI17_MAX_CLASS) return -1;
    if (s_ai17_count[cid] == 0u && (cid + 1) > s_ai17_nclass) s_ai17_nclass = cid + 1;
    s_ai17_count[cid]++;
    n1 = 1.0f / (float)s_ai17_count[cid];
    for (d = 0; d < AI17_EMB_DIM; d++)
        s_ai17_mean[cid][d] += (emb16[d] - s_ai17_mean[cid][d]) * n1;
    return 0;
}

/* classify a spectrum's embedding by nearest class mean; out_d2 = squared dist. */
int ai17_pl_classify(const float *emb16, float *out_d2)
{
    int c, d, best = -1; float best_d2 = 0.0f;
    for (c = 0; c < s_ai17_nclass; c++) {
        float d2 = 0.0f;
        if (s_ai17_count[c] == 0u) continue;
        for (d = 0; d < AI17_EMB_DIM; d++) {
            float diff = emb16[d] - s_ai17_mean[c][d];
            d2 += diff * diff;
        }
        if (best < 0 || d2 < best_d2) { best_d2 = d2; best = c; }
    }
    if (out_d2) *out_d2 = best_d2;
    return best;
}

/* ===================================================================== AI-19 RUL
 * Sintering remaining-time (ETA) regressor. feat[26] = 24 temps/1600 + hold_cum/600
 * + stage/5 -> minutes to firing-complete. A NEW quantity no other model outputs;
 * robust to ramp anomalies that shift the run total (measure-first: 76 min MAE vs a
 * nominal-schedule baseline's 200 min). Returns minutes remaining (>=0). */
float ai19_rul(const float *feat26)
{
    float h0[AI19_HID], h1[AI19_HID], o[1];
    nn_linear(feat26, ai19_w0, ai19_b0, h0, AI19_NX,  AI19_HID); nn_relu(h0, AI19_HID);
    nn_linear(h0,     ai19_w1, ai19_b1, h1, AI19_HID, AI19_HID); nn_relu(h1, AI19_HID);
    nn_linear(h1,     ai19_w2, ai19_b2, o,  AI19_HID, 1);
    o[0] *= AI19_RNORM;
    return (o[0] < 0.0f) ? 0.0f : o[0];
}

/* ============================================================ AI-20 TC integrity
 * Thermocouple-integrity classifier. Input is the RAW L-window
 * win2L = [measured/1600 x L, setpoint/1600 x L]; this recomputes the 8 plausibility
 * features (mirror of feats8() in train_rul_vsensor.py) then the MLP. Class
 * 0 healthy / 1 open-circuit(->0) / 2 erratic(EMI/loose). probs3 may be NULL.
 * Analytical-redundancy SENSOR monitor — questions whether the READING is trustworthy
 * (AI-3 assumes it is real). 'stuck'/'offset' deliberately out of scope (ambiguous). */
static void ai20_feats8(const float *win2L, float *f8)
{
    const float *meas = win2L, *setp = win2L + AI20_L;
    float mm = 0.0f, ms = 0.0f, vm = 0.0f, vs = 0.0f;
    float mn = meas[0], mxd = 0.0f, mxr = 0.0f, low = 0.0f;
    int i;
    for (i = 0; i < AI20_L; i++) { mm += meas[i]; ms += setp[i]; }
    mm /= (float)AI20_L; ms /= (float)AI20_L;
    for (i = 0; i < AI20_L; i++) {
        float dm = meas[i] - mm, ds = setp[i] - ms;
        vm += dm * dm; vs += ds * ds;
        if (meas[i] < mn) mn = meas[i];
        if (i > 0) { float j = fabsf(meas[i] - meas[i - 1]); if (j > mxd) mxd = j; }
        { float rr = fabsf(meas[i] - setp[i]); if (rr > mxr) mxr = rr; }
        if (meas[i] < (40.0f / AI20_TNORM)) low += 1.0f;
    }
    f8[0] = mm - ms;                          /* residual bias (context) */
    f8[1] = sqrtf(vm / (float)AI20_L);        /* std(measured) */
    f8[2] = sqrtf(vs / (float)AI20_L);        /* std(setpoint) */
    f8[3] = mn;                               /* min(measured) -> open-circuit ~0 */
    f8[4] = mxd;                              /* max |Δmeasured| -> erratic large */
    f8[5] = mxr;                              /* max |measured-setpoint| */
    f8[6] = mm;                               /* level */
    f8[7] = low / (float)AI20_L;              /* fraction implausibly low */
}

int ai20_tcfault(const float *win2L, float *probs3)
{
    float f8[AI20_NF], h0[AI20_H], h1[AI20_H], o[AI20_NC];
    int i;
    ai20_feats8(win2L, f8);
    nn_linear(f8, ai20_w0, ai20_b0, h0, AI20_NF, AI20_H); nn_relu(h0, AI20_H);
    nn_linear(h0, ai20_w1, ai20_b1, h1, AI20_H,  AI20_H); nn_relu(h1, AI20_H);
    nn_linear(h1, ai20_w2, ai20_b2, o,  AI20_H,  AI20_NC);
    if (probs3) {
        for (i = 0; i < AI20_NC; i++) probs3[i] = o[i];
        nn_softmax(probs3, AI20_NC);
    }
    return nn_argmax(o, AI20_NC);
}

/* expose the raw logits too (host golden compares logits, like AI-14/15). */
void ai20_tcfault_logits(const float *win2L, float *logit3)
{
    float f8[AI20_NF], h0[AI20_H], h1[AI20_H];
    ai20_feats8(win2L, f8);
    nn_linear(f8, ai20_w0, ai20_b0, h0, AI20_NF, AI20_H); nn_relu(h0, AI20_H);
    nn_linear(h0, ai20_w1, ai20_b1, h1, AI20_H,  AI20_H); nn_relu(h1, AI20_H);
    nn_linear(h1, ai20_w2, ai20_b2, logit3, AI20_H, AI20_NC);
}

/* ============================================================================
 * On-device ONLINE-LEARNING risk head (TinyML continual learning).
 *   risk = softmax(W·f + b),  f in R^OL_F (16),  OL_K (4) classes.
 *   Seeded on the PC (ol_W0/ol_b0); copied to RAM at boot; each operator
 *   correction does ONE SGD step (forward + backward + weight update) on the M7.
 *   This is a REAL gradient update on-chip — not a frozen model. "越用越准".
 * Honest scope: we adapt a tiny last layer (the achievable on-device-learning
 *   result for an M7); we do NOT backprop the whole nano-LM on-chip.
 * ==========================================================================*/
static float ol_W[OL_K][OL_F];     /* RAM working weights (adapt) */
static float ol_b[OL_K];
static int   ol_ready = 0;

static void ol_softmax4(const float *logit, float *p)   /* exact expf, matches numpy */
{
    int i; float mx = logit[0], s = 0.0f;
    for (i = 1; i < OL_K; i++) if (logit[i] > mx) mx = logit[i];
    for (i = 0; i < OL_K; i++) { p[i] = expf(logit[i] - mx); s += p[i]; }
    if (s <= 0.0f) s = 1.0f;
    for (i = 0; i < OL_K; i++) p[i] /= s;
}

void online_reset(void)
{
    int k, i;
    for (k = 0; k < OL_K; k++) {
        for (i = 0; i < OL_F; i++) ol_W[k][i] = ol_W0[k][i];
        ol_b[k] = ol_b0[k];
    }
    ol_ready = 1;
}

int online_predict(const float *f16, float *probs4)
{
    float logit[OL_K]; int k, i;
    if (!ol_ready) online_reset();
    for (k = 0; k < OL_K; k++) {
        float acc = ol_b[k];
        for (i = 0; i < OL_F; i++) acc += ol_W[k][i] * f16[i];
        logit[k] = acc;
    }
    if (probs4) ol_softmax4(logit, probs4);
    return nn_argmax(logit, OL_K);
}

/* one online SGD step on the cross-entropy loss: grad_logit = p - onehot(label). */
void online_update(const float *f16, int label)
{
    float logit[OL_K], p[OL_K]; int k, i;
    if (!ol_ready) online_reset();
    if (label < 0 || label >= OL_K) return;
    for (k = 0; k < OL_K; k++) {
        float acc = ol_b[k];
        for (i = 0; i < OL_F; i++) acc += ol_W[k][i] * f16[i];
        logit[k] = acc;
    }
    ol_softmax4(logit, p);
    for (k = 0; k < OL_K; k++) {
        float g = p[k] - ((k == label) ? 1.0f : 0.0f);
        for (i = 0; i < OL_F; i++) ol_W[k][i] -= OL_LR * g * f16[i];
        ol_b[k] -= OL_LR * g;
    }
}

/* golden self-test: reproduce the numpy online update stream + probe predictions.
 * Leaves the head RESET (seed) so runtime starts clean. Returns 1 on pass. */
int online_selftest(float *w_err)
{
    int i, k, j, ok = 1;
    float err = 0.0f;
    online_reset();
    for (j = 0; j < OL_NUP; j++) online_update(ol_up_x[j], ol_up_y[j]);
    for (k = 0; k < OL_K; k++) {
        float eb = ol_b[k] - ol_b_exp[k]; if (eb < 0.0f) eb = -eb;
        if (eb > err) err = eb;
        for (i = 0; i < OL_F; i++) {
            float e = ol_W[k][i] - ol_W_exp[k][i]; if (e < 0.0f) e = -e;
            if (e > err) err = e;
        }
    }
    for (j = 0; j < OL_NPROBE; j++)
        if (online_predict(ol_probe_x[j], 0) != ol_probe_pred[j]) ok = 0;
    if (w_err) *w_err = err;
    online_reset();
    return (ok && err < 1.0e-3f) ? 1 : 0;
}

/* ------------------------------------------------------------ Robustness perturbations
 * Deterministic, physically-standard input perturbations for the reliability
 * (robustness) demonstration. Shared verbatim by the firmware (lab_sentinel.c
 * vision_task / pl_refresh) AND the host regression (robustness_eval.c) so the
 * on-device "graceful degradation" demo and the report numbers use the SAME code.
 * Non-destructive: the caller passes a private copy of the model input.
 *   *seed is an LCG state the caller advances per call -> reproducible numbers. */
static float rob_unif(uint32_t *s)
{
    *s = (*s) * 1103515245u + 12345u;
    return (float)((*s >> 8) & 0xFFFFFFu) / (float)0x1000000u;   /* [0,1) */
}
/* cheap zero-mean noise in ~[-1,1] (sum of two uniforms - 1). */
static float rob_noise(uint32_t *s) { return rob_unif(s) + rob_unif(s) - 1.0f; }

/* spec64 perturbation. mode: 0 clean / 1 noise (sigma~6% of unit) /
 * 2 occlusion (zero a central 10-bin band) / 3 baseline-drift (additive tilt). */
void rob_perturb_spec(float *spec, int n, int mode, uint32_t *seed)
{
    int i;
    if (mode == 0 || spec == 0 || seed == 0) return;
    if (mode == 1) {
        for (i = 0; i < n; i++) {
            float v = spec[i] + 0.06f * rob_noise(seed);
            spec[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        }
    } else if (mode == 2) {
        int start = (n >= 30) ? (n / 2 - 5) : 0;   /* central 10-bin occlusion */
        for (i = start; i < start + 10 && i < n; i++) spec[i] = 0.0f;
    } else if (mode == 3) {
        for (i = 0; i < n; i++) {
            float v = spec[i] + 0.20f * ((float)i / (float)(n - 1) - 0.5f);
            spec[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        }
    }
}

/* CHW [0,1] image perturbation (n = total floats = 3*H*W, square H=W planes).
 * mode: 0 clean / 1 noise (sigma~8%) / 2 dark (x0.5) / 3 bright (x1.6 clip) /
 * 4 occlusion (zero the top-left quarter of every plane). */
void rob_perturb_img(float *chw, int n, int mode, uint32_t *seed)
{
    int i;
    if (mode == 0 || chw == 0 || seed == 0) return;
    if (mode == 1) {
        for (i = 0; i < n; i++) {
            float v = chw[i] + 0.08f * rob_noise(seed);
            chw[i] = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
        }
    } else if (mode == 2) {
        for (i = 0; i < n; i++) chw[i] *= 0.5f;
    } else if (mode == 3) {
        for (i = 0; i < n; i++) { float v = chw[i] * 1.6f; chw[i] = v > 1.0f ? 1.0f : v; }
    } else if (mode == 4) {
        int plane = n / 3;
        int side = 1; while (side * side < plane) side++;   /* H=W */
        int p, y, x;
        for (p = 0; p < 3; p++)
            for (y = 0; y < side / 2; y++)
                for (x = 0; x < side / 2; x++)
                    chw[p * plane + y * side + x] = 0.0f;
    }
}

/* ---- AI-4 fusion DEPLOYED from the GD32 Embedded AI Tool's TFLite output ----
 * The tool (bundled TinyNeuralNetwork) converted the PyTorch FusionMLP to a .tflite
 * flatbuffer; model/gd32ai_deploy_export.py EXTRACTED the FullyConnected weights
 * straight from that flatbuffer (NOT re-derived from PyTorch) into ai4_tflite_deploy.h
 * and proved they reproduce PyTorch (host check1 max|err| 7.6e-6 = faithful conversion).
 * This runner re-executes the tool's FC graph on-chip:
 *     FC(NIN->NH0) RELU -> FC(NH0->NH1) RELU -> FC(NH1->NOUT)   [no fused act on last]
 * with the tflite FullyConnected weight layout W[out, in] (out = act(x.W^T + b)).
 * Buffers h0/h1 are tiny (48 floats) -> caller stack cost is negligible.
 *
 * So "we ran the GD32 AI Tool end-to-end (PyTorch -> TFLite -> on-chip), host+chip
 * byte-verified" is literally true: the on-chip numbers trace to the tool's flatbuffer. */
void ai4_tflite_deploy(const float *x, float *out)
{
    float h0[TFL_NH0], h1[TFL_NH1];
    int i, j;
    if (x == 0 || out == 0) return;
    for (j = 0; j < TFL_NH0; j++) {
        float s = tfl_b0[j];
        for (i = 0; i < TFL_NIN; i++) s += x[i] * tfl_w0[j * TFL_NIN + i];
        h0[j] = s > 0.0f ? s : 0.0f;                 /* FC0 RELU */
    }
    for (j = 0; j < TFL_NH1; j++) {
        float s = tfl_b1[j];
        for (i = 0; i < TFL_NH0; i++) s += h0[i] * tfl_w1[j * TFL_NH0 + i];
        h1[j] = s > 0.0f ? s : 0.0f;                 /* FC1 RELU */
    }
    for (j = 0; j < TFL_NOUT; j++) {
        float s = tfl_b2[j];
        for (i = 0; i < TFL_NH1; i++) s += h1[i] * tfl_w2[j * TFL_NH1 + i];
        out[j] = s;                                   /* FC2 (no activation) */
    }
}

/* Golden self-test: rerun the tool's TFLite golden input and compare against the
 * golden output emitted from the SAME flatbuffer weights. Sets *err_out to max|err|
 * (NULL ok); returns 1 if the deployed model matches the tool's TFLite output. */
int ai4_tflite_selftest(float *err_out)
{
    float o[TFL_NOUT], e = 0.0f;
    int j;
    ai4_tflite_deploy(tfl_golden_in, o);
    for (j = 0; j < TFL_NOUT; j++) {
        float d = o[j] - tfl_golden_out[j];
        if (d < 0.0f) d = -d;
        if (d > e) e = d;
    }
    if (err_out) *err_out = e;
    return (e < 1.0e-3f) ? 1 : 0;
}
