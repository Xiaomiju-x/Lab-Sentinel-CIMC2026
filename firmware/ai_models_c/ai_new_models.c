/* ai_new_models.c — AI-6/7/8/9/10 edge inference (CIMC Lab-Sentinel)
 * Hand-written float32, reuses nn_ops.c. See ai_new_models.h for the contract and
 * the per-model weights headers for honest provenance. Host-verified vs PyTorch.
 */
#include "ai_new_models.h"
#include "nn_ops.h"
#include "ai6_optical_weights.h"
#include "ai7_thermal_weights.h"
#include "ai8_energy_weights.h"
#include "ai9_recipe_db.h"
#include "ai10_vib_weights.h"
#include <math.h>

/* ----------------------------------------------------------------- AI-6 optical */
void ai6_optical(const float *desc24, float *lambda_em_nm, float *fwhm_nm)
{
    float h1[AI6_H], h2[AI6_H], o[AI6_OUT];
    nn_linear(desc24, ai6_w0, ai6_b0, h1, AI6_IN, AI6_H); nn_relu(h1, AI6_H);
    nn_linear(h1,     ai6_w1, ai6_b1, h2, AI6_H,  AI6_H); nn_relu(h2, AI6_H);
    nn_linear(h2,     ai6_w2, ai6_b2, o,  AI6_H,  AI6_OUT);
    if (lambda_em_nm) *lambda_em_nm = o[0] * ai6_out_sd[0] + ai6_out_mu[0];
    if (fwhm_nm)      *fwhm_nm      = o[1] * ai6_out_sd[1] + ai6_out_mu[1];
}

/* ----------------------------------------------------------------- AI-7 thermal */
int ai7_thermal(const float *desc24, float *thermal_pct)
{
    float h1[AI7_H], h2[AI7_H], o[AI7_OUT];
    float pct;
    nn_linear(desc24, ai7_w0, ai7_b0, h1, AI7_IN, AI7_H); nn_relu(h1, AI7_H);
    nn_linear(h1,     ai7_w1, ai7_b1, h2, AI7_H,  AI7_H); nn_relu(h2, AI7_H);
    nn_linear(h2,     ai7_w2, ai7_b2, o,  AI7_H,  AI7_OUT);
    pct = o[0] * ai7_out_sd + ai7_out_mu;
    if (thermal_pct) *thermal_pct = pct;
    return (pct >= 75.0f) ? 2 : (pct >= 60.0f ? 1 : 0);   /* good / marginal / poor */
}

/* ----------------------------------------------------------------- AI-8 energy */
void ai8_energy(const float *e5, float *kwh, float *co2_kg)
{
    float h1[AI8_H], h2[AI8_H], o[AI8_OUT];
    float e;
    nn_linear(e5, ai8_w0, ai8_b0, h1, AI8_IN, AI8_H); nn_relu(h1, AI8_H);
    nn_linear(h1, ai8_w1, ai8_b1, h2, AI8_H, AI8_H);  nn_relu(h2, AI8_H);
    nn_linear(h2, ai8_w2, ai8_b2, o,  AI8_H, AI8_OUT);
    e = o[0] * ai8_out_sd + ai8_out_mu;
    if (e < 0.0f) e = 0.0f;
    if (kwh)    *kwh    = e;
    if (co2_kg) *co2_kg = e * ai8_co2_per_kwh;
}

/* ----------------------------------------------------------------- AI-9 retrieval */
int ai9_retrieve(const float *q18, float *out_dist)
{
    int i, j, best = 0;
    float bestd = 1e30f;
    for (i = 0; i < AI9_N; i++) {
        const float *ref = ai9_ref + (long)i * AI9_D;
        float d = 0.0f;
        for (j = 0; j < AI9_D; j++) {
            float t = (q18[j] - ref[j]) / ai9_sd[j];
            d += t * t;
        }
        if (d < bestd) { bestd = d; best = i; }
    }
    if (out_dist) *out_dist = bestd;
    return best;
}

int ai9_count(void) { return AI9_N; }

const char *ai9_recipe_name(int idx)
{
    if (idx < 0 || idx >= AI9_N) return "?";
    return ai9_name[idx];
}

/* ----------------------------------------------------------------- AI-10 vibration */
#define VIB_N 64
#define VIB_FS 200.0f
#define VIB_BINS 33      /* N/2 + 1 */

void ai10_features(const float *x, float *feat8)
{
    int n, k;
    float mean = 0.0f, var = 0.0f, ss = 0.0f, peak = 0.0f, rms, sd, crest, kurt;
    float e1 = 0.0f, e2 = 0.0f, e3 = 0.0f, ehi = 0.0f, ebb = 0.0f;

    for (n = 0; n < VIB_N; n++) {
        float a = x[n] < 0.0f ? -x[n] : x[n];
        mean += x[n]; ss += x[n] * x[n];
        if (a > peak) peak = a;
    }
    mean /= (float)VIB_N;
    rms = sqrtf(ss / (float)VIB_N);
    for (n = 0; n < VIB_N; n++) { float d = x[n] - mean; var += d * d; }
    var /= (float)VIB_N;
    sd = sqrtf(var);
    crest = peak / (rms + 1e-9f);
    kurt = 0.0f;
    for (n = 0; n < VIB_N; n++) { float z = (x[n] - mean) / (sd + 1e-9f); kurt += z * z * z * z; }
    kurt /= (float)VIB_N;

    /* magnitude^2 spectrum (Hanning window), bins 0..32; band-sum like numpy rfft */
    for (k = 0; k < VIB_BINS; k++) {
        float re = 0.0f, im = 0.0f, p, f;
        for (n = 0; n < VIB_N; n++) {
            float w = 0.5f - 0.5f * cosf(6.28318530718f * (float)n / 63.0f);
            float ang = 6.28318530718f * (float)k * (float)n / (float)VIB_N;
            float xv = x[n] * w;
            re += xv * cosf(ang);
            im -= xv * sinf(ang);
        }
        p = re * re + im * im;
        f = (float)k * (VIB_FS / (float)VIB_N);       /* k * 3.125 Hz */
        if (f >= 0.0f  && f < 100.0f) ebb += p;
        if (f >= 19.0f && f < 31.0f)  e1  += p;
        if (f >= 44.0f && f < 56.0f)  e2  += p;
        if (f >= 69.0f && f < 81.0f)  e3  += p;
        if (f >= 65.0f && f < 100.0f) ehi += p;
    }
    ebb += 1e-9f;
    feat8[0] = rms; feat8[1] = crest; feat8[2] = kurt;
    feat8[3] = e1 / ebb; feat8[4] = e2 / ebb; feat8[5] = e3 / ebb; feat8[6] = ehi / ebb;
    feat8[7] = ebb;
}

int ai10_vibration(const float *window64, float *probs4)
{
    float f8[8], nf[8], h1[AI10_H], h2[AI10_H], o[AI10_OUT];
    int i, cls;
    ai10_features(window64, f8);
    for (i = 0; i < AI10_IN; i++) nf[i] = (f8[i] - ai10_in_mu[i]) / ai10_in_sd[i];
    nn_linear(nf, ai10_w0, ai10_b0, h1, AI10_IN, AI10_H); nn_relu(h1, AI10_H);
    nn_linear(h1, ai10_w1, ai10_b1, h2, AI10_H,  AI10_H); nn_relu(h2, AI10_H);
    nn_linear(h2, ai10_w2, ai10_b2, o,  AI10_H,  AI10_OUT);
    cls = nn_argmax(o, AI10_OUT);
    if (probs4) { for (i = 0; i < AI10_OUT; i++) probs4[i] = o[i]; nn_softmax(probs4, AI10_OUT); }
    return cls;
}
