/******************************************************************************
 * ai2_ae.c  —  AI-2 Sintering AE forward + conformal anomaly + attribution.
 * Weights / mu / std / q_hat / feat_mae from ai2_ae_weights.h.
 ******************************************************************************/
#include "ai2_ae.h"
#include "nn_ops.h"
#include "ai2_ae_weights.h"
#include "nn_opt.h"    /* force -O3 -Otime on this TU (project default is -O0) */

/* feature-group index ranges (see synth_data.py layout) */
#define G_TEMP_LO  0
#define G_TEMP_HI  8     /* [0,8)   temperature */
#define G_GAS_LO   8
#define G_GAS_HI   14    /* [8,14)  gas/atmosphere */
#define G_ROOM_LO  14
#define G_ROOM_HI  18    /* [14,18) room env (temp/humidity) */
#define G_VIB_LO   18
#define G_VIB_HI   26    /* [18,26) vibration */

void ai2_ae_reconstruct(const float *xn, float *recon)
{
    float h16[AI2_H1];
    float h8[AI2_H2];
    float d16[AI2_H1];
    /* encoder: 32->16 relu ->8 */
    nn_linear(xn,  ai2_enc0_w, ai2_enc0_b, h16, AI2_DIM, AI2_H1);
    nn_relu(h16, AI2_H1);
    nn_linear(h16, ai2_enc1_w, ai2_enc1_b, h8,  AI2_H1, AI2_H2);
    /* decoder: 8->16 relu ->32 */
    nn_linear(h8,  ai2_dec0_w, ai2_dec0_b, d16, AI2_H2, AI2_H1);
    nn_relu(d16, AI2_H1);
    nn_linear(d16, ai2_dec1_w, ai2_dec1_b, recon, AI2_H1, AI2_DIM);
}

/* ---- Adaptive Conformal q_hat (upgrade H) ---- */
static float s_qhat_adaptive = AI2_QHAT_90;   /* starts at the trained prior */

void ai2_ae_adapt(float mse)
{
    const float alpha = 0.10f;    /* target miscoverage (=> 90% CI)        */
    const float gamma = 0.01f;    /* learning rate (slow, stable drift)    */
    const float qmin  = 0.02f;    /* never collapse below floor            */
    const float qmax  = 1.00f;    /* never run away above ceiling          */
    float exceed = (mse > s_qhat_adaptive) ? 1.0f : 0.0f;
    s_qhat_adaptive += gamma * (exceed - alpha);
    if (s_qhat_adaptive < qmin) s_qhat_adaptive = qmin;
    if (s_qhat_adaptive > qmax) s_qhat_adaptive = qmax;
}

float ai2_ae_qhat(void)         { return s_qhat_adaptive; }
float ai2_ae_qhat_trained(void) { return (float)AI2_QHAT_90; }

static float group_mean_abs(const float *ar, int lo, int hi)
{
    float s = 0.0f;
    int i;
    for (i = lo; i < hi; i++) s += ar[i];
    return s / (float)(hi - lo);
}

static float group_baseline(int lo, int hi)
{
    float s = 0.0f;
    int i;
    for (i = lo; i < hi; i++) s += ai2_feat_mae[i];
    return s / (float)(hi - lo);
}

float ai2_ae_score(const float *feat_raw, float out_resid3[3],
                   uint8_t *out_attr, float *out_ratio)
{
    float xn[AI2_FEAT_DIM];
    float recon[AI2_FEAT_DIM];
    float ar[AI2_FEAT_DIM];      /* per-feature abs residual (normalised space) */
    float mse = 0.0f;
    int i;

    /* z-score normalise (std already has +1e-8 baked in from training).
     * Clamp to +/-8 sigma: a single corrupt sensor read (e.g. SHT30 soft-I2C
     * CRC false value) must not blow the reconstruction MSE up to astronomical
     * values. Training data lives well within +/-3 sigma, so +/-8 is a safe
     * robustness cap that never touches in-distribution inputs. */
    for (i = 0; i < AI2_FEAT_DIM; i++) {
        float s = ai2_std[i];
        float z = (s > 1e-6f) ? (feat_raw[i] - ai2_mu[i]) / s : 0.0f;
        if (z >  8.0f) z =  8.0f;
        else if (z < -8.0f) z = -8.0f;
        xn[i] = z;
    }

    ai2_ae_reconstruct(xn, recon);

    for (i = 0; i < AI2_FEAT_DIM; i++) {
        float d = xn[i] - recon[i];
        ar[i] = (d < 0.0f) ? -d : d;
        mse += d * d;
    }
    mse /= (float)AI2_FEAT_DIM;

    if (out_resid3 != 0) {
        float t = group_mean_abs(ar, G_TEMP_LO, G_TEMP_HI);
        float v = group_mean_abs(ar, G_VIB_LO,  G_VIB_HI);
        float g = group_mean_abs(ar, G_GAS_LO,  G_GAS_HI);
        out_resid3[0] = (t > 1.0f) ? 1.0f : t;
        out_resid3[1] = (v > 1.0f) ? 1.0f : v;
        out_resid3[2] = (g > 1.0f) ? 1.0f : g;
    }

    if (out_attr != 0) {
        uint8_t m = 0u;
        const float K = 2.5f;   /* group residual > 2.5x its normal baseline = deviating */
        if (group_mean_abs(ar, G_TEMP_LO, G_TEMP_HI) > K * group_baseline(G_TEMP_LO, G_TEMP_HI)) m |= AI2_ATTR_TEMP;
        if (group_mean_abs(ar, G_ROOM_LO, G_ROOM_HI) > K * group_baseline(G_ROOM_LO, G_ROOM_HI)) m |= AI2_ATTR_HUMID;
        if (group_mean_abs(ar, G_GAS_LO,  G_GAS_HI)  > K * group_baseline(G_GAS_LO,  G_GAS_HI))  m |= AI2_ATTR_GAS;
        if (group_mean_abs(ar, G_VIB_LO,  G_VIB_HI)  > K * group_baseline(G_VIB_LO,  G_VIB_HI))  m |= AI2_ATTR_VIB;
        *out_attr = m;
    }

    if (out_ratio != 0) {
        float r = mse / s_qhat_adaptive;   /* adaptive conformal threshold */
        *out_ratio = (r > 6.0f) ? 6.0f : r;
    }

    return mse;
}
