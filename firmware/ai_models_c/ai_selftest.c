/******************************************************************************
 * ai_selftest.c  —  golden-vector verification of all 5 AI engines.
 ******************************************************************************/
#include "ai_selftest.h"
#include "ai1_crucible.h"
#include "ai1_crucible_golden.h"
#include "ai2_ae.h"
#include "ai3_transformer.h"
#include "ai4_fusion.h"
#include "ai4_calib.h"
#include "ai4_calib_golden.h"
#include "ai1b_ncm.h"
#include "gas_safety.h"
#include "ai5_diagnose.h"
#include "ai5_golden.h"
#include "ai_new_models.h"
#include "new_models_golden.h"
#include "ai10_golden.h"
#include "ai_ext_models.h"
#include "ext_models_golden.h"
#include "ai14_forecast_weights.h"
#include "ai14_forecast_golden.h"
#include "ai15_hostid_weights.h"
#include "ai16_lambda_weights.h"
#include "pl_extra_golden.h"
#include "int8_golden.h"
#include "ai19_rul_weights.h"
#include "ai20_tcfault_weights.h"
#include "ai19_ai20_golden.h"
#include "ai_golden.h"
#include <math.h>

static float max_abs_err(const float *a, const float *b, int n)
{
    float m = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > m) m = e;
    }
    return m;
}

void ai_selftest_run(ai_selftest_result_t *r)
{
    /* ---- AI-1 crucible 3-class CNN (the deployed vision task model) ----
     * 3x64x64 RGB -> 3 logits (empty/loaded/done) + 32-D GAP embedding.
     * Byte-verified vs PyTorch (folded-BN). Weights are trained on REAL phone-
     * shot crucible photos (CIMC/手机拍摄数据, CV 90.7%); this checks the ENGINE
     * reproduces torch. Logit count tracks the golden array (auto, no hard 3). */
    {
        float logits[8], emb[32];
        int   ncls = (int)(sizeof(cru_g_logits) / sizeof(cru_g_logits[0]));
        ai1_crucible_forward(cru_g_input, logits, emb);
        r->ai1_logit_err = max_abs_err(logits, cru_g_logits, ncls);
        r->ai1_emb_err   = max_abs_err(emb,    cru_g_emb,    32);
        r->ai1_pass = (r->ai1_logit_err < AI_SELFTEST_TOL &&
                       r->ai1_emb_err   < AI_SELFTEST_TOL) ? 1 : 0;
    }

    /* ---- AI-2 AE (pure reconstruction on normalised golden input) ---- */
    {
        float recon[32];
        float mse = 0.0f;
        int i;
        ai2_ae_reconstruct(g_ai2_input_norm, recon);
        r->ai2_recon_err = max_abs_err(recon, g_ai2_recon, 32);
        for (i = 0; i < 32; i++) {
            float d = g_ai2_input_norm[i] - recon[i];
            mse += d * d;
        }
        mse /= 32.0f;
        r->ai2_mse_err = fabsf(mse - g_ai2_mse[0]);
        r->ai2_pass = (r->ai2_recon_err < AI_SELFTEST_TOL &&
                       r->ai2_mse_err   < AI_SELFTEST_TOL) ? 1 : 0;
    }

    /* ---- AI-3 TinyTransformer (pure forward on normalised golden input) ---- */
    {
        float logits[5];
        ai3_forward_norm(g_ai3_input_norm, logits);
        r->ai3_logit_err = max_abs_err(logits, g_ai3_logits, 5);
        r->ai3_pass = (r->ai3_logit_err < AI_SELFTEST_TOL) ? 1 : 0;
    }

    /* ---- AI-4 Fusion MLP (raw logits + calibrated/abstention path) ---- */
    {
        float   logits[4];
        float   cprobs[4], cconf = 0.0f;
        uint8_t cabst = 0u;
        float   cerr;
        ai4_forward(g_ai4_input, logits);
        r->ai4_logit_err = max_abs_err(logits, g_ai4_logits, 4);
        /* calibrated fuse on its own golden input (temperature + abstention) */
        (void)ai4_fuse_calibrated(ai4_calib_g_input, cprobs, &cconf, &cabst);
        cerr = max_abs_err(cprobs, ai4_calib_g_probs, 4);
        r->ai4_pass = (r->ai4_logit_err < AI_SELFTEST_TOL &&
                       cerr < AI_SELFTEST_TOL &&
                       cabst == (uint8_t)AI4_CALIB_G_ABSTAIN) ? 1 : 0;
    }

    /* ---- AI-1b NCM mechanism check (on the crucible embedding) ---- */
    {
        float shifted[32];
        float dist = 0.0f;
        int i, pred;
        for (i = 0; i < 32; i++) shifted[i] = cru_g_emb[i] + 5.0f;   /* a distinct class */
        ai1b_reset();
        ai1b_add_sample(0, cru_g_emb);     /* class 0 = the golden crucible embedding */
        ai1b_add_sample(1, shifted);       /* class 1 = shifted version */
        pred = ai1b_classify(cru_g_emb, &dist);   /* should snap back to class 0 */
        r->ncm_pred  = pred;
        r->ncm_pass  = (pred == 0 && ai1b_num_classes() == 2) ? 1 : 0;
        ai1b_reset();                       /* leave clean for runtime enrolment */
    }

    /* ---- gas-safety rule engine (formula-aware furnace gas evolution) ---- */
    r->gas_pass = gas_safety_selftest();

    /* ---- AI-5 root-cause diagnoser (pure forward on 27-D golden vector) ---- */
    {
        float logits[AI5_N_CLASS];
        ai5_forward(g_ai5_input, logits);
        r->ai5_logit_err = max_abs_err(logits, g_ai5_logits, AI5_N_CLASS);
        r->ai5_pass = (r->ai5_logit_err < AI_SELFTEST_TOL) ? 1 : 0;
    }

    /* ---- AI-6/7/8/9/10 new edge models (golden = PyTorch physical outputs) ----
     * Tolerances are absolute on the physical quantity (nm / % / kWh): tight
     * enough to catch a weight/math error, loose enough for M7 float32 vs PC. */
    {
        float lam = 0.0f, fwhm = 0.0f, pct = 0.0f, kwh = 0.0f, co2 = 0.0f;
        float feat[8], probs[4], ferr = 0.0f;
        int   i, nn, cls;

        ai6_optical(g_ai6_desc, &lam, &fwhm);
        r->ai6_pass = (fabsf(lam - g_ai6_lambda) < 0.5f &&
                       fabsf(fwhm - g_ai6_fwhm) < 0.5f) ? 1 : 0;

        (void)ai7_thermal(g_ai6_desc, &pct);
        r->ai7_pass = (fabsf(pct - g_ai7_pct) < 0.1f) ? 1 : 0;

        ai8_energy(g_ai8_e5, &kwh, &co2);
        r->ai8_pass = (fabsf(kwh - g_ai8_kwh) < 0.5f) ? 1 : 0;

        nn = ai9_retrieve(g_ai9_query, &lam);   /* lam reused as dist sink */
        r->ai9_pass = (nn == g_ai9_nn) ? 1 : 0;

        ai10_features(g_ai10_window, feat);
        for (i = 0; i < 8; i++) { float e = fabsf(feat[i] - g_ai10_feat8[i]); if (e > ferr) ferr = e; }
        cls = ai10_vibration(g_ai10_window, probs);
        r->ai10_pass = (cls == g_ai10_cls && ferr < 5.0e-2f) ? 1 : 0;
    }

    /* ---- AI-11/12/13 new edge models (golden = PyTorch eval outputs) ----
     * AI-11 class + P(pure); AI-12 class; AI-13 reconstruction + MSE. */
    {
        float p = 0.0f, pr3[3], rec[64], mse = 0.0f, rerr = 0.0f;
        int   c11, c12, i;
        int   n13 = (int)(sizeof(g_ai13_recon) / sizeof(g_ai13_recon[0]));

        c11 = ai11_purity(g_ai11_desc, &p);
        r->ai11_pass = (c11 == g_ai11_cls && fabsf(p - g_ai11_ppure) < 1.0e-2f) ? 1 : 0;

        c12 = ai12_plclass(g_ai12_spec, pr3);
        r->ai12_pass = (c12 == g_ai12_cls) ? 1 : 0;

        (void)ai13_plqc(g_ai13_spec, rec, &mse);
        for (i = 0; i < n13; i++) { float e = fabsf(rec[i] - g_ai13_recon[i]); if (e > rerr) rerr = e; }
        r->ai13_pass = (rerr < 1.0e-2f && fabsf(mse - g_ai13_mse) < 1.0e-3f) ? 1 : 0;
    }

    /* ---- AI-14/15/16/17 new edge models (golden = PyTorch outputs) ---- */
    {
        float out[AI14_HOR], ferr = 0.0f, lerr = 0.0f, emb[16], eerr = 0.0f;
        int   g, k, hok = 1, mok = 1;

        /* AI-14 forecaster: NG windows x HOR-step regression */
        for (g = 0; g < AI14_NG; g++) {
            ai14_forecast(&ai14_g_in[g * AI14_WIN], out);
            for (k = 0; k < AI14_HOR; k++) {
                float e = fabsf(out[k] - ai14_g_out[g * AI14_HOR + k]);
                if (e > ferr) ferr = e;
            }
        }
        r->ai14_pass = (ferr < 1.0e-3f) ? 1 : 0;

        /* AI-15 host-ID + AI-16 lambda_em on the PL golden spectra */
        for (g = 0; g < PLX_NG; g++) {
            if (ai15_hostid(&plx_g_spec[g * AI15_IN], 0) != plx_g_hostcls[g]) hok = 0;
            {
                float e = fabsf(ai16_lambda(&plx_g_spec[g * AI16_IN]) - plx_g_lambda[g]);
                if (e > lerr) lerr = e;
            }
        }
        r->ai15_pass = hok;
        r->ai16_pass = (lerr < 1.0e-2f) ? 1 : 0;

        /* AI-17 few-shot NCM: embed reproduces PyTorch + classify == Python NCM */
        for (g = 0; g < PLX_NG; g++) {
            ai12_embed(&plx_g_spec[g * 64], emb);
            for (k = 0; k < 16; k++) {
                float e = fabsf(emb[k] - plx_g_emb16[g * 16 + k]);
                if (e > eerr) eerr = e;
            }
        }
        ai17_pl_reset();
        for (g = 0; g < PLX_NSEED; g++) {
            ai12_embed(&plx_seed_spec[g * 64], emb);
            ai17_pl_add_sample(plx_seed_cls[g], emb);
        }
        for (g = 0; g < PLX_NQ; g++) {
            ai12_embed(&plx_q_spec[g * 64], emb);
            if (ai17_pl_classify(emb, 0) != plx_q_pred[g]) mok = 0;
        }
        r->ai17_pass = (eerr < 1.0e-4f && mok) ? 1 : 0;
        ai17_pl_reset();
    }

    /* ---- AI-19/20 new edge models (golden = PyTorch outputs) ----
     * AI-19 RUL: compare normalised remaining; AI-20 TC-integrity: compare logits
     * (the C engine recomputes the 8 plausibility features -> verifies the full path)
     * AND the argmax class. */
    {
        float rerr = 0.0f, lerr = 0.0f, lg[AI20_NC];
        int   g, k, cok = 1;
        for (g = 0; g < AI19_NG; g++) {
            float e = fabsf(ai19_rul(&ai19_g_in[g * AI19_NX]) / AI19_RNORM - ai19_g_out[g]);
            if (e > rerr) rerr = e;
        }
        r->ai19_pass = (rerr < 1.0e-3f) ? 1 : 0;

        for (g = 0; g < AI20_NG; g++) {
            ai20_tcfault_logits(&ai20_g_in[g * (2 * AI20_L)], lg);
            for (k = 0; k < AI20_NC; k++) {
                float e = fabsf(lg[k] - ai20_g_logit[g * AI20_NC + k]);
                if (e > lerr) lerr = e;
            }
            if (ai20_tcfault(&ai20_g_in[g * (2 * AI20_L)], 0) != ai20_g_cls[g]) cok = 0;
        }
        r->ai20_pass = (lerr < 1.0e-2f && cok) ? 1 : 0;
    }

    /* ---- B-track depth features: CAM / INT8 / adaptive conformal ---- */
    {
        float cam[16], cerr = 0.0f;
        int   i, cls = ai1_crucible_cam(cru_g_input, cam);
        for (i = 0; i < 16; i++) { float e = fabsf(cam[i] - cru_g_cam[i]); if (e > cerr) cerr = e; }
        r->cam_pass = (cls == cru_g_cam_cls && cerr < 1.0e-3f) ? 1 : 0;
    }
    {
        int g, ok = 1;
        for (g = 0; g < I8_NG; g++) {
            int c8 = ai12_plclass_int8(&i8_g_spec[g * 64], 0);
            if (c8 != i8_g_cls_int8[g] || c8 != ai12_plclass(&i8_g_spec[g * 64], 0)) ok = 0;
        }
        r->int8_pass = ok;
    }
    {
        /* adaptive conformal: with q tracking, the long-run exceedance of a constant
         * score stream converges to alpha (here all scores 0 -> q decays to qmin and
         * exceedance ~0; feed a 50/50 stream to check it settles near alpha=0.5). */
        aconf_t c; int t, exc = 0;
        unsigned int rng = 7u;
        aconf_init(&c, 0.5f, 0.5f, 0.02f);
        for (t = 0; t < 2000; t++) {
            rng = rng * 1664525u + 1013904223u;
            {
                float s = (float)((rng >> 8) & 0xFFFFFFu) / (float)0x1000000u;  /* U[0,1) */
                int a = aconf_step(&c, s);
                if (t >= 1000) exc += a;
            }
        }
        {
            float fpr = (float)exc / 1000.0f;
            r->aconf_pass = (fpr > 0.40f && fpr < 0.60f) ? 1 : 0;   /* ~alpha=0.5 */
        }
    }

    /* ---- GD32 Embedded AI Tool deployment: re-run the tool's AI-4 TFLite on-chip ----
     * Weights were EXTRACTED from the .tflite flatbuffer (ai4_tflite_deploy.h); this
     * reproduces the tool's golden output -> the deployed model IS the tool's output. */
    r->tflite_pass = ai4_tflite_selftest(&r->tflite_err);

    r->all_pass = (r->ai1_pass && r->ai2_pass && r->ai3_pass &&
                   r->ai4_pass && r->ncm_pass && r->gas_pass && r->ai5_pass &&
                   r->ai6_pass && r->ai7_pass && r->ai8_pass &&
                   r->ai9_pass && r->ai10_pass &&
                   r->ai11_pass && r->ai12_pass && r->ai13_pass &&
                   r->ai14_pass && r->ai15_pass && r->ai16_pass && r->ai17_pass &&
                   r->ai19_pass && r->ai20_pass &&
                   r->cam_pass && r->int8_pass && r->aconf_pass &&
                   r->tflite_pass) ? 1 : 0;
}
