/******************************************************************************
 * ai_selftest.h  —  On-chip golden-vector self-test for all AI engines.
 *
 * Runs each C inference engine on a fixed input baked from PyTorch and
 * compares against the PyTorch (eval-mode) expected output (ai_golden.h).
 * This is the verification anchor: if max|err| is tiny on hardware, the
 * float engines reproduce PyTorch byte-for-byte on the M7 FPU.
 ******************************************************************************/
#ifndef AI_SELFTEST_H
#define AI_SELFTEST_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float ai1_logit_err;   /* max|C-torch| over 10 logits   */
    float ai1_emb_err;     /* max|C-torch| over 32 embedding */
    float ai2_recon_err;   /* max|C-torch| over 32 recon     */
    float ai2_mse_err;     /* |C-torch| reconstruction MSE   */
    float ai3_logit_err;   /* max|C-torch| over 5 logits     */
    float ai4_logit_err;   /* max|C-torch| over 4 logits     */
    int   ncm_pred;        /* AI-1b NCM demo prediction (expect 0) */
    int   ai1_pass;
    int   ai2_pass;
    int   ai3_pass;
    int   ai4_pass;
    int   ncm_pass;
    int   gas_pass;        /* gas-safety rule engine self-test (formula+temp) */
    float ai5_logit_err;   /* max|C-torch| over 9 root-cause logits           */
    int   ai5_pass;        /* AI-5 root-cause diagnoser golden self-test      */
    /* new models (6->11): edge surrogates / retrieval / vib PdM */
    int   ai6_pass;        /* optical surrogate (lambda_em + FWHM)            */
    int   ai7_pass;        /* thermal-quench surrogate                       */
    int   ai8_pass;        /* energy/carbon estimator                        */
    int   ai9_pass;        /* recipe analog retrieval (nearest idx)          */
    int   ai10_pass;       /* vibration PdM (class + features)               */
    /* new models (11->14): phase-purity prior / PL dopant classifier / PL-QC AE */
    int   ai11_pass;       /* phase-purity prior (class + P(pure))            */
    int   ai12_pass;       /* PL dopant classifier (class + probs)            */
    int   ai13_pass;       /* PL-QC autoencoder (reconstruction + MSE)        */
    /* new models (14->18): temp forecaster / PL host-ID / PL lambda / PL few-shot */
    int   ai14_pass;       /* furnace temp multi-step forecaster              */
    int   ai15_pass;       /* PL host-ID classifier                           */
    int   ai16_pass;       /* PL lambda_em regressor                          */
    int   ai17_pass;       /* PL few-shot NCM (embed + classify)              */
    /* new models (18->20): sintering RUL / thermocouple-integrity classifier */
    int   ai19_pass;       /* AI-19 sintering RUL/ETA regressor               */
    int   ai20_pass;       /* AI-20 thermocouple-integrity classifier         */
    /* depth features (B-track): CAM explainability / INT8 path / adaptive conformal */
    int   cam_pass;        /* AI-1 CAM 4x4 heatmap matches PyTorch            */
    int   int8_pass;       /* AI-12 INT8 twin agrees with fp32                */
    int   aconf_pass;      /* adaptive conformal holds target FPR under drift */
    /* GD32 Embedded AI Tool deployment: AI-4 TFLite output re-run on-chip */
    float tflite_err;      /* max|chip - tool-TFLite golden| over 4 logits   */
    int   tflite_pass;     /* deployed TFLite (AI-4) reproduces the tool's output */
    int   all_pass;
} ai_selftest_result_t;

/* Pass threshold on max|err| (float32 accumulation slack). */
#define AI_SELFTEST_TOL 1.0e-2f

void ai_selftest_run(ai_selftest_result_t *r);

#ifdef __cplusplus
}
#endif

#endif /* AI_SELFTEST_H */
