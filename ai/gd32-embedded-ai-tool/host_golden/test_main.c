/* Host verification harness for the on-chip AI engines (clang on PC).
 * Compiles the firmware float engines unchanged and runs the golden self-test. */
#include <stdio.h>
#include "ai_selftest.h"

unsigned char g_host_scratch[262144];   /* stands in for SDRAM_AI_SCRATCH */

int main(void)
{
    ai_selftest_result_t r;
    ai_selftest_run(&r);
    printf("AI-1 CNN     logit_err=%.3e  emb_err=%.3e  pass=%d\n", r.ai1_logit_err, r.ai1_emb_err, r.ai1_pass);
    printf("AI-2 AE      recon_err=%.3e  mse_err=%.3e  pass=%d\n", r.ai2_recon_err, r.ai2_mse_err, r.ai2_pass);
    printf("AI-3 Xformer logit_err=%.3e  pass=%d\n", r.ai3_logit_err, r.ai3_pass);
    printf("AI-4 Fusion  logit_err=%.3e  pass=%d\n", r.ai4_logit_err, r.ai4_pass);
    printf("AI-1b NCM    pred=%d (expect 0)  pass=%d\n", r.ncm_pred, r.ncm_pass);
    printf("GAS safety   rule-engine self-test pass=%d\n", r.gas_pass);
    printf("AI-5 RootCau logit_err=%.3e  pass=%d\n", r.ai5_logit_err, r.ai5_pass);
    printf("AI-6 Optical pass=%d   AI-7 Thermal pass=%d   AI-8 Energy pass=%d\n",
           r.ai6_pass, r.ai7_pass, r.ai8_pass);
    printf("AI-9 Retrieve pass=%d  AI-10 VibPdM pass=%d\n",
           r.ai9_pass, r.ai10_pass);
    printf("AI-11 Purity pass=%d  AI-12 PLclass pass=%d  AI-13 PL-QC pass=%d\n",
           r.ai11_pass, r.ai12_pass, r.ai13_pass);
    printf("AI-14 Forecast pass=%d  AI-15 host-ID pass=%d  AI-16 lambda pass=%d  AI-17 PL-fewshot pass=%d\n",
           r.ai14_pass, r.ai15_pass, r.ai16_pass, r.ai17_pass);
    printf("AI-19 RUL/ETA pass=%d  AI-20 TC-integrity pass=%d   (20 models)\n",
           r.ai19_pass, r.ai20_pass);
    printf("B-depth: CAM pass=%d  INT8 pass=%d  AdaptiveConformal pass=%d\n",
           r.cam_pass, r.int8_pass, r.aconf_pass);
    printf("GD32-AI-Tool: AI-4 TFLite deployed on-chip  err=%.3e  pass=%d\n",
           r.tflite_err, r.tflite_pass);
    printf("-------------------------------------------\n");
    printf("ALL_PASS=%d  (tol=%.0e)\n", r.all_pass, (double)AI_SELFTEST_TOL);
    return r.all_pass ? 0 : 1;
}
