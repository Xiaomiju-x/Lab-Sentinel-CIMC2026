/* tflite_deploy_test.c — host golden for the GD32 Embedded AI Tool deployment leg.
 * ===========================================================================
 * Compiles the SAME on-chip runner (ai4_tflite_deploy / ai4_tflite_selftest in
 * ai_ext_models.c) that the firmware boot self-test calls, and verifies it
 * reproduces the AI-4 TFLite output that the GD32 AI Tool produced. The weights
 * live in ai4_tflite_deploy.h, EXTRACTED from the tool's .tflite flatbuffer by
 * model/gd32ai_deploy_export.py (host check1 already proved 7.6e-6 vs PyTorch).
 *
 * Build (Windows clang, NO -lm; math is in the C runtime):
 *   clang -std=c99 -O2 -Wall -Wextra tflite_deploy_test.c \
 *       ../../firmware/ai_models_c/ai_ext_models.c \
 *       ../../firmware/ai_models_c/nn_ops.c \
 *       -I../../firmware/ai_models_c -I. -o tflite_deploy_test.exe
 *
 * Pass criterion: TFLITE_DEPLOY ALL_PASS  (selftest==1, golden max|err| tiny). */
#include <stdio.h>
#include <math.h>
#include "ai_ext_models.h"
#include "ai4_tflite_deploy.h"   /* tfl_golden_in / tfl_golden_out (static copy here) */

int main(void)
{
    int ok = 1;

    /* 1) the engine's own self-test (same call the MCU boot makes) */
    float serr = -1.0f;
    int spass = ai4_tflite_selftest(&serr);
    printf("[1] ai4_tflite_selftest -> pass=%d  max|err|=%.3e\n", spass, (double)serr);
    if (!spass || serr >= 1.0e-3f) ok = 0;

    /* 2) explicit forward on the tool's golden input vs the tool's golden output */
    {
        float out[TFL_NOUT];
        float e = 0.0f;
        int j;
        ai4_tflite_deploy(tfl_golden_in, out);
        printf("[2] deployed AI-4 TFLite out = [");
        for (j = 0; j < TFL_NOUT; j++) {
            float d = fabsf(out[j] - tfl_golden_out[j]);
            if (d > e) e = d;
            printf(" %.4f", (double)out[j]);
        }
        printf(" ]  golden = [");
        for (j = 0; j < TFL_NOUT; j++) printf(" %.4f", (double)tfl_golden_out[j]);
        printf(" ]  max|err|=%.3e\n", (double)e);
        if (e >= 1.0e-4f) ok = 0;
    }

    /* 3) graph dims sanity (the tool emitted FC 16->32->16->4) */
    printf("[3] graph dims: NIN=%d NH0=%d NH1=%d NOUT=%d\n",
           TFL_NIN, TFL_NH0, TFL_NH1, TFL_NOUT);
    if (!(TFL_NIN == 16 && TFL_NH0 == 32 && TFL_NH1 == 16 && TFL_NOUT == 4)) ok = 0;

    printf("TFLITE_DEPLOY %s\n", ok ? "ALL_PASS" : "FAIL");
    return ok ? 0 : 1;
}
