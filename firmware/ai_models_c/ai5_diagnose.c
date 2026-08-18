/******************************************************************************
 * ai5_diagnose.c  --  AI-5 Root-Cause Diagnoser forward + action lookup.
 * Weights (BN folded) from ai5_diagnose_weights.h.
 *
 * AI-1/2/3/4 say "what's wrong"; AI-5 says "why + what to do". The 27-D input
 * is the diagnostic vector the env/fusion tasks assemble from the live AI
 * snapshot + gas_safety_eval + furnace_sim (layout in ai5_config.json).
 ******************************************************************************/
#include "ai5_diagnose.h"
#include "nn_ops.h"
#include "ai5_diagnose_weights.h"
#include "ai5_golden.h"   /* g_ai5_input / g_ai5_logits for ai5_selftest() */
#include "nn_opt.h"   /* force -O3 -Otime on this TU (project default is -O0) */

void ai5_forward(const float *feat27, float *logits9)
{
    float h0[AI5_H0];   /* 32 */
    float h1[AI5_H1];   /* 16 */
    nn_linear(feat27, ai5_fc0_w, ai5_fc0_b, h0, AI5_IN, AI5_H0);
    nn_relu(h0, AI5_H0);
    nn_linear(h0, ai5_fc1_w, ai5_fc1_b, h1, AI5_H0, AI5_H1);
    nn_relu(h1, AI5_H1);
    nn_linear(h1, ai5_fc2_w, ai5_fc2_b, logits9, AI5_H1, AI5_NCLS);
}

int ai5_diagnose(const float *feat27, float *probs9, float *out_conf)
{
    int cls;
    ai5_forward(feat27, probs9);
    nn_softmax(probs9, AI5_N_CLASS);
    cls = nn_argmax(probs9, AI5_N_CLASS);
    if (out_conf) *out_conf = probs9[cls];
    return cls;
}

/* Corrective actions -- auto-generated from gas_safety reasons +
 * sintering_profiles + the 25228-chunk literature gate (ai5_rootcause_labels.md).
 * ASCII-only (armcc / no %f safe). Index = ai5_rootcause_t. */
static const char *const AI5_ACTION[AI5_N_CLASS] = {
    /* NORMAL        */ "Continue; no operator action required.",
    /* RAMP_TOO_FAST */ "Reduce ramp rate to the host profile; add an intermediate dwell through the reaction window.",
    /* UNDER_TEMP    */ "Raise peak temperature or extend the soak to the host profile; verify with XRD before accepting.",
    /* OXIDIZE_CR6   */ "Switch to a reducing/inert atmosphere (5%H2/N2) or lower peak temperature; ventilate (Cr6+ toxic).",
    /* DECOMP_GAS    */ "Slow the ramp through the decomposition window and ensure venting before continuing to peak.",
    /* VOLATIL_LOSS  */ "Add excess of the volatile component or use a covered/sealed crucible; cap the peak temperature.",
    /* MOISTURE      */ "Dry / pre-calcine precursors; load under low humidity.",
    /* GRIND_INHOMO  */ "Re-grind / extend mixing cycles to the host grinding_cycles.",
    /* SOAK_DRIFT    */ "Check furnace PID / heating element; recalibrate before next batch."
};

const char *ai5_action(int cls)
{
    if (cls < 0 || cls >= AI5_N_CLASS) return "(unknown root cause)";
    return AI5_ACTION[cls];
}

const char *ai5_name(int cls)
{
    if (cls < 0 || cls >= AI5_N_CLASS) return "UNKNOWN";
    return AI5_CLASS_NAME[cls];   /* from ai5_diagnose_weights.h */
}

int ai5_selftest(void)
{
    float logits[AI5_N_CLASS];
    float maxerr = 0.0f, e;
    int i;
    ai5_forward(g_ai5_input, logits);
    for (i = 0; i < AI5_N_CLASS; i++) {
        e = logits[i] - g_ai5_logits[i];
        if (e < 0.0f) e = -e;
        if (e > maxerr) maxerr = e;
    }
    return (maxerr < 1.0e-3f) ? 1 : 0;
}
