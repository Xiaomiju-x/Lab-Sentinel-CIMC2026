/* gas_safety.c -- see gas_safety.h
 * ---------------------------------------------------------------------
 * Formula-aware furnace gas-evolution supervisor for the GD32 edge sentinel.
 * Pure integer logic over the auto-generated GAS_RULES table.
 *
 * Host self-test:
 *   clang -DGAS_SAFETY_HOST_TEST gas_safety.c -o gas_safety_test && ./gas_safety_test
 */
#include "gas_safety.h"
#include <string.h>

static int in_batch(const char *const *batch, int n, const char *mat)
{
    int i;
    for (i = 0; i < n; ++i) {
        if (batch[i] && strcmp(batch[i], mat) == 0) return 1;
    }
    return 0;
}

void gas_safety_eval(const char *const *batch, int n_batch,
                     int temp_c, gas_status_t *out)
{
    unsigned i;
    out->expected_mask = 0;
    out->max_sev = 0;
    out->n_active = 0;
    out->top_gas = GAS_NONE;
    out->top_reason = 0;

    for (i = 0; i < GAS_RULES_N; ++i) {
        const gas_rule_t *r = &GAS_RULES[i];
        if (temp_c < r->onset_c) continue;          /* not hot enough yet */
        if (!in_batch(batch, n_batch, r->mat)) continue; /* not in charge  */

        out->expected_mask |= (uint8_t)(1u << r->gas);
        out->n_active++;
        if (r->sev > out->max_sev) {
            out->max_sev   = r->sev;
            out->top_gas   = r->gas;
            out->top_reason = r->reason;
        }
    }
}

int gas_safety_crosscheck(const gas_status_t *st, int mq_rise)
{
    if (st->expected_mask) {
        if (mq_rise) return GAS_XC_CONFIRMED;             /* chemistry explains it */
        if (st->max_sev >= 3) return GAS_XC_SENSORFLAT;   /* danger expected, flat  */
        return GAS_XC_QUIET;
    }
    /* no gas expected from chemistry */
    return mq_rise ? GAS_XC_UNEXPECTED : GAS_XC_QUIET;
}

int gas_safety_selftest(void)
{
    gas_status_t st;
    int ok = 1;

    /* (1) YAG:Cr charge at sinter temp -> Cr6+ aerosol caution expected. */
    static const char *const yag[] = { "Y2O3", "Al2O3", "Cr2O3" };
    gas_safety_eval(yag, 3, 1500, &st);
    ok &= (st.expected_mask & (1u << GAS_CR6)) != 0;
    ok &= (st.max_sev == 2);
    /* below Cr6 onset: nothing expected */
    gas_safety_eval(yag, 3, 100, &st);
    ok &= (st.expected_mask == 0);
    /* unexpected sensor rise at low temp -> leak/contamination */
    ok &= (gas_safety_crosscheck(&st, 1) == GAS_XC_UNEXPECTED);

    /* (2) fluoride charge at 250C -> HF (danger) + NH3 expected. */
    static const char *const flu[] = { "NH4F", "Y2O3" };
    gas_safety_eval(flu, 2, 250, &st);
    ok &= (st.expected_mask & (1u << GAS_HF))  != 0;
    ok &= (st.expected_mask & (1u << GAS_NH3)) != 0;
    ok &= (st.max_sev == 3);
    ok &= (st.top_gas == GAS_HF);
    /* danger expected but sensor flat -> sensor-fault flag */
    ok &= (gas_safety_crosscheck(&st, 0) == GAS_XC_SENSORFLAT);
    /* danger expected and sensor rises -> confirmed */
    ok &= (gas_safety_crosscheck(&st, 1) == GAS_XC_CONFIRMED);
    /* same charge cold (180C, below 190 NH3 onset) -> nothing */
    gas_safety_eval(flu, 2, 180, &st);
    ok &= (st.expected_mask == 0);

    /* (3) carbonate charge crossing calcination -> CO2 process note. */
    static const char *const carb[] = { "CaCO3", "Al2O3" };
    gas_safety_eval(carb, 2, 800, &st);
    ok &= (st.expected_mask & (1u << GAS_CO2)) != 0;
    ok &= (st.max_sev == 1);

    return ok;
}

#ifdef GAS_SAFETY_HOST_TEST
#include <stdio.h>
static void dump(const char *tag, const char *const *b, int n, int t)
{
    gas_status_t st;
    gas_safety_eval(b, n, t, &st);
    printf("  %-22s T=%4dC  mask=0x%02X sev=%d n=%d top=%-6s : %s\n",
           tag, t, st.expected_mask, st.max_sev, st.n_active,
           GAS_NAME[st.top_gas], st.top_reason ? st.top_reason : "(none)");
}
int main(void)
{
    static const char *const yag[] = { "Y2O3", "Al2O3", "Cr2O3" };
    static const char *const flu[] = { "NH4F", "Y2O3" };
    static const char *const carb[] = { "CaCO3", "Al2O3" };
    printf("gas_safety table: %u rules\n", (unsigned)GAS_RULES_N);
    dump("YAG:Cr @sinter",  yag, 3, 1500);
    dump("YAG:Cr @preheat", yag, 3, 100);
    dump("NH4F flux @250",  flu, 2, 250);
    dump("NH4F flux @180",  flu, 2, 180);
    dump("CaCO3 @calcine",  carb, 2, 800);
    printf("selftest = %s\n", gas_safety_selftest() ? "PASS" : "FAIL");
    return gas_safety_selftest() ? 0 : 1;
}
#endif
