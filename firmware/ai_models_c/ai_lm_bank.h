/* ai_lm_bank.h — SPI-flash swap-load bank of smaller flagship LM sizes.
 * One variable-dim decoder-GPT engine; one bank model bound at a time, loaded
 * from SPI flash into SDRAM. Shares nanolm_vocab.h (V/detok/control) with the
 * internal x1p9. The runtime "active LM" can be x1p9 (internal, ai_nanolm) OR a
 * bank model (here) — the integrator picks. */
#ifndef AI_LM_BANK_H
#define AI_LM_BANK_H

/* Load bank model `idx` (0..BANK_N-1) from SPI flash into SDRAM and bind it.
 * Returns 0 ok, <0 on error. No-op (0) if already bound. */
int bank_load(int idx);

/* Currently bound bank model idx, or -1 if none. */
int bank_current(void);

/* Greedy-generate a Chinese diagnosis from the 12 control-token ids into out
 * (UTF-8, NUL-terminated). Returns #generated tokens; fills *conf (mean top-1).
 * Requires a prior successful bank_load(). */
int bank_generate(const short *ctx12, char *out, int cap, float *conf);

/* Verify every bank model reproduces its deployed INT8 golden token-for-token.
 * per_ok[BANK_N] gets per-model pass; *max_err gets worst logit |err|.
 * Returns 1 iff all pass and max_err < 1.0. */
int bank_selftest(int *per_ok, float *max_err);

/* ── runtime LM roster accessors (decouple lab_sentinel/HMI from nlm_bank.h's
 *    big golden arrays). Roster index 0 = internal x1p9, 1..N = SPI bank models;
 *    a roster index r maps to bank model (r-1) for bank_load(). ───────────────── */
int         lm_roster_count(void);          /* total switchable LMs (= 1 + BANK_N) */
const char *lm_roster_label_s(int r);       /* e.g. "1.8M" / "1.26M" / "0.6M"      */
const char *lm_roster_tag_s(int r);         /* e.g. "x1p9" / "m1p35" / "s0p6"      */
int         lm_roster_ppl_x100(int r);      /* val ppl x100 (256 = 2.56)           */
int         lm_roster_lat_x10(int r);       /* est per-token latency x vs 0.6M, x10 */

#ifdef NLM_HOST_TEST
void bank_set_host_image(const unsigned char *image);   /* host golden: in-RAM image */
#endif

#endif
