/******************************************************************************
 * ai_nanolm.c  —  GD32 on-chip edge nano-LM inference engine.
 *
 * Decoder-only GPT, pre-LayerNorm, learned absolute positions, tied embeddings,
 * exact GELU. Weights are INT8 weight-only (per-output-row symmetric) + fp32
 * scales/bias/LN/pos (nanolm_weights.h). Autoregressive greedy decode with a
 * KV cache, so each new token is O(context), not O(context^2).
 *
 * Memory: the KV cache (~288 KB) lives in SDRAM (the on-chip 1 MB SRAM is for
 * stacks + LVGL + the 20 models); logits + per-step scratch are small .bss.
 * On the host golden test (-DNLM_HOST_TEST) the KV cache is a normal array.
 *
 * Fidelity: GELU uses erff() and attention softmax uses expf() (NOT the fast
 * approx) so the engine reproduces the deployed (INT8-dequant) torch model;
 * verified token-for-token by nanolm_selftest() against nanolm_golden.h.
 ******************************************************************************/
#include "ai_nanolm.h"
#include "lab_build_config.h"

#if !defined(LAB_LM_ENABLE) && !defined(NLM_HOST_TEST)
/* ── UI-dev FAST build: the generative LM is disabled (see lab_build_config.h).
 *    The 1.8 MB flagship INT8 weights below are NOT compiled in (image shrinks),
 *    and the two exported entry points are stubbed. Re-enable LAB_LM_ENABLE for
 *    the production build with the real on-chip generative GPT. ───────────────*/
int nanolm_generate(const short *ctx12, char *out, int cap, float *conf)
{
    (void)ctx12;
    if (conf) *conf = 0.0f;
    if (out && cap > 0) {
        static const char msg[] = "(LM disabled - UI dev build)";
        int i = 0;
        for (; msg[i] && i < cap - 1; i++) out[i] = msg[i];
        out[i] = '\0';
    }
    return 0;
}
int nanolm_selftest(float *logit_err) { if (logit_err) *logit_err = 0.0f; return 1; }
#else
#include "nanolm_weights.h"
#include "nanolm_golden.h"
#include "nn_ops.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#ifndef NLM_HOST_TEST
#include "nn_opt.h"            /* force -O3 -Otime on this TU */
#endif

#define D     NLM_DMODEL
#define DH    NLM_DHEAD
#define NH    NLM_NHEAD
#define NL    NLM_NLAYER
#define FF    NLM_DFF
#define MS    NLM_MAXSEQ
#define V     NLM_VOCAB
#define KVN   (NL * MS * D)    /* K (and V) element count */

/* ── KV cache: SDRAM on the MCU, plain arrays on the host ──────────────────── */
#ifdef NLM_HOST_TEST
static float g_K[KVN];
static float g_V[KVN];
#define KBUF g_K
#define VBUF g_V
#else
#define NLM_SDRAM_BASE 0xC0600000U      /* +6MB, above CAM_VIEW; ~26MB free above */
#define KBUF ((float *)(NLM_SDRAM_BASE))
#define VBUF ((float *)(NLM_SDRAM_BASE + (uint32_t)(KVN * 4)))
#endif

/* per-step scratch (small) */
static float s_x[D], s_h[D], s_q[D], s_k[D], s_v[D], s_y[D];
static float s_att[MS], s_ff[FF];
static float s_log[V];

static float nlm_gelu(float x)            /* exact erf GELU, matches torch.F.gelu */
{
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}

static void softmax_exact(float *a, int n)
{
    int i; float mx = a[0], sum = 0.0f;
    for (i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    for (i = 0; i < n; i++) { a[i] = expf(a[i] - mx); sum += a[i]; }
    if (sum <= 0.0f) sum = 1.0f;
    for (i = 0; i < n; i++) a[i] /= sum;
}

/* One transformer step: token `id` at sequence position `pos`. Updates KV and
 * writes next-token logits into s_log[V]. Causal attention over s=0..pos. */
static void forward_token(int id, int pos)
{
    int l, h, s, j, d;
    const float scale = 1.0f / sqrtf((float)DH);
    const signed char *erow = nlm_tok_q + (long)id * D;

    for (d = 0; d < D; d++)
        s_x[d] = nlm_tok_s[id] * (float)erow[d] + nlm_pos[pos * D + d];

    for (l = 0; l < NL; l++) {
        float *Kp = KBUF + ((long)l * MS + pos) * D;
        float *Vp = VBUF + ((long)l * MS + pos) * D;

        /* attention */
        nn_layernorm(s_x, nlm_ln1_g[l], nlm_ln1_b[l], s_h, D, 1e-5f);
        nn_linear_int8(s_h, nlm_q_q[l], nlm_q_s[l], nlm_q_b[l], s_q, D, D);
        nn_linear_int8(s_h, nlm_k_q[l], nlm_k_s[l], nlm_k_b[l], s_k, D, D);
        nn_linear_int8(s_h, nlm_v_q[l], nlm_v_s[l], nlm_v_b[l], s_v, D, D);
        memcpy(Kp, s_k, D * sizeof(float));
        memcpy(Vp, s_v, D * sizeof(float));

        for (h = 0; h < NH; h++) {
            int off = h * DH;
            for (s = 0; s <= pos; s++) {
                const float *Ks = KBUF + ((long)l * MS + s) * D + off;
                float dot = 0.0f;
                for (j = 0; j < DH; j++) dot += s_q[off + j] * Ks[j];
                s_att[s] = dot * scale;
            }
            softmax_exact(s_att, pos + 1);
            for (j = 0; j < DH; j++) {
                float acc = 0.0f;
                for (s = 0; s <= pos; s++)
                    acc += s_att[s] * (VBUF + ((long)l * MS + s) * D + off)[j];
                s_y[off + j] = acc;
            }
        }
        nn_linear_int8(s_y, nlm_o_q[l], nlm_o_s[l], nlm_o_b[l], s_h, D, D);
        for (d = 0; d < D; d++) s_x[d] += s_h[d];

        /* feed-forward */
        nn_layernorm(s_x, nlm_ln2_g[l], nlm_ln2_b[l], s_h, D, 1e-5f);
        nn_linear_int8(s_h, nlm_f1_q[l], nlm_f1_s[l], nlm_f1_b[l], s_ff, D, FF);
        for (j = 0; j < FF; j++) s_ff[j] = nlm_gelu(s_ff[j]);
        nn_linear_int8(s_ff, nlm_f2_q[l], nlm_f2_s[l], nlm_f2_b[l], s_h, FF, D);
        for (d = 0; d < D; d++) s_x[d] += s_h[d];
    }

    /* final norm + tied head (logits[v] = scale_tok[v] * (tok_q[v] . hf)) */
    nn_layernorm(s_x, nlm_lnf_g, nlm_lnf_b, s_h, D, 1e-5f);
    for (s = 0; s < V; s++) {
        const signed char *row = nlm_tok_q + (long)s * D;
        float dot = 0.0f;
        for (d = 0; d < D; d++) dot += (float)row[d] * s_h[d];
        s_log[s] = nlm_tok_s[s] * dot;
    }
}

/* top-1 probability of the current s_log (cheap confidence proxy). */
static float top1_prob(void)
{
    int i; float mx = s_log[0], sum = 0.0f;
    for (i = 1; i < V; i++) if (s_log[i] > mx) mx = s_log[i];
    for (i = 0; i < V; i++) sum += expf(s_log[i] - mx);
    return (sum > 0.0f) ? (1.0f / sum) : 0.0f;   /* exp(mx-mx)=1 over sum */
}

/* Greedy decode from a full context (incl <bos>..<sep>). Fills gen[] with the
 * generated token ids (after the context), returns the count. */
static int gen_core(const short *ctx, int ctxlen, short *gen, int max_new, float *conf)
{
    int pos, k = 0, nxt;
    float csum = 0.0f;

    for (pos = 0; pos < ctxlen; pos++)
        forward_token(ctx[pos], pos);      /* prefill; last fills s_log */

    pos = ctxlen;
    while (k < max_new && pos < MS) {
        nxt = nn_argmax(s_log, V);
        if (nxt == NLM_EOS) break;
        if (conf) csum += top1_prob();
        gen[k++] = (short)nxt;
        forward_token(nxt, pos);
        pos++;
    }
    if (conf) *conf = (k > 0) ? (csum / (float)k) : 0.0f;
    return k;
}

int nanolm_generate(const short *ctx12, char *out, int cap, float *conf)
{
    short ctx[NLM_NCTX + 2];
    short gen[NLM_MAXSEQ];
    int n, i, p = 0;

    if (!ctx12 || !out || cap < 4) { if (out && cap > 0) out[0] = 0; return 0; }

    ctx[0] = NLM_BOS;
    for (i = 0; i < NLM_NCTX; i++) ctx[i + 1] = ctx12[i];
    ctx[NLM_NCTX + 1] = NLM_SEP;

    n = gen_core(ctx, NLM_NCTX + 2, gen, 48, conf);

    /* detokenize: id -> UTF-8 bytes via the offset table */
    for (i = 0; i < n; i++) {
        int id = gen[i];
        int a = nlm_tok_off[id], b = nlm_tok_off[id + 1];
        for (; a < b && p < cap - 1; a++) out[p++] = (char)nlm_tok_utf8[a];
    }
    out[p] = 0;
    return n;
}

int nanolm_selftest(float *logit_err)
{
    short gen[NLM_MAXSEQ];
    int d, i, ok = 1, exp_n, n;
    float err = 0.0f;

    for (d = 0; d < NLM_NDEMO; d++) {
        n = gen_core(nlm_demo_ctx[d], NLM_CTXLEN, gen, NLM_MAXGEN, 0);
        exp_n = nlm_demo_gen_len[d] - NLM_CTXLEN;
        if (n != exp_n) { ok = 0; continue; }
        for (i = 0; i < n; i++)
            if (gen[i] != nlm_demo_gen[d][NLM_CTXLEN + i]) { ok = 0; break; }
    }

    /* numeric: last-position logits over demo0's full greedy prefix */
    for (i = 0; i < NLM_GOLDEN_PREFIX_LEN; i++)
        forward_token(nlm_golden_prefix[i], i);
    for (i = 0; i < V; i++) {
        float e = s_log[i] - nlm_golden_logits[i];
        if (e < 0.0f) e = -e;
        if (e > err) err = e;
    }
    if (logit_err) *logit_err = err;
    return (ok && err < 1.0f) ? 1 : 0;
}
#endif /* LAB_LM_ENABLE || NLM_HOST_TEST */
