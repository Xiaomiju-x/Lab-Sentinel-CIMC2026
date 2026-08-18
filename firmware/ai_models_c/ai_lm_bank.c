/******************************************************************************
 * ai_lm_bank.c  —  variable-dim swap-load engine for the flagship LM size bank.
 *
 * Same decoder-GPT forward as ai_nanolm.c / ai_llm_cluster.c (exact erff GELU +
 * expf softmax + KV cache), but the model DIMENSIONS are bound at runtime from
 * g_bank[idx] (nlm_bank.h) instead of compile-time macros — because the bank
 * holds DIFFERENT sizes (s0p6 d128/3L, m1p35 d160/4L). Static scratch + KV are
 * sized to the bank maxima (BANK_DMAX/FFMAX/NLMAX/MSMAX); each load binds the
 * active model's weight pointers into the SDRAM working blob.
 *
 * Vocab (V, detok table, control ids, BOS/SEP/EOS) is shared with the internal
 * x1p9 via nanolm_vocab.h — the 3 sizes were trained on the same corpus_v2, so
 * the token space is identical (asserted at export).
 *
 * Fidelity: reproduces the deployed INT8 model token-for-token; bank_selftest()
 * checks each model vs nlm_bank.h golden.
 ******************************************************************************/
#include "ai_lm_bank.h"
#include "nlm_bank.h"          /* g_bank, BANK_*, + nanolm_vocab.h (V/detok/control) */
#include "nn_ops.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#ifndef NLM_HOST_TEST
#include "cimc_spiflash.h"
#include "nn_opt.h"
#endif

#define V    NLM_VOCAB            /* shared with x1p9 (corpus_v2 vocab) */
#define DX   BANK_DMAX
#define FFX  BANK_FFMAX
#define NLX  BANK_NLMAX
#define MSX  BANK_MSMAX
#define KVN  (NLX * MSX * DX)     /* max KV element count */

/* ── bound model: runtime dims + pointers into the SDRAM working blob ───────── */
typedef struct {
    int d, dh, nh, nl, ff, ms;
    const signed char *tok_q; const float *tok_s; const float *pos;
    const float *ln1_g[NLX], *ln1_b[NLX], *ln2_g[NLX], *ln2_b[NLX];
    const signed char *qq[NLX], *kq[NLX], *vq[NLX], *oq[NLX], *f1q[NLX], *f2q[NLX];
    const float *qs[NLX], *qb[NLX], *ks[NLX], *kb[NLX], *vs[NLX], *vb[NLX], *os[NLX], *ob[NLX];
    const float *f1s[NLX], *f1b[NLX], *f2s[NLX], *f2b[NLX];
    const float *lnf_g, *lnf_b;
} bound_t;
static bound_t B;
static int s_cur = -1;

/* ── KV cache + working blob (SDRAM on MCU, arrays on host) ─────────────────── */
#ifdef NLM_HOST_TEST
static float g_K[KVN], g_Vv[KVN];
static unsigned char g_blob[BANK_MAX_BLOB];   /* largest single model blob */
static const unsigned char *g_image = 0;
#define KBUF g_K
#define VBUF g_Vv
#else
/* BANK_PROV_BASE comes from nlm_bank_prov.h (via nlm_bank.h) = single source of truth */
#define BK_SDRAM_BLOB 0xC0900000U             /* +9MB: working blob (active model, <=1.5MB)            */
#define BK_SDRAM_KV   0xC0B00000U             /* +11MB: KV cache (<=655KB)                              */
#define KBUF ((float *)(BK_SDRAM_KV))
#define VBUF ((float *)(BK_SDRAM_KV + (uint32_t)(KVN * 4)))
#endif

static float s_x[DX], s_h[DX], s_q[DX], s_k[DX], s_v[DX], s_y[DX];
static float s_att[MSX], s_ff[FFX], s_log[V];

static void bind(const unsigned char *b, const bank_model_t *m)
{
    int l;
    B.d = m->d; B.dh = m->dh; B.nh = m->nh; B.nl = m->nl; B.ff = m->ff; B.ms = m->ms;
    B.tok_q = (const signed char *)(b + m->off_tok_q);
    B.tok_s = (const float *)(b + m->off_tok_s);
    B.pos   = (const float *)(b + m->off_pos);
    for (l = 0; l < m->nl; l++) {
        const unsigned char *L = b + m->off_layer0 + (uint32_t)l * m->layer_stride;
        B.ln1_g[l] = (const float *)(L + m->lo_ln1g); B.ln1_b[l] = (const float *)(L + m->lo_ln1b);
        B.qq[l] = (const signed char *)(L + m->lo_qq); B.qs[l] = (const float *)(L + m->lo_qs); B.qb[l] = (const float *)(L + m->lo_qb);
        B.kq[l] = (const signed char *)(L + m->lo_kq); B.ks[l] = (const float *)(L + m->lo_ks); B.kb[l] = (const float *)(L + m->lo_kb);
        B.vq[l] = (const signed char *)(L + m->lo_vq); B.vs[l] = (const float *)(L + m->lo_vs); B.vb[l] = (const float *)(L + m->lo_vb);
        B.oq[l] = (const signed char *)(L + m->lo_oq); B.os[l] = (const float *)(L + m->lo_os); B.ob[l] = (const float *)(L + m->lo_ob);
        B.ln2_g[l] = (const float *)(L + m->lo_ln2g); B.ln2_b[l] = (const float *)(L + m->lo_ln2b);
        B.f1q[l] = (const signed char *)(L + m->lo_f1q); B.f1s[l] = (const float *)(L + m->lo_f1s); B.f1b[l] = (const float *)(L + m->lo_f1b);
        B.f2q[l] = (const signed char *)(L + m->lo_f2q); B.f2s[l] = (const float *)(L + m->lo_f2s); B.f2b[l] = (const float *)(L + m->lo_f2b);
    }
    B.lnf_g = (const float *)(b + m->off_lnf_g);
    B.lnf_b = (const float *)(b + m->off_lnf_b);
}

int bank_current(void) { return s_cur; }

int bank_load(int idx)
{
    const bank_model_t *m;
    if (idx < 0 || idx >= BANK_N) return -1;
    if (idx == s_cur) return 0;
    m = &g_bank[idx];
#ifdef NLM_HOST_TEST
    if (!g_image) return -2;
    memcpy(g_blob, g_image + m->spi_off, m->blob_bytes);
    bind(g_blob, m);
#else
    if (cl_spiflash_read((unsigned char *)BK_SDRAM_BLOB,
                         BANK_PROV_BASE + m->spi_off, m->blob_bytes) != 0) return -3;
    bind((const unsigned char *)BK_SDRAM_BLOB, m);
#endif
    s_cur = idx;
    return 0;
}

static float bk_gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

static void bk_softmax(float *a, int n)
{
    int i; float mx = a[0], sum = 0.0f;
    for (i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    for (i = 0; i < n; i++) { a[i] = expf(a[i] - mx); sum += a[i]; }
    if (sum <= 0.0f) sum = 1.0f;
    for (i = 0; i < n; i++) a[i] /= sum;
}

static void forward_token(int id, int pos)
{
    const int d = B.d, dh = B.dh, nh = B.nh, nl = B.nl, ff = B.ff, ms = B.ms;
    int l, h, s, j, e;
    const float scale = 1.0f / sqrtf((float)dh);
    const signed char *erow = B.tok_q + (long)id * d;

    for (e = 0; e < d; e++)
        s_x[e] = B.tok_s[id] * (float)erow[e] + B.pos[pos * d + e];

    for (l = 0; l < nl; l++) {
        float *Kp = KBUF + ((long)l * ms + pos) * d;
        float *Vp = VBUF + ((long)l * ms + pos) * d;

        nn_layernorm(s_x, B.ln1_g[l], B.ln1_b[l], s_h, d, 1e-5f);
        nn_linear_int8(s_h, B.qq[l], B.qs[l], B.qb[l], s_q, d, d);
        nn_linear_int8(s_h, B.kq[l], B.ks[l], B.kb[l], s_k, d, d);
        nn_linear_int8(s_h, B.vq[l], B.vs[l], B.vb[l], s_v, d, d);
        memcpy(Kp, s_k, d * sizeof(float));
        memcpy(Vp, s_v, d * sizeof(float));

        for (h = 0; h < nh; h++) {
            int off = h * dh;
            for (s = 0; s <= pos; s++) {
                const float *Ks = KBUF + ((long)l * ms + s) * d + off;
                float dot = 0.0f;
                for (j = 0; j < dh; j++) dot += s_q[off + j] * Ks[j];
                s_att[s] = dot * scale;
            }
            bk_softmax(s_att, pos + 1);
            for (j = 0; j < dh; j++) {
                float acc = 0.0f;
                for (s = 0; s <= pos; s++)
                    acc += s_att[s] * (VBUF + ((long)l * ms + s) * d + off)[j];
                s_y[off + j] = acc;
            }
        }
        nn_linear_int8(s_y, B.oq[l], B.os[l], B.ob[l], s_h, d, d);
        for (e = 0; e < d; e++) s_x[e] += s_h[e];

        nn_layernorm(s_x, B.ln2_g[l], B.ln2_b[l], s_h, d, 1e-5f);
        nn_linear_int8(s_h, B.f1q[l], B.f1s[l], B.f1b[l], s_ff, d, ff);
        for (j = 0; j < ff; j++) s_ff[j] = bk_gelu(s_ff[j]);
        nn_linear_int8(s_ff, B.f2q[l], B.f2s[l], B.f2b[l], s_h, ff, d);
        for (e = 0; e < d; e++) s_x[e] += s_h[e];
    }

    nn_layernorm(s_x, B.lnf_g, B.lnf_b, s_h, d, 1e-5f);
    for (s = 0; s < V; s++) {
        const signed char *row = B.tok_q + (long)s * d;
        float dot = 0.0f;
        for (j = 0; j < d; j++) dot += (float)row[j] * s_h[j];
        s_log[s] = B.tok_s[s] * dot;
    }
}

static float top1_prob(void)
{
    int i; float mx = s_log[0], sum = 0.0f;
    for (i = 1; i < V; i++) if (s_log[i] > mx) mx = s_log[i];
    for (i = 0; i < V; i++) sum += expf(s_log[i] - mx);
    return (sum > 0.0f) ? (1.0f / sum) : 0.0f;
}

static int gen_core(const short *ctx, int ctxlen, short *gen, int max_new, float *conf)
{
    int pos, k = 0, nxt; float csum = 0.0f;
    for (pos = 0; pos < ctxlen; pos++) forward_token(ctx[pos], pos);
    pos = ctxlen;
    while (k < max_new && pos < B.ms) {
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

int bank_generate(const short *ctx12, char *out, int cap, float *conf)
{
    short ctx[NLM_NCTX + 2], gen[BANK_MSMAX];
    int n, i, p = 0;
    if (s_cur < 0 || !ctx12 || !out || cap < 4) { if (out && cap > 0) out[0] = 0; return 0; }
    ctx[0] = NLM_BOS;
    for (i = 0; i < NLM_NCTX; i++) ctx[i + 1] = ctx12[i];
    ctx[NLM_NCTX + 1] = NLM_SEP;
    n = gen_core(ctx, NLM_NCTX + 2, gen, 48, conf);
    for (i = 0; i < n; i++) {
        int id = gen[i], a = nlm_tok_off[id], b = nlm_tok_off[id + 1];
        for (; a < b && p < cap - 1; a++) out[p++] = (char)nlm_tok_utf8[a];
    }
    out[p] = 0;
    return n;
}

int bank_selftest(int *per_ok, float *max_err)
{
    short gen[BANK_MSMAX];
    int e, dmo, i, all = 1; float worst = 0.0f;

    for (e = 0; e < BANK_N; e++) {
        int ok = 1;
        if (bank_load(e) != 0) { if (per_ok) per_ok[e] = 0; all = 0; continue; }
        for (dmo = 0; dmo < 3; dmo++) {
            int exp_n = bank_gen_len[e][dmo] - BANK_CTXLEN;
            int n = gen_core(bank_demo_ctx[e][dmo], BANK_CTXLEN, gen, BANK_MAXGEN, 0);
            if (n != exp_n) { ok = 0; continue; }
            for (i = 0; i < n; i++)
                if (gen[i] != bank_gen[e][dmo][BANK_CTXLEN + i]) { ok = 0; break; }
        }
        for (i = 0; i < bank_golden_prefix_len[e]; i++)
            forward_token(bank_golden_prefix[e][i], i);
        for (i = 0; i < V; i++) {
            float dd = s_log[i] - bank_golden_logits[e][i];
            if (dd < 0.0f) dd = -dd;
            if (dd > worst) worst = dd;
        }
        if (per_ok) per_ok[e] = ok;
        if (!ok) all = 0;
    }
    if (max_err) *max_err = worst;
    return (all && worst < 1.0f) ? 1 : 0;
}

/* ── roster accessors (keep nlm_bank.h's golden arrays private to this TU) ──── */
int lm_roster_count(void) { return LM_ROSTER_N; }
const char *lm_roster_label_s(int r) { return (r >= 0 && r < LM_ROSTER_N) ? lm_roster_lab[r] : "?"; }
const char *lm_roster_tag_s(int r)   { return (r >= 0 && r < LM_ROSTER_N) ? lm_roster_tag[r] : "?"; }
int lm_roster_ppl_x100(int r) { return (r >= 0 && r < LM_ROSTER_N) ? lm_roster_pplx100[r] : 0; }
int lm_roster_lat_x10(int r)  { return (r >= 0 && r < LM_ROSTER_N) ? lm_roster_latx10[r] : 0; }

#ifdef NLM_HOST_TEST
void bank_set_host_image(const unsigned char *image) { g_image = image; s_cur = -1; }
#endif
