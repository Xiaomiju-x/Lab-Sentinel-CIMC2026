/******************************************************************************
 * ai_llm_cluster.c  —  swap-load inference engine for the 7-expert edge LLM
 * cluster. Same decoder-GPT forward as ai_nanolm.c (exact GELU/softmax, KV
 * cache), but the weights are BOUND from a runtime blob (an expert loaded from
 * SPI flash into SDRAM) instead of compile-time arrays. One engine, 7 experts,
 * one bound at a time = the MCU mirror of a server-class BPU swap-load slot manager.
 *
 * Fidelity: erff() GELU + expf() softmax reproduce the deployed INT8 model;
 * cluster_selftest() verifies each expert token-for-token vs nlm_cluster_golden.h.
 ******************************************************************************/
#include "ai_llm_cluster.h"
#include "nlm_cluster_golden.h"
#include "nn_ops.h"
#include <math.h>
#include <string.h>
#include <stdint.h>
#ifndef NLM_HOST_TEST
#include "cimc_spiflash.h"
#include "nn_opt.h"
#endif

#define D   NLM_DMODEL
#define DH  NLM_DHEAD
#define NH  NLM_NHEAD
#define NL  NLM_NLAYER
#define FF  NLM_DFF
#define MS  NLM_MAXSEQ
#define V   NLM_CL_VOCAB
#define KVN (NL * MS * D)

/* ── bound expert: pointers into the SDRAM working blob ────────────────────── */
typedef struct {
    const signed char *tok_q; const float *tok_s; const float *pos;
    const float *ln1_g[NL], *ln1_b[NL], *ln2_g[NL], *ln2_b[NL];
    const signed char *qq[NL], *kq[NL], *vq[NL], *oq[NL], *f1q[NL], *f2q[NL];
    const float *qs[NL], *qb[NL], *ks[NL], *kb[NL], *vs[NL], *vb[NL], *os[NL], *ob[NL];
    const float *f1s[NL], *f1b[NL], *f2s[NL], *f2b[NL];
    const float *lnf_g, *lnf_b;
} expert_t;
static expert_t E;
static int s_cur = -1;

/* ── KV cache + scratch (SDRAM on MCU, arrays on host) ─────────────────────── */
#ifdef NLM_HOST_TEST
static float g_K[KVN], g_Vv[KVN];
static unsigned char g_blob[NLM_CL_BLOB_BYTES];
static const unsigned char *g_image = 0;
#define KBUF g_K
#define VBUF g_Vv
#else
#define CL_SDRAM_BLOB 0xC0700000U                 /* +7MB: 748KB working blob   */
#define CL_SDRAM_KV   0xC0800000U                 /* +8MB: ~288KB KV cache      */
#define KBUF ((float *)(CL_SDRAM_KV))
#define VBUF ((float *)(CL_SDRAM_KV + (uint32_t)(KVN * 4)))
#endif

static float s_x[D], s_h[D], s_q[D], s_k[D], s_v[D], s_y[D];
static float s_att[MS], s_ff[FF], s_log[V];

static void bind_blob(const unsigned char *b)
{
    int l;
    E.tok_q = (const signed char *)(b + NLM_CL_OFF_TOK_Q);
    E.tok_s = (const float *)(b + NLM_CL_OFF_TOK_S);
    E.pos   = (const float *)(b + NLM_CL_OFF_POS);
    for (l = 0; l < NL; l++) {
        const unsigned char *L = b + NLM_CL_OFF_LAYER0 + (uint32_t)l * NLM_CL_LAYER_STRIDE;
        E.ln1_g[l] = (const float *)(L + NLM_CL_LOFF_LN1_G);
        E.ln1_b[l] = (const float *)(L + NLM_CL_LOFF_LN1_B);
        E.qq[l] = (const signed char *)(L + NLM_CL_LOFF_Q_Q); E.qs[l] = (const float *)(L + NLM_CL_LOFF_Q_S); E.qb[l] = (const float *)(L + NLM_CL_LOFF_Q_B);
        E.kq[l] = (const signed char *)(L + NLM_CL_LOFF_K_Q); E.ks[l] = (const float *)(L + NLM_CL_LOFF_K_S); E.kb[l] = (const float *)(L + NLM_CL_LOFF_K_B);
        E.vq[l] = (const signed char *)(L + NLM_CL_LOFF_V_Q); E.vs[l] = (const float *)(L + NLM_CL_LOFF_V_S); E.vb[l] = (const float *)(L + NLM_CL_LOFF_V_B);
        E.oq[l] = (const signed char *)(L + NLM_CL_LOFF_O_Q); E.os[l] = (const float *)(L + NLM_CL_LOFF_O_S); E.ob[l] = (const float *)(L + NLM_CL_LOFF_O_B);
        E.ln2_g[l] = (const float *)(L + NLM_CL_LOFF_LN2_G);
        E.ln2_b[l] = (const float *)(L + NLM_CL_LOFF_LN2_B);
        E.f1q[l] = (const signed char *)(L + NLM_CL_LOFF_F1_Q); E.f1s[l] = (const float *)(L + NLM_CL_LOFF_F1_S); E.f1b[l] = (const float *)(L + NLM_CL_LOFF_F1_B);
        E.f2q[l] = (const signed char *)(L + NLM_CL_LOFF_F2_Q); E.f2s[l] = (const float *)(L + NLM_CL_LOFF_F2_S); E.f2b[l] = (const float *)(L + NLM_CL_LOFF_F2_B);
    }
    E.lnf_g = (const float *)(b + NLM_CL_OFF_LNF_G);
    E.lnf_b = (const float *)(b + NLM_CL_OFF_LNF_B);
}

#ifdef NLM_HOST_TEST
void cluster_set_host_image(const unsigned char *image) { g_image = image; s_cur = -1; }
#endif

int cluster_current(void) { return s_cur; }

int cluster_load_expert(int id)
{
    if (id < 0 || id >= NLM_CL_NEXPERT) return -1;
    if (id == s_cur) return 0;                       /* already bound */
#ifdef NLM_HOST_TEST
    if (!g_image) return -2;
    memcpy(g_blob, g_image + (uint32_t)id * NLM_CL_STRIDE, NLM_CL_BLOB_BYTES);
    bind_blob(g_blob);
#else
    /* stream the expert blob from SPI flash into SDRAM, then bind */
    if (cl_spiflash_read((unsigned char *)CL_SDRAM_BLOB,
                         NLM_CL_SPI_BASE + (uint32_t)id * NLM_CL_STRIDE,
                         NLM_CL_BLOB_BYTES) != 0) return -3;
    bind_blob((const unsigned char *)CL_SDRAM_BLOB);
#endif
    s_cur = id;
    return 0;
}

static float cl_gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

static void cl_softmax(float *a, int n)
{
    int i; float mx = a[0], sum = 0.0f;
    for (i = 1; i < n; i++) if (a[i] > mx) mx = a[i];
    for (i = 0; i < n; i++) { a[i] = expf(a[i] - mx); sum += a[i]; }
    if (sum <= 0.0f) sum = 1.0f;
    for (i = 0; i < n; i++) a[i] /= sum;
}

static void forward_token(int id, int pos)
{
    int l, h, s, j, d;
    const float scale = 1.0f / sqrtf((float)DH);
    const signed char *erow = E.tok_q + (long)id * D;

    for (d = 0; d < D; d++)
        s_x[d] = E.tok_s[id] * (float)erow[d] + E.pos[pos * D + d];

    for (l = 0; l < NL; l++) {
        float *Kp = KBUF + ((long)l * MS + pos) * D;
        float *Vp = VBUF + ((long)l * MS + pos) * D;

        nn_layernorm(s_x, E.ln1_g[l], E.ln1_b[l], s_h, D, 1e-5f);
        nn_linear_int8(s_h, E.qq[l], E.qs[l], E.qb[l], s_q, D, D);
        nn_linear_int8(s_h, E.kq[l], E.ks[l], E.kb[l], s_k, D, D);
        nn_linear_int8(s_h, E.vq[l], E.vs[l], E.vb[l], s_v, D, D);
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
            cl_softmax(s_att, pos + 1);
            for (j = 0; j < DH; j++) {
                float acc = 0.0f;
                for (s = 0; s <= pos; s++)
                    acc += s_att[s] * (VBUF + ((long)l * MS + s) * D + off)[j];
                s_y[off + j] = acc;
            }
        }
        nn_linear_int8(s_y, E.oq[l], E.os[l], E.ob[l], s_h, D, D);
        for (d = 0; d < D; d++) s_x[d] += s_h[d];

        nn_layernorm(s_x, E.ln2_g[l], E.ln2_b[l], s_h, D, 1e-5f);
        nn_linear_int8(s_h, E.f1q[l], E.f1s[l], E.f1b[l], s_ff, D, FF);
        for (j = 0; j < FF; j++) s_ff[j] = cl_gelu(s_ff[j]);
        nn_linear_int8(s_ff, E.f2q[l], E.f2s[l], E.f2b[l], s_h, FF, D);
        for (d = 0; d < D; d++) s_x[d] += s_h[d];
    }

    nn_layernorm(s_x, E.lnf_g, E.lnf_b, s_h, D, 1e-5f);
    for (s = 0; s < V; s++) {
        const signed char *row = E.tok_q + (long)s * D;
        float dot = 0.0f;
        for (d = 0; d < D; d++) dot += (float)row[d] * s_h[d];
        s_log[s] = E.tok_s[s] * dot;
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

int cluster_generate(const short *ctx12, char *out, int cap, float *conf)
{
    short ctx[NLM_NCTX + 2], gen[NLM_MAXSEQ];
    int n, i, p = 0;
    if (s_cur < 0 || !ctx12 || !out || cap < 4) { if (out && cap > 0) out[0] = 0; return 0; }
    ctx[0] = NLM_BOS;
    for (i = 0; i < NLM_NCTX; i++) ctx[i + 1] = ctx12[i];
    ctx[NLM_NCTX + 1] = NLM_SEP;
    n = gen_core(ctx, NLM_NCTX + 2, gen, 48, conf);
    for (i = 0; i < n; i++) {
        int id = gen[i], a = nlm_cl_tok_off[id], b = nlm_cl_tok_off[id + 1];
        for (; a < b && p < cap - 1; a++) out[p++] = (char)nlm_cl_tok_utf8[a];
    }
    out[p] = 0;
    return n;
}

int cluster_selftest(int *per_ok, float *max_err)
{
    short gen[NLM_MAXSEQ];
    int e, dmo, i, all = 1; float worst = 0.0f;

    for (e = 0; e < NLM_CL_NEXPERT; e++) {
        int ok = 1;
        if (cluster_load_expert(e) != 0) { if (per_ok) per_ok[e] = 0; all = 0; continue; }
        for (dmo = 0; dmo < NLM_CL_NDEMO; dmo++) {
            int exp_n = nlm_cl_gen_len[e][dmo] - NLM_CTXLEN;
            int n = gen_core(nlm_cl_demo_ctx[e][dmo], NLM_CTXLEN, gen, NLM_CL_MAXGEN, 0);
            if (n != exp_n) { ok = 0; continue; }
            for (i = 0; i < n; i++)
                if (gen[i] != nlm_cl_gen[e][dmo][NLM_CTXLEN + i]) { ok = 0; break; }
        }
        /* numeric: demo0 logits */
        for (i = 0; i < nlm_cl_golden_prefix_len[e]; i++)
            forward_token(nlm_cl_golden_prefix[e][i], i);
        for (i = 0; i < V; i++) {
            float d = s_log[i] - nlm_cl_golden_logits[e][i];
            if (d < 0.0f) d = -d;
            if (d > worst) worst = d;
        }
        if (per_ok) per_ok[e] = ok;
        if (!ok) all = 0;
    }
    if (max_err) *max_err = worst;
    return (all && worst < 1.0f) ? 1 : 0;
}
