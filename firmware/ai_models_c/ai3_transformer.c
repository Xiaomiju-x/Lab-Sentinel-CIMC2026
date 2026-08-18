/******************************************************************************
 * ai3_transformer.c  —  AI-3 TinyTransformer forward pass.
 * Reproduces train_transformer.py TinyTransformer exactly (pre-norm, manual
 * 4-head attention, mean-pool). Weights from ai3_transformer_weights.h.
 *
 * ALL activation + temp buffers live in a single static .bss struct (~98 KB in
 * AXI-SRAM @0x24000000), so the engine uses almost no call stack — safe to call
 * from a small task stack. (History: a 1.5 KB-stack version overflowed
 * task_init's 1 KB stack -> HardFault -> reboot; a later version parked the
 * struct in SDRAM @0xC0400000 which was stack-safe but ~100x slower per access
 * over EXMC -> AI-3 forward took 1.18 s. Static AXI-SRAM .bss is both
 * stack-safe AND fast, with no D-cache/TLI-framebuffer coherency hazard that an
 * MPU-cacheable SDRAM region would create.)
 ******************************************************************************/
#include "ai3_transformer.h"
#include "nn_ops.h"
#include "ai3_transformer_weights.h"
#include <math.h>
#include "nn_opt.h"    /* force -O3 -Otime on this TU (project default is -O0) */

#define D   AI3_DMODEL          /* 64 */
#define T   AI3_SEQ             /* 64 */
#define H   AI3_NHEAD           /* 4  */
#define DH  AI3_DHEAD           /* 16 */
#define FF  AI3_FF              /* 128 */
#define LN_EPS 1e-5f

/* ---- on-chip section profiler (firmware only). Reads DWT->CYCCNT (0xE0001004)
 * directly so this TU needs no device header and still compiles on the host,
 * where AI3_CYC is 0, the counters stay 0, and the golden self-test is
 * bit-identical. CYCCNT is enabled by the boot latency probe before this runs.
 * Counters are summed over both blocks per forward; lab_sentinel.c prints them. */
#if defined(__ARMCC_VERSION)
#define AI3_CYC (*(volatile unsigned int *)0xE0001004u)
#else
#define AI3_CYC 0u
#endif
unsigned int ai3_prof_proj, ai3_prof_ln, ai3_prof_qkv, ai3_prof_attn,
             ai3_prof_op, ai3_prof_ffn, ai3_prof_final;

/* ---- attention saliency (explainability) ----
 * Mean attention weight RECEIVED by each of the 64 timesteps in the LAST block,
 * averaged over all heads and query positions. A normalised saliency over the
 * temperature window: "which minutes of the curve the classifier focused on".
 * Filled per forward; read with ai3_get_attention() for the HMI / report. */
static float s_attn[T];

/* Activation + scratch in static .bss (AXI-SRAM, not the stack, not SDRAM). */
typedef struct {
    float x[T * D];     /* running representation                 */
    float nrm[T * D];   /* pre-norm output (reused norm1/norm2)   */
    float q[T * D];
    float k[T * D];
    float v[T * D];
    float ao[T * D];    /* multi-head attention output (pre out_proj) */
    float ffb[T * FF];  /* batched FFN hidden [T][FF] (weight-stationary GEMM) */
    float tmpD[D];      /* per-row linear output scratch (legacy, unused)  */
    float ffh[FF];      /* FFN hidden (legacy, unused)             */
    float scores[T];    /* per-(head,query) attention scores       */
    float pooled[D];    /* mean-pooled representation              */
    float rown[D];      /* per-row final-norm output               */
} ai3_scratch_t;

static ai3_scratch_t      s_ai3_scratch;          /* ~130 KB in .bss / AXI-SRAM */
static ai3_scratch_t *const S = &s_ai3_scratch;

/* one transformer block, in place on S->x */
static void ai3_block(int b)
{
    const float *n1w, *n1b, *qw, *kw, *vw, *ow, *ob, *n2w, *n2b, *ff0w, *ff0b, *ff1w, *ff1b;
    int i, j, h, d;
    unsigned int _t;

    if (b == 0) {
        n1w = ai3_b0_n1_w; n1b = ai3_b0_n1_b;
        qw = ai3_b0_q_w; kw = ai3_b0_k_w; vw = ai3_b0_v_w; ow = ai3_b0_o_w; ob = ai3_b0_o_b;
        n2w = ai3_b0_n2_w; n2b = ai3_b0_n2_b;
        ff0w = ai3_b0_ff0_w; ff0b = ai3_b0_ff0_b; ff1w = ai3_b0_ff1_w; ff1b = ai3_b0_ff1_b;
    } else {
        n1w = ai3_b1_n1_w; n1b = ai3_b1_n1_b;
        qw = ai3_b1_q_w; kw = ai3_b1_k_w; vw = ai3_b1_v_w; ow = ai3_b1_o_w; ob = ai3_b1_o_b;
        n2w = ai3_b1_n2_w; n2b = ai3_b1_n2_b;
        ff0w = ai3_b1_ff0_w; ff0b = ai3_b1_ff0_b; ff1w = ai3_b1_ff1_w; ff1b = ai3_b1_ff1_b;
    }

    /* ---- attention sublayer (pre-norm) ---- */
    /* n1 = LayerNorm(x) row-wise */
    _t = AI3_CYC;
    for (i = 0; i < T; i++)
        nn_layernorm(&S->x[i * D], n1w, n1b, &S->nrm[i * D], D, LN_EPS);
    ai3_prof_ln += AI3_CYC - _t;

    /* q,k,v = Linear(n1)  (q/k/v have no bias). Batched (weight-stationary):
     * each D*D weight matrix is read once and reused across all T rows, instead
     * of being re-streamed from Flash 64x by a per-token loop. */
    _t = AI3_CYC;
    nn_linear_batch(S->nrm, qw, 0, S->q, T, D, D);
    nn_linear_batch(S->nrm, kw, 0, S->k, T, D, D);
    nn_linear_batch(S->nrm, vw, 0, S->v, T, D, D);
    ai3_prof_qkv += AI3_CYC - _t;

    /* per head, per query row: scores -> softmax -> weighted sum of v */
    _t = AI3_CYC;
    {
        const float scale = 1.0f / sqrtf((float)DH);
        for (h = 0; h < H; h++) {
            int off = h * DH;
            for (i = 0; i < T; i++) {
                const float *qi = &S->q[i * D + off];
                for (j = 0; j < T; j++) {
                    const float *kj = &S->k[j * D + off];
                    float s = 0.0f;
                    for (d = 0; d < DH; d++) s += qi[d] * kj[d];
                    S->scores[j] = s * scale;
                }
                nn_softmax(S->scores, T);
                /* attention saliency: accumulate the weight each key j receives,
                 * but only in the last block (closest to the decision). */
                if (b == AI3_NLAYER - 1) {
                    int jj;
                    for (jj = 0; jj < T; jj++) s_attn[jj] += S->scores[jj];
                }
                /* ao[i, off:off+DH] = Σ_j scores[j] * v[j, off:off+DH] */
                {
                    float *aoi = &S->ao[i * D + off];
                    for (d = 0; d < DH; d++) aoi[d] = 0.0f;
                    for (j = 0; j < T; j++) {
                        const float a = S->scores[j];
                        const float *vj = &S->v[j * D + off];
                        for (d = 0; d < DH; d++) aoi[d] += a * vj[d];
                    }
                }
            }
        }
    }
    ai3_prof_attn += AI3_CYC - _t;

    /* out_proj + residual (batched): x += out_proj(ao). nrm is reused as the
     * T*D temp (its norm1 contents are dead now). */
    _t = AI3_CYC;
    nn_linear_batch(S->ao, ow, ob, S->nrm, T, D, D);
    for (i = 0; i < T * D; i++) S->x[i] += S->nrm[i];
    ai3_prof_op += AI3_CYC - _t;

    /* ---- feed-forward sublayer (pre-norm), batched ----
     * n2 LayerNorm stays per-row (cheap: ~0.4 ms total); the two big matmuls
     * go weight-stationary. ff1 writes into ao (free after out_proj). */
    _t = AI3_CYC;
    for (i = 0; i < T; i++)                                               /* n2 */
        nn_layernorm(&S->x[i * D], n2w, n2b, &S->nrm[i * D], D, LN_EPS);
    nn_linear_batch(S->nrm, ff0w, ff0b, S->ffb, T, D, FF);               /* 64->128 */
    nn_relu(S->ffb, T * FF);
    nn_linear_batch(S->ffb, ff1w, ff1b, S->ao, T, FF, D);                /* 128->64 */
    for (i = 0; i < T * D; i++) S->x[i] += S->ao[i];
    ai3_prof_ffn += AI3_CYC - _t;
}

void ai3_forward_norm(const float *seqn, float *logits5)
{
    int i, d;
    unsigned int _t;

    ai3_prof_proj = ai3_prof_ln = ai3_prof_qkv = ai3_prof_attn =
        ai3_prof_op = ai3_prof_ffn = ai3_prof_final = 0u;
    for (i = 0; i < T; i++) s_attn[i] = 0.0f;   /* reset attention saliency */

    /* x = proj(seq) + PE */
    _t = AI3_CYC;
    for (i = 0; i < T; i++) {
        nn_linear(&seqn[i * AI3_N_FEAT], ai3_proj_w, ai3_proj_b, &S->x[i * D], AI3_N_FEAT, D);
        for (d = 0; d < D; d++) S->x[i * D + d] += ai3_pe[i * D + d];
    }
    ai3_prof_proj += AI3_CYC - _t;

    ai3_block(0);
    ai3_block(1);

    /* final LayerNorm row-wise then mean-pool over time */
    _t = AI3_CYC;
    for (d = 0; d < D; d++) S->pooled[d] = 0.0f;
    for (i = 0; i < T; i++) {
        nn_layernorm(&S->x[i * D], ai3_normf_w, ai3_normf_b, S->rown, D, LN_EPS);
        for (d = 0; d < D; d++) S->pooled[d] += S->rown[d];
    }
    for (d = 0; d < D; d++) S->pooled[d] /= (float)T;

    /* head: 64 -> 5 */
    nn_linear(S->pooled, ai3_head_w, ai3_head_b, logits5, D, AI3_N_CLASS);
    ai3_prof_final += AI3_CYC - _t;

    /* normalise attention saliency to a distribution over the 64 timesteps
     * (each of the H*T softmax rows sums to 1, so the accumulator sums to H*T). */
    for (i = 0; i < T; i++) s_attn[i] /= (float)(H * T);
}

void ai3_get_attention(float *out_seq)
{
    int i;
    if (out_seq == 0) return;
    for (i = 0; i < T; i++) out_seq[i] = s_attn[i];
}

int ai3_classify(const float *seq_raw, float *probs5)
{
    static float seqn[T * AI3_N_FEAT];     /* normalised copy (static: 512 floats) */
    int i, f;
    for (i = 0; i < T; i++) {
        for (f = 0; f < AI3_N_FEAT; f++) {
            float s = ai3_std[f];
            seqn[i * AI3_N_FEAT + f] = (s > 1e-9f)
                ? (seq_raw[i * AI3_N_FEAT + f] - ai3_mu[f]) / s : 0.0f;
        }
    }
    ai3_forward_norm(seqn, probs5);
    nn_softmax(probs5, AI3_N_CLASS);
    return nn_argmax(probs5, AI3_N_CLASS);
}
