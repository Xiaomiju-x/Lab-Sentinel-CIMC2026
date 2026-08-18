/******************************************************************************
 * nn_ops.c  —  CIMC Lab-Sentinel float32 neural-net kernels
 * See nn_ops.h for the API contract. Kept deliberately simple and readable:
 * these models are small, the M7 FPU is fast, correctness > micro-tuning.
 ******************************************************************************/
#include "nn_ops.h"
#include <math.h>
#include "nn_opt.h"    /* force -O3 -Otime on this TU (project default is -O0) */

void nn_linear(const float *x, const float *W, const float *b,
               float *y, int in_dim, int out_dim)
{
    int o, i;
    for (o = 0; o < out_dim; o++) {
        const float *wrow = W + (long)o * in_dim;
        float acc = (b != 0) ? b[o] : 0.0f;
        for (i = 0; i < in_dim; i++) {
            acc += wrow[i] * x[i];
        }
        y[o] = acc;
    }
}

void nn_linear_nobias(const float *x, const float *W,
                      float *y, int in_dim, int out_dim)
{
    nn_linear(x, W, 0, y, in_dim, out_dim);
}

/* Batched (weight-stationary) linear: Y[rows][out_dim] = X[rows][in_dim] . W^T + b,
 * with W laid out [out_dim][in_dim] (same as nn_linear). Loop order is out_dim-OUTER
 * so each weight row W[o][*] is fetched from Flash ONCE and reused across all `rows`,
 * instead of being re-streamed once per row (which a loop of per-row nn_linear() does).
 * On AI-3 (rows=T=64) this cuts weight Flash traffic 64x: the difference between
 * D-cache thrash (~70-94 cyc/MAC measured on the 48-64 KB weight matrices) and
 * cache-resident throughput. Numerically identical to calling nn_linear() per row
 * (each output's MAC accumulation order is unchanged). */
void nn_linear_batch(const float *X, const float *W, const float *b,
                     float *Y, int rows, int in_dim, int out_dim)
{
    int o, r, i;
    for (o = 0; o < out_dim; o++) {
        const float *wrow = W + (long)o * in_dim;
        const float  bo   = (b != 0) ? b[o] : 0.0f;
        for (r = 0; r < rows; r++) {
            const float *xr = X + (long)r * in_dim;
            float acc = bo;
            for (i = 0; i < in_dim; i++) acc += wrow[i] * xr[i];
            Y[(long)r * out_dim + o] = acc;
        }
    }
}

/* Weight-only INT8 linear (B1). W[o,i] ~= scale[o] * Wq[o,i], Wq in [-127,127]
 * (per-output-channel symmetric PTQ). y[o] = b[o] + scale[o] * Σ Wq[o,i]*x[i].
 * Honest scheme for Cortex-M7: the M7 FPU is fast but has NO int8 SIMD (Helium is
 * M55/M85), so compute stays float — INT8 here is a 4x FLASH-SIZE win + a weight
 * MEMORY-TRAFFIC win (eases the D-cache thrash on big matmuls, cf. AI-3). Wq is
 * promoted to float for the MAC; one dequant multiply per output. Numerically a
 * close approximation of nn_linear (per-output max-abs quantisation). */
void nn_linear_int8(const float *x, const signed char *Wq, const float *scale,
                    const float *b, float *y, int in_dim, int out_dim)
{
    int o, i;
    for (o = 0; o < out_dim; o++) {
        const signed char *wrow = Wq + (long)o * in_dim;
        float acc = 0.0f;
        for (i = 0; i < in_dim; i++) acc += (float)wrow[i] * x[i];
        y[o] = ((b != 0) ? b[o] : 0.0f) + scale[o] * acc;
    }
}

void nn_relu(float *x, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        if (x[i] < 0.0f) x[i] = 0.0f;
    }
}

void nn_layernorm(const float *x, const float *g, const float *b,
                  float *y, int n, float eps)
{
    int i;
    float mean = 0.0f, var = 0.0f, inv;
    for (i = 0; i < n; i++) mean += x[i];
    mean /= (float)n;
    for (i = 0; i < n; i++) {
        float d = x[i] - mean;
        var += d * d;
    }
    var /= (float)n;                 /* biased variance, matches torch */
    inv = 1.0f / sqrtf(var + eps);
    for (i = 0; i < n; i++) {
        y[i] = g[i] * ((x[i] - mean) * inv) + b[i];
    }
}

/* Fast float exp() for softmax. The microlib expf() (see -D__MICROLIB in the
 * build) is a size-optimised library routine costing thousands of cycles on
 * this Cortex-M7 — and it is the dominant cost of AI-3's attention softmax
 * (512 softmax calls x 64 elems = 32768 exp evals per forward; AI-3 spent
 * ~0.5 s here while the matmuls, at -O3, cost only ~18 ms). No -O flag touches
 * a precompiled library call, so we inline our own:
 *   exp(x) = 2^(x*log2e) — the integer power drops straight into the IEEE-754
 *   exponent field; the [0,1) fractional part uses a degree-5 Taylor poly of
 *   2^f (coeffs c_n = (ln2)^n / n!). Max relative error ~1.5e-4 over [-87,0],
 *   far inside the AI golden tolerance (1e-2) and never flips a softmax argmax.
 * Source: standard 2^x range reduction (cf. Cephes single-precision expf). */
static float nn_fast_expf(float x)
{
    float f, p;
    int i;
    union { int u; float f; } bits;

    if (x <= -87.0f) return 0.0f;           /* exp underflows to 0          */

    x *= 1.4426950409f;                     /* x * log2(e): base-2 exponent */
    i = (int)x;
    if ((float)i > x) i--;                  /* floor (x is usually negative) */
    f = x - (float)i;                       /* fractional part in [0,1)      */

    p = 1.0f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f
          + f * (0.0096181f + f * 0.0013333f))));
    bits.u = (i + 127) << 23;               /* 2^i as a float               */
    return bits.f * p;
}

void nn_softmax(float *x, int n)
{
    int i;
    float mx = x[0], sum = 0.0f;
    for (i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    for (i = 0; i < n; i++) { x[i] = nn_fast_expf(x[i] - mx); sum += x[i]; }
    if (sum <= 0.0f) sum = 1.0f;
    for (i = 0; i < n; i++) x[i] /= sum;
}

int nn_argmax(const float *x, int n)
{
    int i, best = 0;
    float bv = x[0];
    for (i = 1; i < n; i++) {
        if (x[i] > bv) { bv = x[i]; best = i; }
    }
    return best;
}

void nn_conv2d(const float *in, int IC, int IH, int IW,
               const float *w, const float *b,
               int OC, int KH, int KW, int pad, float *out)
{
    int OH = IH + 2 * pad - KH + 1;
    int OW = IW + 2 * pad - KW + 1;
    int oc, ic, oy, ox, ky, kx;

    for (oc = 0; oc < OC; oc++) {
        float *omap = out + (long)oc * OH * OW;
        for (oy = 0; oy < OH; oy++) {
            for (ox = 0; ox < OW; ox++) {
                float acc = b[oc];
                for (ic = 0; ic < IC; ic++) {
                    const float *imap = in + (long)ic * IH * IW;
                    const float *kern = w + (((long)oc * IC + ic) * KH) * KW;
                    for (ky = 0; ky < KH; ky++) {
                        int iy = oy + ky - pad;
                        if (iy < 0 || iy >= IH) continue;
                        for (kx = 0; kx < KW; kx++) {
                            int ix = ox + kx - pad;
                            if (ix < 0 || ix >= IW) continue;
                            acc += imap[iy * IW + ix] * kern[ky * KW + kx];
                        }
                    }
                }
                omap[oy * OW + ox] = acc;
            }
        }
    }
}

void nn_maxpool2d(const float *in, int C, int IH, int IW,
                  int k, int s, float *out)
{
    int OH = (IH - k) / s + 1;
    int OW = (IW - k) / s + 1;
    int c, oy, ox, ky, kx;

    for (c = 0; c < C; c++) {
        const float *imap = in + (long)c * IH * IW;
        float *omap = out + (long)c * OH * OW;
        for (oy = 0; oy < OH; oy++) {
            for (ox = 0; ox < OW; ox++) {
                int iy0 = oy * s, ix0 = ox * s;
                float mx = imap[iy0 * IW + ix0];
                for (ky = 0; ky < k; ky++) {
                    for (kx = 0; kx < k; kx++) {
                        float v = imap[(iy0 + ky) * IW + (ix0 + kx)];
                        if (v > mx) mx = v;
                    }
                }
                omap[oy * OW + ox] = mx;
            }
        }
    }
}
