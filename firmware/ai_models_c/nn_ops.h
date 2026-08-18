/******************************************************************************
 * nn_ops.h  —  CIMC Lab-Sentinel float32 neural-net kernels
 *
 * Hand-written float kernels for the 5 on-chip AI models. The GD32H759 M7
 * has a single-precision FPU @600 MHz; these KB-scale models run in <5 ms
 * each and reproduce PyTorch byte-for-byte (validated by ai_golden.h).
 *
 * All weight tensors follow the PyTorch convention exactly:
 *   Linear  weight = [out, in]  →  y[o] = b[o] + Σ_i W[o*in+i] * x[i]
 *   Conv2d  weight = [out_ch, in_ch, kh, kw]   (NCHW activations)
 ******************************************************************************/
#ifndef NN_OPS_H
#define NN_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Fully-connected: y = W·x + b.  W is [out_dim, in_dim] row-major. */
void nn_linear(const float *x, const float *W, const float *b,
               float *y, int in_dim, int out_dim);

/* Fully-connected without bias (attention q/k/v projections). */
void nn_linear_nobias(const float *x, const float *W,
                      float *y, int in_dim, int out_dim);

/* Batched fully-connected: Y[rows][out_dim] = X[rows][in_dim] . W^T + b (b may be
 * NULL). Weight-stationary loop order reads each weight once and reuses it across
 * all rows -> far less Flash traffic than a per-row nn_linear() loop on big
 * matrices. Numerically identical to per-row nn_linear(). */
void nn_linear_batch(const float *X, const float *W, const float *b,
                     float *Y, int rows, int in_dim, int out_dim);

/* Weight-only INT8 linear (B1 quantisation). Wq is [out_dim,in_dim] int8 (per-output
 * symmetric PTQ), scale is [out_dim]. y[o] = b[o] + scale[o]*Σ Wq[o,i]*x[i]. Compute
 * stays float (M7 has no int8 SIMD); the win is 4x Flash + less weight D-cache traffic. */
void nn_linear_int8(const float *x, const signed char *Wq, const float *scale,
                    const float *b, float *y, int in_dim, int out_dim);

/* In-place ReLU over n elements. */
void nn_relu(float *x, int n);

/* LayerNorm over a vector of length n:  y = g * (x-mean)/sqrt(var+eps) + b.
 * var uses the biased (population) estimate, matching torch.nn.LayerNorm. */
void nn_layernorm(const float *x, const float *g, const float *b,
                  float *y, int n, float eps);

/* In-place numerically-stable softmax over n elements. */
void nn_softmax(float *x, int n);

/* Index of the maximum element. */
int nn_argmax(const float *x, int n);

/* Conv2d, stride 1, square padding `pad`, NCHW.
 *   in     : [IC][IH][IW]
 *   w      : [OC][IC][KH][KW]      b: [OC]
 *   out    : [OC][OH][OW]   with OH=IH+2*pad-KH+1, OW=IW+2*pad-KW+1
 * Caller guarantees `out` is large enough. */
void nn_conv2d(const float *in, int IC, int IH, int IW,
               const float *w, const float *b,
               int OC, int KH, int KW, int pad, float *out);

/* MaxPool2d, square window k, stride s, NCHW.
 *   in : [C][IH][IW]  →  out: [C][OH][OW]  OH=(IH-k)/s+1 */
void nn_maxpool2d(const float *in, int C, int IH, int IW,
                  int k, int s, float *out);

#ifdef __cplusplus
}
#endif

#endif /* NN_OPS_H */
