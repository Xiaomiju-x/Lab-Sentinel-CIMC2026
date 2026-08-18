/*
 * network.h
 * AI-1 Vision CNN inference API  (Phase 0 stub)
 *
 * Phase 0: stub implementations — validates Keil integration, no HardFault.
 * Phase 2: replace with GD32 AI Tool-generated nn_model_configure code or
 *           Python-generated C from tiny_cnn_mnist_float32.tflite.
 *
 * Input : float[1*28*28]  — NHWC after transposing, values normalized [0,1]
 * Output: float[10]       — logits for digits 0-9
 */

#ifndef NETWORK_H
#define NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

void  network_init(void);
void  network_run(const float *input, float *output);

#ifdef __cplusplus
}
#endif

#endif /* NETWORK_H */
