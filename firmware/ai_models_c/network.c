/*
 * network.c
 * AI-1 Vision CNN — legacy API shim (network_init / network_run).
 *
 * 2026-05-29: replaced the Phase-0 zeros stub with the real hand-written
 * float CNN engine (ai1_cnn.c). The legacy network_run() signature is kept
 * so the existing sensor_task boot smoke-test keeps compiling; new code
 * should call ai1_cnn_forward() directly (it also returns the 32-D embedding).
 *
 *   Architecture : Conv4-MaxPool-Conv8-MaxPool-FC32-FC10  (MNIST 97.3%)
 *   Input  : float[1*28*28]  normalised externally
 *   Output : float[10]       raw logits
 */
#include "network.h"
#include "ai1_cnn.h"

void network_init(void)
{
    /* No persistent state — weights live in Flash (.rodata). */
}

void network_run(const float *input, float *output)
{
    ai1_cnn_forward(input, output, 0);   /* discard embedding for legacy callers */
}
