# Quantization and optimization

The project uses optimization per asset family rather than checking every
possible technique in a form.

- AI-12 has a true-board FP32/INT8 comparison: about 0.488 ms to 0.153 ms,
  approximately 3.2×, with identical decisions on the frozen board fixture.
- HOST predictive/support contracts use W8A8 where their package/engine states
  so; P096 reports W8 weights with a CMSIS-NN W8A8 execution contract.
- HOST NanoLM contracts use W8 weights.
- INT4 was not admitted to the formal contract.
- No global “all models were pruned/fused/distilled” claim is made. Distillation
  applies only to the named existing nano-LM assets.

For every comparison record model ID, input set, preprocessing, compiler,
cache/warmup state, included pre/post-processing, cycles, size, RAM and accuracy.

