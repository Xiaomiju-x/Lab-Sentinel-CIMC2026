# Benchmark definitions

| Metric | Test contract | Result |
|---|---|---:|
| AI-12 OOF Accuracy | 281 historical measured PL spectra, 5-fold out-of-fold | 98.22% |
| AI-12 OOF Macro-F1 | same | 97.37% |
| AI-12 FP32 DWT | true board, frozen input/runtime | 0.488 ms representative |
| AI-12 INT8 DWT | true board, same boundary | 0.153 ms representative |
| AI-3 DWT | true board, frozen asset | 67.81 ms representative |
| P096 quantized mIoU | public Carinthia, HOST–EXACT | 0.7794 |
| P096 Boundary-F1 | same | 0.9153 |
| P096 small-defect recall | same | 0.8913 |
| VeriProcess | HOST evidence-chain cases | 69 / 69 |

The AI-4 97.45% result belongs to an `n=2000` synthetic process-fault test, not
the 37 XRD-labeled material records. It is deliberately not used as the main
real-material headline.

The archived Cpk 6.96 belongs to a deterministic process-profile replay. It is
not measured capability of a real 1500 °C furnace or production line.

Every new benchmark should state sample count, split unit, mean/worst case,
baseline, quantization, hardware/runtime, timing boundary and confidence or
variation—not only a best number.

