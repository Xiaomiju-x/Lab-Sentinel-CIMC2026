# Model and weight licensing

Model source code, contracts and model cards authored by the project are
covered by Apache-2.0 unless marked otherwise. Training data rights do not
automatically transfer to weights. Consequently:

- Small board-runtime weight headers included in `firmware/ai_models_c/` are
  released for research and reproducibility under Apache-2.0 where the team
  owns the implementation and training artifact.
- Large ONNX, checkpoint, ModelBank and quantized packages are distributed only
  in a reviewed GitHub Release, each with a manifest and SHA-256 digest.
- HOST–EXACT means the asset has exact source/label/split binding and a host
  contract; it does **not** mean board deployment or production validation.
- HOST–SIM_ONLY assets are interface/research candidates and must not be used as
  evidence of experimental or fab performance.
- Teacher/API outputs are candidates, never ground truth.

## DeepSeek teacher-output provenance

Some teacher samples used in the NanoLM research pipeline originated from the
DeepSeek Open Platform. The applicable public terms effective 2026-04-29 state
in section 4.2 that transferable rights in generated output are assigned to the
user and that the output may be used for derivative development and model
distillation:

- <https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html>
- <https://cdn.deepseek.com/policies/en-US/deepseek-terms-of-use.html>

This record does not turn teacher output into experimental truth. The project
remains responsible for rights in every input, third-party material that may
appear in an output, marking released samples as AI-generated, and human review
before a sample can enter a research corpus. No DeepSeek trademark or logo is
used to endorse this project.

The reviewed v1.0.1 technical-evidence archive contains ten NanoLM teacher
candidate files named `corpus_e1.jsonl` through `corpus_e7.jsonl` (including
the documented `_x5` variants) and `corpus_v2.jsonl` under
`04_training_export_scripts/nanolm/`. They are released as AI-generated
research/provenance artifacts under this same boundary: they are not
experimental labels or ground truth, do not establish scientific claims by
themselves, and must not be promoted without independent human validation.

Every new model contribution must provide a data card, task contract, split and
leakage audit, baseline, quantization/golden evidence, and `authority=0`.
