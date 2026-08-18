# Contributing to Lab-Sentinel

Thank you for helping make embedded materials AI more reproducible.

## Before opening a pull request

1. Open an issue describing the objective, target hardware and evidence class.
2. Use a focused branch: `feat/`, `fix/`, `docs/`, `model/` or `hardware/`.
3. Run:

   ```bash
   python tools/verify_release.py --strict
   python -m unittest discover -s tests -v
   ```

4. Include exact reproduction commands, expected output, logs/screenshots and
   the license/provenance of every new input.
5. Add `Signed-off-by: Name <email>` to commits (Developer Certificate of
   Origin 1.1).

## Model contributions

A model PR must contain:

- a unique task contract and logical-model identity;
- truth class and source/license record;
- group/family split and leakage audit;
- frozen baseline and at least the stated seed protocol;
- quantization, golden and ABI evidence;
- a model card with limits, failure cases and `authority=0`.

Seeds, checkpoints, quantized copies, prompts and runtime sizes are not separate
logical models. Do not replace missing labels with fixtures, teachers or API
responses.

## Hardware contributions

Provide editable source, schematic/PCB export, BOM, pin-conflict review,
bring-up steps and observable evidence. New actuation must remain behind the
deterministic safety path. Do not silently change validated pin assignments.

## Never submit

Credentials, private endpoints, personal data, unlicensed vendor code,
unverifiable performance claims, production data, identity-bearing competition
materials or a path that lets a learned model bypass deterministic safety.

## Review standard

Reviewers evaluate truthfulness and reproducibility before novelty. A clear
fail-closed result is more valuable than a passing number without provenance.

