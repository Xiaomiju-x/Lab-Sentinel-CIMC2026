# HOST evidence verification

The default check validates the public evidence/contract layer:

```bash
python tools/summarize_ledgers.py
python tools/verify_evidence.py
```

Expected summary:

```text
BOARD: 30 runtime assets / 28 logical models
HOST: 170 = P112 + G30 + S28
HOST-EXACT: 78
HOST-SIM_ONLY: 92
NEW BOARD EXECUTION: 0
```

For a specific model:

1. read its task contract and data license;
2. acquire the exact source data or Release fixture;
3. create a fresh environment and record versions;
4. run baseline before trained model;
5. verify split/leakage and package/golden hashes;
6. report HOST output as HOST—never as board timing.

## Download the reviewed technical archive

With GitHub CLI:

```bash
mkdir -p artifacts/v1.0.1
gh release download v1.0.1 \
  --repo Xiaomiju-x/Lab-Sentinel-CIMC2026 \
  --pattern 'lab-sentinel-cimc2026-technical-evidence-v1.0.1.zip*' \
  --dir artifacts/v1.0.1
```

Verify the sidecar before extraction. On Linux/macOS:

```bash
cd artifacts/v1.0.1
sha256sum --check lab-sentinel-cimc2026-technical-evidence-v1.0.1.zip.sha256
```

On PowerShell:

```powershell
$expected = 'fcaa3292c6a404258780fd5cc003df1fc608087f2de50b78eed5165d36bd49a5'
$actual = (Get-FileHash .\artifacts\v1.0.1\lab-sentinel-cimc2026-technical-evidence-v1.0.1.zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actual -ne $expected) { throw "technical archive SHA-256 mismatch" }
```

Expected archive SHA-256:

```text
fcaa3292c6a404258780fd5cc003df1fc608087f2de50b78eed5165d36bd49a5
```

The archive's own `SHA256SUMS.csv` covers every payload member. The GD32
Embedded AI Tool representative chain is under
`03_gd32_embedded_ai_tool/`; its recorded host outputs are under
`host_golden/`. These records support evidence review. Re-running the C harness
requires a compatible C compiler and the corresponding source/weight set; it
does not turn HOST evidence into BOARD timing.

## Reproducibility boundary

The repository does **not** redistribute the 281 historical measured PL spectra
under an unrestricted public-data license. Therefore the AI-12 five-fold OOF
Accuracy 98.22% is traceable to contracts, reports and hashes but cannot be
fully retrained from public raw spectra alone. The firmware contains three
representative real-spectrum golden fixtures for implementation/parity checks;
three fixtures are not a substitute for the 281-spectrum evaluation set.

GPU training scripts are historical reproducibility artifacts, not a mandate to
rerun all 244 tasks. Rejected dispositions must remain rejected unless new,
properly licensed evidence satisfies the original contract.
