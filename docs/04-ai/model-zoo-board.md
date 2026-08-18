# BOARD model zoo

The frozen board ledger contains **30 runtime assets / 28 logical models**.
Twenty assets are discriminative runtime functions; the remaining runtime
assets cover the flagship nano-LM deployment bank and seven domain experts.

Why 30 is not 30 logical models: three package sizes are deployment profiles of
one flagship logical model. They are assets, not three separately counted
models.

## Frozen asset inventory

| Asset | Runtime role | Logical family |
|---|---|---|
| ICM-001 | AI-1 Crucible CNN | ICM-001 |
| ICM-002 | AI-1b Few-shot NCM | ICM-002 |
| ICM-003 | AI-2 Sinter AE | ICM-003 |
| ICM-004 | AI-3 TinyTransformer | ICM-004 |
| ICM-005 | AI-4 Risk Fusion | ICM-005 |
| ICM-006 | AI-5 Root-cause | ICM-006 |
| ICM-007 | AI-6 Optical TS | ICM-007 |
| ICM-008 | AI-7 Thermal Quench | ICM-008 |
| ICM-009 | AI-8 Energy/CO2 | ICM-009 |
| ICM-010 | AI-9 Recipe kNN | ICM-010 |
| ICM-011 | AI-10 Vibration PdM | ICM-011 |
| ICM-012 | AI-11 Phase Purity | ICM-012 |
| ICM-013 | AI-12 PL Dopant | ICM-013 |
| ICM-014 | AI-13 PL-QC AE | ICM-014 |
| ICM-015 | AI-14 Temperature Forecast | ICM-015 |
| ICM-016 | AI-15 PL Host-ID | ICM-016 |
| ICM-017 | AI-16 PL Lambda | ICM-017 |
| ICM-018 | AI-17 PL Few-shot | ICM-018 |
| ICM-019 | AI-19 RUL/ETA | ICM-019 |
| ICM-020 | AI-20 TC Integrity | ICM-020 |
| ICM-021 | x1p9 flagship nano-LM deployment profile | FLAGSHIP_NLM_FAMILY |
| ICM-022 | m1p35 flagship nano-LM deployment profile | FLAGSHIP_NLM_FAMILY |
| ICM-023 | s0p6 flagship nano-LM deployment profile | FLAGSHIP_NLM_FAMILY |
| ICM-024 | E1 Fault Diagnosis LM | ICM-024 |
| ICM-025 | E2 Recipe Advice LM | ICM-025 |
| ICM-026 | E3 Energy Carbon LM | ICM-026 |
| ICM-027 | E4 Batch QC LM | ICM-027 |
| ICM-028 | E5 Operator Brief LM | ICM-028 |
| ICM-029 | E6 Formula Chemistry LM | ICM-029 |
| ICM-030 | E7 Equipment Maintenance LM | ICM-030 |

This table is generated from the frozen first 30 rows of
`ai/contracts/model_roster_200.v1.tsv`. All entries have `authority=0`; a model
result cannot directly actuate the heater, relay, fan, motor or alarm.

Representative board evidence:

| Asset | Function | Latest representative DWT time |
|---|---|---:|
| AI-1 | crucible vision | ~17.31 ms |
| AI-2 | light tabular task | ~0.262 ms |
| AI-3 | heaviest frozen runtime | ~67.81 ms |
| AI-4 | four-level process risk | ~0.230 ms |
| AI-12 FP32 | PL material classification | ~0.488 ms |
| AI-12 INT8 | quantized PL classification | ~0.153 ms |
| AI-20 | lightweight auxiliary task | ~0.097 ms |

Numbers are model- and measurement-contract specific. Re-run DWT and golden
self-test after any compiler, linker, cache or weight change.
