# Runtime versus HOST: the three-ledger rule

Never combine these rows into one “deployed model” number.

| Ledger | Predictive | Generative | Support | Total |
|---|---:|---:|---:|---:|
| BOARD runtime assets | — | — | — | 30 assets / 28 logical models |
| HOST–EXACT | 25 | 25 | 28 | 78 |
| HOST–SIM_ONLY | 87 | 5 | 0 | 92 |
| HOST total | 112 | 30 | 28 | 170 |

The 170 HOST packages have unique package/payload hashes and `authority=0`.
They were copied to microSD as files. On the unified board attempt, FAT32 and
catalog headers were reached, but sustained catalog-body reads failed CRC. Both
catalogs were rejected before entry/payload loading. Therefore:

- new HOST board execution = 0;
- accepted BOARD count remains 30 assets / 28 logical models;
- the failure is a correct fail-closed result, not a partial deployment;
- copying a file to storage is not the same as loading or executing a model.

Primary evidence lives under `evidence/public/`.

