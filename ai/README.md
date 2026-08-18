# AI research and deployment artifacts

- `contracts/` — preregistered tasks, identities, quantization and authority;
- `pipeline/` — source/data gates, training/export, ModelBank and acceptance;
- `training/` — curated competition training/export scripts;
- `gd32-embedded-ai-tool/` — the named AI-4 conversion/golden path;
- `firmware_integration/` — C integration for ModelBank/RAG/VeriProcess;
- `tests/` and `tools/` — original host validation utilities.

Some files are immutable historical plans and therefore contain pre-execution
status fields. Current public state is defined by `evidence/public/`, especially
`release_gap_audit.v7.json` and the final board rejection receipt.

No credential or private remote endpoint is included. Optional teacher/remote
scripts read secrets from environment variables and enforce known-host policy.
Teacher output is never ground truth.

