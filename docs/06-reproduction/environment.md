# Reproduction environment

Reproduction has three levels. Passing a lower level does not imply the next.

## Level 1 — public repository integrity

Python 3.10+ with no third-party package:

```bash
python tools/verify_release.py --strict
python -m unittest discover -s tests -v
```

This checks public files, JSON evidence, fact ledgers, paths, sizes and secret
patterns. It is suitable for CI and does not emulate a board.

Release operators can add a private, untracked denylist without publishing its
values. Store one UTF-8 value per line outside the repository, then set
`LAB_SENTINEL_PRIVATE_DENYLIST` to that file before running the strict check.
The generic CI patterns always run; the private list is an additional local
gate for known names, identifiers and retired credentials.

## Level 2 — HOST evidence and selected pipelines

Create an isolated environment and install only the dependencies required by a
selected pipeline. Large reviewed assets are in the v1.0.1 GitHub Release.
Verify their SHA-256 before extracting. Source-bound datasets must be obtained
under their upstream license.

Public PyTorch loaders use `weights_only=True`. The reviewed technical archive
is also audited so every distributed checkpoint can pass that restricted
loader; do not weaken this setting for an unverified checkpoint.

The default commands verify public ledgers, contracts and receipts; they do not
execute all 170 assets or retrain the reported metrics. The 281 historical PL
spectra are not redistributed as an unrestricted public dataset, so AI-12's
98.22% five-fold result is evidence-verifiable but not fully retrainable from
this public tree alone. See `host-smoke.md` for exact Release download and hash
verification.

## Level 3 — GD32H759 board

Requires Keil MDK, the official GD32 device support, the competition hardware,
DAPLink/CMSIS-DAP and the user's licensed camera/font implementation. Build the
`CIMC_GD32_Template` target (archived release line R2.1) and follow
`board-acceptance.md`. CI never reports this level as
passing without a physical board receipt.

Do not set remote passwords or API keys unless running an explicitly selected
historical helper. Use `.env.example`, environment variables and known-host
verification; secrets must never enter logs or Git.
