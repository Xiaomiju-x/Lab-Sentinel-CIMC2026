# Forge200 ModelBank v4 candidate integration

Status: `HOST_COMPILED_BOARD_PENDING`. These files are an isolated adapter and
are not yet part of the production Keil target.

The loader implements the frozen 256-byte `ICMF` ABI and the required order:
schema/authority/limits -> payload SHA-256 -> generation -> golden -> commit.
It loads into the inactive SDRAM slot and changes the active slot only after
the engine-specific golden callback succeeds. Failure keeps the previously
active model unchanged. The callbacks expose no heater, fan, motor, relay,
alarm, GPIO, or deterministic-control operation; every package remains
`authority=0`.

Catalog generation and package generation are intentionally distinct. The
catalog generation protects A/B catalog rollback while the package header
generation protects stale package formats. Multiple models in one accepted
catalog may therefore share package generation `1` and still be swapped.

`forge200_shared_spi.*` freezes the arbitration boundary for the existing
shared bus: SCK PB10, MOSI PC1, MISO PC2, SD CS PC5, MAX31856 CS PG3. Both CS
signals are deasserted before every owner/mode transition; contention is
refused. Final production integration must wrap FatFs/SD transactions and
MAX31856 transactions with this same arbiter and confirm the electrical
invariant on the powered GD32. The host dry-run is not a timing or board claim.

Final board work, performed once rather than per model, must still verify:

1. Keil target `R2.1`, DAPLink/CMSIS-DAP settings unchanged.
2. Full startup mode restored from `LAB_HARDWARE_BRINGUP=1`.
3. A/B catalog and package hashes from `MANIFEST.v4.json`.
4. One small engine-1 package, one engine-5 nano-LM, then the complete bank.
5. Golden output, arena/canary/cache behavior, rollback and power-loss cases.
6. SD/FatFs plus MAX31856 interleaving on the shared bus and deterministic
   control-chain non-interference.

Do not copy these files into production or burn a board until the release-floor
and unified board-readiness gates are satisfied.
