# Physical board acceptance

Use an immutable firmware/model identity and capture raw serial output.

## Required gates

1. board/firmware identity and full-mode configuration;
2. offline cold boot and 30 asset / 28 logical-model registration;
3. golden self-test for the selected runtime;
4. DWT cycles for a defined input, including whether preprocessing is timed;
5. Flash map, static RW/ZI, peak arena/heap and stack margins;
6. sensor value + fault + freshness for the observable channels;
7. touch/voice command path and checksum-failure no-action test;
8. deterministic stop, interlock and watchdog behavior;
9. long-run reset/leak observation;
10. signed/hash-indexed receipt that keeps failed gates visible.

## ModelBank distinction

Storage presence, FAT mount, catalog-header parse, entry load, payload hash,
runtime initialization and inference are separate gates. The archived v9 board
attempt stopped at sustained catalog-body CRC. Do not skip directly from “file
exists” to “deployed.”

## Physical safety

Use only low-voltage prototype loads, a current-limited supply and manual power
disconnect. This runbook does not authorize connection to a real furnace.

