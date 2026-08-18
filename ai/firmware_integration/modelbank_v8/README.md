# Forge200 ModelBank v8 runtime candidate

This isolated candidate closes the embedded-payload gap found after ModelBank
v7. `F2RT` is an uncompressed, bounds-checkable tensor table inside the frozen
256-byte `ICMF` envelope. The runtime supports the three engines actually used
by the 170-model host set: W8 dense, W8 convolution, and the tied-embedding W8
nano-transformer. It has no actuator, GPIO, heater, fan, relay, motor, alarm,
or deterministic-control API.

The host runner executes the exact C source used for the Cortex-M7 build against
one binary `F2GV` case per package. Passing this runner is still host evidence;
GD32 timing, SDRAM, FatFs, cache, MPU, shared SPI, and board output remain
pending until the single unified board window.
