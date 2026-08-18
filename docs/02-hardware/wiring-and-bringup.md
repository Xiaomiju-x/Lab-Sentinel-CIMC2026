# Wiring and bring-up

Bring up one trust domain at a time; never start by enabling all actuators.

1. **Unpowered inspection:** polarity, shorts, connector orientation and common
   ground.
2. **Controller only:** 3.3/5 V rails, reset, SWD and debug UART.
3. **Memory/display:** SDRAM test, LCD framebuffer and GT911.
4. **Buses:** TCA scan by channel; shared SPI with all chip selects inactive.
5. **Sensors:** read identity/status/fault and freshness before interpreting a
   physical value.
6. **Actuators:** low-voltage load, one path at a time, with a physical current
   limit and manual disconnect.
7. **AI:** golden self-test before live sensor routing.
8. **Safety:** watchdog, stale sensor, bad voice checksum, stop, recovery and
   power-cycle behavior.

Do not infer that two equal I²C design addresses are both healthy merely because
the mux isolates them. Archive the per-channel scan and configuration result.
The latest competition evidence showed the fan INA channel ready while the PTC
INA channel needed address re-verification.

