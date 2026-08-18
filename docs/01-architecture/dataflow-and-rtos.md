# Dataflow and RTOS

The firmware uses FreeRTOS to isolate acquisition, inference, UI and control
cadences. Sensor drivers expose raw value, validity/fault and freshness rather
than silently reusing a stale number.

Important buses:

- TCA9548A multiplexes same-address and high-bandwidth I²C sensors;
- MAX31856 and microSD share SPI signals but use independent chip selects and
  mutual exclusion;
- OV5640 uses DCI + DMA into SDRAM;
- LCD refresh uses the TLI path from an SDRAM framebuffer;
- DWT measures inference cycles on the MCU.

Inference publishes bounded risk/advice. A deterministic task applies profile,
interlock, debounce and watchdog rules before any actuator change. UI/voice
commands enter the same controlled dispatcher.

See source under `firmware/keil_proj/Function/`, `HardWare/` and `User/` for the
exact task and driver organization.

