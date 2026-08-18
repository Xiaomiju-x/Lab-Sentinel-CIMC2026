# The 13-page local HMI

The public page list follows the current source and prototype, not an older
report label:

1. Home
2. Recipe
3. Trend
4. Control
5. Quality
6. System
7. Camera
8. Pre-flt
9. PL
10. Models
11. E-Twin
12. Edge LM
13. Robust

Important boundaries:

- PL automatically rotates three representative historical measured spectra.
  It is marked `REAL_DATA_REPLAY`; 281 spectra belong to the evaluation set,
  not to a live on-board spectrometer.
- Camera may show a live sensor or an explicit fallback pattern. A thermal-like
  color mode derived from RGB luminance is pseudo-color, not MLX90640 imaging.
- Quality's visible batch window can be RAM state even when VeriProcess writes a
  separate A/B TraceLedger/WAL to storage.
- ABORT uses controlled confirmation; UI effects are not proof of electrical
  actuation unless the physical/telemetry evidence also changes.

