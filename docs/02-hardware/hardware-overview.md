# Hardware overview

![Prototype top view](../../assets/hardware/prototype-top-view.webp)

The prototype centers on the competition GD32H759 platform and a team-designed
integration/expansion PCB that replaces early jumper-wire assembly.

| Domain | Device | Role | Public boundary |
|---|---|---|---|
| Point temperature | K-type + MAX31856 | compensated contact temperature and fault bits | low-temperature bench; does not own furnace control |
| Thermal field | MLX90640 | 32×24 min/mean/max/hotspot observation | engineering observation, not calibrated industrial thermography |
| Environment | SHT30 | temperature/humidity context with CRC | outside-process context, not furnace temperature |
| Electrical | INA226 ×2 design | PTC/fan bus/shunt response | report raw/validated channels; no precision power claim without calibration |
| Vibration | ADXL345 | 3-axis response to eccentric motor | diagnostic input, not a certified vibration instrument |
| Vision | OV5640 | 320×240 RGB565 via DCI/DMA | do not claim 5 MP real-time inference |
| Air trend | MQ-135 | gas/smoke trend | no calibrated ppm or gas-specific identification |
| Interaction | 800×480 LCD + GT911, CI1302 | local touch and fixed offline voice | no cloud HMI or free-form voice dialogue |
| Actuation | low-voltage PTC, fans, H-bridge motor, alarm | deterministic physical response | prototype loads only; AI authority remains zero |

The early OV5640 table, SimHei-derived font bitmap and unused optional LVGL LRU
derivative are excluded for license reasons. See `THIRD_PARTY_NOTICES.md`.
