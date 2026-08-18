# Truth boundaries

| Tempting statement | Accurate public statement |
|---|---|
| “200 models run on the MCU” | 30 BOARD runtime assets / 28 logical models, plus 170 separate HOST research assets |
| “170 models were deployed from SD” | files were prepositioned; catalog-body CRC failed before entry/payload load; execution 0 |
| “170 real-data models” | 78 HOST–EXACT and 92 HOST–SIM_ONLY |
| “1500 °C closed-loop furnace” | low-temperature physical bench + deterministic 1500 °C profile replay |
| “online PL spectrometer” | three representative historical measured spectra replayed on the board UI |
| “thermal camera from RGB page” | OV5640 luminance pseudo-color unless explicitly showing MLX90640 data |
| “17-command voice AI” | 17 fixed offline commands defined by CI1302 protocol; not free dialogue |
| “AI controls the furnace” | AI authority=0; deterministic safety and operator retain actuation |
| “ProofPass is digital signature” | tamper-evident payload/trace record, not identity or security certification |
| “IC-material production validated” | method/prototype expansion; no wafer-fab or packaging-line validation |

Accurate boundaries make the work stronger: they tell another team exactly what
can be reproduced and what still requires independent validation.

