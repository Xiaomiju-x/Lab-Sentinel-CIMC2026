# Pinout and bus map

Treat the project source and current hardware revision as authoritative; verify
before rewiring.

| Function | Pins / channel |
|---|---|
| Debug UART | TX PB13, RX PB5, 115200 8N1 |
| CI1302 UART3 | PC10 / PC11, 115200 8N1 |
| Shared SPI | PB10 SCK, PC1 MOSI, PC2 MISO |
| MAX31856 CS | PG3 |
| microSD CS | PC5 |
| TCA9548A upstream | PH7 / PH8; address 0x70 |
| TCA CH2 | DS3231 0x68 |
| TCA CH3 | fan INA226 design address 0x44 |
| TCA CH4 | MLX90640 0x33 |
| TCA CH5 | SHT30 0x44 |
| TCA CH6 | ADXL345 0x53 |
| TCA CH7 | PTC INA226 design channel; re-scan actual address before claim |
| PTC | relay PD12 + MOSFET PH12 |
| legacy fan | PG14 |
| 12 V fan | enable PE2, PWM-interface PE5, tach PH11 |
| H-bridge motor | PG11 / PG6 |
| external alarm | PD13 through relay; verify active polarity on current unit |
| GT911 | PD5 / PD7, INT PH15, RST PH13 |

SDRAM and RGB-LCD pins are high-risk shared resources. Do not assign a new GPIO
from memory; run a source/pin conflict review first.

