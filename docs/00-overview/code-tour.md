# Code tour

This map is the shortest route from the public story to the implementation.

| Question | Start here | What to inspect next |
|---|---|---|
| Where does firmware start? | [`firmware/keil_proj/User/main.c`](../../firmware/keil_proj/User/main.c) | clock/cache/SDRAM initialization and task creation |
| Where is the 13-page HMI and deterministic command dispatch? | [`firmware/keil_proj/HardWare/Lab_Sentinel/lab_sentinel.c`](../../firmware/keil_proj/HardWare/Lab_Sentinel/lab_sentinel.c) | page construction, process-profile replay, ABORT handling and UI evidence |
| Where is fixed-vocabulary voice decoded? | [`firmware/keil_proj/HardWare/CI1302/ci1302.c`](../../firmware/keil_proj/HardWare/CI1302/ci1302.c) | 8-byte UART protocol, checksum failure and command mapping |
| Where are physical sensors read? | [`firmware/keil_proj/HardWare/Sensors/sensors_i2c.c`](../../firmware/keil_proj/HardWare/Sensors/sensors_i2c.c) | TCA9548A channel selection, freshness and device boundaries |
| Where are deterministic actuators implemented? | [`relay.c`](../../firmware/keil_proj/HardWare/Relay/relay.c) and [`motor.c`](../../firmware/keil_proj/HardWare/Motor/motor.c) | safe defaults, relay/fan behavior and motor braking |
| Where are the 20 discriminative C runtimes checked? | [`firmware/ai_models_c/ai_selftest.c`](../../firmware/ai_models_c/ai_selftest.c) | fixed golden inputs, tolerance and aggregate pass logic |
| Where are the nano-LM assets? | [`firmware/ai_models_c/ai_nanolm.c`](../../firmware/ai_models_c/ai_nanolm.c) and [`ai_llm_cluster.c`](../../firmware/ai_models_c/ai_llm_cluster.c) | flagship runtime and swap-loaded experts |
| Where is the 244→170 contract ledger? | [`ai/contracts/model_roster_200.v1.tsv`](../../ai/contracts/model_roster_200.v1.tsv) | asset identity, logical family, status, authority and data gate |
| Where is the ModelBank loader candidate? | [`ai/firmware_integration/modelbank_v8_gd32/`](../../ai/firmware_integration/modelbank_v8_gd32/) | catalog/bus guards and the fail-closed board port |
| Where is evidence-chain persistence? | [`ai/firmware_integration/veriprocess_v9/`](../../ai/firmware_integration/veriprocess_v9/) | A/B trace ledger, WAL and host/board adapters |
| Where are public acceptance facts checked? | [`tools/verify_evidence.py`](../../tools/verify_evidence.py) | 30/28, 170 split, authority, hashes and board execution zero |
| Where are public-release privacy gates? | [`tools/verify_release.py`](../../tools/verify_release.py) | secrets, archives, Keil references, EDA metadata and manifests |

The public OV5640 adapter is intentionally camera-disabled because the archived
initialization table had no established redistribution license. See the camera
directory README before replacing it. Likewise, the public CJK font header uses
an ASCII fallback until the user regenerates glyphs from an OFL font.

