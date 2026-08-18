# Failure modes and responses

| Failure | Detection | Safe response |
|---|---|---|
| Bad voice checksum | frame checksum | no action; log/feedback only |
| Sensor missing/stale | presence, fault, freshness | mark missing; no stale-value claim; deterministic fallback |
| Model package corruption | catalog/package/payload hash | refuse load; keep frozen runtime |
| microSD sustained-read CRC | catalog-body gate | both catalogs invalid; no entry/payload execution |
| Inference deadline miss | DWT/deadline | refuse/degrade advice; control task continues |
| Low heap/stack margin | runtime watermarks | stop promotion, reduce workload, preserve safety task |
| UI freeze | watchdog/task liveness | controlled reset or local safe state |
| PC/cloud absent | offline design | board runtime and deterministic chain continue |
| Unexpected actuator polarity | powered bring-up check | default-off, disconnect load, correct revision-specific mapping |

Failure injection must use fixtures or safe low-voltage stimuli; do not unplug or
short a sensor in a way that can damage the board.

