# Complete demonstration and chapters

The `v1.0.2` media archive publishes the complete 5:06 demonstration in two
forms: one continuous file and five independently seekable chapters. Both keep
the full project demonstration content. The default public derivatives remove
precise GPS and device fields. At the maintainer's explicit archival direction,
the v1.0.2 Release also contains a clearly named `privacy-sensitive` ZIP with
the exact seven supplied source media files. That archive may retain precise
metadata, identifiable people, badges and visible time/location overlays; read
its warning before downloading.

**Continuous version:**
[`00-full-demo-privacy-sanitized.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.2/00-full-demo-privacy-sanitized.mp4)

| Time in source | Release asset | What to observe | Boundary |
|---|---|---|---|
| 00:00–01:30 | [`01-process-profile-and-ai-state.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.2/01-process-profile-and-ai-state.mp4) | Home, deterministic process profile, AI state | high-temperature curve is replay, not a real furnace |
| 01:30–02:30 | [`02-hardware-sensors-and-safety.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.2/02-hardware-sensors-and-safety.mp4) | controller, PCB, touch, sensor/robustness interaction | low-voltage prototype; channel freshness governs claims |
| 02:30–03:30 | [`03-camera-and-vision-pipeline.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.2/03-camera-and-vision-pipeline.mp4) | OV5640, Camera, Pre-flt, true-color path | 320×240 runtime; pseudo-color is not MLX thermal imaging |
| 03:30–04:25 | [`04-edge-lm-and-mcu-inference.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.2/04-edge-lm-and-mcu-inference.mp4) | Edge LM and on-device inference UI | bounded nano-LM task, not cloud/free dialogue |
| 04:25–05:06 | [`05-control-telemetry-and-closeout.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.2/05-control-telemetry-and-closeout.mp4) | fans, Control/System telemetry and finish | AI advice does not directly actuate |

The continuous file and chapters are 720p H.264/AAC and metadata-stripped. The
continuous version is stored in the Release because it exceeds GitHub's normal
repository-file limit; the chapters remain convenient for direct linking.
`SHA256SUMS.txt` covers every v1.0.2 media asset, and the release tag anchors its
own checksum. CI repeats exact-set, digest and privacy-marker checks whenever a
Release is published. Privacy-marker checks apply to the default derivatives;
the exact-source ZIP is separately name-gated, checksummed and required to carry
its internal warning and manifest.

The reviewed model/evidence archive remains independently available in the
immutable [`v1.0.1` technical release](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/tag/v1.0.1).
