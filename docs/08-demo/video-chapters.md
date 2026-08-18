# Demonstration video chapters

The privacy-clean videos are attached to the `v1.0.1` GitHub Release. The raw
phone file is not published because its container stored precise GPS and device
metadata.

| Time in source | Release asset | What to observe | Boundary |
|---|---|---|---|
| 00:00–01:30 | [`01-process-profile-and-ai-state.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.1/01-process-profile-and-ai-state.mp4) | Home, deterministic process profile, AI state | high-temperature curve is replay, not a real furnace |
| 01:30–02:30 | [`02-hardware-sensors-and-safety.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.1/02-hardware-sensors-and-safety.mp4) | controller, PCB, touch, sensor/robustness interaction | low-voltage prototype; channel freshness governs claims |
| 02:30–03:30 | [`03-camera-and-vision-pipeline.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.1/03-camera-and-vision-pipeline.mp4) | OV5640, Camera, Pre-flt, true-color path | 320×240 runtime; pseudo-color is not MLX thermal imaging |
| 03:30–04:25 | [`04-edge-lm-and-mcu-inference.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.1/04-edge-lm-and-mcu-inference.mp4) | Edge LM and on-device inference UI | bounded nano-LM task, not cloud/free dialogue |
| 04:25–05:06 | [`05-control-telemetry-and-closeout.mp4`](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/download/v1.0.1/05-control-telemetry-and-closeout.mp4) | fans, Control/System telemetry and finish | AI advice does not directly actuate |

Each file is 720p H.264/AAC, independently seekable, metadata-stripped and
below GitHub's ordinary-file limit; it is stored in the Release to keep Git
history lean. `SHA256SUMS.txt` beside the Release assets provides integrity.
Verify it before playback or extraction; CI repeats that check whenever a
Release is published.
