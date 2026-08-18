# Lab-Sentinel engineering documentation

The documentation is organized from system intent to reproducibility. The
landing page is complete in Chinese (`README.md`) and English (`README_EN.md`);
the detailed engineering documents use English as their primary language.

| Area | Documents |
|---|---|
| Overview | [Project story](00-overview/project-story.md) · [Code tour](00-overview/code-tour.md) · [Competition and awards](00-overview/competition-and-awards.md) · [Glossary](00-overview/glossary.md) |
| Architecture | [System](01-architecture/system-architecture.md) · [Runtime vs host](01-architecture/runtime-vs-host.md) · [Dataflow/RTOS](01-architecture/dataflow-and-rtos.md) · [Memory budget](01-architecture/memory-budget.md) |
| Hardware | [Overview](02-hardware/hardware-overview.md) · [BOM](02-hardware/bom.csv) · [Pinout](02-hardware/pinout.md) · [PCB](02-hardware/pcb.md) · [Bring-up](02-hardware/wiring-and-bringup.md) |
| Firmware | [Keil build](03-firmware/build-keil-r21.md) · [Flash/serial](03-firmware/flash-and-serial.md) · [FreeRTOS](03-firmware/freertos-tasks.md) · [HMI](03-firmware/hmi-13-pages.md) · [Voice](03-firmware/voice-17-commands.md) |
| AI | [BOARD model zoo](04-ai/model-zoo-board.md) · [nano-LM](04-ai/nanolm.md) · [Quantization](04-ai/quantization.md) · [HOST assets](04-ai/icmat-forge-host.md) · [Model card schema](04-ai/model-card-schema.md) |
| Data | [Data card](05-data/data-card.md) · [Provenance](05-data/provenance.md) · [Licenses](05-data/licenses.md) · [Split/leakage](05-data/splits-and-leakage.md) |
| Reproduction | [Environment](06-reproduction/environment.md) · [HOST evidence verification](06-reproduction/host-smoke.md) · [Board acceptance](06-reproduction/board-acceptance.md) · [Benchmarks](06-reproduction/benchmarks.md) · [Expected hashes](06-reproduction/expected-hashes.md) |
| Safety | [Truth boundaries](07-safety/truth-boundaries.md) · [Deterministic control](07-safety/deterministic-control.md) · [Failure modes](07-safety/failure-modes.md) · [Responsible use](07-safety/responsible-use.md) |
| Demo | [Five-minute demo](08-demo/five-minute-demo.md) · [Video chapters](08-demo/video-chapters.md) · [Prototype tour](08-demo/prototype-tour.md) |
| Competition | [Submission overview](09-competition/submission-overview.md) · [Awards](09-competition/awards.md) · [Gallery](09-competition/gallery.md) |

Start with [runtime vs host](01-architecture/runtime-vs-host.md) before quoting
any model count or performance number.
