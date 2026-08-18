# System architecture

The architecture keeps observation, inference and actuation as separate trust
domains.

```mermaid
flowchart TB
  O["Thermal · environmental · electrical · vibration · vision"] --> IO["DMA / DCI / I²C mux / shared SPI"]
  IO --> R["FreeRTOS acquisition and freshness"]
  R --> M["BOARD runtime models"]
  M --> J["Risk / diagnosis / explanation · authority=0"]
  R --> C["Deterministic process and interlocks"]
  J -. "bounded request" .-> C
  C --> A["Low-voltage PTC · fans · H-bridge motor · alarm"]
  M --> E["SPC · trace ledger · HMI · fixed voice"]
  C --> E
```

The PC/cloud side trains and performs heavy preprocessing. The MCU performs
deployment inference, local interaction and deterministic coordination. HOST
assets are linked by contracts/evidence only; they are not hidden remote calls.

The prototype is offline-capable. A PC may display serial logs during
development, but it is not a substitute inference backend.

