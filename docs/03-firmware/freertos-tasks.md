# FreeRTOS task model

The exact task names can change across revisions, but the architectural
contract is stable:

- **acquisition** reads sensor value + fault + freshness;
- **vision** moves camera frames through DCI/DMA and bounded preprocessing;
- **inference** runs one selected runtime asset and publishes bounded results;
- **control** owns profile, interlock, debounce and actuator dispatch;
- **HMI** renders local state and accepts touch intents;
- **voice** validates fixed CI1302 frames and emits the same intents;
- **evidence** updates SPC/trace state without becoming an actuator authority;
- **watchdog** verifies liveness independently of model confidence.

Shared TCA and SPI access must be serialized. An inference deadline miss may
degrade or refuse a result; it must not stall the deterministic safety path.

