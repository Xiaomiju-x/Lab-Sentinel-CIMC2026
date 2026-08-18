# Memory and timing budget

The frozen competition image is a dense MCU build, not a claim that all HOST
assets are resident. The archived evidence reports approximately:

- linked ROM image: 3,358,532 bytes in the competition linker configuration;
- static RW + ZI: about 672.4 KB;
- latest observed minimum heap: 22,968 bytes;
- critical-stack remaining margin: approximately 2.38 KB;
- deterministic control p99 target/measurement boundary: 100 ms.

These values belong to the frozen 30-asset package and its exact map/runtime
measurement. They must not be applied to the 170 HOST assets. External 32 MB
SDRAM and microSD storage are not counted as on-chip SRAM or Flash.

Rebuilds can change the map. Always archive the `.map`, build configuration,
tool version and board log together.

