# Build the public firmware snapshot

## Prerequisites

- Windows and Keil MDK capable of building the supplied GD32H759 project;
- GD32H7xx DFP 1.5.0 (the pack ID embedded in the project);
- an ARM Compiler 5 compatible Keil installation (the archived build used
  ARM Compiler 5.06 update 6, build 750);
- DAPLink/CMSIS-DAP for the user's own board;
- no vendor installer is mirrored by this repository.

## Build

1. Clone to a short path without unusual permissions.
2. Open `firmware/keil_proj/project/CIMC_GD32_Template.uvprojx`.
3. Select target **CIMC_GD32_Template**. “R2.1” is the archived competition
   release line, not the XML target name in this public project.
4. Confirm `firmware/ai_models_c/lab_build_config.h` before compiling. Do not
   change an acceptance define just to make a test pass.
5. Build and archive the build log plus `.map` outside Git.
6. Review Flash/RW/ZI against the frozen reference before flashing.

`tools/verify_release.py --strict` parses every public `.uvprojx` `FilePath`
and fails on a missing source. That is a **source-reference closure check**, not
a substitute for a licensed Keil compile. This archive was not rebuilt in CI;
save the compiler version, build log and map from your local licensed toolchain.

Release qualification on 2026-08-19 used a clean Keil UV4 rebuild of the
public `CIMC_GD32_Template` target with ARM Compiler 5.06 update 6 (build 750).
It completed with **0 errors and 24 warnings**; the linker reported
`Code=253268`, `RO-data=2921220`, `RW-data=16980`, and `ZI-data=440988` bytes.
The generated AXF, map, dependency files and local build log are intentionally
excluded from Git because they contain machine-specific paths; rebuild them
locally instead of treating an archived binary as proof of reproducibility.

The public source omits an unlicensed OV5640 initialization table, a
SimHei-derived CJK bitmap, and LVGL's unused optional LRU derivative. A
camera-disabled adapter returns an explicit probe failure, the default font
header maps to an ASCII LVGL font, and the unused SDL/Tiny-TTF adapters that
depend on the excluded cache are not redistributed. The retained font sources
are limited to the four Montserrat/Font Awesome sizes used by this Keil target;
their OFL/CC notices are in `THIRD_PARTY_NOTICES.md`. Follow the README files in
those source directories to provide permissively licensed live-camera and OFL
CJK implementations. Never label a fallback frame as live input or restore
excluded code without a documented redistribution license.

The lwIP copy is likewise a target-specific subset: core runtime sources plus
the FreeRTOS port used by the Keil project. Desktop contrib examples and tools
are deliberately not part of the public firmware build.

## Flashing boundary

Flashing changes physical device state and is never part of CI. Verify target,
power, SWD connection and recovery path locally. The competition setup used the
existing DAPLink/CMSIS-DAP configuration and a 115200 8N1 debug UART.
