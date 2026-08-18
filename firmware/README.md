# Firmware

This directory is the curated GD32H759 source snapshot used by Lab-Sentinel.
The primary project is `keil_proj/project/CIMC_GD32_Template.uvprojx`, target
`CIMC_GD32_Template` (the archived competition release line was R2.1).
`ai_models_c/` contains the small board-runtime implementations and
weight headers; `lvgl_ui/` contains the selected LVGL source subset.

Build products, user-specific Keil state, vendor installers, the unlicensed
OV5640 initialization table and SimHei-derived CJK bitmap are intentionally
excluded. The public tree substitutes an explicit camera-disabled adapter and
an ASCII font fallback; live camera and Chinese glyphs require reviewed
replacements. Read `docs/03-firmware/build-keil-r21.md` before building.

All learned code paths remain `authority=0`. Source availability is not a
license to connect the prototype to a real furnace.
