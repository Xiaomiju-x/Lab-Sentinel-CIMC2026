# Third-party notices

This file records components intentionally present in the public tree. It is
not a substitute for each component's own copyright header or license file.

| Component | Location | License / notice |
|---|---|---|
| Arm CMSIS | `firmware/keil_proj/CMSIS/` | Apache-2.0; upstream license retained |
| GD32H7xx firmware library, demo-derived support and startup files | `firmware/keil_proj/Library/`, selected `CMSIS/` and `User/` support files, `Startup/` | Original GigaDevice BSD-3-Clause / Apache-2.0 notices retained per source file |
| FreeRTOS kernel | `firmware/keil_proj/FreeRTOS/` | MIT; upstream headers retained |
| lwIP runtime subset + FreeRTOS port | `firmware/keil_proj/lwip/src/`, `contrib/ports/freertos/` | BSD-3-Clause; `COPYING` retained; unused contrib tools/examples excluded |
| LVGL 8.3.11 source subset | `firmware/lvgl_ui/lvgl-8.3.11/` | MIT for LVGL-authored files; `LICENCE.txt` retained; nested components are itemized below; optional unlicensed LRU derivative excluded |
| Montserrat glyphs embedded in the four retained LVGL fonts | `firmware/lvgl_ui/lvgl-8.3.11/src/font/lv_font_montserrat_{14,18,20,28}.c` | SIL Open Font License 1.1; Copyright 2011 The Montserrat Project Authors; generated/subsetted from [Montserrat](https://github.com/JulietaUla/Montserrat); OFL text retained |
| Font Awesome 5 Free glyphs embedded in those LVGL fonts | same four generated font sources | Font files under SIL Open Font License 1.1; icon artwork under CC BY 4.0; Copyright Fonticons, Inc.; generated/subsetted from [Font Awesome Free](https://github.com/FortAwesome/Font-Awesome); [CC BY 4.0 text](LICENSES/CC-BY-4.0.txt) retained |
| Arm-2D LVGL integration source | `firmware/lvgl_ui/lvgl-8.3.11/src/draw/arm2d/lv_gpu_arm2d.c` | Apache-2.0; Arm copyright and SPDX header retained |
| LodePNG | `firmware/lvgl_ui/lvgl-8.3.11/src/extra/libs/png/lodepng.{c,h}` | zlib license; source notice retained |
| TJpgDec R0.03 | `firmware/lvgl_ui/lvgl-8.3.11/src/extra/libs/sjpg/tjpgd*` | ChaN permissive license; source notice retained; represented as `LicenseRef-TJpgDec` in the SPDX inventory |
| TLSF allocator 3.1 | `firmware/lvgl_ui/lvgl-8.3.11/src/misc/lv_tlsf.{c,h}` | BSD-3-Clause; Matthew Conte copyright and notice retained |
| mpaland embedded printf | `firmware/lvgl_ui/lvgl-8.3.11/src/misc/lv_printf.{c,h}` | MIT; Copyright Marco Paland / PALANDesign, 2014–2019; full source notice retained |
| NXP LVGL PXP and VG-Lite adapters | `firmware/lvgl_ui/lvgl-8.3.11/src/draw/nxp/` | MIT; Copyright NXP, 2020–2023; full source notices retained; adapters are disabled in the selected GD32 build |
| FatFs R0.16p2 | historical competition firmware dependency; source not redistributed in this Git tree | FatFs permissive license; obtain from the upstream/vendor delivery |
| Melexis MLX90640 API | historical competition firmware dependency; source not redistributed in this Git tree | Apache-2.0 upstream implementation; obtain from Melexis |

## Intentionally excluded

- **OV5640 initialization table:** an early file stated that its table was
  adapted from a vendor tutorial, but a redistribution license could not be
  established. The public tree contains the project interface and integration
  notes, not that table. Users must supply a licensed implementation.
- **SimHei-derived embedded bitmap:** the development generator used a Windows
  system font. The generated bitmap is excluded. The public font workflow uses
  an OFL-licensed Noto/Source Han font supplied by the user.
- **Bundled LVGL SimSun/Korean font artifacts:** upstream source snapshots
  contained font binaries/generated glyph data whose independent font rights
  were not clear enough for this release. They are excluded without affecting
  the Montserrat source fonts used by the selected configuration.
- **Arial test font:** LVGL's FreeType example snapshot included a Microsoft /
  Monotype `arial.ttf` whose embedded terms prohibit redistribution. It is
  excluded; examples must use a separately licensed OFL font supplied by the
  user.
- **Optional LVGL LRU implementation:** LVGL 8.3.11 identifies its optional
  `lv_lru.c/.h` implementation as modified from C-LRU-Cache, whose upstream
  repository defines no redistribution license. The selected GD32 build uses
  neither SDL rendering nor Tiny-TTF, so those files, their unused adapters and
  the Keil source reference are excluded. Do not restore them without
  documented permission.
- **LVGL font subset:** only the Montserrat 14/18/20/28 generated sources used
  by the selected Keil target are retained. The unused DejaVu and Unscii
  generated sources are excluded to keep the public license surface minimal.
- **lwIP contrib subset:** only the FreeRTOS port required by the selected Keil
  target is retained. Unused desktop examples/tools are excluded, including an
  old MIB compiler snapshot that contained a private strong-name key and
  separately licensed SharpSnmpLib sources. The key is removed; this project
  does not trust or support artifacts signed with it.
- **Disabled codec/network helpers:** the selected target has LVGL GIF and QR
  widgets, lwIP PPP/PPPoE, and the host-only `makefsdata` utility disabled.
  Their unused source trees are excluded so the public archive does not imply
  a build or license surface that the project does not exercise.
- **Keil MDK, GD32 Embedded AI Tool, DAPLink packages, datasheets, competition
  templates and vendor installers:** obtain them from their official sources.
- **API teachers, remote GPU credentials and private endpoints:** none are part
  of this release. Teacher outputs are never treated as ground truth.

External datasets and model terms are documented separately in
`DATA_LICENSES.md` and `MODEL_LICENSES.md`.
