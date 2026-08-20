# Changelog

All notable public-release changes are documented here.

## [1.0.2] — 2026-08-20

### Media archive

- Added a content-complete, continuous 5:06 privacy-clean demonstration while
  retaining the five independently seekable chapters.
- Published the team portrait as an archival project record at the maintainer's
  explicit direction; badge names and barcodes remain redacted.
- Added exact-set Release checks, SHA-256 anchoring and binary privacy-marker
  gates for the complete media archive.
- Kept precise GPS, phone-model, capture-system and EXIF metadata out of the
  default public derivatives.
- Added a clearly named `privacy-sensitive` exact-source ZIP at the maintainer's
  explicit archival direction. It preserves all seven supplied source files,
  carries an internal warning/manifest, stays out of Git history and receives
  no default media-license grant.
- Kept the immutable v1.0.1 technical evidence package unchanged and linked it
  from the v1.0.2 media archive.

## [1.0.1] — 2026-08-19

### Security

- Removed the GPU worker's caller-selectable Python executable; child trainers
  now inherit the already-running, trusted interpreter.
- Restricted nano-LM experiment tags to filename-safe identifiers and enabled
  PyTorch's weights-only checkpoint loader on the affected public tools.
- Replaced a backtracking-prone privacy-scan expression with a linear character
  class and added explicit integer/range guards to two public C runtimes.
- Removed LVGL's unused optional LRU derivative after confirming that its cited
  upstream project did not define a redistribution license; also excluded the
  unused SDL/Tiny-TTF adapters that depend on it and added a hash gate to
  prevent accidental reintroduction.
- Reduced the vendored LVGL font snapshot to the four enabled Montserrat/Font
  Awesome sizes and added explicit OFL/CC attribution.
- Reduced lwIP contrib to the required FreeRTOS port and removed an unused
  desktop MIB compiler snapshot containing a private strong-name key.
- Enabled GitHub secret scanning, push protection, Dependabot updates, private
  vulnerability reporting, protected `main`, pinned Actions, and CodeQL.

## [1.0.0] — 2026-08-19 (withdrawn)

This first public tag was withdrawn during the same-day release audit. It must
not be restored or used as a dependency; `v1.0.1` is the first supported
public baseline.

### Added

- Post-competition archival release for the 2026 CIMC “Siemens Cup” Industrial
  Hardware R&D Problem 2 project.
- Curated GD32H759 firmware, board-runtime model code and original hardware
  design files.
- Machine-readable BOARD / HOST–EXACT / HOST–SIM_ONLY evidence ledgers.
- ICMat-Forge contracts and reproducibility pipeline with refusal evidence.
- Bilingual README, engineering documentation, safety policy and media gallery.
- Privacy-clean demonstration chapters and a reviewed technical Release asset.

### Intentionally excluded

Credentials, private endpoints, personal Keil state, build products, vendor
installers, competition templates, raw phone media with location/device
metadata, unlicensed OV5640 initialization code and SimHei-derived glyph data.

### Known limitations

170 HOST assets are not new board-runtime models. The board loader rejected the
catalog body on sustained-read CRC and executed none of those assets. The
frozen 30-asset / 28-logical-model board chain remains the public board count.
