# v1.0.1 release notes

Lab-Sentinel v1.0.1 is the clean-history post-competition archival release.

## Highlights

- Curated GD32H759 source with all public Keil project references closed,
  plus explicit camera-disabled and ASCII-font fallbacks for material whose
  redistribution rights could not be established;
- 30 BOARD runtime assets / 28 logical models with true-board evidence;
- 170 HOST research assets with explicit 78 EXACT / 92 SIM_ONLY separation;
- ICMat-Forge, SinterGraph and VeriProcess contracts, pipelines and receipts;
- bilingual landing READMEs, English engineering documentation and
  privacy-clean media;
- source/data/model/media license separation, SBOM and secret/path gates.

## Release assets

- `lab-sentinel-cimc2026-technical-evidence-v1.0.1.zip` — reviewed competition
  technical artifact bundle, too large for ordinary Git;
- five metadata-clean 720p demonstration chapters and cover images;
- `SHA256SUMS.txt` — checksums for every attached asset.

The release tag anchors the outer asset manifest at
`release-manifests/v1.0.1/SHA256SUMS.txt.sha256`. Its SHA-256 is recorded in
that tag-anchored file.
The technical evidence ZIP is independently anchored at
the SHA-256 printed in its adjacent `.sha256` sidecar and in
`docs/06-reproduction/host-smoke.md`.

The repository verifies source/reference integrity in CI. A licensed Keil/GD32
toolchain and physical board are still required for a real firmware build and
board acceptance; CI never represents that level as completed.

## Truth boundary

The 170 HOST assets were prepositioned as files on microSD. A sustained catalog
read CRC failure stopped the board loader before entry/payload load; new board
execution is zero. The public board count remains 30 assets / 28 logical models.
