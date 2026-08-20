# v1.0.2 release notes

Lab-Sentinel v1.0.2 completes the public project-media archive without changing
the frozen technical claims or board/host ledgers.

## Highlights

- One continuous 5:06, 720p H.264/AAC privacy-clean demonstration;
- the same complete demonstration split into five independently seekable
  chapters with five cover images;
- an identifier-redacted, metadata-clean team portrait as a competition record;
- SHA-256 checksums, a tag-anchored outer manifest and Release exact-set checks;
- unchanged truth boundaries: 30 BOARD assets / 28 logical models and 170 HOST
  assets (78 EXACT + 92 SIM_ONLY), all HOST authority=0.

## Release assets

- `00-full-demo-privacy-sanitized.mp4` — continuous complete demonstration;
- five metadata-clean 720p chapters and their cover images;
- `team-photo-sanitized.webp` — Release copy of the identifier-redacted team
  portrait also tracked under `assets/competition/`;
- `lab-sentinel-original-media-sources-privacy-sensitive-v1.0.2.zip` and its
  sidecar — exact-byte archive of all seven supplied source media files, with an
  internal warning and manifest; it may retain precise metadata and identifiers
  and is not covered by the default media license;
- `MEDIA_MANIFEST.md` — media provenance, transformation and rights boundary;
- `SHA256SUMS.txt` — checksums for every attached asset.

The release tag anchors the outer asset manifest at
`release-manifests/v1.0.2/SHA256SUMS.txt.sha256`. Its SHA-256 is recorded in
that tag-anchored file.

The reviewed technical evidence ZIP remains unchanged in the immutable
[`v1.0.1` Release](https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/releases/tag/v1.0.1), independently anchored by its adjacent `.sha256` sidecar and
`docs/06-reproduction/host-smoke.md`.

The repository verifies source/reference integrity in CI. A licensed Keil/GD32
toolchain and physical board are still required for a real firmware build and
board acceptance; CI never represents that level as completed.

## Truth boundary

The 170 HOST assets were prepositioned as files on microSD. A sustained catalog
read CRC failure stopped the board loader before entry/payload load; new board
execution is zero. The public board count remains 30 assets / 28 logical models.
