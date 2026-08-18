# Expected hashes and manifests

The authoritative machine-readable inventory is generated at release time:

- `PUBLIC_RELEASE_MANIFEST.json` — path, bytes and SHA-256 for the Git tree;
- `SBOM.spdx.json` — source-package/file inventory and declared licenses;
- GitHub Release checksums — large technical package and video chapters;
- evidence JSON under `evidence/public/` — project-specific content roots and
  acceptance state.

Run:

```bash
python tools/build_public_manifest.py --check
python tools/verify_release.py --strict
```

Do not copy a hash from this prose into a new artifact. Regenerate the manifest
after any approved change and review the diff before tagging.

