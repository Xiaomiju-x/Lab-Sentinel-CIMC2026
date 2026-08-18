.PHONY: verify test manifest sbom assets metadata release-check

verify:
	python tools/verify_release.py --strict
	python tools/check_local_links.py

test:
	python -m unittest discover -s tests -v
	python -m compileall -q tools tests ai/pipeline ai/tools

manifest:
	python tools/build_public_manifest.py

sbom:
	python tools/build_sbom.py

assets:
	python tools/build_asset_checksums.py

metadata: sbom manifest assets

release-check: metadata verify test
