#!/usr/bin/env python3
"""Create a deterministic SPDX 2.3 file-inventory SBOM.

This is deliberately a mixed-license inventory: files are assigned to the
project or a known vendored component by path. Unknown/photographic rights stay
NOASSERTION instead of being silently relicensed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from _common import ROOT, iter_public_files, sha256_file


OUTPUT = ROOT / "SBOM.spdx.json"
CREATED = "2026-08-19T00:00:00Z"


@dataclass(frozen=True)
class PackageRule:
    spdx_id: str
    name: str
    version: str
    license_id: str
    prefixes: tuple[str, ...]


RULES = (
    PackageRule(
        "SPDXRef-Package-GD32-SPL",
        "GD32H7xx firmware library and demo-derived support",
        "1.4.0",
        "BSD-3-Clause",
        (
            "firmware/keil_proj/Library/",
            "firmware/keil_proj/CMSIS/gd32h7xx_libopt.h",
            "firmware/keil_proj/User/gd32h7xx_it.c",
            "firmware/keil_proj/User/gd32h7xx_it.h",
            "firmware/keil_proj/User/gd32h7xx_libopt.h",
            "firmware/keil_proj/User/systick.c",
            "firmware/keil_proj/User/systick.h",
        ),
    ),
    PackageRule("SPDXRef-Package-CMSIS", "Arm CMSIS + GD32 CMSIS device support", "archived", "Apache-2.0", ("firmware/keil_proj/CMSIS/",)),
    PackageRule("SPDXRef-Package-GD32-Startup", "GD32H7xx startup files", "1.4.0", "Apache-2.0", ("firmware/keil_proj/Startup/",)),
    PackageRule("SPDXRef-Package-FreeRTOS", "FreeRTOS Kernel", "archived", "MIT", ("firmware/keil_proj/FreeRTOS/",)),
    PackageRule("SPDXRef-Package-lwIP", "lwIP", "archived", "BSD-3-Clause", ("firmware/keil_proj/lwip/",)),
    PackageRule(
        "SPDXRef-Package-LVGL-Fonts",
        "LVGL generated Montserrat and Font Awesome glyph subset",
        "8.3.11-generated",
        "OFL-1.1 AND CC-BY-4.0",
        (
            "firmware/lvgl_ui/lvgl-8.3.11/src/font/lv_font_montserrat_14.c",
            "firmware/lvgl_ui/lvgl-8.3.11/src/font/lv_font_montserrat_18.c",
            "firmware/lvgl_ui/lvgl-8.3.11/src/font/lv_font_montserrat_20.c",
            "firmware/lvgl_ui/lvgl-8.3.11/src/font/lv_font_montserrat_28.c",
        ),
    ),
    PackageRule(
        "SPDXRef-Package-Arm2D",
        "Arm-2D LVGL integration source",
        "archived",
        "Apache-2.0",
        ("firmware/lvgl_ui/lvgl-8.3.11/src/draw/arm2d/lv_gpu_arm2d.c",),
    ),
    PackageRule(
        "SPDXRef-Package-LodePNG",
        "LodePNG",
        "20201017",
        "Zlib",
        (
            "firmware/lvgl_ui/lvgl-8.3.11/src/extra/libs/png/lodepng.c",
            "firmware/lvgl_ui/lvgl-8.3.11/src/extra/libs/png/lodepng.h",
        ),
    ),
    PackageRule(
        "SPDXRef-Package-TJpgDec",
        "TJpgDec",
        "R0.03",
        "LicenseRef-TJpgDec",
        (
            "firmware/lvgl_ui/lvgl-8.3.11/src/extra/libs/sjpg/tjpgd.c",
            "firmware/lvgl_ui/lvgl-8.3.11/src/extra/libs/sjpg/tjpgd.h",
            "firmware/lvgl_ui/lvgl-8.3.11/src/extra/libs/sjpg/tjpgdcnf.h",
        ),
    ),
    PackageRule(
        "SPDXRef-Package-TLSF",
        "TLSF memory allocator",
        "3.1",
        "BSD-3-Clause",
        (
            "firmware/lvgl_ui/lvgl-8.3.11/src/misc/lv_tlsf.c",
            "firmware/lvgl_ui/lvgl-8.3.11/src/misc/lv_tlsf.h",
        ),
    ),
    PackageRule(
        "SPDXRef-Package-mpaland-printf",
        "mpaland embedded printf",
        "archived",
        "MIT",
        (
            "firmware/lvgl_ui/lvgl-8.3.11/src/misc/lv_printf.c",
            "firmware/lvgl_ui/lvgl-8.3.11/src/misc/lv_printf.h",
        ),
    ),
    PackageRule(
        "SPDXRef-Package-NXP-LVGL-Adapters",
        "NXP LVGL PXP and VG-Lite adapters",
        "8.3.11-archived",
        "MIT",
        ("firmware/lvgl_ui/lvgl-8.3.11/src/draw/nxp/",),
    ),
    PackageRule("SPDXRef-Package-LVGL", "LVGL", "8.3.11", "MIT", ("firmware/lvgl_ui/lvgl-8.3.11/",)),
    PackageRule("SPDXRef-Package-Competition-Media", "Lab-Sentinel competition-photo derivatives", "1.0.1", "NOASSERTION", ("assets/competition/",)),
    PackageRule(
        "SPDXRef-Package-Docs-Media",
        "Lab-Sentinel documentation and licensed media",
        "1.0.1",
        "CC-BY-4.0",
        (
            "docs/",
            "assets/",
            "hardware/README.md",
            "hardware/design/README.md",
            "firmware/keil_proj/lwip/PUBLIC_SUBSET.md",
        ),
    ),
    PackageRule("SPDXRef-Package-Hardware", "Lab-Sentinel original hardware design", "1.0.1", "CERN-OHL-S-2.0", ("hardware/",)),
    PackageRule("SPDXRef-Package-License-Metadata", "License texts and release-rights metadata", "1.0.1", "NOASSERTION", ("LICENSE", "LICENSES/")),
    PackageRule("SPDXRef-Package-Software", "Lab-Sentinel original software and release tooling", "1.0.1", "Apache-2.0", ()),
)


def rule_for(relative: str) -> PackageRule:
    if relative in {
        "hardware/README.md",
        "hardware/design/README.md",
        "firmware/keil_proj/lwip/PUBLIC_SUBSET.md",
    }:
        return next(
            rule for rule in RULES
            if rule.spdx_id == "SPDXRef-Package-Docs-Media"
        )
    if "/" not in relative and (relative.endswith(".md") or relative == "CITATION.cff"):
        return next(rule for rule in RULES if rule.spdx_id == "SPDXRef-Package-Docs-Media")
    for rule in RULES:
        if rule.prefixes and relative.startswith(rule.prefixes):
            return rule
    return RULES[-1]


def package_json(rule: PackageRule) -> dict:
    return {
        "name": rule.name,
        "SPDXID": rule.spdx_id,
        "versionInfo": rule.version,
        "downloadLocation": "https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026",
        "filesAnalyzed": True,
        "licenseConcluded": rule.license_id,
        "licenseDeclared": rule.license_id,
        "copyrightText": "NOASSERTION",
    }


def main() -> int:
    files = []
    relationships = []
    used_packages: set[str] = set()
    for index, path in enumerate(iter_public_files(), start=1):
        relative = path.relative_to(ROOT).as_posix()
        spdx_id = f"SPDXRef-File-{index}"
        rule = rule_for(relative)
        used_packages.add(rule.spdx_id)
        files.append({
            "SPDXID": spdx_id,
            "fileName": f"./{relative}",
            "checksums": [{"algorithm": "SHA256", "checksumValue": sha256_file(path)}],
            "licenseConcluded": rule.license_id,
            "licenseInfoInFiles": rule.license_id.split(" AND "),
            "copyrightText": "NOASSERTION",
        })
        relationships.append({
            "spdxElementId": rule.spdx_id,
            "relationshipType": "CONTAINS",
            "relatedSpdxElement": spdx_id,
        })

    packages = [package_json(rule) for rule in RULES if rule.spdx_id in used_packages]
    describes = [{
        "spdxElementId": "SPDXRef-DOCUMENT",
        "relationshipType": "DESCRIBES",
        "relatedSpdxElement": package["SPDXID"],
    } for package in packages]
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "Lab-Sentinel-CIMC2026-v1.0.1-file-inventory",
        "documentNamespace": "https://github.com/Xiaomiju-x/Lab-Sentinel-CIMC2026/sbom/v1.0.1",
        "creationInfo": {
            "created": CREATED,
            "creators": ["Tool: Lab-Sentinel-build_sbom.py"],
        },
        "documentComment": (
            "Path-based file-inventory SBOM. See THIRD_PARTY_NOTICES.md and the "
            "data/model/media license files for authoritative reuse boundaries."
        ),
        "packages": packages,
        "files": files,
        "relationships": [*describes, *relationships],
        "hasExtractedLicensingInfos": [{
            "licenseId": "LicenseRef-TJpgDec",
            "name": "TJpgDec permissive license",
            "extractedText": (
                "Copyright (C) 2021, ChaN, all right reserved.\n\n"
                "* The TJpgDec module is a free software and there is NO WARRANTY.\n"
                "* No restriction on use. You can use, modify and redistribute it for\n"
                "  personal, non-profit or commercial products UNDER YOUR RESPONSIBILITY.\n"
                "* Redistributions of source code must retain the above copyright notice."
            ),
        }],
    }
    OUTPUT.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"WROTE {OUTPUT}: {len(files)} files in {len(packages)} packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
