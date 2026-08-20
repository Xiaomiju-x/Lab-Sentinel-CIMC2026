from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEAM_PHOTO = ROOT / "assets" / "competition" / "team-photo.webp"
RELEASE_CONTRACT = (
    ROOT / "release-manifests" / "v1.0.2" / "RELEASE_ASSETS.json"
)


class MediaContractTests(unittest.TestCase):
    def test_team_photo_is_metadata_clean_webp(self) -> None:
        payload = TEAM_PHOTO.read_bytes()
        self.assertTrue(payload.startswith(b"RIFF"))
        self.assertEqual(payload[8:12], b"WEBP")
        upper = payload.upper()
        self.assertNotIn(b"EXIF", upper)
        self.assertNotIn(b"XMP ", upper)
        for marker in (
            b"LOCATION",
            b"XIAOMI",
            b"REDMI",
            b"ANDROID",
            b"CREATION_TIME",
        ):
            self.assertNotIn(marker, upper)

    def test_v102_media_release_contract_is_exact(self) -> None:
        contract = json.loads(RELEASE_CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(contract["tag"], "v1.0.2")
        assets = contract["assets"]
        names = [item["name"] for item in assets]
        self.assertEqual(len(names), 16)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("00-full-demo-privacy-sanitized.mp4", names)
        self.assertIn("team-photo-sanitized.webp", names)
        self.assertIn(
            "lab-sentinel-original-media-sources-privacy-sensitive-v1.0.2.zip",
            names,
        )
        self.assertIn("SHA256SUMS.txt", names)
        for item in assets:
            self.assertGreater(item["bytes"], 0)
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(item["media_type"])
            self.assertTrue(item["role"])


if __name__ == "__main__":
    unittest.main()
