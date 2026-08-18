"""Unit tests for the selective EasyEDA Pro metadata sanitizer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
import hashlib
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from sanitize_easyeda_pro import (  # noqa: E402
    NEUTRAL_USER,
    PUBLIC_TEAM_ID_SHA256,
    REDACTED_TEAM_ID,
    SanitizationError,
    _redact_exact_text_values,
    audit_archive,
    sanitize_archive,
)


class EasyEdaSanitizerTests(unittest.TestCase):
    SAMPLE_TEAM_ID = "0123456789"

    def _archive(self, root: Path, user: object) -> Path:
        path = root / "fixture.epro2"
        header = {"type": "DOCHEAD", "ticket": 1}
        payload = {
            "docType": "PCB",
            "uuid": "document-uuid-must-survive",
            "link": {"uuid": "linked-document-uuid-must-survive"},
            "text": self.SAMPLE_TEAM_ID,
            "user": user,
        }
        record = (
            json.dumps(header, separators=(",", ":"))
            + "||"
            + json.dumps(payload, separators=(",", ":"))
            + "|"
        ).encode()
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("project2.json", '{"title":"fixture"}')
            archive.writestr("design.epru", record)
            archive.writestr("IMAGE/pixel.bin", b"unchanged-binary")
        return path

    def test_only_user_metadata_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = self._archive(
                Path(temp),
                {
                    "uuid": "private-account-id",
                    "username": "private-user",
                    "nickname": "Private User",
                    "avatar": "/private/avatar.png",
                },
            )
            sample_digest = hashlib.sha256(self.SAMPLE_TEAM_ID.encode()).hexdigest()
            with patch("sanitize_easyeda_pro.PUBLIC_TEAM_ID_SHA256", sample_digest):
                before = audit_archive(source)
                result = sanitize_archive(source, source, overwrite=True)
                after = audit_archive(source)
            self.assertEqual(result.changed_user_fields, 1)
            self.assertEqual(result.changed_team_id_fields, 1)
            self.assertEqual(after.user_fields, after.neutral_user_fields)
            self.assertEqual(after.literal_team_id_hits, 0)
            self.assertEqual(after.redacted_team_id_fields, 1)
            self.assertEqual(before.semantic_digest, after.semantic_digest)
            self.assertEqual(before.non_user_uuid_digest, after.non_user_uuid_digest)
            self.assertEqual(before.raw_member_digests, after.raw_member_digests)
            with zipfile.ZipFile(source) as archive:
                raw = archive.read("design.epru")
                self.assertIn(b"document-uuid-must-survive", raw)
                self.assertIn(b"linked-document-uuid-must-survive", raw)
                self.assertIn(NEUTRAL_USER["uuid"].encode(), raw)
                self.assertIn(REDACTED_TEAM_ID.encode(), raw)
                self.assertNotIn(self.SAMPLE_TEAM_ID.encode(), raw)
                self.assertNotIn(b"private-user", raw)
                self.assertEqual(archive.read("IMAGE/pixel.bin"), b"unchanged-binary")
                self.assertIsNone(archive.testzip())

    def test_rejects_ambiguous_non_object_user_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = self._archive(Path(temp), "private-user")
            with self.assertRaises(SanitizationError):
                sanitize_archive(source, source, overwrite=True)

    def test_exact_text_redaction_does_not_rewrite_keys_or_uuids(self) -> None:
        sample = self.SAMPLE_TEAM_ID
        value = {
            sample: "key-is-preserved",
            "uuid": sample,
            "text": sample,
            "description": f"prefix-{sample}-suffix",
        }
        sample_digest = hashlib.sha256(sample.encode()).hexdigest()
        with patch("sanitize_easyeda_pro.PUBLIC_TEAM_ID_SHA256", sample_digest):
            encountered, changed = _redact_exact_text_values(value)
        self.assertEqual((encountered, changed), (1, 1))
        self.assertEqual(value["text"], REDACTED_TEAM_ID)
        self.assertEqual(value["uuid"], sample)
        self.assertIn(sample, value)
        self.assertEqual(value["description"], f"prefix-{sample}-suffix")


if __name__ == "__main__":
    unittest.main()
