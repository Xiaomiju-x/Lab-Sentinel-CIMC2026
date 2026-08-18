#!/usr/bin/env python3
"""Remove narrowly scoped private metadata from an EasyEDA Pro archive.

The sanitizer is deliberately narrow: it replaces the value of every nested
JSON ``user`` key with one neutral UUID.  It also replaces the exact public
competition-registration token only when it is the complete string value of an
EasyEDA ``text`` field.  It never rewrites JSON keys, ``uuid`` values, partial
string matches, numbers or arbitrary strings.  All other JSON values—including
document, component, symbol and link UUIDs—remain unchanged.

The output archive is written to a temporary file, verified, and then installed
with ``os.replace`` so an interrupted run cannot leave a partial archive at the
destination.  Verification fails closed if the registration token remains in
any member, rather than broadening the redaction beyond the approved text field.

Examples::

    python tools/sanitize_easyeda_pro.py --check
    python tools/sanitize_easyeda_pro.py --in-place
    python tools/sanitize_easyeda_pro.py source.epro2 --output public.epro2
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from _common import ROOT


DEFAULT_ARCHIVE = ROOT / "hardware/design/lab-sentinel-hardware.epro2"
NEUTRAL_USER = {"uuid": "00000000000000000000000000000000"}
# Store only a one-way digest of the private registration number.  This keeps
# the public sanitizer capable of detecting/redacting the exact identifier
# without re-publishing that identifier in source code.
PUBLIC_TEAM_ID_SHA256 = "25ab08fca36dbd68b0ddbe372a972491da3d255576744b835176b90cb9fdcb15"
REDACTED_TEAM_ID = "TEAM-ID-REDACTED"
JSON_SUFFIXES = {".json"}
TEN_DIGIT_TOKEN = re.compile(rb"(?<![0-9])[0-9]{10}(?![0-9])")


class SanitizationError(RuntimeError):
    """Raised when an archive cannot be sanitized without ambiguity."""


@dataclass(frozen=True)
class ArchiveAudit:
    members: tuple[str, ...]
    epru_members: int
    json_records: int
    user_fields: int
    neutral_user_fields: int
    exposed_team_id_fields: int
    redacted_team_id_fields: int
    literal_team_id_hits: int
    semantic_digest: str
    non_user_uuid_digest: str
    raw_member_digests: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SanitizeResult:
    destination: Path
    user_fields: int
    changed_user_fields: int
    changed_team_id_fields: int
    removed_identity_token_hits: int
    semantic_digest: str
    non_user_uuid_digest: str


def _is_public_team_id(value: str) -> bool:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() == PUBLIC_TEAM_ID_SHA256


def _count_public_team_id_bytes(raw: bytes) -> int:
    return sum(
        hashlib.sha256(match.group(0)).hexdigest() == PUBLIC_TEAM_ID_SHA256
        for match in TEN_DIGIT_TOKEN.finditer(raw)
    )


def _split_line_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1], line[-1]
    return line, ""


def _neutralize_user_fields(value: Any) -> tuple[int, int, set[str]]:
    """Mutate only ``user`` dictionary values and return audit counters."""
    encountered = 0
    changed = 0
    identity_tokens: set[str] = set()

    def walk(node: Any) -> None:
        nonlocal encountered, changed
        if isinstance(node, dict):
            for key in list(node):
                child = node[key]
                if key == "user":
                    if not isinstance(child, dict):
                        raise SanitizationError(
                            f"refusing ambiguous non-object user field: {type(child).__name__}"
                        )
                    encountered += 1
                    # UUID-only user objects sometimes reuse a document UUID.  Do
                    # not treat those UUIDs as standalone identity-search tokens,
                    # because their document occurrence must remain untouched.
                    rich_identity = any(
                        field in child for field in ("username", "nickname", "avatar")
                    )
                    if rich_identity:
                        for token in child.values():
                            if isinstance(token, str) and len(token) >= 4:
                                identity_tokens.add(token)
                    if child != NEUTRAL_USER:
                        node[key] = dict(NEUTRAL_USER)
                        changed += 1
                    continue
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return encountered, changed, identity_tokens


def _redact_exact_text_values(value: Any) -> tuple[int, int]:
    """Redact only an exact registration token stored under a ``text`` key.

    Keys, UUID fields, list items, partial matches, and all other string-valued
    properties are intentionally outside this transformation.
    """
    encountered = 0
    changed = 0

    def walk(node: Any) -> None:
        nonlocal encountered, changed
        if isinstance(node, dict):
            for key in list(node):
                child = node[key]
                if key == "text" and isinstance(child, str):
                    if _is_public_team_id(child):
                        encountered += 1
                        node[key] = REDACTED_TEAM_ID
                        changed += 1
                    elif child == REDACTED_TEAM_ID:
                        encountered += 1
                    continue
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    walk(value)
    return encountered, changed


def _masked_semantics(value: Any) -> Any:
    """Return a semantic view in which only user metadata is masked."""
    if isinstance(value, dict):
        masked: dict[str, Any] = {}
        for key, child in value.items():
            if key == "user":
                masked[key] = "<USER_METADATA>"
            elif (
                key == "text"
                and isinstance(child, str)
                and (_is_public_team_id(child) or child == REDACTED_TEAM_ID)
            ):
                masked[key] = "<TEAM_ID_METADATA>"
            else:
                masked[key] = _masked_semantics(child)
        return masked
    if isinstance(value, list):
        return [_masked_semantics(child) for child in value]
    return value


def _collect_non_user_uuids(value: Any, path: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    records: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "user":
                continue
            child_path = path + (key,)
            if key == "uuid":
                records.append(("/".join(child_path), child))
            records.extend(_collect_non_user_uuids(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(_collect_non_user_uuids(child, path + (str(index),)))
    return records


def _parse_epru(raw: bytes, member: str) -> tuple[bool, list[tuple[str, str, Any]]]:
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    body = raw[3:] if has_bom else raw
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SanitizationError(f"{member}: not valid UTF-8: {exc}") from exc

    records: list[tuple[str, str, Any]] = []
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        content, ending = _split_line_ending(line)
        if "||" not in content:
            raise SanitizationError(f"{member}:{line_number}: unsupported .epru record framing")
        framed = content[:-1] if content.endswith("|") else content
        first, second = framed.split("||", 1)
        for part_number, encoded in enumerate((first, second), start=1):
            if not encoded:
                records.append((f"line:{line_number}:part:{part_number}", ending, None))
                continue
            try:
                value = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise SanitizationError(
                    f"{member}:{line_number}:part:{part_number}: invalid JSON: {exc}"
                ) from exc
            records.append((f"line:{line_number}:part:{part_number}", ending, value))
    return has_bom, records


def _transform_epru(
    raw: bytes, member: str
) -> tuple[bytes, int, int, set[str], int]:
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    body = raw[3:] if has_bom else raw
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SanitizationError(f"{member}: not valid UTF-8: {exc}") from exc

    output: list[str] = []
    encountered = 0
    changed = 0
    changed_team_ids = 0
    identity_tokens: set[str] = set()
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        content, ending = _split_line_ending(line)
        if "||" not in content:
            raise SanitizationError(f"{member}:{line_number}: unsupported .epru record framing")
        has_terminal_pipe = content.endswith("|")
        framed = content[:-1] if has_terminal_pipe else content
        parts = list(framed.split("||", 1))
        for part_number, encoded in enumerate(parts, start=1):
            if not encoded:
                continue
            try:
                value = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise SanitizationError(
                    f"{member}:{line_number}:part:{part_number}: invalid JSON: {exc}"
                ) from exc
            found, modified, tokens = _neutralize_user_fields(value)
            encountered += found
            changed += modified
            identity_tokens.update(tokens)
            _, modified_team_ids = _redact_exact_text_values(value)
            changed_team_ids += modified_team_ids
            if modified or modified_team_ids:
                parts[part_number - 1] = json.dumps(
                    value, ensure_ascii=False, separators=(",", ":")
                )
        output.append("||".join(parts) + ("|" if has_terminal_pipe else "") + ending)
    encoded_output = "".join(output).encode("utf-8")
    if has_bom:
        encoded_output = b"\xef\xbb\xbf" + encoded_output
    return encoded_output, encountered, changed, identity_tokens, changed_team_ids


def _transform_json(
    raw: bytes, member: str
) -> tuple[bytes, int, int, set[str], int]:
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    body = raw[3:] if has_bom else raw
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SanitizationError(f"{member}: invalid JSON member: {exc}") from exc
    encountered, changed, tokens = _neutralize_user_fields(value)
    _, changed_team_ids = _redact_exact_text_values(value)
    if not changed and not changed_team_ids:
        return raw, encountered, changed, tokens, changed_team_ids
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if has_bom:
        encoded = b"\xef\xbb\xbf" + encoded
    return encoded, encountered, changed, tokens, changed_team_ids


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    infos = archive.infolist()
    seen: set[str] = set()
    for info in infos:
        name = info.filename.replace("\\", "/")
        pure = PurePosixPath(name)
        if info.flag_bits & 0x1:
            raise SanitizationError(f"encrypted archive member is not supported: {name}")
        if pure.is_absolute() or ".." in pure.parts or (pure.parts and ":" in pure.parts[0]):
            raise SanitizationError(f"unsafe archive member path: {name}")
        folded = name.casefold()
        if folded in seen:
            raise SanitizationError(f"duplicate archive member path: {name}")
        seen.add(folded)
    bad_member = archive.testzip()
    if bad_member is not None:
        raise SanitizationError(f"CRC failure in archive member: {bad_member}")
    return infos


def _canonical_digest(records: Iterable[Any]) -> str:
    encoded = json.dumps(
        list(records), ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def audit_archive(path: Path) -> ArchiveAudit:
    semantic_records: list[Any] = []
    uuid_records: list[Any] = []
    raw_member_digests: list[tuple[str, str]] = []
    user_fields = 0
    neutral_user_fields = 0
    exposed_team_id_fields = 0
    redacted_team_id_fields = 0
    literal_team_id_hits = 0
    json_records = 0
    epru_members = 0

    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise SanitizationError(f"cannot open {path}: {exc}") from exc
    with archive:
        infos = _safe_members(archive)
        members = tuple(info.filename for info in infos)
        for info in infos:
            raw = archive.read(info)
            literal_team_id_hits += _count_public_team_id_bytes(raw)
            suffix = Path(info.filename).suffix.lower()
            parsed: list[tuple[str, Any]] = []
            if suffix == ".epru":
                epru_members += 1
                _, records = _parse_epru(raw, info.filename)
                parsed = [(location, value) for location, _, value in records if value is not None]
            elif suffix in JSON_SUFFIXES and not info.is_dir():
                has_bom = raw.startswith(b"\xef\xbb\xbf")
                body = raw[3:] if has_bom else raw
                try:
                    parsed = [("document", json.loads(body.decode("utf-8")))]
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise SanitizationError(f"{info.filename}: invalid JSON member: {exc}") from exc
            else:
                raw_member_digests.append((info.filename, hashlib.sha256(raw).hexdigest()))

            for location, value in parsed:
                json_records += 1

                def inspect(node: Any) -> None:
                    nonlocal user_fields, neutral_user_fields
                    nonlocal exposed_team_id_fields, redacted_team_id_fields
                    if isinstance(node, dict):
                        for key, child in node.items():
                            if key == "user":
                                if not isinstance(child, dict):
                                    raise SanitizationError(
                                        f"{info.filename}:{location}: non-object user field"
                                    )
                                user_fields += 1
                                if child == NEUTRAL_USER:
                                    neutral_user_fields += 1
                                continue
                            if key == "text" and isinstance(child, str):
                                if _is_public_team_id(child):
                                    exposed_team_id_fields += 1
                                elif child == REDACTED_TEAM_ID:
                                    redacted_team_id_fields += 1
                            inspect(child)
                    elif isinstance(node, list):
                        for child in node:
                            inspect(child)

                inspect(value)
                semantic_records.append(
                    (info.filename, location, _masked_semantics(value))
                )
                uuid_records.extend(
                    (info.filename, location, key_path, uuid)
                    for key_path, uuid in _collect_non_user_uuids(value)
                )

    if epru_members == 0:
        raise SanitizationError("archive contains no .epru design member")
    if user_fields == 0:
        raise SanitizationError("archive contains no user metadata fields to audit")
    return ArchiveAudit(
        members=members,
        epru_members=epru_members,
        json_records=json_records,
        user_fields=user_fields,
        neutral_user_fields=neutral_user_fields,
        exposed_team_id_fields=exposed_team_id_fields,
        redacted_team_id_fields=redacted_team_id_fields,
        literal_team_id_hits=literal_team_id_hits,
        semantic_digest=_canonical_digest(semantic_records),
        non_user_uuid_digest=_canonical_digest(uuid_records),
        raw_member_digests=tuple(raw_member_digests),
    )


def _count_token_hits(path: Path, tokens: set[str]) -> int:
    if not tokens:
        return 0
    hits = 0
    with zipfile.ZipFile(path) as archive:
        for info in _safe_members(archive):
            raw = archive.read(info)
            for token in tokens:
                hits += raw.count(token.encode("utf-8"))
    return hits


def sanitize_archive(source: Path, destination: Path, *, overwrite: bool = False) -> SanitizeResult:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise SanitizationError(f"source archive does not exist: {source}")
    if destination.exists() and destination != source and not overwrite:
        raise SanitizationError(f"destination exists; pass --force to replace it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    before = audit_archive(source)
    total_users = 0
    changed_users = 0
    changed_team_ids = 0
    identity_tokens: set[str] = set()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with zipfile.ZipFile(source) as input_archive:
            infos = _safe_members(input_archive)
            with zipfile.ZipFile(temporary_path, "w", allowZip64=True) as output_archive:
                output_archive.comment = input_archive.comment
                for info in infos:
                    raw = input_archive.read(info)
                    suffix = Path(info.filename).suffix.lower()
                    if suffix == ".epru":
                        raw, encountered, changed, tokens, text_redactions = _transform_epru(
                            raw, info.filename
                        )
                    elif suffix in JSON_SUFFIXES and not info.is_dir():
                        raw, encountered, changed, tokens, text_redactions = _transform_json(
                            raw, info.filename
                        )
                    else:
                        encountered, changed, tokens, text_redactions = 0, 0, set(), 0
                    total_users += encountered
                    changed_users += changed
                    changed_team_ids += text_redactions
                    identity_tokens.update(tokens)
                    output_archive.writestr(copy.copy(info), raw)

        # Flush archive bytes before any verification or replacement.
        with temporary_path.open("r+b") as handle:
            os.fsync(handle.fileno())
        after = audit_archive(temporary_path)
        if before.members != after.members:
            raise SanitizationError("archive member order/names changed")
        if before.semantic_digest != after.semantic_digest:
            raise SanitizationError("non-user JSON semantics changed")
        if before.non_user_uuid_digest != after.non_user_uuid_digest:
            raise SanitizationError("a document/link UUID outside user metadata changed")
        if before.raw_member_digests != after.raw_member_digests:
            raise SanitizationError("a non-JSON archive member changed")
        if after.neutral_user_fields != after.user_fields:
            raise SanitizationError("one or more user fields were not neutralized")
        if after.literal_team_id_hits:
            raise SanitizationError(
                f"registration token remains in {after.literal_team_id_hits} archive location(s); "
                "refusing to broaden the exact text-field redaction"
            )
        if after.exposed_team_id_fields:
            raise SanitizationError("one or more exact text-field registration tokens remain")
        token_hits = _count_token_hits(temporary_path, identity_tokens)
        if token_hits:
            raise SanitizationError(
                f"{token_hits} author identity token occurrence(s) remain after sanitization"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return SanitizeResult(
            destination=destination,
            user_fields=total_users,
            changed_user_fields=changed_users,
            changed_team_id_fields=changed_team_ids,
            removed_identity_token_hits=token_hits,
            semantic_digest=after.semantic_digest,
            non_user_uuid_digest=after.non_user_uuid_digest,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="verify that all user fields are neutral")
    mode.add_argument("--in-place", action="store_true", help="atomically replace the input archive")
    mode.add_argument("--output", type=Path, help="write an atomically verified sanitized copy")
    parser.add_argument("--force", action="store_true", help="replace an existing --output file")
    args = parser.parse_args()

    try:
        archive = args.archive.resolve()
        if args.in_place:
            result = sanitize_archive(archive, archive, overwrite=True)
            print(
                "PASS sanitized EasyEDA Pro archive: "
                f"{result.user_fields} user fields, "
                f"{result.changed_user_fields} changed, "
                f"{result.changed_team_id_fields} exact team-ID text field(s) redacted, "
                "identity/team-ID hits 0; "
                f"non-user UUID digest {result.non_user_uuid_digest}"
            )
            return 0
        if args.output is not None:
            result = sanitize_archive(archive, args.output, overwrite=args.force)
            print(
                "PASS sanitized EasyEDA Pro archive: "
                f"{result.destination}; {result.user_fields} user fields, "
                f"{result.changed_user_fields} changed, "
                f"{result.changed_team_id_fields} exact team-ID text field(s) redacted, "
                "identity/team-ID hits 0; "
                f"non-user UUID digest {result.non_user_uuid_digest}"
            )
            return 0

        audit = audit_archive(archive)
        if audit.neutral_user_fields != audit.user_fields:
            print(
                "FAIL EasyEDA Pro privacy check: "
                f"{audit.user_fields - audit.neutral_user_fields} of {audit.user_fields} "
                "user fields still contain author metadata"
            )
            return 1
        if audit.literal_team_id_hits or audit.exposed_team_id_fields:
            print(
                "FAIL EasyEDA Pro privacy check: "
                f"registration token hits={audit.literal_team_id_hits}, "
                f"exact text-field hits={audit.exposed_team_id_fields}"
            )
            return 1
        print(
            "PASS EasyEDA Pro privacy check: "
            f"{audit.epru_members} design member(s), {audit.user_fields} neutral user fields, "
            f"{audit.redacted_team_id_fields} team-ID text field(s) redacted, "
            f"CRC OK; non-user UUID digest {audit.non_user_uuid_digest}"
        )
        return 0
    except SanitizationError as exc:
        print(f"FAIL EasyEDA Pro sanitizer: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
