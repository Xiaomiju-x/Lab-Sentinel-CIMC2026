#!/usr/bin/env python3
"""Inventory licensed phosphor supplementary files without creating task labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from pypdf import PdfReader


TERMS = {
    "activation_energy": r"activation energy|arrhenius",
    "cie_color": r"\bcie\b|chromaticity|color coordinate",
    "concentration": r"concentration|mol%|at%|doping level",
    "decay_lifetime": r"decay|lifetime|microsecond|millisecond",
    "dsc_tga": r"\bdsc\b|\btga\b|thermogravimet",
    "grain_particle": r"grain size|particle size|\bsem\b|micromorphology",
    "intensity": r"integrated intensity|emission intensity|luminescence intensity",
    "nir": r"near-infrared|\bnir\b|nir-ii",
    "phase_rietveld": r"rietveld|phase fraction|site occupancy|x-ray diffraction|\bxrd\b",
    "quantum_efficiency": r"quantum efficiency|quantum yield|\biqe\b|\beqe\b|\bplqy\b",
    "synthesis_schedule": r"anneal|calcination|sinter|solid-state|hydrothermal|co-precipitation",
    "thermal_quenching": r"thermal quench|temperature-dependent|temperature dependent|thermal stability",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_root(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def safe_destination(root: Path, name: str) -> Path:
    pure = PurePosixPath(name.replace("\\", "/"))
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe archive member: {name}")
    destination = root.joinpath(*pure.parts).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise ValueError(f"archive member escapes root: {name}")
    return destination


def extract_payload(root: Path, name: str, data: bytes) -> Path:
    destination = safe_destination(root, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return destination


def xml_text(data: bytes) -> tuple[str, int]:
    document = ET.fromstring(data)
    text = " ".join(part.strip() for part in document.itertext() if part.strip())
    tables = sum(1 for node in document.iter() if node.tag.rsplit("}", 1)[-1] == "tbl")
    return text, tables


def docx_text(data: bytes) -> tuple[str, int]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return xml_text(archive.read("word/document.xml"))


def pdf_text(data: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages), len(reader.pages)


def terminal_payloads(name: str, data: bytes) -> list[tuple[str, bytes]]:
    if name.lower().endswith(".zip"):
        rows: list[tuple[str, bytes]] = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"CRC failure in nested ZIP: {name}")
            for item in archive.infolist():
                if not item.is_dir():
                    rows.extend(terminal_payloads(item.filename, archive.read(item)))
        return rows
    return [(name, data)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = root / "data/raw/pmc_phosphor_supp_v1"
    extract_root = source_root / "extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    records = []

    for outer_path in sorted(source_root.glob("PMC*_supplementaryFiles.zip")):
        pmcid = outer_path.name.split("_", 1)[0]
        metadata_path = root / f"data/metadata/ccby_multidomain_v2/{pmcid}.json"
        xml_path = root / f"data/raw/ccby_multidomain_v2/{pmcid}.xml"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("pmcid") != pmcid or "cc by" not in metadata.get("license", "").lower():
            raise ValueError(f"identity or license gate failed: {pmcid}")
        article_root = ET.parse(xml_path).getroot()
        table_texts = [
            " ".join(part.strip() for part in node.itertext() if part.strip())
            for node in article_root.iter()
            if node.tag.rsplit("}", 1)[-1] == "table-wrap"
        ]
        figure_captions = [
            " ".join(part.strip() for part in node.itertext() if part.strip())
            for node in article_root.iter()
            if node.tag.rsplit("}", 1)[-1] == "caption"
        ]
        main_text = " ".join(part.strip() for part in article_root.itertext() if part.strip())

        attachments = []
        supplement_texts = []
        docx_tables = 0
        pdf_pages = 0
        with zipfile.ZipFile(outer_path) as outer:
            if outer.testzip() is not None:
                raise ValueError(f"CRC failure: {outer_path.name}")
            terminal = []
            for item in outer.infolist():
                if not item.is_dir():
                    terminal.extend(terminal_payloads(item.filename, outer.read(item)))
        family_root = extract_root / pmcid
        if family_root.exists():
            shutil.rmtree(family_root)
        family_root.mkdir(parents=True)
        for name, data in terminal:
            destination = extract_payload(family_root, name, data)
            suffix = destination.suffix.lower()
            extracted_text = ""
            pages = 0
            tables = 0
            if suffix == ".pdf":
                extracted_text, pages = pdf_text(data)
                pdf_pages += pages
            elif suffix == ".docx":
                extracted_text, tables = docx_text(data)
                docx_tables += tables
            supplement_texts.append(extracted_text)
            attachments.append({
                "path": destination.relative_to(root).as_posix(),
                "filename": name,
                "suffix": suffix,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "text_chars": len(extracted_text),
                "pdf_pages": pages,
                "docx_tables": tables,
            })

        searchable = "\n".join([main_text, *supplement_texts]).lower()
        term_hits = {key: len(re.findall(pattern, searchable, flags=re.IGNORECASE)) for key, pattern in TERMS.items()}
        records.append({
            "pmcid": pmcid,
            "doi": metadata.get("doi"),
            "title": re.sub(r"<[^>]+>", "", metadata.get("title", "")),
            "license": metadata.get("license"),
            "metadata": {
                "path": metadata_path.relative_to(root).as_posix(),
                "bytes": metadata_path.stat().st_size,
                "sha256": sha256_file(metadata_path),
            },
            "jats_xml": {
                "path": xml_path.relative_to(root).as_posix(),
                "bytes": xml_path.stat().st_size,
                "sha256": sha256_file(xml_path),
                "text_chars": len(main_text),
                "table_wraps": len(table_texts),
                "figure_captions": len(figure_captions),
            },
            "outer_archive": {
                "path": outer_path.relative_to(root).as_posix(),
                "bytes": outer_path.stat().st_size,
                "sha256": sha256_file(outer_path),
                "crc_verified": True,
            },
            "attachments": attachments,
            "attachment_count": len(attachments),
            "attachment_bytes": sum(item["bytes"] for item in attachments),
            "pdf_pages": pdf_pages,
            "docx_tables": docx_tables,
            "term_hits": term_hits,
            "split_family": pmcid,
            "training_label_authorized": False,
        })

    if len(records) != 14 or len({record["pmcid"] for record in records}) != len(records):
        raise ValueError("expected fourteen unique PMCID document families")
    audit = {
        "schema": "cimc.forge200.pmc-phosphor-supplement-inventory.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_LICENSE_AND_SUPPLEMENT_INVENTORY_PASS_CONTRACT_BINDING_PENDING",
        "source_family_count": len(records),
        "split_unit": "PMCID_DOCUMENT_FAMILY_BEFORE_RECORD_EXTRACTION",
        "cross_split_overlap": None,
        "records": records,
        "totals": {
            "attachments": sum(record["attachment_count"] for record in records),
            "attachment_bytes": sum(record["attachment_bytes"] for record in records),
            "pdf_pages": sum(record["pdf_pages"] for record in records),
            "docx_tables": sum(record["docx_tables"] for record in records),
            "jats_tables": sum(record["jats_xml"]["table_wraps"] for record in records),
        },
        "truth_class": "LITERATURE_CURATED_EXPERIMENT_SOURCE_MATERIAL_ONLY",
        "training_actions": 0,
        "task_contract_bindings": 0,
        "host_promotions": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Extract only explicitly source-bound numeric records, freeze PMCID-family splits, and audit each candidate against its exact input, label, baseline, and metric contract before training.",
    }
    audit["content_root_sha256"] = canonical_root(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "families": len(records), **audit["totals"], "content_root_sha256": audit["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
