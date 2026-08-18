#!/usr/bin/env python3
"""Freeze the exact-contract decision for the PMC13157481 NIR supplement."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def cell_text(cell: ET.Element) -> str:
    return " ".join("".join(node.text or "" for node in cell.findall(".//w:t", NS)).split())


def parse_docx(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
        media = [name for name in archive.namelist() if name.startswith("word/media/") and not name.endswith("/")]
    tables: list[list[list[str]]] = []
    for table in root.findall(".//w:tbl", NS):
        rows: list[list[str]] = []
        for row in table.findall("./w:tr", NS):
            rows.append([cell_text(cell) for cell in row.findall("./w:tc", NS)])
        tables.append(rows)
    paragraphs = [
        " ".join("".join(node.text or "" for node in para.findall(".//w:t", NS)).split())
        for para in root.findall(".//w:p", NS)
    ]
    return {
        "paragraph_count": len(paragraphs),
        "nonempty_paragraph_count": sum(bool(item) for item in paragraphs),
        "table_count": len(tables),
        "table_shapes": [[len(table), max((len(row) for row in table), default=0)] for table in tables],
        "table_headers": [table[0] for table in tables],
        "media_count": len(media),
        "paragraphs": paragraphs,
        "tables": tables,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    inventory_path = root / "evidence" / "pmc_phosphor_supplement_inventory.v1.json"
    inventory = load_json(inventory_path)
    record = next(item for item in inventory["records"] if item["pmcid"] == "PMC13157481")
    attachment = record["attachments"][0]
    docx_path = root / attachment["path"]

    actual_sha = sha256_file(docx_path)
    if actual_sha != attachment["sha256"] or docx_path.stat().st_size != attachment["bytes"]:
        raise RuntimeError("PMC13157481_ATTACHMENT_HASH_GATE")
    if "cc by" not in record["license"].lower():
        raise RuntimeError("PMC13157481_LICENSE_GATE")

    parsed = parse_docx(docx_path)
    expected_headers = [
        ["Phosphor", "Space Group", "Parameters (Å)", "Volume (Å3)", "R"],
        ["Phosphor", "λex (nm)", "λem (nm)", "FWHM (nm)", "IT/I298K", "Ea (eV)", "Ref."],
        ["λem (nm)", "A1", "τ1 (μs)", "A2", "τ2 (μs)", "τ (μs)", "R2"],
    ]
    if parsed["table_shapes"] != [[7, 5], [15, 7], [8, 7]] or parsed["table_headers"] != expected_headers:
        raise RuntimeError("PMC13157481_DOCX_STRUCTURE_GATE")

    full_text = "\n".join(parsed["paragraphs"])
    required_facts = [
        "internal quantum efficiency is 78%",
        "absorption efficiency is 61%",
        "external quantum efficiency is 48%",
        "SSBO:0.02Fe3+,0.15Yb3+",
    ]
    if any(fact not in full_text for fact in required_facts):
        raise RuntimeError("PMC13157481_TEXT_FACT_GATE")

    dispositions = [
        {
            "candidate_id": "CAND-P-061",
            "status": "EXACT_REJECTED",
            "reason": "No sample-level normalized NIR intensity series with complete composition, process, phase, and measurement-normalization fields; figure curves and cross-paper summary rows are not interchangeable records.",
        },
        {
            "candidate_id": "CAND-P-062",
            "status": "EXACT_REJECTED",
            "reason": "Table S2 gives isolated retention/Ea summaries, not the required multi-temperature PL intensity curves and fitted kinetic parameters for independently split source families.",
        },
        {
            "candidate_id": "CAND-P-079",
            "status": "EXACT_REJECTED",
            "reason": "Concentration-dependent intensity and decay information is figure-only; critical concentration and quenching-rate records with full process fields are not tabulated.",
        },
        {
            "candidate_id": "CAND-P-080",
            "status": "EXACT_REJECTED_FOR_THIS_SOURCE",
            "reason": "The supplement reports one derived 78% IQY point (61% absorption, 48% external efficiency) without reusable calibrated excitation/emission photon-count records or a target interval.",
        },
        {
            "candidate_id": "CAND-P-081",
            "status": "EXACT_REJECTED",
            "reason": "Table S3 contains 28.5461-107.3200 microsecond prompt luminescence lifetimes, not afterglow t10 and integrated-persistence labels with trap/process features.",
        },
    ]
    if any(item["status"] == "EXACT_ADMITTED" for item in dispositions):
        raise RuntimeError("PMC13157481_FAIL_CLOSED_GATE")

    receipt = {
        "schema": "cimc.forge200.pmc13157481-nir-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_LICENSE_AND_HASH_PASS_FIVE_EXACT_CONTRACTS_REJECTED",
        "source": {
            "pmcid": record["pmcid"],
            "doi": record["doi"],
            "license": record["license"],
            "split_family": record["split_family"],
            "metadata": record["metadata"],
            "jats_xml": record["jats_xml"],
            "supplement_docx": {
                "path": attachment["path"],
                "bytes": attachment["bytes"],
                "sha256": attachment["sha256"],
            },
        },
        "structural_review": {
            "docx_table_count": parsed["table_count"],
            "docx_table_shapes": parsed["table_shapes"],
            "docx_table_headers": parsed["table_headers"],
            "embedded_media_count": parsed["media_count"],
            "libreoffice_visual_render_available": False,
            "visual_render_limitation": "LibreOffice and Microsoft Word were not installed; review used immutable OOXML table/text extraction and did not claim page-layout verification.",
        },
        "observed_source_facts": {
            "table_s2_this_work_rows": 2,
            "table_s3_emission_wavelength_rows": 7,
            "table_s3_lifetime_range_us": [28.5461, 107.32],
            "reported_internal_quantum_efficiency_percent": 78,
            "reported_absorption_efficiency_percent": 61,
            "reported_external_quantum_efficiency_percent": 48,
            "raw_calibrated_photon_count_records": 0,
            "numeric_multi_temperature_curve_records": 0,
            "numeric_concentration_quench_series_records": 0,
            "afterglow_t10_records": 0,
        },
        "candidate_dispositions": dispositions,
        "figure_digitization_used_as_ground_truth": False,
        "cross_paper_summary_rows_used_as_training_records": False,
        "task_contract_bindings_created": 0,
        "training_actions": 0,
        "host_promotions": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "board_actions": 0,
    }
    output = root / "evidence" / "pmc13157481_nir_source_contract_audit.v1.json"
    write_json(output, receipt)
    print(json.dumps({"status": receipt["status"], "candidate_dispositions": len(dispositions)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
