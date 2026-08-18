#!/usr/bin/env python3
"""Index CC BY phosphor JATS tables without authorizing task labels."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ARTICLE_TERMS = (
    "phosphor", "luminescen", "photoluminescen", "rare-earth dop",
    "near-infrared emission", "upconversion",
)
TABLE_TAGS = {
    "activation_energy": r"activation energy|arrhenius|\be_a\b|\bea\s*\(ev\)",
    "cie_color": r"\bcie\b|chromaticity|color coordinate|colour coordinate",
    "concentration": r"concentration|mol\s*%|at\s*%|doping level|dopant fraction",
    "decay_lifetime": r"decay|lifetime|\btau\b|τ|microsecond|millisecond|\bms\b|\bµs\b",
    "dsc_tga": r"\bdsc\b|\btga\b|thermogravimet|weight loss",
    "emission_peak": r"emission (center|peak|maximum)|λ\s*em|wavelength.*nm|emission.*nm",
    "grain_particle": r"grain size|particle size|crystallite size|crystalline size|\bsem\b|micromorphology",
    "intensity": r"integrated intensity|emission intensity|luminescence intensity|relative intensity",
    "nir": r"near-infrared|\bnir\b|nir-ii|800\s*nm|900\s*nm|1000\s*nm",
    "oxygen_vacancy": r"oxygen vacanc|vacancy concentration|defect concentration",
    "phase_rietveld": r"rietveld|phase fraction|site occupancy|x-ray diffraction|\bxrd\b|space group",
    "quantum_efficiency": r"quantum efficiency|quantum yield|\biqe\b|\beqe\b|\bplqy\b",
    "site_occupancy": r"site occupancy|occupancy factor|preferential site|substitution site",
    "synthesis_schedule": r"anneal|calcination|sinter|solid-state|hydrothermal|co-precipitation|temperature.*°c",
    "thermal_quenching": r"thermal quench|temperature-dependent|temperature dependent|thermal stability|temperature range",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_root(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def local_name(node: ET.Element) -> str:
    return node.tag.rsplit("}", 1)[-1]


def node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(part.strip() for part in node.itertext() if part.strip())).strip()


def first_descendant(node: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in node.iter() if local_name(item) == name), None)


def table_rows(table: ET.Element) -> list[list[str]]:
    rows = []
    for tr in (node for node in table.iter() if local_name(node) == "tr"):
        cells = [node_text(cell) for cell in tr if local_name(cell) in {"th", "td"}]
        if cells:
            rows.append(cells)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata_root = root / "data/metadata/ccby_multidomain_v2"
    xml_root = root / "data/raw/ccby_multidomain_v2"
    families = []
    tables = []

    for metadata_path in sorted(metadata_root.glob("PMC*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        title = re.sub(r"<[^>]+>", "", html.unescape(metadata.get("title", "")))
        abstract = re.sub(r"<[^>]+>", " ", html.unescape(metadata.get("abstractText", "")))
        searchable = f"{title} {abstract}".lower()
        if not any(term in searchable for term in ARTICLE_TERMS):
            continue
        pmcid = metadata.get("pmcid")
        if not pmcid or "cc by" not in metadata.get("license", "").lower():
            raise ValueError(f"identity or license gate failed: {metadata_path.name}")
        xml_path = xml_root / f"{pmcid}.xml"
        article = ET.parse(xml_path).getroot()
        article_tables = [node for node in article.iter() if local_name(node) == "table-wrap"]
        family_record = {
            "pmcid": pmcid,
            "doi": metadata.get("doi"),
            "title": title,
            "publication_type": (metadata.get("pubTypeList") or {}).get("pubType", []),
            "license": metadata.get("license"),
            "has_supplement": metadata.get("hasSuppl") == "Y",
            "metadata_path": metadata_path.relative_to(root).as_posix(),
            "metadata_sha256": sha256_file(metadata_path),
            "jats_xml_path": xml_path.relative_to(root).as_posix(),
            "jats_xml_sha256": sha256_file(xml_path),
            "table_count": len(article_tables),
            "split_family": pmcid,
        }
        families.append(family_record)
        for index, table in enumerate(article_tables, start=1):
            rows = table_rows(table)
            label = node_text(first_descendant(table, "label"))
            caption = node_text(first_descendant(table, "caption"))
            foot = " | ".join(node_text(node) for node in table.iter() if local_name(node) in {"table-wrap-foot", "fn"})
            combined = " | ".join([label, caption, foot, *(" | ".join(row) for row in rows)])
            tags = [name for name, pattern in TABLE_TAGS.items() if re.search(pattern, combined, flags=re.IGNORECASE)]
            numeric_cells = sum(
                bool(re.search(r"[-+]?\d+(?:[.,]\d+)?", cell))
                for row in rows for cell in row
            )
            tables.append({
                "table_id": f"{pmcid}:T{index:02d}",
                "pmcid": pmcid,
                "doi": metadata.get("doi"),
                "article_title": title,
                "label": label,
                "caption": caption,
                "footnotes": foot,
                "rows": rows,
                "row_count": len(rows),
                "max_columns": max((len(row) for row in rows), default=0),
                "numeric_cell_count": numeric_cells,
                "tags": tags,
                "split_family": pmcid,
                "record_truth_authorized": False,
                "review_or_external_comparison_table": bool(re.search(r"comparison|reported|literature|previous|other phosphor|ref\.?$", combined, flags=re.IGNORECASE)),
            })

    if len(families) != 186 or len(tables) != 360:
        raise ValueError(f"corpus inventory changed: families={len(families)} tables={len(tables)}")
    index_document = {
        "schema": "cimc.forge200.pmc-phosphor-jats-table-index.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INDEXED_CONTRACT_BINDING_PENDING",
        "selection_rule": "CC BY JATS article title or abstract contains a phosphor/luminescence term",
        "split_unit": "PMCID_DOCUMENT_FAMILY_BEFORE_RECORD_EXTRACTION",
        "families": families,
        "tables": tables,
        "totals": {
            "families": len(families),
            "families_with_tables": sum(item["table_count"] > 0 for item in families),
            "tables": len(tables),
            "numeric_cells": sum(item["numeric_cell_count"] for item in tables),
            "tag_counts": {tag: sum(tag in table["tags"] for table in tables) for tag in TABLE_TAGS},
        },
        "training_actions": 0,
        "task_contract_bindings": 0,
        "host_promotions": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    index_document["content_root_sha256"] = content_root(index_document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence = {
        "schema": "cimc.forge200.pmc-phosphor-jats-table-index-receipt.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_INDEX_ONLY_NO_TASK_LABELS_AUTHORIZED",
        "index": {
            "path": args.output.resolve().relative_to(root).as_posix(),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "content_root_sha256": index_document["content_root_sha256"],
        },
        "totals": index_document["totals"],
        "split_unit": index_document["split_unit"],
        "training_actions": 0,
        "task_contract_bindings": 0,
        "host_promotions": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Review source article tables, exclude reviews and external comparison rows, bind only explicit in-study records to exact frozen contracts, then freeze PMCID-family splits.",
    }
    evidence["content_root_sha256"] = content_root(evidence)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], **index_document["totals"], "content_root_sha256": evidence["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
