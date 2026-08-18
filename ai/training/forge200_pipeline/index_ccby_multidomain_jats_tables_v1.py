#!/usr/bin/env python3
"""Index every licensed CC BY JATS table without authorizing task labels.

This is a discovery index.  A keyword hit is never a source/label binding: each
candidate still needs an exact, in-study, record-level contract audit and a
PMCID-family split frozen before training.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


TABLE_TAGS = {
    "activation_energy": r"activation energ|arrhenius|\be[_\s-]?a\b|\bea\s*\(\s*ev\s*\)",
    "afm_height_roughness": r"atomic force microscop|\bafm\b|surface roughness|\brms roughness\b|height map|root.mean.square roughness",
    "cmp_polishing": r"chemical.mechanical polish|\bcmp\b|material removal rate|polishing rate|slurry",
    "deposition_recipe": r"deposition|sputter|evaporation|atomic layer deposition|\bald\b|chemical vapor deposition|\bcvd\b|plasma.enhanced|precursor|base pressure",
    "dsc_tga_onset": r"\bdsc\b|\btga\b|thermogravimet|differential scanning calorim|reaction onset|onset temperature|weight loss",
    "etch_endpoint_plasma": r"etch|plasma|optical emission spect|\boes\b|endpoint|end.point|rf power|rf bias|gas flow|mass flow|bosch",
    "fatigue_cycles_failure": r"fatigue|cycles? to failure|number of cycles|thermal cycling|survival|weibull|crack propagation|solder joint",
    "four_point_probe": r"four.point|4.point|sheet resistance|electrical resistivity|resistivity|conductivity",
    "grain_particle_size": r"grain size|particle size|crystallite size|crystalline size|average grain|grain growth",
    "image_defect_cd_overlay": r"defect|critical dimension|\bcd\b|overlay|segmentation|mask|image quality|micrograph|\bsem\b",
    "lifetime_quantum_efficiency": r"decay lifetime|fluorescence lifetime|quantum efficiency|quantum yield|\biqe\b|\beqe\b|\bplqy\b",
    "mechanical_cte_modulus": r"coefficient of thermal expansion|\bcte\b|young.?s modulus|elastic modulus|shear modulus|tensile strength",
    "moisture_diffusion": r"moisture|humidity|relative humidity|\brh\b|diffusion coefficient|water uptake|sorption|saturation concentration|hygroscopic",
    "nir_emission_intensity": r"near.infrared|\bnir\b|emission intensity|integrated intensity|luminescence intensity|emission peak",
    "oxygen_vacancy_xps": r"oxygen vacanc|vacancy concentration|\bxps\b|o\s*1s|relative area|peak area",
    "phase_fraction_rietveld": r"phase fraction|rietveld|quantitative phase|site occupancy|occupancy factor|space group|x.ray diffraction|\bxrd\b",
    "pvd_thickness": r"physical vapor deposition|\bpvd\b|film thickness|coating thickness|layer thickness|thickness\s*\(?\s*nm",
    "relative_density_shrinkage": r"relative density|green density|bulk density|densification|linear shrinkage|volumetric shrinkage|sintering shrinkage",
    "site_occupancy_solubility": r"site occupancy|occupancy factor|substitution site|solubility limit|solid solubility|dopant incorporation|preferential site",
    "temperature_time_schedule": r"anneal|calcination|sinter|heat treatment|holding time|dwell time|temperature\s*\(?\s*.?c|heating rate",
    "thickness_target": r"film thickness|coating thickness|layer thickness|etch depth|etched depth|depth etched|thickness\s*\(?\s*(?:nm|um|渭m)",
    "underfill_flow_void": r"underfill|capillary flow|flow front|fill time|void fraction|void content|encapsulant",
    "xct_registration": r"x.ray computed tomograph|\bxct\b|micro.ct|computed tomography|registration error|registration accuracy|voxel",
}

REVIEW_TYPES = {"review", "systematic review", "meta-analysis", "scoping review"}
COMPARISON_PATTERN = re.compile(
    r"comparison with|comparison of .*reported|previously reported|reported in the literature|"
    r"other studies|other materials|selected literature|literature data|references?\s*$|\bref\.\s*$",
    flags=re.IGNORECASE,
)


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


def publication_types(metadata: dict) -> list[str]:
    raw = (metadata.get("pubTypeList") or {}).get("pubType", [])
    if isinstance(raw, str):
        return [raw]
    return [str(item) for item in raw]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    metadata_root = root / "data/metadata/ccby_multidomain_v2"
    xml_root = root / "data/raw/ccby_multidomain_v2"
    families: list[dict] = []
    tables: list[dict] = []

    for metadata_path in sorted(metadata_root.glob("PMC*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        pmcid = metadata.get("pmcid")
        license_text = metadata.get("license", "")
        if not pmcid or "cc by" not in license_text.lower():
            raise ValueError(f"identity or license gate failed: {metadata_path.name}")
        xml_path = xml_root / f"{pmcid}.xml"
        if not xml_path.is_file():
            raise FileNotFoundError(xml_path)
        title = re.sub(r"<[^>]+>", "", html.unescape(metadata.get("title", "")))
        pub_types = publication_types(metadata)
        is_review = any(item.lower() in REVIEW_TYPES for item in pub_types) or bool(
            re.search(r"\b(systematic |scoping )?review\b|meta-analysis", title, flags=re.IGNORECASE)
        )
        article = ET.parse(xml_path).getroot()
        article_tables = [node for node in article.iter() if local_name(node) == "table-wrap"]
        families.append({
            "pmcid": pmcid,
            "doi": metadata.get("doi"),
            "title": title,
            "publication_type": pub_types,
            "is_review_article": is_review,
            "license": license_text,
            "has_supplement": metadata.get("hasSuppl") == "Y",
            "metadata_path": metadata_path.relative_to(root).as_posix(),
            "metadata_sha256": sha256_file(metadata_path),
            "jats_xml_path": xml_path.relative_to(root).as_posix(),
            "jats_xml_sha256": sha256_file(xml_path),
            "table_count": len(article_tables),
            "split_family": pmcid,
        })
        for index, table in enumerate(article_tables, start=1):
            rows = table_rows(table)
            label = node_text(first_descendant(table, "label"))
            caption = node_text(first_descendant(table, "caption"))
            footnotes = " | ".join(
                node_text(node) for node in table.iter() if local_name(node) in {"table-wrap-foot", "fn"}
            )
            row_text = " | ".join(" | ".join(row) for row in rows)
            combined = " | ".join([label, caption, footnotes, row_text])
            tags = [name for name, pattern in TABLE_TAGS.items() if re.search(pattern, combined, flags=re.IGNORECASE)]
            numeric_cells = sum(
                bool(re.search(r"[-+]?\d+(?:[.,]\d+)?", cell)) for row in rows for cell in row
            )
            has_reference_column = bool(rows and any(re.search(r"\bref(?:erence)?\b", cell, re.I) for cell in rows[0]))
            comparison_risk = bool(COMPARISON_PATTERN.search(combined)) or has_reference_column
            tables.append({
                "table_id": f"{pmcid}:T{index:02d}",
                "pmcid": pmcid,
                "doi": metadata.get("doi"),
                "article_title": title,
                "publication_type": pub_types,
                "is_review_article": is_review,
                "label": label,
                "caption": caption,
                "footnotes": footnotes,
                "rows": rows,
                "row_count": len(rows),
                "max_columns": max((len(row) for row in rows), default=0),
                "numeric_cell_count": numeric_cells,
                "tags": tags,
                "split_family": pmcid,
                "record_truth_authorized": False,
                "external_comparison_risk": comparison_risk,
                "source_scope_status": "MANUAL_IN_STUDY_RECORD_AUDIT_REQUIRED",
            })

    if len(families) != 1080 or sum(item["table_count"] > 0 for item in families) != 797 or len(tables) != 2701:
        raise ValueError(
            "corpus inventory changed: "
            f"families={len(families)} families_with_tables={sum(item['table_count'] > 0 for item in families)} "
            f"tables={len(tables)}"
        )
    totals = {
        "families": len(families),
        "families_with_tables": sum(item["table_count"] > 0 for item in families),
        "review_families": sum(item["is_review_article"] for item in families),
        "tables": len(tables),
        "review_article_tables": sum(item["is_review_article"] for item in tables),
        "external_comparison_risk_tables": sum(item["external_comparison_risk"] for item in tables),
        "numeric_cells": sum(item["numeric_cell_count"] for item in tables),
        "tag_counts": {tag: sum(tag in table["tags"] for table in tables) for tag in TABLE_TAGS},
    }
    index_document = {
        "schema": "cimc.forge200.ccby-multidomain-jats-table-index.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INDEXED_CONTRACT_BINDING_PENDING",
        "selection_rule": "All locally verified CC BY metadata and matching JATS XML document families",
        "split_unit": "PMCID_DOCUMENT_FAMILY_BEFORE_RECORD_EXTRACTION",
        "keyword_hits_are_labels": False,
        "families": families,
        "tables": tables,
        "totals": totals,
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
        "schema": "cimc.forge200.ccby-multidomain-jats-table-index-receipt.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_INDEX_ONLY_NO_TASK_LABELS_AUTHORIZED",
        "index": {
            "path": args.output.resolve().relative_to(root).as_posix(),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "content_root_sha256": index_document["content_root_sha256"],
        },
        "totals": totals,
        "split_unit": index_document["split_unit"],
        "training_actions": 0,
        "task_contract_bindings": 0,
        "host_promotions": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Audit non-review, non-comparison, in-study rows against exact frozen candidate fields; freeze PMCID-family splits before any training.",
    }
    evidence["content_root_sha256"] = content_root(evidence)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": evidence["status"], **totals, "content_root_sha256": evidence["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
