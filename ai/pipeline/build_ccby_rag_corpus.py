#!/usr/bin/env python3
"""Build a six-domain, source-bound CC BY corpus through Europe PMC REST.

Every document is explicitly selected, license-checked in both metadata and
JATS XML, hashed, and split by document family before chunking.  Article text
is evidence-bearing source text; it is never promoted to independent material
ground truth without the cited source and claim-state fields.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DOCUMENTS = {
    "PHOSPHOR": ["PMC13258628", "PMC13149538", "PMC12609268", "PMC12961452", "PMC12632419", "PMC13261445", "PMC13075092", "PMC13258048"],
    "FURNACE": ["PMC13027571", "PMC13258771", "PMC13185808", "PMC13280825", "PMC13295240", "PMC13165069", "PMC13074225"],
    "SEMIMAT": ["PMC13231439", "PMC13144534", "PMC13257496", "PMC13296610", "PMC13208880", "PMC13270641", "PMC13164758", "PMC13247828"],
    "METROLOGY": ["PMC13208757", "PMC12656061", "PMC13211111", "PMC13264615", "PMC13115415", "PMC13209095", "PMC13185705", "PMC13165091", "PMC13209852", "PMC12728232", "PMC12723346"],
    "PACKAGING": ["PMC12898862", "PMC12389694", "PMC12287503", "PMC11052395", "PMC10745373", "PMC12610736", "PMC12943695"],
    "FABQUALITY": ["PMC8199536", "PMC10181745", "PMC10575315", "PMC10611205", "PMC10675586", "PMC13119986"],
}
DOCUMENT_SPLITS = {
    "PMC12609268": "train", "PMC12961452": "train", "PMC12632419": "train",
    "PMC13149538": "validation", "PMC13261445": "validation", "PMC13258048": "validation",
    "PMC13258628": "test", "PMC13075092": "test",
    "PMC13027571": "train", "PMC13185808": "train", "PMC13165069": "train",
    "PMC13258771": "validation", "PMC13074225": "validation",
    "PMC13280825": "test", "PMC13295240": "test",
    "PMC13231439": "train", "PMC13144534": "train", "PMC13208880": "train",
    "PMC13257496": "validation", "PMC13270641": "validation", "PMC13247828": "validation",
    "PMC13296610": "test", "PMC13164758": "test",
    "PMC13208757": "train", "PMC12656061": "train", "PMC13211111": "validation", "PMC13115415": "validation",
    "PMC13209095": "validation", "PMC13185705": "validation", "PMC13165091": "validation", "PMC13264615": "test", "PMC13209852": "test",
    "PMC12728232": "test", "PMC12723346": "test",
    "PMC12287503": "train", "PMC11052395": "train", "PMC10745373": "train", "PMC12610736": "train",
    "PMC12389694": "validation", "PMC12898862": "test", "PMC12943695": "test",
    "PMC10181745": "train", "PMC10575315": "train", "PMC10611205": "train", "PMC10675586": "train",
    "PMC13119986": "validation", "PMC8199536": "test",
}
API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
USER_AGENT = "CIMC-Forge200-LicenseAudit/1.0 (competition research; per-document REST access)"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/xml;q=0.9"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def element_text(element: ET.Element | None) -> str:
    return normalize_text("".join(element.itertext())) if element is not None else ""


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def direct_children(element: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in list(element) if local_name(child.tag) == name)


def collect_passages(root: ET.Element) -> list[tuple[str, str]]:
    passages: list[tuple[str, str]] = []
    abstract = next((item for item in root.iter() if local_name(item.tag) == "abstract"), None)
    if abstract is not None:
        text = element_text(abstract)
        if len(text) >= 120:
            passages.append(("ABSTRACT", text))
    body = next((item for item in root.iter() if local_name(item.tag) == "body"), None)
    if body is None:
        return passages

    def walk(section: ET.Element, inherited_title: str) -> None:
        title_element = next(iter(direct_children(section, "title")), None)
        title = element_text(title_element) or inherited_title or "BODY"
        for paragraph in direct_children(section, "p"):
            text = element_text(paragraph)
            if len(text) >= 120:
                passages.append((title, text))
        for child in direct_children(section, "sec"):
            walk(child, title)

    for paragraph in direct_children(body, "p"):
        text = element_text(paragraph)
        if len(text) >= 120:
            passages.append(("BODY", text))
    for section in direct_children(body, "sec"):
        walk(section, "BODY")
    return passages


def chunk_passages(passages: list[tuple[str, str]], minimum: int = 500, maximum: int = 1200) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for section, passage in passages:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", passage) if item.strip()]
        buffer = ""
        for sentence in sentences:
            if len(sentence) > maximum:
                pieces = [sentence[index : index + maximum] for index in range(0, len(sentence), maximum)]
            else:
                pieces = [sentence]
            for piece in pieces:
                if buffer and len(buffer) + 1 + len(piece) > maximum:
                    if len(buffer) >= minimum:
                        chunks.append((section, buffer))
                    buffer = piece
                else:
                    buffer = f"{buffer} {piece}".strip()
        if len(buffer) >= minimum:
            chunks.append((section, buffer))
    return chunks


def split_documents(documents: list[str]) -> dict[str, str]:
    if len(documents) < 3:
        raise RuntimeError("each domain requires at least three document families")
    missing = [pmcid for pmcid in documents if pmcid not in DOCUMENT_SPLITS]
    if missing:
        raise RuntimeError(f"explicit document split missing: {missing}")
    result = {pmcid: DOCUMENT_SPLITS[pmcid] for pmcid in documents}
    if set(result.values()) != {"train", "validation", "test"}:
        raise RuntimeError("each domain must contain all three explicit splits")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    raw_root = root / "data" / "raw" / "ccby_multidomain_v1"
    metadata_root = root / "data" / "metadata" / "ccby_multidomain_v1"
    raw_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    document_records: list[dict[str, Any]] = []
    chunk_records: list[dict[str, Any]] = []
    for domain, pmcids in DOCUMENTS.items():
        split_map = split_documents(pmcids)
        for pmcid in pmcids:
            query = urllib.parse.quote(f"PMCID:{pmcid}")
            metadata_url = f"{API}/search?query={query}&format=json&resultType=core&pageSize=1"
            xml_url = f"{API}/{pmcid}/fullTextXML"
            metadata_path = metadata_root / f"{pmcid}.json"
            xml_path = raw_root / f"{pmcid}.xml"
            if args.refresh or not metadata_path.is_file():
                metadata_bytes = fetch(metadata_url)
                metadata_path.write_bytes(metadata_bytes)
            if args.refresh or not xml_path.is_file():
                xml_bytes = fetch(xml_url)
                xml_path.write_bytes(xml_bytes)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            results = metadata.get("resultList", {}).get("result", [])
            if len(results) != 1 or results[0].get("pmcid") != pmcid:
                raise RuntimeError(f"{pmcid}: metadata identity")
            article = results[0]
            if str(article.get("license", "")).lower().replace("-", " ") != "cc by":
                raise RuntimeError(f"{pmcid}: metadata license is not CC BY")
            xml_bytes = xml_path.read_bytes()
            xml_lower = xml_bytes.lower()
            if b"creativecommons.org/licenses/by/" not in xml_lower and b"creative commons attribution" not in xml_lower:
                raise RuntimeError(f"{pmcid}: JATS CC BY statement missing")
            article_root = ET.fromstring(xml_bytes)
            passages = collect_passages(article_root)
            chunks = chunk_passages(passages)
            if len(chunks) < 8:
                raise RuntimeError(f"{pmcid}: insufficient extractable chunks ({len(chunks)})")
            doc_record = {
                "domain": domain,
                "pmcid": pmcid,
                "doi": article.get("doi"),
                "title": normalize_text(str(article.get("title", ""))),
                "publication_year": article.get("pubYear"),
                "license": "CC-BY-4.0_OR_SOURCE_CC_BY_VERSION",
                "license_metadata_value": article.get("license"),
                "metadata_url": metadata_url,
                "fulltext_url": xml_url,
                "metadata_path": str(metadata_path.relative_to(root)).replace("\\", "/"),
                "metadata_sha256": sha256_file(metadata_path),
                "xml_path": str(xml_path.relative_to(root)).replace("\\", "/"),
                "xml_sha256": sha256_file(xml_path),
                "xml_bytes": xml_path.stat().st_size,
                "split": split_map[pmcid],
                "chunks": len(chunks),
                "truth_class": "LITERATURE_CURATED_EXPERIMENT",
                "claim_state": "SOURCE_BOUND_PUBLICATION_TEXT_NOT_INDEPENDENT_GROUND_TRUTH",
                "training_role": "LICENSED_LANGUAGE_AND_EVIDENCE_CORPUS",
                "authority": 0,
            }
            document_records.append(doc_record)
            for index, (section, text) in enumerate(chunks):
                chunk_id = f"{pmcid}:{index:04d}"
                chunk_records.append(
                    {
                        "chunk_id": chunk_id,
                        "domain": domain,
                        "pmcid": pmcid,
                        "doi": article.get("doi"),
                        "title": doc_record["title"],
                        "section": section,
                        "text": text,
                        "split": split_map[pmcid],
                        "license": doc_record["license"],
                        "truth_class": doc_record["truth_class"],
                        "claim_state": doc_record["claim_state"],
                        "source_sha256": doc_record["xml_sha256"],
                        "authority": 0,
                    }
                )
            time.sleep(0.05)
    corpus_path = root / "data" / "corpora" / "ccby_multidomain_v1.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in chunk_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    domain_counts = defaultdict(lambda: defaultdict(int))
    for record in chunk_records:
        domain_counts[record["domain"]][record["split"]] += 1
    manifest = {
        "schema": "cimc.forge200.ccby-multidomain-corpus.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_api": API,
        "selection_policy": "EXPLICIT_PMCI_DS_METADATA_CC_BY_AND_JATS_LICENSE_CHECK",
        "split_unit": "PMCID_DOCUMENT_FAMILY_BEFORE_CHUNKING",
        "documents": document_records,
        "document_count": len(document_records),
        "chunk_count": len(chunk_records),
        "domain_split_counts": {domain: dict(counts) for domain, counts in sorted(domain_counts.items())},
        "corpus_path": str(corpus_path.relative_to(root)).replace("\\", "/"),
        "corpus_bytes": corpus_path.stat().st_size,
        "corpus_sha256": sha256_file(corpus_path),
        "document_family_overlap": 0,
        "teacher_outputs": 0,
        "authority_nonzero": 0,
        "content_root_sha256": sha256_bytes(canonical_bytes(document_records)),
    }
    write_json(root / "data" / "ledgers" / "ccby_multidomain_corpus.v1.json", manifest)
    source_path = root / "data" / "ledgers" / "source_ledger.v1.json"
    license_path = root / "data" / "ledgers" / "license_ledger.v1.json"
    source_ledger = json.loads(source_path.read_text(encoding="utf-8"))
    source_ledger["records"] = [item for item in source_ledger["records"] if item.get("source_id") != "europe_pmc_ccby_multidomain_v1"]
    source_ledger["records"].append(
        {
            "source_id": "europe_pmc_ccby_multidomain_v1",
            "version": "v1-explicit-47-document-families",
            "pid": "EUROPE_PMC_EXPLICIT_PMCID_SET",
            "canonical_url": API,
            "path": manifest["corpus_path"],
            "artifact_bytes": manifest["corpus_bytes"],
            "artifact_sha256": manifest["corpus_sha256"],
            "metadata_snapshot": "data/ledgers/ccby_multidomain_corpus.v1.json",
            "metadata_snapshot_sha256": manifest["content_root_sha256"],
            "license": "CC-BY_PER_DOCUMENT_METADATA_AND_JATS_VERIFIED",
            "license_url": "PER_DOCUMENT_JATS_LICENSE_ELEMENT",
            "truth_class": "LITERATURE_CURATED_EXPERIMENT",
            "training_allowed": True,
            "rag_allowed": True,
            "scope": "Licensed language, retrieval, exact cited spans, and controlled structure-derived labels only; source text is not independent experimental ground truth.",
            "decision": "TRAIN_RAG_SOURCE_BOUND_GO",
        }
    )
    source_ledger["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(source_path, source_ledger)
    license_ledger = json.loads(license_path.read_text(encoding="utf-8"))
    license_ledger["records"] = [item for item in license_ledger["records"] if item.get("source_id") != "europe_pmc_ccby_multidomain_v1"]
    license_ledger["records"].append(
        {
            "source_id": "europe_pmc_ccby_multidomain_v1",
            "license": "CC-BY_PER_DOCUMENT_METADATA_AND_JATS_VERIFIED",
            "license_url": "PER_DOCUMENT_JATS_LICENSE_ELEMENT",
            "metadata_snapshot_sha256": manifest["content_root_sha256"],
            "training_allowed": True,
            "rag_allowed": True,
            "decision": "TRAIN_RAG_SOURCE_BOUND_GO",
        }
    )
    write_json(license_path, license_ledger)
    print(json.dumps({"status": manifest["status"], "documents": manifest["document_count"], "chunks": manifest["chunk_count"], "bytes": manifest["corpus_bytes"], "sha256": manifest["corpus_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
