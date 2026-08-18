#!/usr/bin/env python3
"""Build the contract-size six-domain Europe PMC CC BY RAG corpus.

Discovery uses the official Europe PMC REST search API.  Admission is still
per document: the core metadata must identify a PMCID and an exact CC BY
license, the JATS XML must independently contain a CC BY license URI, and the
article must yield enough source-bound chunks.  Search relevance is never
treated as an experimental label or material ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_ccby_rag_corpus import (
    API,
    USER_AGENT,
    canonical_bytes,
    chunk_passages,
    collect_passages,
    fetch,
    normalize_text,
    sha256_bytes,
    sha256_file,
    write_json,
)
import xml.etree.ElementTree as ET


DOMAIN_QUERIES = {
    "PHOSPHOR": '((phosphor OR luminescence OR photoluminescence) AND (synthesis OR sintering OR dopant))',
    "FURNACE": '((sintering OR furnace OR thermal-processing OR densification) AND (ceramic OR powder OR material))',
    "SEMIMAT": '((semiconductor OR dielectric OR thin-film OR optoelectronic) AND (material OR defect OR interface))',
    "METROLOGY": '((X-ray-diffraction OR scanning-electron-microscopy OR spectroscopy OR metrology) AND material)',
    "PACKAGING": '((electronic-packaging OR underfill OR solder OR interconnect OR warpage) AND reliability)',
    "FABQUALITY": '((semiconductor-manufacturing OR wafer OR deposition OR CMP) AND (defect OR yield OR process-control))',
}
REQUIRED_FILTER = 'OPEN_ACCESS:Y AND IN_EPMC:Y AND LICENSE:"CC BY"'


def search_domain(query: str, limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    cursor = "*"
    while len(results) < limit * 3:
        params = urllib.parse.urlencode(
            {
                "query": f"{REQUIRED_FILTER} AND {query}",
                "format": "json",
                "resultType": "core",
                "pageSize": 100,
                "cursorMark": cursor,
            }
        )
        payload = json.loads(fetch(f"{API}/search?{params}").decode("utf-8"))
        page = payload.get("resultList", {}).get("result", [])
        if not page:
            break
        results.extend(page)
        next_cursor = payload.get("nextCursorMark")
        if not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
        time.sleep(0.1)
    return results


def exact_cc_by(value: Any) -> bool:
    normalized = str(value or "").strip().lower().replace("_", "-").replace(" ", "-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized in {"cc-by", "creative-commons-attribution"}


def split_for(pmcid: str) -> str:
    bucket = int(hashlib.sha256(pmcid.encode("ascii")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--documents-per-domain", type=int, default=90)
    parser.add_argument("--minimum-documents-per-domain", type=int, default=55)
    parser.add_argument("--minimum-chunks", type=int, default=20_000)
    parser.add_argument("--maximum-chunks", type=int, default=60_000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root = args.root.resolve()
    raw_root = root / "data" / "raw" / "ccby_multidomain_v2"
    metadata_root = root / "data" / "metadata" / "ccby_multidomain_v2"
    raw_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    discovered: dict[str, list[dict[str, Any]]] = {}
    search_receipts = []
    for domain, query in DOMAIN_QUERIES.items():
        rows = search_domain(query, args.documents_per_domain)
        discovered[domain] = rows
        search_receipts.append(
            {
                "domain": domain,
                "query": f"{REQUIRED_FILTER} AND {query}",
                "returned": len(rows),
            }
        )

    admitted: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    tasks: list[tuple[str, dict[str, Any]]] = []
    for domain, rows in discovered.items():
        selected = 0
        for article in rows:
            pmcid = str(article.get("pmcid") or "")
            if not pmcid.startswith("PMC") or pmcid in seen:
                continue
            if not exact_cc_by(article.get("license")):
                rejected.append(
                    {"domain": domain, "pmcid": pmcid, "reason": "METADATA_NOT_EXACT_CC_BY"}
                )
                continue
            seen.add(pmcid)
            tasks.append((domain, article))
            selected += 1
            # Fetch a bounded reserve because JATS license and chunk gates can
            # still reject metadata-qualified records.
            if selected >= args.documents_per_domain * 2:
                break

    def materialize(task: tuple[str, dict[str, Any]]) -> dict[str, Any]:
        domain, article = task
        pmcid = str(article["pmcid"])
        metadata_path = metadata_root / f"{pmcid}.json"
        xml_path = raw_root / f"{pmcid}.xml"
        try:
            if not metadata_path.is_file():
                metadata_path.write_text(
                    json.dumps(article, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
            if not xml_path.is_file():
                xml_path.write_bytes(fetch(f"{API}/{pmcid}/fullTextXML"))
            xml_bytes = xml_path.read_bytes()
            lower = xml_bytes.lower()
            has_cc_by = (
                b"creativecommons.org/licenses/by/" in lower
                and b"creativecommons.org/licenses/by-nc" not in lower
                and b"creativecommons.org/licenses/by-nd" not in lower
            ) or b"creative commons attribution license" in lower
            if not has_cc_by:
                raise RuntimeError("JATS_EXACT_CC_BY_MISSING")
            article_chunks = chunk_passages(
                collect_passages(ET.fromstring(xml_bytes))
            )
            if len(article_chunks) < 8:
                raise RuntimeError(f"INSUFFICIENT_CHUNKS_{len(article_chunks)}")
        except Exception as exc:
            return {
                "ok": False,
                "domain": domain,
                "pmcid": pmcid,
                "reason": str(exc),
            }
        return {
            "ok": True,
            "domain": domain,
            "article": article,
            "pmcid": pmcid,
            "metadata_path": metadata_path,
            "xml_path": xml_path,
            "xml_bytes": xml_bytes,
            "article_chunks": article_chunks,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        materialized = list(executor.map(materialize, tasks))

    domain_counts: dict[str, int] = defaultdict(int)
    for result in materialized:
        domain = result["domain"]
        pmcid = result["pmcid"]
        if not result["ok"]:
            rejected.append(
                {"domain": domain, "pmcid": pmcid, "reason": result["reason"]}
            )
            continue
        if domain_counts[domain] >= args.documents_per_domain:
            continue
        article = result["article"]
        metadata_path = result["metadata_path"]
        xml_path = result["xml_path"]
        xml_bytes = result["xml_bytes"]
        article_chunks = result["article_chunks"]
        split = split_for(pmcid)
        document = {
            "domain": domain,
            "pmcid": pmcid,
            "doi": article.get("doi"),
            "title": normalize_text(str(article.get("title", ""))),
            "publication_year": article.get("pubYear"),
            "license": "CC-BY_PER_METADATA_AND_JATS_VERIFIED",
            "license_metadata_value": article.get("license"),
            "metadata_url": f"{API}/search?query=PMCID:{pmcid}&format=json&resultType=core&pageSize=1",
            "fulltext_url": f"{API}/{pmcid}/fullTextXML",
            "metadata_path": str(metadata_path.relative_to(root)).replace("\\", "/"),
            "metadata_sha256": sha256_file(metadata_path),
            "xml_path": str(xml_path.relative_to(root)).replace("\\", "/"),
            "xml_sha256": sha256_file(xml_path),
            "xml_bytes": len(xml_bytes),
            "split": split,
            "chunks": len(article_chunks),
            "truth_class": "LITERATURE_CURATED_EXPERIMENT",
            "claim_state": "SOURCE_BOUND_PUBLICATION_TEXT_NOT_INDEPENDENT_GROUND_TRUTH",
            "training_role": "LICENSED_RAG_LANGUAGE_AND_EVIDENCE_CORPUS",
            "discovery_domain_is_experimental_label": False,
            "authority": 0,
        }
        admitted.append(document)
        for index, (section, text) in enumerate(article_chunks):
            chunks.append(
                {
                    "chunk_id": f"{pmcid}:{index:04d}",
                    "domain": domain,
                    "pmcid": pmcid,
                    "doi": article.get("doi"),
                    "title": document["title"],
                    "section": section,
                    "text": text,
                    "split": split,
                    "license": document["license"],
                    "truth_class": document["truth_class"],
                    "claim_state": document["claim_state"],
                    "source_sha256": document["xml_sha256"],
                    "authority": 0,
                }
            )
        domain_counts[domain] += 1

    document_counts = defaultdict(int)
    chunk_counts = defaultdict(lambda: defaultdict(int))
    for document in admitted:
        document_counts[document["domain"]] += 1
    for chunk in chunks:
        chunk_counts[chunk["domain"]][chunk["split"]] += 1
    errors = []
    for domain in DOMAIN_QUERIES:
        if document_counts[domain] < args.minimum_documents_per_domain:
            errors.append(f"{domain}:DOCUMENTS_{document_counts[domain]}")
        if set(chunk_counts[domain]) != {"train", "validation", "test"}:
            errors.append(f"{domain}:SPLIT_COVERAGE")
    if not args.minimum_chunks <= len(chunks) <= args.maximum_chunks:
        errors.append(f"CHUNK_GATE_{len(chunks)}")
    corpus_path = root / "data" / "corpora" / "ccby_multidomain_v2.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in chunks:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    document_splits = {item["pmcid"]: item["split"] for item in admitted}
    overlap = sum(
        1
        for pmcid in document_splits
        if len({chunk["split"] for chunk in chunks if chunk["pmcid"] == pmcid}) != 1
    )
    if overlap:
        errors.append("DOCUMENT_FAMILY_OVERLAP")
    manifest = {
        "schema": "cimc.forge200.ccby-multidomain-corpus.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "source_api": API,
        "user_agent": USER_AGENT,
        "search_receipts": search_receipts,
        "selection_policy": "REST_DISCOVERY_THEN_PER_DOCUMENT_METADATA_AND_JATS_EXACT_CC_BY_GATE",
        "split_policy": "SHA256_PMCID_70_15_15_BEFORE_CHUNKING",
        "split_unit": "PMCID_DOCUMENT_FAMILY",
        "documents": admitted,
        "rejected": rejected,
        "document_count": len(admitted),
        "document_counts_by_domain": dict(sorted(document_counts.items())),
        "chunk_count": len(chunks),
        "chunk_counts_by_domain_split": {
            domain: dict(sorted(values.items())) for domain, values in sorted(chunk_counts.items())
        },
        "corpus_path": str(corpus_path.relative_to(root)).replace("\\", "/"),
        "corpus_bytes": corpus_path.stat().st_size,
        "corpus_sha256": sha256_file(corpus_path),
        "document_family_overlap": overlap,
        "teacher_outputs": 0,
        "authority_nonzero": 0,
        "content_root_sha256": sha256_bytes(canonical_bytes(admitted)),
        "errors": errors,
    }
    write_json(root / "data" / "ledgers" / "ccby_multidomain_corpus.v2.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "documents": manifest["document_count"],
                "chunks": manifest["chunk_count"],
                "bytes": manifest["corpus_bytes"],
                "sha256": manifest["corpus_sha256"],
                "by_domain": manifest["document_counts_by_domain"],
                "errors": errors,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
