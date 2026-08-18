#!/usr/bin/env python3
"""Compile and exercise the frozen six-domain RAG state machine on host.

Host timing is diagnostic only.  The receipt deliberately leaves GD32 DWT,
microSD throughput, shared-SPI behavior, and power-loss recovery board-pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOMAINS = ("PHOSPHOR", "FURNACE", "SEMIMAT", "METROLOGY", "PACKAGING", "FABQUALITY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def command(root: Path, executable: Path, files: tuple[Path, Path, Path, Path]) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.run(
        [str(executable), *(str(path) for path in files)],
        cwd=root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=600,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(f"HOST_RUN_FAILED:{process.returncode}:{process.stderr}:{process.stdout[:1000]}")
    parsed = json.loads(process.stdout)
    parsed["wall_seconds"] = elapsed
    return parsed


def mutation_case(
    root: Path,
    executable: Path,
    files: tuple[Path, Path, Path, Path],
    index: int,
    name: str,
    mutation_root: Path,
) -> dict[str, Any]:
    mutated = mutation_root / f"{name}.bin"
    raw = bytearray(files[index].read_bytes())
    offset = len(raw) - 17
    raw[offset] ^= 0x5A
    mutated.write_bytes(raw)
    trial = list(files)
    trial[index] = mutated
    process = subprocess.run(
        [str(executable), *(str(path) for path in trial)],
        cwd=root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=120,
        check=False,
    )
    return {
        "name": name,
        "target": files[index].name,
        "mutation_offset": offset,
        "rejected": process.returncode != 0,
        "returncode": process.returncode,
        "mutated_sha256": sha256_file(mutated),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--zig", type=Path, required=True)
    parser.add_argument("--cimc", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release = args.release.resolve()
    cimc = args.cimc.resolve()
    manifest_path = release / "MANIFEST.v9.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["content_root_sha256"] != "e8a9e983451551bd22f82f4e59b10cbd1c4fb92a45b0763d700284779b77d956":
        raise RuntimeError("UNEXPECTED_RELEASE_CONTENT_ROOT")

    source = root / "firmware_integration/rag_v9"
    runtime = root / "firmware_integration/modelbank_v8"
    lab = cimc / "firmware/keil_proj/HardWare/Lab_Sentinel"
    temporary = root / ".tmp/rag_v9_acceptance"
    temporary.mkdir(parents=True, exist_ok=True)
    executable = temporary / "forge200_rag_host_runner_v9.exe"
    compile_command = [
        str(args.zig.resolve()), "cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
        "-I", str(source), "-I", str(runtime), "-I", str(lab),
        str(source / "forge200_rag_v9.c"),
        str(source / "forge200_rag_host_runner_v9.c"),
        str(runtime / "forge200_runtime_v8.c"),
        str(lab / "sha256.c"), "-lm", "-o", str(executable),
    ]
    compiled = subprocess.run(compile_command, cwd=root, text=True, capture_output=True, timeout=180, check=False)
    if compiled.returncode != 0:
        raise RuntimeError(f"HOST_COMPILE_FAILED:{compiled.stdout}:{compiled.stderr}")

    domain_runs = []
    all_queries = []
    paths_by_domain: list[tuple[Path, Path, Path, Path]] = []
    lm_by_domain = {item["candidate_id"]: item for item in manifest["lm_packages"]}
    for domain_id, domain in enumerate(DOMAINS):
        workload = next(item for item in manifest["workloads"] if item["domain_id"] == domain_id)
        support = next(item for item in manifest["support_bundles"] if item["domain_id"] == domain_id)
        lm_id = next(item["lm_candidate_id"] for item in manifest["workload_records"] if item["domain"] == domain)
        lm = lm_by_domain[lm_id]
        files = (
            release / support["path"],
            release / workload["path"],
            release / lm["path"],
            release / lm["golden_path"],
        )
        paths_by_domain.append(files)
        result = command(root, executable, files)
        queries = result["queries"]
        if len(queries) != 20:
            raise RuntimeError(f"DOMAIN_QUERY_COUNT:{domain}:{len(queries)}")
        all_queries.extend(queries)
        domain_runs.append({
            "domain": domain,
            "domain_id": domain_id,
            "lm_candidate_id": lm_id,
            "wall_seconds": result["wall_seconds"],
            **result["summary"],
        })

    mutation_root = temporary / "mutations"
    mutation_root.mkdir(parents=True, exist_ok=True)
    first = paths_by_domain[0]
    mutations = [
        mutation_case(root, executable, first, 0, "support_body_corrupt", mutation_root),
        mutation_case(root, executable, first, 1, "workload_body_corrupt", mutation_root),
        mutation_case(root, executable, first, 2, "lm_payload_corrupt", mutation_root),
        mutation_case(root, executable, first, 3, "lm_golden_corrupt", mutation_root),
    ]

    total = len(all_queries)
    safe = sum(item["safe"] for item in all_queries)
    negative = [item for item in all_queries if item["expected_refusal"]]
    positive = [item for item in all_queries if not item["expected_refusal"]]
    published = [item for item in all_queries if item["published"]]
    source_bound = [item for item in all_queries if item["source_bound"]]
    exact_generation = sum(item["generation_exact"] for item in all_queries)
    state_complete = sum(item["state_mask"] == 255 for item in all_queries)
    zeroized = sum(item["lm_zeroized"] and item["workspace_zeroized"] for item in all_queries)
    support_complete = sum(item["support_models"] == 13 for item in all_queries)
    cold_read_max = max(item["cold_read_bytes"] for item in all_queries)
    # This deterministic transfer-plus-stage budget is a planning calculation,
    # never represented as a board latency measurement.
    cold_projection_seconds = cold_read_max / 524288.0 + 11.0
    warm_projection_seconds = 7.5
    gates = {
        "query_count_120": total == 120,
        "balanced_60_60": len(negative) == 60 and len(positive) == 60,
        "safe_outcomes_120": safe == 120,
        "negative_refusal_60": sum(item["refused"] for item in negative) == 60,
        "published_are_source_bound": len(published) == len(source_bound),
        "at_least_one_source_bound_answer": len(source_bound) > 0,
        "state_machine_8_of_8": state_complete == 120,
        "support_bundle_13_executed": support_complete == 120,
        "lm_and_workspace_zeroized": zeroized == 120,
        "generation_tokens_max_24": max(item["generation_tokens"] for item in all_queries) <= 24,
        "cold_read_le_4mib": cold_read_max <= 4 * 1024 * 1024,
        "support_bundle_le_1mib": manifest["max_support_bundle_bytes"] <= 1024 * 1024,
        "lm_le_2mib": manifest["max_lm_package_bytes"] <= 2 * 1024 * 1024,
        "index_le_1mib": manifest["max_index_evidence_bytes"] <= 1024 * 1024,
        "authority_zero": manifest["authority_nonzero"] == 0,
        "four_hash_mutations_rejected": all(item["rejected"] for item in mutations),
        "host_compile_werror": compiled.returncode == 0,
    }
    status = "PASS_HOST_RAG_BOARD_TIMING_AND_SHARED_SPI_PENDING" if all(gates.values()) else "REJECTED"
    receipt = {
        "schema": "cimc.forge200.rag-runtime-host-acceptance.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "release": {
            "path": release.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "content_root_sha256": manifest["content_root_sha256"],
        },
        "host_executable": {
            "path": executable.relative_to(root).as_posix(),
            "sha256": sha256_file(executable),
            "strict_flags": ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror"],
        },
        "workload": {
            "total": total,
            "positive": len(positive),
            "negative": len(negative),
            "safe": safe,
            "published": len(published),
            "source_bound": len(source_bound),
            "refused": sum(item["refused"] for item in all_queries),
            "positive_refused": sum(item["refused"] for item in positive),
            "negative_refused": sum(item["refused"] for item in negative),
            "generation_exact": exact_generation,
            "state_complete": state_complete,
            "zeroized": zeroized,
            "support_models_13": support_complete,
        },
        "domain_runs": domain_runs,
        "resource_limits": {
            "max_cold_read_bytes": cold_read_max,
            "max_support_bundle_bytes": manifest["max_support_bundle_bytes"],
            "max_lm_package_bytes": manifest["max_lm_package_bytes"],
            "max_index_evidence_bytes": manifest["max_index_evidence_bytes"],
            "max_executing_models": manifest["max_executing_models"],
            "max_resident_packages": manifest["resident_packages_max"],
        },
        "timing_boundary": {
            "host_wall_seconds_total": sum(item["wall_seconds"] for item in domain_runs),
            "cold_projection_at_512kib_s_seconds": cold_projection_seconds,
            "warm_projection_seconds": warm_projection_seconds,
            "projection_only": True,
            "gd32_dwt_measured": False,
            "micro_sd_measured_this_run": False,
            "shared_spi_measured_this_run": False,
        },
        "mutations": mutations,
        "gates": gates,
        "authority": 0,
        "board_accepted": False,
        "board_pending": [
            "GD32_DWT_STAGE_LATENCY",
            "MICROSD_512KIB_S_OR_HIGHER",
            "MAX31856_SHARED_SPI_NONINTERFERENCE",
            "POWER_LOSS_RECOVERY",
            "24H_CONCURRENT_STABILITY",
        ],
    }
    receipt["content_root_sha256"] = hashlib.sha256(canonical(receipt)).hexdigest()
    output = root / "evidence/rag_runtime_host_acceptance.v9.json"
    write_json(output, receipt)
    print(json.dumps({
        "status": status,
        "queries": total,
        "safe": safe,
        "source_bound": len(source_bound),
        "negative_refused": sum(item["refused"] for item in negative),
        "mutations_rejected": sum(item["rejected"] for item in mutations),
        "max_cold_read_bytes": cold_read_max,
        "content_root_sha256": receipt["content_root_sha256"],
    }, sort_keys=True))
    return 0 if status.startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
