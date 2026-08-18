#!/usr/bin/env python3
"""Host conformance and mutation checks for the frozen Forge200 interfaces."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
MODEL_ID = re.compile(r"^ICM-[0-9]{3}$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_root(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("content_root_sha256", None)
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_evidence(card: dict, schema: dict) -> None:
    required = set(schema["required"])
    allowed = set(schema["properties"])
    require(required <= set(card), "required")
    require(set(card) <= allowed, "additional")
    require(card["schema"] == "cimc.evidence-card.v2", "schema")
    require(card["authority"] == 0, "authority")
    require(isinstance(card["seq"], int) and card["seq"] >= 0, "seq")
    require(isinstance(card["monotonic_ms"], int) and card["monotonic_ms"] >= 0, "monotonic")
    require(isinstance(card["age_ms"], int) and card["age_ms"] >= 0, "age")
    require(0 <= float(card["quality"]) <= 1, "quality")
    require(1 <= len(card["session_id"]) <= 40, "session")
    require(1 <= len(card["run_id"]) <= 40, "run")
    require(1 <= len(card["source_id"]) <= 64, "source")
    require(1 <= len(card["independence_family"]) <= 64, "independence")
    require(card["stage"] in schema["properties"]["stage"]["enum"], "stage")
    require(card["truth_class"] in schema["properties"]["truth_class"]["enum"], "truth")
    require(bool(HEX64.fullmatch(card["summary_sha256"])), "summary")
    datetime.fromisoformat(card["rtc_utc"].replace("Z", "+00:00"))
    if card.get("model_id") is not None:
        require(bool(MODEL_ID.fullmatch(card["model_id"])), "model_id")
    if card.get("model_release_root") is not None:
        require(bool(HEX64.fullmatch(card["model_release_root"])), "release_root")
    parents = card.get("parent_evidence_ids", [])
    require(len(parents) <= 8 and len(parents) == len(set(parents)), "parents")
    interval = card.get("interval")
    if interval is not None:
        require(set(interval) == {"lower", "upper", "coverage"}, "interval_fields")
        require(interval["lower"] <= interval["upper"], "interval_order")
        require(0 < interval["coverage"] < 1, "coverage")


def validate_sintergraph(request: dict, sinter_schema: dict, evidence_schema: dict) -> None:
    required = set(sinter_schema["required"])
    allowed = set(sinter_schema["properties"])
    require(required <= set(request), "required")
    require(set(request) <= allowed, "additional")
    require(request["schema"] == "cimc.sintergraph-psp.r1", "schema")
    require(request["authority"] == 0, "authority")
    require(request["forbidden_same_run_post_sinter_sources"] == [
        "XRD", "PL", "SEM", "EDS", "POST_RUN_QUALITY"
    ], "forbidden_sources")
    require(len(request["planned_curve"]) >= 2, "curve")
    previous_t = -1.0
    for point in request["planned_curve"]:
        require({"t_s", "temperature_c"} <= set(point), "curve_point")
        require(float(point["t_s"]) >= previous_t, "curve_order")
        previous_t = float(point["t_s"])
    forbidden = {"XRD", "PL", "SEM", "EDS", "POST_RUN_QUALITY"}
    for card in request["evidence_cards"]:
        validate_evidence(card, evidence_schema)
        require(card["seq"] <= request["as_of_seq"], "future_seq")
        require(card["monotonic_ms"] <= request["as_of_monotonic_ms"], "future_time")
        if card["run_id"] == request["run_id"]:
            source = card["source_id"].upper()
            require(not any(token in source for token in forbidden), "same_run_postburn_source")
            require(card["stage"] not in {"POST_RUN", "METROLOGY"}, "same_run_postburn_stage")


def validate_chrono(spec: dict) -> None:
    require(spec["schema"] == "cimc.chronospec-r4.events.v1", "schema")
    require(spec["authority"] == 0, "authority")
    ids = [event["id"] for event in spec["events"]]
    names = [event["name"] for event in spec["events"]]
    require(len(ids) == len(set(ids)), "duplicate_id")
    require(len(names) == len(set(names)), "duplicate_name")
    require(all(event["deadline_ms"] >= 0 for event in spec["events"]), "deadline")
    required = {
        "MODEL_LOAD_BEGIN", "MODEL_SCHEMA_VERIFIED", "MODEL_SHA256_VERIFIED",
        "MODEL_GENERATION_VERIFIED", "MODEL_GOLDEN_VERIFIED", "MODEL_COMMIT",
        "MODEL_ROLLBACK_REFUSE", "RAG_SOURCE_FRESHNESS_CHECK", "RAG_QUERY_TIMEOUT",
        "PTC_COMMAND_ISSUED", "PTC_CURRENT_OBSERVED", "THERMAL_RESPONSE_OBSERVED",
        "SINTERGRAPH_PREDICTION_FROZEN", "SINTERGRAPH_FULFILLMENT_AVAILABLE",
        "PROOFPASS_WAL_PREPARE", "PROOFPASS_SYNC_COMMIT", "PROOFPASS_RECOVERY",
    }
    require(required <= set(names), "required_events")


def validate_commit_trace(trace: list[str]) -> None:
    required = [
        "MODEL_LOAD_BEGIN", "MODEL_SCHEMA_VERIFIED", "MODEL_SHA256_VERIFIED",
        "MODEL_GENERATION_VERIFIED", "MODEL_GOLDEN_VERIFIED", "MODEL_COMMIT",
    ]
    cursor = 0
    for event in trace:
        if cursor < len(required) and event == required[cursor]:
            cursor += 1
    require(cursor == len(required), "commit_order")
    require(trace[-1] == "MODEL_COMMIT", "commit_terminal")


def run_case(name: str, fn: Callable[[], Any], expected_pass: bool) -> dict:
    try:
        fn()
        actual_pass = True
        detail = "accepted"
    except (ValueError, KeyError, TypeError) as exc:
        actual_pass = False
        detail = f"rejected:{exc}"
    if actual_pass != expected_pass:
        raise RuntimeError(f"CASE_MISMATCH:{name}:{detail}")
    return {"name": name, "expected": "PASS" if expected_pass else "REJECT", "result": detail}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    schema_dir = root / "contracts" / "schemas"
    evidence_path = schema_dir / "evidence_card_v2.schema.json"
    sinter_path = schema_dir / "sintergraph_psp_r1.schema.json"
    chrono_path = schema_dir / "chronospec_r4.events.v1.json"
    evidence_schema = json.loads(evidence_path.read_text(encoding="utf-8"))
    sinter_schema = json.loads(sinter_path.read_text(encoding="utf-8"))
    chrono = json.loads(chrono_path.read_text(encoding="utf-8"))

    card = {
        "schema": "cimc.evidence-card.v2", "session_id": "session-1",
        "run_id": "run-1", "seq": 7, "monotonic_ms": 7000,
        "rtc_utc": "2026-08-03T08:00:00Z", "stage": "SOAK",
        "source_id": "MAX31856", "model_id": None, "model_release_root": None,
        "truth_class": "TEAM_MEASURED", "quality": 0.98, "age_ms": 10,
        "independence_family": "thermocouple-k", "parent_evidence_ids": [],
        "interval": {"lower": 799.0, "upper": 801.0, "coverage": 0.95},
        "as_of_seq": 7, "as_of_monotonic_ms": 7000, "authority": 0,
        "summary_sha256": "a" * 64,
    }
    request = {
        "schema": "cimc.sintergraph-psp.r1", "run_id": "run-1",
        "as_of_seq": 10, "as_of_monotonic_ms": 10000,
        "recipe": {"material_family": "phosphor"},
        "planned_curve": [{"t_s": 0, "temperature_c": 25}, {"t_s": 900, "temperature_c": 800}],
        "evidence_cards": [card],
        "forbidden_same_run_post_sinter_sources": ["XRD", "PL", "SEM", "EDS", "POST_RUN_QUALITY"],
        "authority": 0,
    }
    cases: list[dict] = []
    cases.append(run_case("evidence_valid", lambda: validate_evidence(card, evidence_schema), True))
    for name, mutate in [
        ("evidence_authority_nonzero", lambda x: x.update(authority=1)),
        ("evidence_unknown_field", lambda x: x.update(actuator_command=True)),
        ("evidence_bad_release_root", lambda x: x.update(model_release_root="bad")),
        ("evidence_nonunique_parent", lambda x: x.update(parent_evidence_ids=["a", "a"])),
        ("evidence_invalid_interval", lambda x: x.update(interval={"lower": 2, "upper": 1, "coverage": .95})),
    ]:
        bad = copy.deepcopy(card); mutate(bad)
        cases.append(run_case(name, lambda bad=bad: validate_evidence(bad, evidence_schema), False))

    cases.append(run_case("sintergraph_valid", lambda: validate_sintergraph(request, sinter_schema, evidence_schema), True))
    mutations = [
        ("sintergraph_authority_nonzero", lambda x: x.update(authority=1)),
        ("sintergraph_future_seq", lambda x: x["evidence_cards"][0].update(seq=11)),
        ("sintergraph_future_time", lambda x: x["evidence_cards"][0].update(monotonic_ms=10001)),
        ("sintergraph_same_run_xrd", lambda x: x["evidence_cards"][0].update(source_id="XRD")),
        ("sintergraph_same_run_postrun", lambda x: x["evidence_cards"][0].update(stage="POST_RUN")),
        ("sintergraph_forbidden_list_changed", lambda x: x.update(forbidden_same_run_post_sinter_sources=[])),
        ("sintergraph_curve_too_short", lambda x: x.update(planned_curve=x["planned_curve"][:1])),
        ("sintergraph_unknown_field", lambda x: x.update(control_output=1)),
    ]
    for name, mutate in mutations:
        bad = copy.deepcopy(request); mutate(bad)
        cases.append(run_case(name, lambda bad=bad: validate_sintergraph(bad, sinter_schema, evidence_schema), False))

    cases.append(run_case("chronospec_valid", lambda: validate_chrono(chrono), True))
    for name, mutate in [
        ("chronospec_authority_nonzero", lambda x: x.update(authority=1)),
        ("chronospec_duplicate_id", lambda x: x["events"][1].update(id=x["events"][0]["id"])),
        ("chronospec_negative_deadline", lambda x: x["events"][0].update(deadline_ms=-1)),
        ("chronospec_required_event_missing", lambda x: x.update(events=x["events"][:-1])),
    ]:
        bad = copy.deepcopy(chrono); mutate(bad)
        cases.append(run_case(name, lambda bad=bad: validate_chrono(bad), False))

    valid_trace = [
        "MODEL_LOAD_BEGIN", "MODEL_SCHEMA_VERIFIED", "MODEL_SHA256_VERIFIED",
        "MODEL_GENERATION_VERIFIED", "MODEL_GOLDEN_VERIFIED", "MODEL_COMMIT",
    ]
    cases.append(run_case("commit_trace_valid", lambda: validate_commit_trace(valid_trace), True))
    cases.append(run_case("commit_without_golden", lambda: validate_commit_trace(valid_trace[:-2] + ["MODEL_COMMIT"]), False))
    cases.append(run_case("commit_wrong_order", lambda: validate_commit_trace([valid_trace[0], valid_trace[2], valid_trace[1]] + valid_trace[3:]), False))

    result = {
        "schema": "cimc.forge200.interface-freeze-verification.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_HOST_CONFORMANCE_AND_MUTATION_BOARD_PENDING",
        "schemas": {
            "EvidenceCard_v2": {"path": str(evidence_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(evidence_path)},
            "SinterGraph_PSP_R1": {"path": str(sinter_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(sinter_path)},
            "ChronoSpec_R4": {"path": str(chrono_path.relative_to(root)).replace("\\", "/"), "sha256": sha256(chrono_path)},
        },
        "case_count": len(cases), "cases_passed_as_expected": len(cases),
        "cases": cases, "authority_nonzero": 0, "board_accepted": False,
        "claim_boundary": "Host interface conformance only; firmware and physical timing remain board pending.",
    }
    result["content_root_sha256"] = canonical_root(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cases": len(cases), "content_root_sha256": result["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
