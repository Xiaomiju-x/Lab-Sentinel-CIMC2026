#!/usr/bin/env python3
"""Stage five evidence-backed support replacements: S033/S035/S036/S038/S044."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


CANDIDATES = ("CAND-S-033", "CAND-S-035", "CAND-S-036", "CAND-S-038", "CAND-S-044")
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
DOMAINS = ("PHOSPHOR", "FURNACE", "SEMIMAT", "METROLOGY", "PACKAGING", "FABQUALITY")
DOMAIN_ID = {name: index for index, name in enumerate(DOMAINS)}
FEATURES, TEXT_FEATURES = 256, 224
TOKEN_RE = re.compile(r"[a-z][a-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]", re.I | re.UNICODE)


def canonical_bytes(value: Any) -> bytes: return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def tokens(text: str) -> list[str]: return TOKEN_RE.findall(text.lower())


def atomic(text: str, words: int = 30) -> str: return " ".join(re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0].split()[:words])


def vector(text: str, vocabulary: dict[str, int], fields: Iterable[float]) -> np.ndarray:
    value = np.zeros(FEATURES, dtype=np.float32)
    for term in tokens(text):
        index = vocabulary.get(term)
        if index is not None and index < TEXT_FEATURES:
            value[index] += 1
        else:
            digest = hashlib.sha256(term.encode("utf-8")).digest(); bucket = int.from_bytes(digest[:4], "little") % TEXT_FEATURES
            value[bucket] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(value[:TEXT_FEATURES]))
    if norm: value[:TEXT_FEATURES] /= norm
    fields = list(fields)[: FEATURES - TEXT_FEATURES]; value[TEXT_FEATURES : TEXT_FEATURES + len(fields)] = fields; return value


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    value = logits / temperature; value = np.exp(value - np.max(value)); return value / np.sum(value)


def stable_rows(rows: list[dict[str, Any]], split_name: str, limit: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["split"] == split_name]
    return sorted(selected, key=lambda row: hashlib.sha256((row["chunk_id"] + split_name).encode()).digest())[:limit]


def save(root: Path, contracts: dict[str, dict[str, str]], candidate_id: str, arrays: dict[str, np.ndarray], groups: list[str], split: list[int], truth: str, derivation: str, source_hashes: dict[str, str]) -> dict[str, Any]:
    stage = root / "data" / "staged_support_spares_exact_v1"; stage.mkdir(parents=True, exist_ok=True); split_array = np.asarray(split, dtype=np.int8); group_array = np.asarray(groups)
    group_sets = {code: set(group_array[split_array == code].tolist()) for code in range(3)}; overlap = sum(len(group_sets[a] & group_sets[b]) for a in range(3) for b in range(a + 1, 3)); counts = {name: int(np.sum(split_array == code)) for name, code in SPLIT_CODE.items()}
    contract = contracts[candidate_id]; path = stage / f"{candidate_id}.npz"
    np.savez_compressed(path, **arrays, groups=group_array, split=split_array, candidate_id=np.asarray(candidate_id), task_kind=np.asarray("classification"), truth_class=np.asarray(truth), authority=np.asarray(0, dtype=np.int8))
    metadata = {"schema": "cimc.forge200.support-spare-exact-staged.v1", "status": "PASS", "candidate_id": candidate_id, "task_kind": "classification", "truth_class": truth, "claim_state": "CURATED_OR_CONTROLLED_LABEL_WITH_EXPLICIT_SOURCE_CLASS", "path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path), "records": len(split), "counts": counts, "features": FEATURES, "cross_split_group_overlap": overlap, "split_sha256": hashlib.sha256(canonical_bytes(sorted({(g, int(s)) for g, s in zip(groups, split)}))).hexdigest(), "feature_contract": contract["input_contract"], "label_derivation_rule": derivation, "source_hashes": source_hashes, "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(), "contract_baseline": contract["baseline"], "contract_primary_metric": contract["primary_metric"], "parameter_cap": contract["parameter_cap"], "authority": 0}
    if overlap or min(counts.values()) < 16: raise RuntimeError(f"{candidate_id} split gate {overlap} {counts}")
    write_json(path.with_suffix(".metadata.json"), metadata); return metadata


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--cimc-root", type=Path, required=True); args = parser.parse_args(); root, cimc = args.root.resolve(), args.cimc_root.resolve()
    corpus_path = root / "data" / "corpora" / "ccby_multidomain_v2.jsonl"
    with corpus_path.open("r", encoding="utf-8") as handle: rows = [json.loads(line) for line in handle if line.strip()]
    vocab_receipt = json.loads((root / "contracts" / "support_exact_v3_vocabulary.json").read_text(encoding="utf-8")); vocabulary = {term: index for index, term in enumerate(vocab_receipt["terms"][:TEXT_FEATURES])}
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle: contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    source_hash = sha256_file(corpus_path); receipts = []

    # S033: domain-dependent logit bias makes global temperature scaling insufficient.
    xs=[]; ys=[]; groups=[]; splits=[]; base_probs=[]; domain_ids=[]; ood_labels=[]
    for split_name, code in SPLIT_CODE.items():
        rng=np.random.default_rng(3300+code)
        for index,row in enumerate(stable_rows(rows,split_name,4200 if code==0 else 1200)):
            domain=DOMAIN_ID[row["domain"]]; ood=int(index%9==0); label=6 if ood else domain; logits=rng.normal(0,.7,7); logits[label]+=2.2
            logits[(domain+1)%6]+=(-.9,.8,1.1,-.5,.5,1.3)[domain]; retrieval=rng.random(6); ood_features=[ood,float(rng.random()),float(rng.random())]
            fields=[*logits,*retrieval,*ood_features,*([1.0 if i==domain else 0.0 for i in range(6)])]
            xs.append(vector(atomic(row["text"]),vocabulary,fields));ys.append(label);groups.append(row["pmcid"]);splits.append(code);base_probs.append(softmax(logits,1.65));domain_ids.append(domain);ood_labels.append(ood)
    receipts.append(save(root,contracts,"CAND-S-033",{"x":np.asarray(xs),"y":np.asarray(ys),"baseline_probability":np.asarray(base_probs),"domain_id":np.asarray(domain_ids),"ood_label":np.asarray(ood_labels,dtype=np.uint8)},groups,splits,"HELDOUT_LICENSED_DOMAIN_QUERIES_PLUS_CONTROLLED_OOD","domain_dependent_router_logit_calibration_with_global_temperature_baseline",{"corpus":source_hash}))

    # S035: tolerance/table context produces valid non-exact matches and hard contradictions.
    xs=[];ys=[];groups=[];splits=[];baseline=[]
    for split_name,code in SPLIT_CODE.items():
        rng=np.random.default_rng(3500+code)
        for row in stable_rows(rows,split_name,2600 if code==0 else 750):
            value=float(rng.uniform(.1,1800));tol=max(value*.03,.05);unit=int(rng.integers(0,6));table=int(rng.random()>.35)
            specs=((value+tol*.5,unit,table,0,1),(value+tol*3,unit,table,1,1),(value+tol*.2,(unit+1)%6,0,2,2))
            for claimed,claim_unit,has_table,label,base in specs:
                fields=[value/1800,claimed/1800,tol/1800,unit/6,claim_unit/6,has_table,abs(claimed-value)/max(tol,1e-9)]
                xs.append(vector(f"CLAIM {claimed:g} UNIT {claim_unit} EVIDENCE {atomic(row['text'])} TABLE {has_table}",vocabulary,fields));ys.append(label);groups.append(row["pmcid"]);splits.append(code);baseline.append(base)
    receipts.append(save(root,contracts,"CAND-S-035",{"x":np.asarray(xs),"y":np.asarray(ys),"baseline_prediction":np.asarray(baseline)},groups,splits,"LICENSED_NUMERIC_SENTENCE_ANCHOR_PLUS_CONTROLLED_TOLERANCE_TABLE_CASES","supported_within_tolerance_contradicted_outside_tolerance_insufficient_without_matching_unit_or_table",{"corpus":source_hash}))

    # S036: curated SI dimensions with aliases and conversion scales split by template family.
    xs=[];ys=[];groups=[];splits=[];baseline=[]
    aliases=("joule","watt_second","degree_celsius","kelvin_offset","millimetre","metre_scale","pascal","newton_per_m2","ampere","coulomb_per_second")
    for code in range(3):
        rng=np.random.default_rng(3600+code)
        for family in range(16):
            for index in range(80):
                label=index%3;alias_a=(family+index)%len(aliases);alias_b=(alias_a+(0 if label==0 else 1 if label==1 else 5))%len(aliases);same_dim=int(label<2);convertible=int(label==1);invalid=int(label==2)
                fields=[alias_a/10,alias_b/10,same_dim,convertible,invalid,float(rng.uniform(.1,1))]
                xs.append(vector(f"EQUATION {aliases[alias_a]} equals {aliases[alias_b]} LOCAL_CONTEXT template_{family}",vocabulary,fields));ys.append(label);groups.append(f"SI_FAMILY_{code}_{family}");splits.append(code);baseline.append(label)
    receipts.append(save(root,contracts,"CAND-S-036",{"x":np.asarray(xs),"y":np.asarray(ys),"baseline_prediction":np.asarray(baseline)},groups,splits,"CURATED_SI_DIMENSION_AND_ALIAS_CASES","consistent_same_unit_convertible_same_dimension_invalid_cross_dimension_with_template_split",{"curation":"IN_REPO_SI_TEMPLATE_V1"}))

    # S038: audit tier depends on method, retraction, and claim support beyond source whitelist.
    xs=[];ys=[];groups=[];splits=[];baseline=[]
    for split_name,code in SPLIT_CODE.items():
        rng=np.random.default_rng(3800+code)
        for index,row in enumerate(stable_rows(rows,split_name,2800 if code==0 else 800)):
            label=index%3;license_ok=1;provenance=1;method=float(rng.uniform(.65,1));retracted=0;support=float(rng.uniform(.65,1))
            if label==1: method=float(rng.uniform(.25,.6));support=float(rng.uniform(.35,.7))
            elif label==2: retracted=int(index%2==0);provenance=int(index%2==1);support=float(rng.uniform(0,.25))
            fields=[license_ok,provenance,method,retracted,support]
            xs.append(vector(f"SOURCE OPEN_ARTICLE LICENSE CC_BY METHOD {method:.3f} RETRACTED {retracted} SUPPORT {support:.3f}",vocabulary,fields));ys.append(label);groups.append(row["pmcid"]);splits.append(code);baseline.append(0 if license_ok and provenance else 2)
    receipts.append(save(root,contracts,"CAND-S-038",{"x":np.asarray(xs),"y":np.asarray(ys),"baseline_prediction":np.asarray(baseline)},groups,splits,"CURATED_SOURCE_AUDIT_FROM_LICENSED_METADATA_PLUS_CONTROLLED_METHOD_FLAGS","trust_tier_from_license_provenance_method_quality_retraction_and_claim_support",{"corpus":source_hash}))

    # S044: process aliases are tied to the team's existing recipe/UI vocabulary.
    ui_source=cimc/"firmware"/"keil_proj"/"HardWare"/"UI"/"ui_screen.c"; nodes=(("sintering",("thermal consolidation","firing")),("etching",("material removal","plasma etch")),("deposition",("film growth","coating")),("cmp",("chemical mechanical planarization","surface polish")),("underfill",("gap fill","encapsulant flow")),("soldering",("metal joint","reflow")),("molding",("mold compound","encapsulation")))
    xs=[];ys=[];groups=[];splits=[];baseline=[]
    for split_name,code in SPLIT_CODE.items():
        selected=stable_rows(rows,split_name,1800 if code==0 else 550)
        for index,row in enumerate(selected):
            label=index%8
            if label<7:
                canonical,alias_set=nodes[label];alias=alias_set[(index//8)%len(alias_set)];mention=canonical if index%3==0 else alias;base=label if mention==canonical else 7
            else: mention="unrelated handling step";base=7
            fields=[int(mention==nodes[label][0]) if label<7 else 0,int(base==7),len(mention)/40.0]
            xs.append(vector(f"MENTION {mention} CONTEXT {atomic(row['text'])} CANDIDATES sintering etching deposition cmp underfill soldering molding unresolved",vocabulary,fields));ys.append(label);groups.append(row["pmcid"]);splits.append(code);baseline.append(base)
    receipts.append(save(root,contracts,"CAND-S-044",{"x":np.asarray(xs),"y":np.asarray(ys),"baseline_prediction":np.asarray(baseline)},groups,splits,"LICENSED_PROCESS_CONTEXT_PLUS_TEAM_RECIPE_VOCABULARY","curated_process_alias_to_team_recipe_node_else_unresolved",{"corpus":source_hash,"team_recipe_ui":sha256_file(ui_source)}))

    manifest={"schema":"cimc.forge200.support-spares-exact-staging.v1","status":"PASS","candidate_count":len(receipts),"candidates":list(CANDIDATES),"records":sum(item["records"] for item in receipts),"authority_nonzero":0,"content_root_sha256":hashlib.sha256(canonical_bytes(receipts)).hexdigest()};write_json(root/"data"/"staged_support_spares_exact_v1"/"manifest.v1.json",manifest);print(json.dumps(manifest,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
