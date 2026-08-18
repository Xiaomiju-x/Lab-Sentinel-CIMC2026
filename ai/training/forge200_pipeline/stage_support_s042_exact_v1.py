#!/usr/bin/env python3
"""Stage S042 from frozen team evidence revisions and licensed source records."""
from __future__ import annotations

import argparse,csv,hashlib,json,re
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json

VERSION_RE=re.compile(r"(?:^|[._-])v(\d+)(?=[._-]|$)",re.I)
LABELS=("fresh","stale","superseded","time_irrelevant")
SCOPES=("CURRENT","AS_OF_PRIOR_REVISION","LATEST_FOR_BATCH","TIME_IRRELEVANT")

def family(path:Path)->str:return VERSION_RE.sub(".vN",path.stem)
def split_for(name:str)->int:
    value=int(hashlib.sha256(name.encode("utf-8")).hexdigest()[:8],16)%100
    return 0 if value<70 else 1 if value<85 else 2

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();root=args.root.resolve();paths=sorted(list((root/"evidence").glob("*.json"))+list((root/"data"/"ledgers").glob("*.json")))
    source=[]
    for path in paths:
        try:data=json.loads(path.read_text(encoding="utf-8"))
        except Exception:continue
        created=data.get("created_at_utc") or data.get("generated_at_utc") or data.get("timestamp_utc")
        try:stamp=datetime.fromisoformat(str(created).replace("Z","+00:00")) if created else datetime.fromtimestamp(path.stat().st_mtime,timezone.utc)
        except Exception:stamp=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc)
        match=VERSION_RE.search(path.name);version=int(match.group(1)) if match else 1;source_type=0 if path.parent.name=="evidence" else 1;source.append({"path":path,"family":family(path),"version":version,"stamp":stamp,"source_type":source_type,"status":str(data.get("status","UNKNOWN"))})
    grouped=defaultdict(list)
    for record in source:grouped[record["family"]].append(record)
    records=[];frozen=datetime(2026,8,3,8,0,0,tzinfo=timezone.utc)
    for fam,items in sorted(grouped.items()):
        max_version=max(item["version"] for item in items);max_stamp=max(item["stamp"] for item in items);code=split_for(fam)
        for item in items:
            age_days=max((frozen-item["stamp"]).total_seconds()/86400.,0.);revision_gap=max_version-item["version"]
            for scope_index,scope in enumerate(SCOPES):
                if scope=="TIME_IRRELEVANT":label=3
                elif revision_gap>0:label=2
                elif scope=="AS_OF_PRIOR_REVISION" and item["version"]==max_version and max_version>1:label=1
                elif age_days>30.:label=1
                else:label=0
                status_hash=int(hashlib.sha256(item["status"].encode()).hexdigest()[:4],16)%17/16.;name_hash=int(hashlib.sha256(fam.encode()).hexdigest()[:4],16)%31/30.
                features=[min(age_days/365.,2.),min(item["version"]/10.,2.),min(max_version/10.,2.),min(revision_gap/10.,1.),float(item["stamp"]==max_stamp),float(item["source_type"]),float(scope_index==0),float(scope_index==1),float(scope_index==2),float(scope_index==3),status_hash,name_hash]
                records.append((features,label,code,fam,str(item["path"].relative_to(root)).replace('\\','/'),scope))
    x=np.asarray([record[0] for record in records],np.float32);y=np.asarray([record[1] for record in records],np.int64);split=np.asarray([record[2] for record in records],np.int8);groups=np.asarray([record[3] for record in records]);source_path=np.asarray([record[4] for record in records]);scope=np.asarray([record[5] for record in records]);sets={code:set(groups[split==code]) for code in range(3)};overlap=sum(len(sets[a]&sets[b]) for a,b in ((0,1),(0,2),(1,2)));split_counts={name:int(np.sum(split==code)) for code,name in enumerate(("train","validation","test"))};class_counts={name:{LABELS[label]:int(np.sum((split==code)&(y==label))) for label in range(4)} for code,name in enumerate(("train","validation","test"))}
    if overlap or min(split_counts.values())<100 or min(min(value.values()) for value in class_counts.values())<10:raise RuntimeError(f"DATA_GATE:{overlap}:{split_counts}:{class_counts}")
    baseline=np.where(x[:,9]>.5,0,np.where(x[:,0]>(30./365.),1,0)).astype(np.int64)
    with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as handle:contracts={row["candidate_id"]:row for row in csv.DictReader(handle,delimiter="\t")}
    cid="CAND-S-042";contract=contracts[cid];output=root/"data"/"staged_support_s042_exact_v1"/f"{cid}.npz";output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,x=x,y=y,baseline_pred=baseline,group=groups,split=split,source_path=source_path,query_scope=scope)
    frozen_sources=[{"path":str(record["path"].relative_to(root)).replace('\\','/'),"sha256":sha256_file(record["path"]),"family":record["family"],"version":record["version"]} for record in source];metadata={"schema":"cimc.forge200.support-s042-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"STRUCTURE_DERIVED","source_gate":contract["source_gate"],"source_gate_match":"TEAM_LEDGER_PLUS_LICENSED_REVISION_CASES_WITH_SOURCE_SPLIT","frozen_cutoff_utc":frozen.isoformat(),"records":len(y),"source_files":len(source),"source_families":len(grouped),"source_manifest_root_sha256":hashlib.sha256(canonical_bytes(frozen_sources)).hexdigest(),"split_counts":split_counts,"class_counts":class_counts,"cross_split_source_family_overlap":overlap,"label_derivation":"revision_order_timestamp_and_query_temporal_scope","baseline_execution":"FIXED_30_DAY_AGE_THRESHOLD_IGNORING_REVISION_AND_QUERY_SCOPE","input_contract":contract["input_contract"],"target_label":contract["target_label"],"primary_metric":contract["primary_metric"],"parameter_cap":contract["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(contract)).hexdigest(),"source_license":"TEAM_OWNED_LEDGER_AND_LICENSED_SOURCE_METADATA","authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(output)}
    write_json(output.with_suffix(".metadata.json"),metadata);write_json(root/"evidence"/"support_s042_exact_staging.v1.json",{"schema":"cimc.forge200.support-s042-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":metadata,"authority_nonzero":0,"board_actions":0});print(json.dumps({"status":"PASS","candidate_id":cid,"records":len(y),"split_counts":split_counts,"class_counts":class_counts},sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
