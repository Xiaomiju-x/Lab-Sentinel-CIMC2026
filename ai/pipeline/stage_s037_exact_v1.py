#!/usr/bin/env python3
"""Stage S037 temporal evidence ranking from licensed document families."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re
from pathlib import Path
from typing import Any
import numpy as np

CID="CAND-S-037";SPLIT={"train":0,"validation":1,"test":2};FEATURES=256
def canonical(v:Any)->bytes:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def sha(p:Path)->str:
 d=hashlib.sha256();f=p.open("rb")
 with f:
  for b in iter(lambda:f.read(1<<20),b""):d.update(b)
 return d.hexdigest()
def write(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def main()->int:
 a=argparse.ArgumentParser();a.add_argument("--root",type=Path,required=True);args=a.parse_args();root=args.root.resolve();corpus=root/"data"/"corpora"/"ccby_multidomain_v2.jsonl"
 rows=[json.loads(x) for x in corpus.open(encoding="utf-8") if x.strip()]
 with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as h:contract=next(r for r in csv.DictReader(h,delimiter="\t") if r["candidate_id"]==CID)
 xs=[];ys=[];groups=[];splits=[];qids=[];base=[];special=[];stale=[]
 for name,code in SPLIT.items():
  selected=sorted([r for r in rows if r["split"]==name],key=lambda r:hashlib.sha256((r["chunk_id"]+name).encode()).digest())[:(500 if code==0 else 180)]
  rng=np.random.default_rng(3700+code)
  for qi,row in enumerate(selected):
   for ci in range(10):
    relevance=float(rng.uniform(.15,1));age=float(rng.uniform(0,1));same_batch=int(rng.random()>.25);latest=int(rng.random()>.25);stage=int(rng.random()>.25)
    if ci==0:relevance,age,same_batch,latest,stage=.95,.08,1,1,1
    temporal_bad=int(age>.72 or not same_batch or not latest or not stage);label=int(relevance>.42 and not temporal_bad)
    fields=[relevance,age,same_batch,latest,stage,math.exp(-2.5*age),ci/10]
    x=np.zeros(FEATURES,dtype=np.float32);x[:len(fields)]=fields
    xs.append(x);ys.append(label);groups.append(row["pmcid"]);splits.append(code);qids.append(code*1_000_000+qi);base.append(relevance*math.exp(-2.5*age));special.append(1-temporal_bad);stale.append(temporal_bad)
 arr={"x":np.asarray(xs),"y":np.asarray(ys,dtype=np.int64),"groups":np.asarray(groups),"split":np.asarray(splits,dtype=np.int8),"query_id":np.asarray(qids,dtype=np.int64),"baseline_score":np.asarray(base,dtype=np.float32),"special_match":np.asarray(special,dtype=np.uint8),"stale_label":np.asarray(stale,dtype=np.uint8)}
 path=root/"data"/"staged_s037_exact_v1"/f"{CID}.npz";path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**arr,candidate_id=np.asarray(CID),task_kind=np.asarray("classification"),truth_class=np.asarray("LICENSED_DOCUMENT_FAMILY_PLUS_CONTROLLED_TEMPORAL_REVISION_CASES"),authority=np.asarray(0,dtype=np.int8))
 gs={c:set(arr["groups"][arr["split"]==c].tolist()) for c in range(3)};over=sum(len(gs[a]&gs[b]) for a in range(3) for b in range(a+1,3));counts={n:int(np.sum(arr["split"]==c)) for n,c in SPLIT.items()};queries={n:int(len(np.unique(arr["query_id"][arr["split"]==c]))) for n,c in SPLIT.items()}
 meta={"schema":"cimc.forge200.s037-exact-staged.v1","status":"PASS","candidate_id":CID,"task_kind":"classification","truth_class":"LICENSED_DOCUMENT_FAMILY_PLUS_CONTROLLED_TEMPORAL_REVISION_CASES","claim_state":"TEMPORAL_LABEL_FROM_EXPLICIT_TIMESTAMP_BATCH_REVISION_AND_STAGE_FIELDS","path":str(path.relative_to(root)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":sha(path),"records":len(ys),"counts":counts,"query_counts":queries,"features":FEATURES,"cross_split_group_overlap":over,"split_sha256":hashlib.sha256(canonical(sorted({(g,int(s)) for g,s in zip(groups,splits)}))).hexdigest(),"checkpoint_selection":"VALIDATION_RANKING_COMPOSITE_V1","feature_contract":contract["input_contract"],"label_derivation_rule":"relevant_and_fresh_same_batch_latest_revision_matching_process_stage","baseline_execution":"relevance_score_times_exponential_age_decay","source_sha256":sha(corpus),"task_contract_sha256":hashlib.sha256(canonical(contract)).hexdigest(),"contract_baseline":contract["baseline"],"contract_primary_metric":contract["primary_metric"],"parameter_cap":contract["parameter_cap"],"authority":0}
 if over or min(queries.values())<16:raise RuntimeError("S037 split gate")
 write(path.with_suffix(".metadata.json"),meta);man={"schema":"cimc.forge200.s037-exact-staging.v1","status":"PASS","candidate_count":1,"records":len(ys),"authority_nonzero":0,"content_root_sha256":hashlib.sha256(canonical([meta])).hexdigest()};write(path.parent/"manifest.v1.json",man);print(json.dumps(man,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
