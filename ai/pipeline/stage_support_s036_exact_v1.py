#!/usr/bin/env python3
"""Create source-gate-exact curated SI unit cases for S036."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json

SEED=20260803
CONTEXTS=("length","time","mass","temperature","current","pressure","energy","power")
UNITS={
 "length":[("m",1.),("mm",1e-3),("um",1e-6),("cm",1e-2)],"time":[("s",1.),("ms",1e-3),("min",60.),("h",3600.)],"mass":[("kg",1.),("g",1e-3),("mg",1e-6)],"temperature":[("K",1.),("degC",1.),("C",1.)],"current":[("A",1.),("mA",1e-3),("uA",1e-6)],"pressure":[("Pa",1.),("kPa",1e3),("MPa",1e6),("bar",1e5)],"energy":[("J",1.),("mJ",1e-3),("eV",1.602176634e-19)],"power":[("W",1.),("mW",1e-3),("kW",1e3)]}
AMBIGUOUS={"m":{"length","time"},"C":{"temperature","current"},"J":{"energy","current"},"W":{"power","energy"}}

def split_for(name):
 value=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)%100;return 0 if value<70 else 1 if value<85 else 2
def hashed(token,bins=24):
 out=np.zeros(bins,np.float32);digest=hashlib.sha256(token.encode()).digest();out[int.from_bytes(digest[:2],"little")%bins]=1. if digest[2]&1 else -1.;return out

def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve();rng=np.random.default_rng(SEED);records=[];families=[]
 for family_index in range(160):
  family=f"SI-TEMPLATE-{family_index:03d}";context=CONTEXTS[family_index%len(CONTEXTS)];code=split_for(family);families.append({"family":family,"context":context,"split":code})
  units=UNITS[context]
  for sample in range(30):
   label=sample%3;left_name,left_scale=units[int(rng.integers(len(units)))]
   if label==0:right_name,right_scale=left_name,left_scale
   elif label==1:
    choices=[value for value in units if value[0]!=left_name];right_name,right_scale=choices[int(rng.integers(len(choices)))]
   else:
    other=CONTEXTS[(CONTEXTS.index(context)+1+int(rng.integers(len(CONTEXTS)-1)))%len(CONTEXTS)];right_name,right_scale=UNITS[other][int(rng.integers(len(UNITS[other])))]
   value=float(10**rng.uniform(-3,3));ratio=math.log10(max(left_scale,1e-30)/max(right_scale,1e-30));context_onehot=[1. if context==item else 0. for item in CONTEXTS];right_context=[1. if right_name in {name for name,_ in UNITS[item]} else 0. for item in CONTEXTS]
   features=np.concatenate(([math.tanh(math.log10(value+1e-12)/4.),math.tanh(ratio/12.),float(left_name==right_name),float(left_name in AMBIGUOUS),float(right_name in AMBIGUOUS)],context_onehot,right_context,hashed(left_name),hashed(right_name))).astype(np.float32)
   naive_left=next((index for index,item in enumerate(CONTEXTS) if left_name in {name for name,_ in UNITS[item]}),-1);naive_right=next((index for index,item in enumerate(CONTEXTS) if right_name in {name for name,_ in UNITS[item]}),-2);baseline=2 if naive_left!=naive_right else 0 if left_name==right_name else 1
   records.append((features,label,baseline,code,family,left_name,right_name,context))
 x=np.asarray([r[0] for r in records],np.float32);y=np.asarray([r[1] for r in records],np.int64);baseline=np.asarray([r[2] for r in records],np.int64);split=np.asarray([r[3] for r in records],np.int8);group=np.asarray([r[4] for r in records]);left=np.asarray([r[5] for r in records]);right=np.asarray([r[6] for r in records]);context=np.asarray([r[7] for r in records]);sets={c:set(group[split==c]) for c in range(3)};overlap=sum(len(sets[i]&sets[j]) for i,j in ((0,1),(0,2),(1,2)));counts={name:int(np.sum(split==c)) for c,name in enumerate(("train","validation","test"))};classes={name:{str(k):int(np.sum((split==c)&(y==k))) for k in range(3)} for c,name in enumerate(("train","validation","test"))}
 if overlap or min(counts.values())<300 or min(min(v.values()) for v in classes.values())<100:raise RuntimeError(f"DATA_GATE:{overlap}:{counts}:{classes}")
 with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as h:contracts={r["candidate_id"]:r for r in csv.DictReader(h,delimiter="\t")}
 cid="CAND-S-036";contract=contracts[cid];out=root/"data"/"staged_support_s036_exact_v1"/f"{cid}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,x=x,y=y,baseline_pred=baseline,split=split,group=group,left_unit=left,right_unit=right,context=context);meta={"schema":"cimc.forge200.support-s036-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"CONTROLLED_FIXTURE","source_gate":contract["source_gate"],"source_gate_match":"CURATED_SI_UNIT_CASES","records":len(y),"template_families":len(families),"template_manifest_root_sha256":hashlib.sha256(canonical_bytes(families)).hexdigest(),"split_counts":counts,"class_counts":classes,"cross_split_template_overlap":overlap,"label_derivation":"SI_DIMENSION_AND_SCALE_TABLE_WITH_CONTEXTUAL_AMBIGUITY","baseline_execution":"DETERMINISTIC_FIRST_MATCH_DIMENSION_PARSER_WITHOUT_CONTEXT_DISAMBIGUATION","input_contract":contract["input_contract"],"target_label":contract["target_label"],"primary_metric":contract["primary_metric"],"parameter_cap":contract["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(contract)).hexdigest(),"source_license":"TEAM_OWNED_CURATED_SI_CASES","authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(out)};write_json(out.with_suffix(".metadata.json"),meta);write_json(root/"evidence"/"support_s036_exact_staging.v1.json",{"schema":"cimc.forge200.support-s036-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":meta,"authority_nonzero":0,"board_actions":0});print(json.dumps({"status":"PASS","candidate_id":cid,"records":len(y),"split_counts":counts,"class_counts":classes},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
