#!/usr/bin/env python3
"""Generate SIM_ONLY TIM thermal-resistance task P110."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json
SEED=20260803;FAMILIES=100;PER=40;AREA=1e-4;SPLIT_NAMES=("train","validation","test")
def split_for(g):
 b=int(hashlib.sha256(g.encode("ascii")).hexdigest()[:8],16)%100;return 0 if b<70 else 1 if b<85 else 2
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve();rng=np.random.default_rng(SEED);x=[];y=[];base=[];groups=[];splits=[];manifest=[]
 for fi in range(FAMILIES):
  g=f"TIM-{fi:03d}";contact=float(rng.uniform(.015,.45));exponent=float(rng.uniform(.25,.9));tempcoef=float(rng.uniform(-.001,.005));code=split_for(g);manifest.append({"family":g,"contact_coefficient_K_W":contact,"pressure_exponent":exponent,"temperature_coefficient":tempcoef,"split":SPLIT_NAMES[code]})
  for _ in range(PER):
   thickness=float(rng.uniform(20.,500.));conductivity=float(np.exp(rng.uniform(math.log(.4),math.log(15.))));pressure=float(rng.uniform(.08,3.));void=float(rng.uniform(0.,.22));temperature=float(rng.uniform(-20.,150.));bulk=thickness*1e-6/(conductivity*AREA*max((1-void)**1.5,.1));cr=contact/max(pressure**exponent,.05)*(1+3.5*void)*max(1+tempcoef*(temperature-25.),.5);target=bulk+cr;baseline=thickness*1e-6/(conductivity*AREA);x.append([thickness/500.,math.log1p(conductivity)/3.,pressure/3.,void/.25,(temperature+20.)/180.]);y.append(target);base.append(baseline);groups.append(g);splits.append(code)
 x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);base=np.asarray(base,np.float32);group=np.asarray(groups);split=np.asarray(splits,np.int8);sets={c:set(group[split==c]) for c in (0,1,2)};overlap=sum(len(sets[l]&sets[r]) for l,r in ((0,1),(0,2),(1,2)));counts={SPLIT_NAMES[c]:int(np.sum(split==c)) for c in (0,1,2)}
 if overlap or min(counts.values())<400:raise RuntimeError(f"DATA_GATE:{overlap}:{counts}")
 with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as h:contracts={r["candidate_id"]:r for r in csv.DictReader(h,delimiter="\t")}
 cid="CAND-P-110";out=root/"data"/"staged_physics_p110_exact_v1"/f"{cid}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,x=x,y=y,baseline_pred=base,group=group,split=split);c=contracts[cid];meta={"schema":"cimc.forge200.physics-p110-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","simulator":"TEAM_OWNED_TIM_BULK_PLUS_PRESSURE_VOID_TEMPERATURE_CONTACT_RESISTANCE_MODEL","fixed_interface_area_m2":AREA,"generation_seed":SEED,"records":len(y),"families":FAMILIES,"family_manifest_root_sha256":hashlib.sha256(canonical_bytes(manifest)).hexdigest(),"split_counts":counts,"cross_split_component_overlap":overlap,"cross_split_family_overlap":overlap,"baseline_execution":"THICKNESS_M_OVER_BULK_CONDUCTIVITY_W_MK_AND_FIXED_AREA_M2","input_contract":c["input_contract"],"target_label":c["target_label"],"primary_metric":c["primary_metric"],"parameter_cap":c["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(c)).hexdigest(),"source_license":"TEAM_OWNED_GENERATED_SIMULATION","experimental_records":0,"teacher_outputs":0,"authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(out)};write_json(out.with_suffix(".metadata.json"),meta);write_json(root/"evidence"/"physics_p110_exact_staging.v1.json",{"schema":"cimc.forge200.physics-p110-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":meta,"authority_nonzero":0,"board_actions":0});print(json.dumps({"status":"PASS","records":len(y),"split_counts":counts},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
