#!/usr/bin/env python3
"""Generate the SIM_ONLY rheology gel-point contract P102."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json
SEED=20260803;POINTS=16;FAMILIES=100;PER_FAMILY=40;SPLIT_NAMES=("train","validation","test")
def split_for(g):
 b=int(hashlib.sha256(g.encode("ascii")).hexdigest()[:8],16)%100;return 0 if b<70 else 1 if b<85 else 2
def f1(y,p):
 tp=np.sum((y==1)&(p==1));fp=np.sum((y==0)&(p==1));fn=np.sum((y==1)&(p==0));return 2*tp/max(2*tp+fp+fn,1)
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve();rng=np.random.default_rng(SEED);x=[];y=[];event=[];base=[];groups=[];splits=[];manifest=[]
 for fi in range(FAMILIES):
  g=f"RHEO-{fi:03d}";eta0=float(np.exp(rng.uniform(math.log(.08),math.log(20.))));gel=float(rng.uniform(.38,.78));tau=float(rng.uniform(15.,180.));sensitivity=float(rng.uniform(1.2,5.));order=float(rng.uniform(.8,2.2));divergence=float(rng.uniform(1.2,3.5));code=split_for(g);manifest.append({"family":g,"eta0_Pas":eta0,"gel_conversion":gel,"tau_min":tau,"temperature_sensitivity":sensitivity,"reaction_order":order,"viscosity_divergence":divergence,"split":SPLIT_NAMES[code]})
  for _ in range(PER_FAMILY):
   start=float(rng.uniform(20.,40.));peak=float(rng.uniform(65.,205.));ramp=float(rng.uniform(20.,150.));dwell=float(rng.uniform(5.,240.));total=ramp+dwell;time=np.linspace(0,total,POINTS);temp=np.minimum(start+(peak-start)*time/ramp,peak);alpha=np.zeros(POINTS);viscosity=np.zeros(POINTS);viscosity[0]=eta0;gel_index=None
   for i in range(1,POINTS):
    dt=time[i]-time[i-1];rate=math.exp(np.clip(sensitivity*(temp[i]-120.)/100.,-5.,5.))/tau;alpha[i]=min(1.,alpha[i-1]+rate*max(1-alpha[i-1],0.)**order*dt);distance=max(1.-alpha[i]/gel,.012);viscosity[i]=min(1e8,eta0*math.exp(np.clip((80.-temp[i])/100.,-2.,2.))/distance**divergence)
    if gel_index is None and alpha[i]>=gel:gel_index=i
   ev=int(gel_index is not None);idx=gel_index if gel_index is not None else POINTS-1;target=[float(time[idx]),float(temp[idx]),float(alpha[idx])];threshold=np.flatnonzero(viscosity>=1e4);bev=int(len(threshold)>0);bidx=int(threshold[0]) if bev else POINTS-1;baseline=[float(time[bidx]),float(temp[bidx]),.6,float(bev)];features=np.concatenate((np.log1p(viscosity)/20.,temp/220.,time/400.,np.asarray([float(rng.uniform(.1,100.))/100.],dtype=np.float64))).astype(np.float32);x.append(features);y.append(target);event.append(ev);base.append(baseline);groups.append(g);splits.append(code)
 x=np.asarray(x,np.float32);y=np.asarray(y,np.float32);event=np.asarray(event,np.int8);base=np.asarray(base,np.float32);group=np.asarray(groups);split=np.asarray(splits,np.int8);sets={c:set(group[split==c]) for c in (0,1,2)};overlap=sum(len(sets[l]&sets[r]) for l,r in ((0,1),(0,2),(1,2)));counts={SPLIT_NAMES[c]:int(np.sum(split==c)) for c in (0,1,2)};classes={SPLIT_NAMES[c]:{"gel":int(np.sum(event[split==c])),"no_gel":int(np.sum((1-event)[split==c]))} for c in (0,1,2)}
 if overlap or min(counts.values())<400 or min(min(v.values()) for v in classes.values())<40:raise RuntimeError(f"DATA_GATE:{overlap}:{counts}:{classes}")
 with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as h:contracts={r["candidate_id"]:r for r in csv.DictReader(h,delimiter="\t")}
 cid="CAND-P-102";out=root/"data"/"staged_physics_p102_exact_v1"/f"{cid}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,x=x,y=y,event=event,baseline_pred=base,group=group,split=split);c=contracts[cid];meta={"schema":"cimc.forge200.physics-p102-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","simulator":"TEAM_OWNED_TEMPERATURE_DEPENDENT_CURE_KINETICS_WITH_MATERIAL_SPECIFIC_GEL_CONVERSION_AND_VISCOSITY_DIVERGENCE","generation_seed":SEED,"records":len(y),"families":FAMILIES,"trajectory_points":POINTS,"family_manifest_root_sha256":hashlib.sha256(canonical_bytes(manifest)).hexdigest(),"split_counts":counts,"class_counts":classes,"cross_split_component_overlap":overlap,"cross_split_family_overlap":overlap,"baseline_execution":"FIXED_10000_PA_S_VISCOSITY_THRESHOLD_AND_FIXED_0.6_CONVERSION","input_contract":c["input_contract"],"target_label":c["target_label"],"primary_metric":c["primary_metric"],"parameter_cap":c["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(c)).hexdigest(),"source_license":"TEAM_OWNED_GENERATED_SIMULATION","experimental_records":0,"teacher_outputs":0,"authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(out)};write_json(out.with_suffix(".metadata.json"),meta);write_json(root/"evidence"/"physics_p102_exact_staging.v1.json",{"schema":"cimc.forge200.physics-p102-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":meta,"authority_nonzero":0,"board_actions":0});print(json.dumps({"status":"PASS","records":len(y),"split_counts":counts,"class_counts":classes},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
