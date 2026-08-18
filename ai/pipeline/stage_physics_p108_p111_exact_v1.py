#!/usr/bin/env python3
"""Generate SIM_ONLY interface reliability tasks P108 and P111."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json
SEED=20260803;FAMILIES=100;PER=48;SPLIT_NAMES=("train","validation","test")
def split_for(g):
 b=int(hashlib.sha256(g.encode("ascii")).hexdigest()[:8],16)%100;return 0 if b<70 else 1 if b<85 else 2
def sigmoid(v):return 1/(1+np.exp(-np.clip(v,-30,30)))
def build_p108(rng):
 x=[];prob=[];risk=[];loc=[];base_prob=[];base_loc=[];groups=[];splits=[];manifest=[]
 for fi in range(FAMILIES):
  g=f"DELAM-STACK-{fi:03d}";cte1,cte2=rng.uniform(3,35,2);modulus=float(rng.uniform(5,160));thickness=float(rng.uniform(.1,2.));toughness=float(rng.uniform(.05,2.));sus=rng.uniform(.85,1.15,4);sus[fi%4]+=3.0;code=split_for(g);manifest.append({"family":g,"cte_ppm":[float(cte1),float(cte2)],"modulus_GPa":modulus,"thickness_mm":thickness,"toughness":toughness,"location_susceptibility":sus.tolist(),"split":SPLIT_NAMES[code]})
  for _ in range(PER):
   moisture=float(rng.uniform(0,1));cure=float(rng.uniform(.45,1));delta=float(rng.uniform(40,230));cycles=float(np.exp(rng.uniform(math.log(1),math.log(3000))));interface=float(rng.uniform(.2,1));mismatch=abs(cte1-cte2)/32;drive=(mismatch*delta/180)**2*(modulus/100)*(thickness/1.)*(1+2.8*moisture)*(cycles/100)**.16/(max(toughness*cure*interface,.03));p=float(sigmoid(2.2*(drive-1)));rl=int(drive>1);scores=sus*np.asarray([1+2*moisture,1+.8*mismatch,1+cycles**.12/3,1+(1-cure)*2]);lc=int(np.argmax(scores));bp=float(sigmoid(5*(.55*moisture+.7*mismatch+.002*delta-.9)));bl=int(np.argmax([moisture,mismatch,cycles/3000,1-cure]));x.append([moisture,cure,delta/250,math.log1p(cycles)/9,mismatch,modulus/180,thickness/2.2,toughness/2.2,interface,*((sus-.85)/3.3)]);prob.append(p);risk.append(rl);loc.append(lc);base_prob.append(bp);base_loc.append(bl);groups.append(g);splits.append(code)
 return {"x":np.asarray(x,np.float32),"y_probability":np.asarray(prob,np.float32),"risk_label":np.asarray(risk,np.int8),"location_label":np.asarray(loc,np.int8),"baseline_probability":np.asarray(base_prob,np.float32),"baseline_location":np.asarray(base_loc,np.int8),"group":np.asarray(groups),"split":np.asarray(splits,np.int8)},manifest
def build_p111(rng):
 x=[];strength=[];weak=[];groups=[];splits=[];manifest=[]
 for fi in range(FAMILIES):
  g=f"BOND-PAIR-{fi:03d}";intrinsic=float(rng.uniform(18,160));oxide_s=float(rng.uniform(.05,.35));optimum=float(rng.uniform(.2,3));temp_scale=float(rng.uniform(35,120));code=split_for(g);manifest.append({"family":g,"intrinsic_strength_MPa":intrinsic,"oxide_sensitivity":oxide_s,"roughness_optimum_um":optimum,"temperature_scale_C":temp_scale,"split":SPLIT_NAMES[code]})
  for _ in range(PER):
   ra=float(rng.uniform(.03,8));rq=ra*float(rng.uniform(1.05,1.8));oxide=float(rng.uniform(0,12));clean=float(rng.uniform(0,1));pressure=float(rng.uniform(.05,8));temperature=float(rng.uniform(25,450));rough=math.exp(-.5*((ra-optimum)/max(optimum*.8,.3))**2);oxide_factor=math.exp(-oxide_s*oxide*(1-.8*clean));press=1-math.exp(-pressure/1.5);thermal=1-math.exp(-max(temperature-20,0)/temp_scale);s=intrinsic*(.15+.85*rough)*oxide_factor*(.2+.8*press)*(.25+.75*thermal);wl=int(s<20);x.append([ra/8,rq/12,oxide/12,clean,pressure/8,temperature/450]);strength.append(s);weak.append(wl);groups.append(g);splits.append(code)
 arrays={"x":np.asarray(x,np.float32),"y_strength":np.asarray(strength,np.float32),"weak_label":np.asarray(weak,np.int8),"group":np.asarray(groups),"split":np.asarray(splits,np.int8)};train=arrays["split"]==0;design=np.column_stack((arrays["x"][train,:3],np.ones(np.sum(train))));coef=np.linalg.lstsq(design,arrays["y_strength"][train],rcond=None)[0];pred=np.column_stack((arrays["x"][:,:3],np.ones(len(x))))@coef;scale=max(float(np.std(arrays["y_strength"][train]-pred[train])),1.);arrays["baseline_strength"]=pred.astype(np.float32);arrays["baseline_weak_probability"]=sigmoid((20-pred)/scale).astype(np.float32);return arrays,manifest
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve();rng=np.random.default_rng(SEED)
 with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as h:contracts={r["candidate_id"]:r for r in csv.DictReader(h,delimiter="\t")}
 defs={"CAND-P-108":(*build_p108(rng),"TEAM_OWNED_COHESIVE_DRIVING_FORCE_DELAMINATION_SIM","FIXED_MOISTURE_CTE_THRESHOLD_RULE"),"CAND-P-111":(*build_p111(rng),"TEAM_OWNED_NONLINEAR_SURFACE_OXIDE_PRESSURE_TEMPERATURE_BOND_SIM","TRAIN_ONLY_LINEAR_ROUGHNESS_OXIDE_REGRESSION")};evidence=[]
 for cid,(arrays,manifest,simulator,baseline) in defs.items():
  group=arrays["group"];split=arrays["split"];sets={c:set(group[split==c]) for c in (0,1,2)};overlap=sum(len(sets[l]&sets[r]) for l,r in ((0,1),(0,2),(1,2)));counts={SPLIT_NAMES[c]:int(np.sum(split==c)) for c in (0,1,2)}
  if cid.endswith("108"):classes={SPLIT_NAMES[c]:{"risk_positive":int(np.sum(arrays["risk_label"][split==c])),"risk_negative":int(np.sum(1-arrays["risk_label"][split==c])),**{f"location_{k}":int(np.sum(arrays["location_label"][split==c]==k)) for k in range(4)}} for c in (0,1,2)}
  else:classes={SPLIT_NAMES[c]:{"weak":int(np.sum(arrays["weak_label"][split==c])),"strong":int(np.sum(1-arrays["weak_label"][split==c]))} for c in (0,1,2)}
  if overlap or min(counts.values())<400 or min(min(v.values()) for v in classes.values())<25:raise RuntimeError(f"{cid}:DATA_GATE:{overlap}:{counts}:{classes}")
  out=root/"data"/"staged_physics_p108_p111_exact_v1"/f"{cid}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays);c=contracts[cid];meta={"schema":"cimc.forge200.physics-interface-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","simulator":simulator,"generation_seed":SEED,"records":len(group),"families":FAMILIES,"family_manifest_root_sha256":hashlib.sha256(canonical_bytes(manifest)).hexdigest(),"split_counts":counts,"class_counts":classes,"cross_split_component_overlap":overlap,"cross_split_family_overlap":overlap,"baseline_execution":baseline,"input_contract":c["input_contract"],"target_label":c["target_label"],"primary_metric":c["primary_metric"],"parameter_cap":c["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(c)).hexdigest(),"source_license":"TEAM_OWNED_GENERATED_SIMULATION","experimental_records":0,"teacher_outputs":0,"authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(out)};write_json(out.with_suffix(".metadata.json"),meta);evidence.append(meta)
 write_json(root/"evidence"/"physics_p108_p111_exact_staging.v1.json",{"schema":"cimc.forge200.physics-p108-p111-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","records":evidence,"authority_nonzero":0,"board_actions":0});print(json.dumps({"status":"PASS","tasks":{m["candidate_id"]:m["split_counts"] for m in evidence}},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
