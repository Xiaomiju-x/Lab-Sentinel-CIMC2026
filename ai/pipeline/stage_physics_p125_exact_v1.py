#!/usr/bin/env python3
"""Generate stack-split, SIM_ONLY beam-FEA labels for P125."""

from __future__ import annotations

import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json

SEED=20260803; FAMILIES=120; PER_FAMILY=40; ELEMENTS=10; SPLIT_NAMES=("train","validation","test")

def split_for(g:str)->int:
    b=int(hashlib.sha256(g.encode("ascii")).hexdigest()[:8],16)%100; return 0 if b<70 else 1 if b<85 else 2

def curvature(da:float,dt:float,e1:float,e2:float,t1:float,t2:float)->float:
    n=6.*da*dt*e1*e2*t1*t2*(t1+t2); d=(e1*t1+e2*t2)*(e1*t1**3+e2*t2**3)+3.*e1*e2*t1*t2*(t1+t2)**2; return n/max(d,1e-20)

def beam_fea(kappa:float,e1:float,e2:float,t1:float,t2:float,width:float,length:float)->tuple[float,float]:
    neutral=(e1*t1*(t1/2)+e2*t2*(t1+t2/2))/max(e1*t1+e2*t2,1e-20); i1=width*(t1**3/12+t1*(neutral-t1/2)**2); i2=width*(t2**3/12+t2*(t1+t2/2-neutral)**2); ei=e1*i1+e2*i2; le=length/ELEMENTS; ndof=2*(ELEMENTS+1); k=np.zeros((ndof,ndof),dtype=np.float64); local=ei/le**3*np.asarray([[12,6*le,-12,6*le],[6*le,4*le**2,-6*le,2*le**2],[-12,-6*le,12,-6*le],[6*le,2*le**2,-6*le,4*le**2]],dtype=np.float64)
    for element in range(ELEMENTS):
        dof=[2*element,2*element+1,2*element+2,2*element+3]; k[np.ix_(dof,dof)]+=local
    force=np.zeros(ndof,dtype=np.float64); force[-1]=ei*kappa; free=np.arange(2,ndof); displacement=np.zeros(ndof,dtype=np.float64); displacement[free]=np.linalg.solve(k[np.ix_(free,free)],force[free]); return float(displacement[-2]*1e6),float(ei)

def sigmoid(v:float)->float: return 1./(1.+math.exp(-max(min(v,40.),-40.)))

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve();rng=np.random.default_rng(SEED);x=[];y=[];risk_label=[];baseline=[];groups=[];splits=[];manifest=[]
    for fi in range(FAMILIES):
        g=f"FEA-STACK-{fi:03d}"; cte1,cte2=rng.uniform(3.,30.,2); e1,e2=rng.uniform(15.,190.,2)*1e9; t1,t2=rng.uniform(.06,.85,2)*1e-3; tg1,tg2=rng.uniform(70.,210.,2); toughness=float(rng.uniform(.08,2.5)); code=split_for(g);manifest.append({"family":g,"cte_ppm":[float(cte1),float(cte2)],"E_GPa":[float(e1/1e9),float(e2/1e9)],"thickness_mm":[float(t1*1e3),float(t2*1e3)],"Tg_C":[float(tg1),float(tg2)],"Gc_J_m2":toughness,"split":SPLIT_NAMES[code]})
        for _ in range(PER_FAMILY):
            tmin=float(rng.uniform(-65.,10.));tmax=float(rng.uniform(90.,205.));reference=float(rng.uniform(15.,35.));length=float(rng.uniform(8.,40.))*1e-3;width=float(rng.uniform(5.,30.))*1e-3;peak=max(abs(tmin-reference),abs(tmax-reference));direction=1. if abs(tmax-reference)>=abs(tmin-reference) else -1.;temperature=tmax if direction>0 else tmin; e1t=e1*(.1+.9/(1.+math.exp(max(min((temperature-tg1)/8.,40.),-40.))));e2t=e2*(.1+.9/(1.+math.exp(max(min((temperature-tg2)/8.,40.),-40.))));kap=curvature((cte1-cte2)*1e-6,direction*peak,e1t,e2t,t1,t2);warpage,ei=beam_fea(kap,e1t,e2t,t1,t2,width,length);release=.5*(ei/width)*kap**2;risk=sigmoid((release-toughness)/max(.18*toughness,.02));blkap=curvature((cte1-cte2)*1e-6,direction*peak,e1,e2,t1,t2);blwarp,_=beam_fea(blkap,e1,e2,t1,t2,width,length);bli=.5*(ei/width)*blkap**2;blrisk=sigmoid((bli-toughness)/max(.18*toughness,.02));features=np.asarray([tmin/220.,tmax/220.,reference/50.,cte1/35.,cte2/35.,e1/2e11,e2/2e11,t1/1e-3,t2/1e-3,tg1/220.,tg2/220.,length/.05,width/.04,toughness/3.],dtype=np.float32);x.append(features);y.append([warpage,risk]);risk_label.append(int(release>toughness));baseline.append([blwarp,blrisk]);groups.append(g);splits.append(code)
    x=np.asarray(x,dtype=np.float32);y=np.asarray(y,dtype=np.float32);risk_label=np.asarray(risk_label,dtype=np.int8);baseline=np.asarray(baseline,dtype=np.float32);group=np.asarray(groups);split=np.asarray(splits,dtype=np.int8);sets={c:set(group[split==c]) for c in (0,1,2)};overlap=sum(len(sets[l]&sets[r]) for l,r in ((0,1),(0,2),(1,2)));counts={SPLIT_NAMES[c]:int(np.sum(split==c)) for c in (0,1,2)};class_counts={SPLIT_NAMES[c]:{"positive":int(np.sum(risk_label[split==c])),"negative":int(np.sum((1-risk_label)[split==c]))} for c in (0,1,2)}
    if overlap or min(counts.values())<400 or min(min(v.values()) for v in class_counts.values())<40:raise RuntimeError(f"DATA_GATE:{overlap}:{counts}:{class_counts}")
    with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as h:contracts={r["candidate_id"]:r for r in csv.DictReader(h,delimiter="\t")}
    cid="CAND-P-125";out=root/"data"/"staged_physics_p125_exact_v1"/f"{cid}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,x=x,y=y,risk_label=risk_label,baseline_pred=baseline,group=group,split=split);c=contracts[cid];meta={"schema":"cimc.forge200.physics-p125-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"PHYSICS_SIM_FEA","public_claim_scope":"SIM_ONLY","simulator":"TEAM_OWNED_10_ELEMENT_EULER_BERNOULLI_COMPOSITE_BEAM_FEA_WITH_TEMPERATURE_DEPENDENT_MODULUS_AND_INTERFACE_ENERGY_RELEASE","generation_seed":SEED,"records":len(y),"families":FAMILIES,"finite_elements":ELEMENTS,"family_manifest_root_sha256":hashlib.sha256(canonical_bytes(manifest)).hexdigest(),"split_counts":counts,"class_counts":class_counts,"cross_split_component_overlap":overlap,"cross_split_family_overlap":overlap,"baseline_execution":"CONSTANT_MODULUS_LAMINATED_BEAM_PLUS_ANALYTICAL_ENERGY_THRESHOLD","input_contract":c["input_contract"],"target_label":c["target_label"],"primary_metric":c["primary_metric"],"parameter_cap":c["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(c)).hexdigest(),"source_license":"TEAM_OWNED_GENERATED_FEA","experimental_records":0,"teacher_outputs":0,"authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(out)};write_json(out.with_suffix(".metadata.json"),meta);write_json(root/"evidence"/"physics_p125_exact_staging.v1.json",{"schema":"cimc.forge200.physics-p125-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":meta,"authority_nonzero":0,"board_actions":0});print(json.dumps({"status":"PASS","records":len(y),"split_counts":counts,"class_counts":class_counts},sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
