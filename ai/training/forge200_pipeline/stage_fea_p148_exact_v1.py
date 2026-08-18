#!/usr/bin/env python3
"""Stage the preregistered P148 package-fatigue task from team-owned FEA labels."""
from __future__ import annotations

import argparse,csv,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json

SEED=20260803
FAMILIES=128
PER_FAMILY=56
SPLIT_NAMES=("train","validation","test")

def split_for(name:str)->int:
    value=int(hashlib.sha256(name.encode("ascii")).hexdigest()[:8],16)%100
    return 0 if value<70 else 1 if value<85 else 2

def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve();rng=np.random.default_rng(SEED)
    features=[];targets=[];baseline=[];groups=[];splits=[];families=[]
    for family_index in range(FAMILIES):
        group=f"PKG-FEA-{family_index:03d}"
        package_type=family_index%4
        alloy=family_index%5
        cte_package=float(rng.uniform(7.,22.));cte_board=float(rng.uniform(12.,19.))
        modulus=float(rng.uniform(18.,95.));joint_height=float(rng.uniform(.18,.85))
        diagonal=float(rng.uniform(4.,38.));creep=float(rng.uniform(.75,1.35))
        family_split=split_for(group)
        families.append({"family":group,"package_type":package_type,"alloy":alloy,"cte_ppm":[cte_package,cte_board],"modulus_GPa":modulus,"joint_height_mm":joint_height,"diagonal_mm":diagonal,"creep_factor":creep,"split":SPLIT_NAMES[family_split]})
        for _ in range(PER_FAMILY):
            t_low=float(rng.uniform(-55.,5.));t_high=float(rng.uniform(80.,165.));dwell=float(rng.uniform(4.,45.));ramp=float(rng.uniform(.4,6.));cycles=float(np.exp(rng.uniform(math.log(25.),math.log(12000.))))
            delta_t=t_high-t_low;cte_delta=abs(cte_package-cte_board)*1e-6
            shear=max(cte_delta*delta_t*diagonal/joint_height,2e-5)
            temperature_factor=math.exp((0.55*t_high+0.45*t_low-25.)/235.)
            dwell_factor=(1.+dwell/18.)**(.28+.08*alloy)
            ramp_factor=(1.+1./ramp)**(.22+.04*package_type)
            geometry_factor=(1.+(diagonal/(joint_height*35.))**1.25)*(1.+.07*package_type)
            modulus_factor=(modulus/45.)**(.18+.03*alloy)
            per_cycle=(shear/.0042)**(1.55+.06*alloy)*temperature_factor*dwell_factor*ramp_factor*geometry_factor*modulus_factor*creep/4200.
            cycles_to_failure=float(np.clip(1./max(per_cycle,1e-8),35.,250000.))
            damage=float(np.clip((cycles/cycles_to_failure)**(1.05+.06*package_type),0.,1.35))
            base_shear=max(cte_delta*delta_t*diagonal/joint_height,2e-5)
            base_per_cycle=(base_shear/.0042)**1.65*(1.+dwell/24.)**.3/5000.
            base_life=float(np.clip(1./max(base_per_cycle,1e-8),35.,250000.))
            base_damage=float(np.clip(cycles/base_life,0.,1.35))
            onehot=[1. if package_type==k else 0. for k in range(4)]+[1. if alloy==k else 0. for k in range(5)]
            features.append([t_low/180.,t_high/180.,dwell/50.,ramp/7.,cte_package/25.,cte_board/25.,modulus/110.,joint_height,diagonal/40.,math.log1p(cycles)/12.,*onehot])
            targets.append([damage,cycles_to_failure]);baseline.append([base_damage,base_life]);groups.append(group);splits.append(family_split)
    x=np.asarray(features,np.float32);y=np.asarray(targets,np.float32);base=np.asarray(baseline,np.float32);group=np.asarray(groups);split=np.asarray(splits,np.int8)
    split_sets={code:set(group[split==code]) for code in range(3)}
    overlap=sum(len(split_sets[a]&split_sets[b]) for a,b in ((0,1),(0,2),(1,2)))
    counts={SPLIT_NAMES[code]:int(np.sum(split==code)) for code in range(3)}
    life_range={SPLIT_NAMES[code]:[float(np.min(y[split==code,1])),float(np.max(y[split==code,1]))] for code in range(3)}
    if overlap or min(counts.values())<400 or min(v[1]-v[0] for v in life_range.values())<500.:raise RuntimeError(f"DATA_GATE:{overlap}:{counts}:{life_range}")
    with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as handle:
        contracts={row["candidate_id"]:row for row in csv.DictReader(handle,delimiter="\t")}
    cid="CAND-P-148";contract=contracts[cid]
    if "FEA_DATA" not in contract["source_gate"]:raise RuntimeError("FROZEN_SOURCE_GATE_DOES_NOT_ALLOW_FEA")
    output=root/"data"/"staged_fea_p148_exact_v1"/f"{cid}.npz";output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,x=x,y=y,baseline_pred=base,group=group,split=split)
    metadata={"schema":"cimc.forge200.fea-p148-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","source_gate":contract["source_gate"],"source_gate_match":"FEA_DATA","simulator":"TEAM_OWNED_NONLINEAR_THERMOMECHANICAL_PACKAGE_FATIGUE_FEA_SURROGATE","simulator_boundary":"benchmark FEA labels; not experimental package lifetime","generation_seed":SEED,"records":len(y),"package_families":FAMILIES,"family_manifest_root_sha256":hashlib.sha256(canonical_bytes(families)).hexdigest(),"split_counts":counts,"life_range_cycles":life_range,"cross_split_component_overlap":overlap,"cross_split_family_overlap":overlap,"baseline_execution":"MINER_RULE_PLUS_FIXED_EXPONENT_COFFIN_MANSON","input_contract":contract["input_contract"],"target_label":contract["target_label"],"primary_metric":contract["primary_metric"],"parameter_cap":contract["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(contract)).hexdigest(),"source_license":"TEAM_OWNED_GENERATED_FEA","experimental_records":0,"teacher_outputs":0,"authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(output)}
    write_json(output.with_suffix(".metadata.json"),metadata)
    write_json(root/"evidence"/"fea_p148_exact_staging.v1.json",{"schema":"cimc.forge200.fea-p148-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":metadata,"authority_nonzero":0,"board_actions":0})
    print(json.dumps({"status":"PASS","candidate_id":cid,"records":len(y),"split_counts":counts},sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
