#!/usr/bin/env python3
"""Generate the SIM_ONLY kinematic EUV scatterometry inverse task P100."""

from __future__ import annotations

import argparse, csv, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage_matbench_experimental_v1 import canonical_bytes, sha256_file, write_json


SEED=20260803; POINTS=32; FAMILIES=96; SAMPLES_PER_FAMILY=50; SPLIT_NAMES=("train","validation","test")


def split_for(group:str)->int:
    bucket=int(hashlib.sha256(group.encode("ascii")).hexdigest()[:8],16)%100
    return 0 if bucket<70 else 1 if bucket<85 else 2


def forward_curve(target:np.ndarray,sim:np.ndarray)->np.ndarray:
    width,swa,height=(float(v) for v in target); pitch,wavelength,contrast=(float(v) for v in sim); theta=np.deg2rad(np.linspace(2.,20.,POINTS)); side=max(math.tan(math.radians(90.-swa)),0.); top=max(width-2.*height*side,1.); effective=.5*(width+top); q=2.*math.pi*np.sin(theta)/pitch; envelope=np.sinc(q*effective/(2.*math.pi)); taper=np.sinc(q*max(width-top,1.)/(4.*math.pi)); phase=4.*math.pi*height*np.cos(theta)/(wavelength*10.); reference=1.-contrast*effective/pitch; complex_amplitude=reference+contrast*(effective/pitch)*envelope*taper*np.exp(1j*phase); harmonic=.12*contrast*np.sinc(q*effective/math.pi)*np.exp(2j*phase); intensity=np.abs(complex_amplitude+harmonic)**2; return (intensity/np.maximum(np.mean(intensity),1e-9)).astype(np.float32)


def nearest_lut(x:np.ndarray,y:np.ndarray,train:np.ndarray)->np.ndarray:
    mean=x[train].mean(0); std=x[train].std(0); std[std<1e-7]=1.; z=(x-mean)/std; result=np.empty_like(y)
    for start in range(0,len(x),128):
        query=z[start:start+128]; distance=np.sum((query[:,None,:]-z[train][None,:,:])**2,axis=2); result[start:start+len(query)]=y[train[np.argmin(distance,axis=1)]]
    return result


def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); root=a.root.resolve(); rng=np.random.default_rng(SEED)
    curves=[]; sim_params=[]; priors=[]; targets=[]; groups=[]; splits=[]; manifest=[]
    for fi in range(FAMILIES):
        group=f"EUV-STACK-{fi:03d}"; pitch=float(rng.uniform(42.,92.)); wavelength=float(rng.uniform(13.,14.)); contrast=float(rng.uniform(.25,.9)); code=split_for(group); manifest.append({"family":group,"pitch_nm":pitch,"wavelength_nm":wavelength,"optical_contrast":contrast,"split":SPLIT_NAMES[code]})
        for _ in range(SAMPLES_PER_FAMILY):
            width=float(rng.uniform(.18,.72)*pitch); height=float(rng.uniform(18.,105.)); swa=float(rng.uniform(78.,89.5)); target=np.asarray([width,swa,height],dtype=np.float32); sim=np.asarray([pitch,wavelength,contrast],dtype=np.float32); curve=forward_curve(target,sim); curve=np.maximum(curve+rng.normal(0.,.003,size=POINTS),0.).astype(np.float32); prior=target+np.asarray([rng.normal(0.,8.),rng.normal(0.,2.5),rng.normal(0.,14.)],dtype=np.float32)
            curves.append(curve); sim_params.append(sim); priors.append(prior); targets.append(target); groups.append(group); splits.append(code)
    curve=np.asarray(curves,dtype=np.float32); sim=np.asarray(sim_params,dtype=np.float32); prior=np.asarray(priors,dtype=np.float32); y=np.asarray(targets,dtype=np.float32); group=np.asarray(groups); split=np.asarray(splits,dtype=np.int8); x=np.column_stack((curve,sim,prior)).astype(np.float32); train=np.flatnonzero(split==0); baseline=nearest_lut(x,y,train); sets={code:set(group[split==code]) for code in (0,1,2)}; overlap=sum(len(sets[l]&sets[r]) for l,r in ((0,1),(0,2),(1,2))); counts={SPLIT_NAMES[c]:int(np.sum(split==c)) for c in (0,1,2)}
    if overlap or min(counts.values())<400: raise RuntimeError(f"SPLIT_GATE:{overlap}:{counts}")
    with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as h: contracts={r["candidate_id"]:r for r in csv.DictReader(h,delimiter="\t")}
    cid="CAND-P-100"; output=root/"data"/"staged_physics_p100_exact_v1"/f"{cid}.npz"; output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,x=x,y=y,curve=curve,sim_params=sim,baseline_pred=baseline,group=group,split=split)
    contract=contracts[cid]; metadata={"schema":"cimc.forge200.physics-p100-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","simulator":"TEAM_OWNED_KINEMATIC_EUV_LINE_GRATING_SURROGATE_NOT_RCWA","simulator_boundary":"benchmark inverse problem only; not calibrated EUV metrology","generation_seed":SEED,"records":len(y),"curve_points":POINTS,"families":FAMILIES,"family_manifest_root_sha256":hashlib.sha256(canonical_bytes(manifest)).hexdigest(),"split_counts":counts,"cross_split_component_overlap":overlap,"cross_split_family_overlap":overlap,"baseline_execution":"FULL_TRAIN_SPLIT_NEAREST_LUT_IN_STANDARDIZED_CURVE_SIMULATION_PRIOR_SPACE","input_contract":contract["input_contract"],"target_label":contract["target_label"],"primary_metric":contract["primary_metric"],"parameter_cap":contract["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(contract)).hexdigest(),"source_license":"TEAM_OWNED_GENERATED_SIMULATION","experimental_records":0,"teacher_outputs":0,"authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(output)}; write_json(output.with_suffix(".metadata.json"),metadata); write_json(root/"evidence"/"physics_p100_exact_staging.v1.json",{"schema":"cimc.forge200.physics-p100-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":metadata,"authority_nonzero":0,"board_actions":0}); print(json.dumps({"status":"PASS","records":len(y),"split_counts":counts},sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
