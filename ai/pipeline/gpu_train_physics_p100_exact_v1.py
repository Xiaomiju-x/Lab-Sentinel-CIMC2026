#!/usr/bin/env python3
"""Train/package the SIM_ONLY P100 inverse scatterometry model."""

from __future__ import annotations

import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state
from stage_physics_p100_exact_v1 import forward_curve

CAP=86_000; WEIGHT_CAP=104*1024; SCALE=np.asarray([70.,12.,90.],dtype=np.float32)

def metrics(y:np.ndarray,pred:np.ndarray,curve:np.ndarray,sim:np.ndarray)->dict[str,float]:
    pred=np.column_stack((np.clip(pred[:,0],1.,90.),np.clip(pred[:,1],70.,90.),np.clip(pred[:,2],5.,130.))).astype(np.float32); normalized=(pred-y)/SCALE; nrmse=float(np.sqrt(np.mean(normalized**2))); parameter_mae=float(np.mean(np.abs(normalized))); reconstruction=np.asarray([forward_curve(p,s) for p,s in zip(pred,sim,strict=True)]); chi=float(np.mean((reconstruction-curve)**2)); result={"normalized_RMSE":nrmse,"normalized_parameter_MAE":parameter_mae,"profile_chi_square":chi,"NRMSE_score":1./(1.+nrmse),"parameter_score":1./(1.+parameter_mae),"profile_score":1./(1.+100.*chi)}; result["primary_composite"]=float(np.mean([result["NRMSE_score"],result["parameter_score"],result["profile_score"]])); return result

def run(a:argparse.Namespace)->dict[str,Any]:
    import onnx,torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    root=a.root.resolve(); dataset=root/"data"/"staged_physics_p100_exact_v1"/"CAND-P-100.npz"; meta=json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"));
    if meta["status"]!="PASS" or meta["truth_class"]!="PHYSICS_SIM" or meta["cross_split_family_overlap"] or sha256_file(dataset)!=meta["sha256"]: raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
    device=torch.device(a.device); props=torch.cuda.get_device_properties(device); raw=np.load(dataset,allow_pickle=False); xraw=raw["x"].astype(np.float32); y=raw["y"].astype(np.float32); curve=raw["curve"].astype(np.float32); sim=raw["sim_params"].astype(np.float32); baseline=raw["baseline_pred"].astype(np.float32); split=raw["split"].astype(np.int8); train,val,test=(np.flatnonzero(split==c) for c in (0,1,2)); mean,std=xraw[train].mean(0),xraw[train].std(0); std[std<1e-7]=1.; x=np.clip((xraw-mean)/std,-12,12).astype(np.float32); ymean,ystd=y[train].mean(0),y[train].std(0); ys=(y-ymean)/ystd
    output=a.artifact_root.resolve()/"CAND-P-100"; output.mkdir(parents=True,exist_ok=True); hb=output/"heartbeat.json"; started=time.perf_counter()
    class MLP(nn.Module):
        def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(x.shape[1],160),nn.GELU(),nn.Linear(160,96),nn.GELU(),nn.Linear(96,3))
        def forward(self,v): return self.net(v)
    params=sum(p.numel() for p in MLP().parameters());
    if params>CAP: raise RuntimeError(f"PARAMETER_CAP:{params}")
    base_val=metrics(y[val],baseline[val],curve[val],sim[val]); base_test=metrics(y[test],baseline[test],curve[test],sim[test]); reports=[]; states={}; data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(ys[train].astype(np.float32)))
    for seed in SEEDS:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); model=MLP().to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=3e-4); loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed)); ck=output/f"train_seed_{seed}"/"best.pt"; ck.parent.mkdir(parents=True,exist_ok=True); best=-1e30; patience=0
        for epoch in range(a.max_epochs):
            model.train()
            for bx,by in loader: opt.zero_grad(set_to_none=True); pr=model(bx.to(device)); loss=nn.functional.smooth_l1_loss(pr,by.to(device),beta=.1); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); opt.step()
            model.eval()
            with torch.no_grad(): vp=model(torch.from_numpy(x[val]).to(device)).cpu().numpy()*ystd+ymean
            score=metrics(y[val],vp,curve[val],sim[val])["primary_composite"]
            if score>best+1e-5: best=score; patience=0; torch.save(model.state_dict(),ck)
            else: patience+=1
            heartbeat(hb,"CAND-P-100","TRAIN_PHYSICS_P100_EXACT",seed,epoch)
            if epoch+1>=a.min_epochs and patience>=a.early_stop_patience: break
        model.load_state_dict(torch.load(ck,map_location=device,weights_only=True)); model.eval();
        with torch.no_grad(): tp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()*ystd+ymean
        m=metrics(y[test],tp,curve[test],sim[test]); reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":m,"beats_baseline":m["primary_composite"]>base_test["primary_composite"]+1e-4}); states[seed]={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    comp=np.asarray([r["test"]["primary_composite"] for r in reports]); agg={"mean":float(comp.mean()),"variance":float(comp.var()),"std":float(comp.std()),"worst":float(comp.min())}; aggpass=agg["mean"]>base_test["primary_composite"]+1e-4; bestseed=int(max(reports,key=lambda r:r["validation_primary_composite"])["seed"]); model=MLP().to(device); model.load_state_dict(states[bestseed]); model.eval();
    with torch.no_grad(): fp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()*ystd+ymean
    fpm=metrics(y[test],fp,curve[test],sim[test]); q,s=quantize_state(states[bestseed]); b=io.BytesIO(); np.savez_compressed(b,**q,**{f"scale::{k}":v for k,v in s.items()}); payload=b.getvalue();
    if len(payload)>WEIGHT_CAP: raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch,q,s));
    with torch.no_grad(): qp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()*ystd+ymean
    qm=metrics(y[test],qp,curve[test],sim[test]); qdelta=fpm["primary_composite"]-qm["primary_composite"]; qpass=qm["primary_composite"]>base_test["primary_composite"]+1e-4 and qdelta<=.03; passed=aggpass and qpass; golden=output/"golden_vectors.npz"; np.savez_compressed(golden,x=x[test[:64]],y=y[test[:64]],fp32=fp[:64],quantized=qp[:64],curve=curve[test[:64]],sim_params=sim[test[:64]]); model.load_state_dict(states[bestseed]); onnx_path=output/"fp32.onnx"; torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["scatterometry_curve_simulation_and_profile_prior"],output_names=["normalized_profile_parameters"],dynamic_axes={"scatterometry_curve_simulation_and_profile_prior":{0:"batch"},"normalized_profile_parameters":{0:"batch"}},opset_version=17,dynamo=False); onnx.checker.check_model(onnx.load(onnx_path)); schema={"task_kind":"inverse_profile_regression","shape":[None,3],"semantics":["line_width_nm","sidewall_angle_deg","height_nm"],"postprocess":"multiply_y_std_add_y_mean_then_contract_clamp","authority":0,"public_claim_scope":"SIM_ONLY"}; release=hashlib.sha256(canonical_bytes({"candidate":"CAND-P-100","dataset":meta["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(golden),"mean":agg["mean"]})).hexdigest(); package=build_package(output,"CAND-P-100",payload,sha256_file(golden),release,hashlib.sha256(canonical_bytes(schema)).hexdigest()); status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE_SIM_ONLY"; audit={"schema":"cimc.forge200.physics-p100-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-P-100","truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","baseline":{"kind":meta["baseline_execution"],"validation":base_val,"test":base_test},"seed_reports":reports,"aggregate":agg,"g3_aggregate_mean_gate":aggpass,"quantized_best_seed":{"seed":bestseed,"test":qm,"metric_delta":qdelta,"gate":qpass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False}; write_json(output/"contract_exact_audit.json",audit); write_json(output/"source_manifest.json",meta); write_json(output/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist(),"y_mean":ymean.tolist(),"y_std":ystd.tolist()}); write_json(output/"output_schema.json",schema); write_json(output/"quantization_parity.json",{"primary_composite_delta":qdelta,"gate":qpass}); (output/"model_card.md").write_text(f"# CAND-P-100 SIM_ONLY\n\n- Status: `{status}`.\n- Kinematic surrogate benchmark, not calibrated EUV metrology or RCWA.\n- Mean `{agg['mean']:.6f}` vs full train LUT `{base_test['primary_composite']:.6f}`.\n- Parameters `{params}`; W8 `{len(payload)}` bytes; authority `0`; board pending.\n",encoding="utf-8"); promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-P-100","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"PHYSICS_SIM","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(golden),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}}; write_json(output/"promotion_receipt.json",promotion); manifest(output); heartbeat(hb,"CAND-P-100","COMPLETE"); write_json(root/"evidence"/"physics_p100_exact_closure.v1.json",{"schema":"cimc.forge200.physics-p100-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0}); print(json.dumps({"candidate_id":"CAND-P-100","status":status,"mean_composite":agg["mean"],"baseline_composite":base_test["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True)); return promotion


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--artifact-root",type=Path,required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--max-epochs",type=int,default=100); p.add_argument("--min-epochs",type=int,default=30); p.add_argument("--early-stop-patience",type=int,default=14); p.add_argument("--learning-rate",type=float,default=8e-4); a=p.parse_args(); r=run(a); return 0 if r["host_contract_pass"] else 2


if __name__=="__main__": raise SystemExit(main())
