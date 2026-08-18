#!/usr/bin/env python3
"""Train/package SIM_ONLY vector trajectory tasks P103 and P109."""

from __future__ import annotations

import argparse, hashlib, io, json, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_job import SEEDS, build_package, canonical_bytes, heartbeat, sha256_file, write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state, manifest, quantize_state


SPECS = {"CAND-P-103": {"cap": 56_000, "weight": 68 * 1024, "metric": "stress", "semantics": "residual_stress_trajectory_MPa"}, "CAND-P-109": {"cap": 58_000, "weight": 72 * 1024, "metric": "warpage", "semantics": "warpage_curve_um_vs_temperature"}}


def metric(candidate_id: str, y: np.ndarray, prediction: np.ndarray, interval_halfwidth: np.ndarray | None = None) -> dict[str, float]:
    error = prediction - y; rmse = float(np.sqrt(np.mean(error**2))); peak = float(np.mean(np.abs(np.max(np.abs(prediction), axis=1) - np.max(np.abs(y), axis=1)))); scale = max(float(np.std(y)), 1e-6)
    result = {"trajectory_RMSE": rmse, "peak_MAE": peak, "RMSE_skill_vs_target_std": 1.0 - rmse / scale, "peak_score": 1.0 / (1.0 + peak / scale)}
    if candidate_id == "CAND-P-103":
        coverage = float(np.mean(np.abs(error) <= interval_halfwidth)) if interval_halfwidth is not None else 0.0
        result.update({"trajectory_RMSE_MPa": rmse, "peak_MAE_MPa": peak, "interval_coverage_80": coverage, "coverage_score": 1.0 - abs(coverage - 0.8)})
        result["primary_composite"] = float(np.mean([result["RMSE_skill_vs_target_std"], result["peak_score"], result["coverage_score"]]))
    else:
        mask = np.abs(y) > max(1e-4, 0.01 * float(np.std(y))); sign = float(np.mean((prediction[mask] >= 0) == (y[mask] >= 0))) if np.any(mask) else 0.0
        result.update({"trajectory_RMSE_um": rmse, "peak_MAE_um": peak, "sign_accuracy": sign})
        result["primary_composite"] = float(np.mean([result["RMSE_skill_vs_target_std"], result["peak_score"], sign]))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import onnx, torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    candidate_id=args.candidate_id; spec=SPECS[candidate_id]; root=args.root.resolve(); dataset=root/"data"/"staged_physics_p103_p109_exact_v1"/f"{candidate_id}.npz"; metadata=json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["status"]!="PASS" or metadata["truth_class"]!="PHYSICS_SIM" or metadata["cross_split_family_overlap"] or sha256_file(dataset)!=metadata["sha256"]: raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
    device=torch.device(args.device); props=torch.cuda.get_device_properties(device); raw=np.load(dataset,allow_pickle=False); x_raw=raw["x"].astype(np.float32); y=raw["y"].astype(np.float32); baseline=raw["baseline"].astype(np.float32); split=raw["split"].astype(np.int8); train,validation,test=(np.flatnonzero(split==code) for code in (0,1,2))
    mean,std=x_raw[train].mean(0),x_raw[train].std(0); std[std<1e-7]=1.; x=np.clip((x_raw-mean)/std,-12,12).astype(np.float32); y_scale=np.maximum(np.quantile(np.abs(y[train]),.95,axis=0).astype(np.float32),.1); y_scaled=y/y_scale
    output=args.artifact_root.resolve()/candidate_id; output.mkdir(parents=True,exist_ok=True); heartbeat_path=output/"heartbeat.json"; started=time.perf_counter()
    class MLP(nn.Module):
        def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(x.shape[1],128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,y.shape[1]))
        def forward(self,value): return self.net(value)
    parameter_count=sum(p.numel() for p in MLP().parameters());
    if parameter_count>spec["cap"]: raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")
    baseline_half=np.quantile(np.abs(baseline[validation]-y[validation]),.8,axis=0) if candidate_id=="CAND-P-103" else None; baseline_validation=metric(candidate_id,y[validation],baseline[validation],baseline_half); baseline_test=metric(candidate_id,y[test],baseline[test],baseline_half)
    train_data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(y_scaled[train])); reports=[]; states={}; calibration={}
    for seed in SEEDS:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); model=MLP().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=3e-4); loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed)); checkpoint=output/f"train_seed_{seed}"/"best.pt"; checkpoint.parent.mkdir(parents=True,exist_ok=True); best=-float("inf"); patience=0
        for epoch in range(args.max_epochs):
            model.train()
            for bx,by in loader:
                optimizer.zero_grad(set_to_none=True); pred=model(bx.to(device)); loss=nn.functional.smooth_l1_loss(pred,by.to(device),beta=.1); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
            model.eval()
            with torch.no_grad(): vp=model(torch.from_numpy(x[validation]).to(device)).cpu().numpy()*y_scale
            half=np.quantile(np.abs(vp-y[validation]),.8,axis=0) if candidate_id=="CAND-P-103" else None; current=metric(candidate_id,y[validation],vp,half)
            if current["primary_composite"]>best+1e-5: best=current["primary_composite"]; patience=0; torch.save(model.state_dict(),checkpoint)
            else: patience+=1
            heartbeat(heartbeat_path,candidate_id,"TRAIN_PHYSICS_TRAJECTORY_EXACT",seed,epoch)
            if epoch+1>=args.min_epochs and patience>=args.early_stop_patience: break
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True)); model.eval()
        with torch.no_grad(): vp=model(torch.from_numpy(x[validation]).to(device)).cpu().numpy()*y_scale; tp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()*y_scale
        half=np.quantile(np.abs(vp-y[validation]),.8,axis=0) if candidate_id=="CAND-P-103" else None; calibration[seed]=half; report=metric(candidate_id,y[test],tp,half); reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":report,"beats_baseline":report["primary_composite"]>baseline_test["primary_composite"]+1e-4}); states[seed]={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    composites=np.asarray([r["test"]["primary_composite"] for r in reports]); aggregate={"mean":float(composites.mean()),"variance":float(composites.var()),"std":float(composites.std()),"worst":float(composites.min())}; aggregate_pass=aggregate["mean"]>baseline_test["primary_composite"]+1e-4; best_seed=int(max(reports,key=lambda r:r["validation_primary_composite"])["seed"]); model=MLP().to(device); model.load_state_dict(states[best_seed]); model.eval()
    with torch.no_grad(): fp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()*y_scale
    fp_metrics=metric(candidate_id,y[test],fp,calibration[best_seed]); q,s=quantize_state(states[best_seed]); buf=io.BytesIO(); np.savez_compressed(buf,**q,**{f"scale::{k}":v for k,v in s.items()}); payload=buf.getvalue();
    if len(payload)>spec["weight"]: raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch,q,s));
    with torch.no_grad(): qp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()*y_scale
    qm=metric(candidate_id,y[test],qp,calibration[best_seed]); qdelta=fp_metrics["primary_composite"]-qm["primary_composite"]; qpass=qm["primary_composite"]>baseline_test["primary_composite"]+1e-4 and qdelta<=.03; passed=aggregate_pass and qpass
    golden=output/"golden_vectors.npz"; np.savez_compressed(golden,x=x[test[:64]],y=y[test[:64]],fp32=fp[:64],quantized=qp[:64]); model.load_state_dict(states[best_seed]); onnx_path=output/"fp32.onnx"; torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["thermomechanical_features"],output_names=[spec["semantics"]],dynamic_axes={"thermomechanical_features":{0:"batch"},spec["semantics"]:{0:"batch"}},opset_version=17,dynamo=False); onnx.checker.check_model(onnx.load(onnx_path))
    schema={"task_kind":"trajectory_regression","shape":[None,y.shape[1]],"semantics":spec["semantics"],"postprocess":"multiply_train_only_output_scale","authority":0,"public_claim_scope":"SIM_ONLY"}; release=hashlib.sha256(canonical_bytes({"candidate_id":candidate_id,"dataset":metadata["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(golden),"mean":aggregate["mean"]})).hexdigest(); package=build_package(output,candidate_id,payload,sha256_file(golden),release,hashlib.sha256(canonical_bytes(schema)).hexdigest()); status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE_SIM_ONLY"
    audit={"schema":"cimc.forge200.physics-trajectory-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":candidate_id,"truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","baseline":{"kind":metadata["baseline_execution"],"validation":baseline_validation,"test":baseline_test},"seed_reports":reports,"aggregate":aggregate,"g3_aggregate_mean_gate":aggregate_pass,"quantized_best_seed":{"seed":best_seed,"test":qm,"metric_delta":qdelta,"gate":qpass},"parameter_count":parameter_count,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False}; write_json(output/"contract_exact_audit.json",audit); write_json(output/"source_manifest.json",metadata); write_json(output/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist(),"y_scale_train_q95":y_scale.tolist(),"interval_halfwidth_validation_q80":calibration[best_seed].tolist() if calibration[best_seed] is not None else None}); write_json(output/"output_schema.json",schema); write_json(output/"quantization_parity.json",{"primary_composite_delta":qdelta,"gate":qpass}); (output/"model_card.md").write_text(f"# {candidate_id} SIM_ONLY model\n\n- Status: `{status}`.\n- Truth: `PHYSICS_SIM`; no experimental claim.\n- Three-seed mean: `{aggregate['mean']:.6f}`; baseline: `{baseline_test['primary_composite']:.6f}`.\n- Parameters: `{parameter_count}`; W8: `{len(payload)}` bytes.\n- Authority: `0`; board pending.\n",encoding="utf-8")
    promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":candidate_id,"authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"PHYSICS_SIM","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(golden),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}}; write_json(output/"promotion_receipt.json",promotion); manifest(output); heartbeat(heartbeat_path,candidate_id,"COMPLETE"); write_json(root/"evidence"/f"physics_{candidate_id[-4:].lower()}_exact_closure.v1.json",{"schema":"cimc.forge200.physics-trajectory-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0}); print(json.dumps({"candidate_id":candidate_id,"status":status,"mean_composite":aggregate["mean"],"baseline_composite":baseline_test["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True)); return promotion


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument("--artifact-root",type=Path,required=True); p.add_argument("--candidate-id",choices=sorted(SPECS),required=True); p.add_argument("--device",default="cuda:0"); p.add_argument("--batch-size",type=int,default=256); p.add_argument("--max-epochs",type=int,default=100); p.add_argument("--min-epochs",type=int,default=30); p.add_argument("--early-stop-patience",type=int,default=14); p.add_argument("--learning-rate",type=float,default=8e-4); a=p.parse_args(); receipt=run(a); return 0 if receipt["host_contract_pass"] else 2


if __name__=="__main__": raise SystemExit(main())
