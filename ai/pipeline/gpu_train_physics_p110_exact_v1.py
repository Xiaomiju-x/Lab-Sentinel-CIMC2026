#!/usr/bin/env python3
"""Train/package SIM_ONLY P110 thermal interface model."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state
def metrics(y,p,half):
 err=np.abs(p-y);mae=float(np.mean(err));mape=float(np.mean(err/np.maximum(np.abs(y),1e-4))*100);cov=float(np.mean(err<=half));scale=max(float(np.std(y)),1e-6);r={"MAPE_percent":mape,"MAE_K_per_W":mae,"interval_coverage_80":cov,"MAPE_score":1./(1.+mape/100.),"MAE_skill":1.-mae/scale,"coverage_score":1.-abs(cov-.8)};r["primary_composite"]=float(np.mean([r["MAPE_score"],r["MAE_skill"],r["coverage_score"]]));return r
def run(a):
 import onnx,torch
 from torch import nn
 from torch.utils.data import DataLoader,TensorDataset
 root=a.root.resolve();d=root/"data"/"staged_physics_p110_exact_v1"/"CAND-P-110.npz";m=json.loads(d.with_suffix(".metadata.json").read_text(encoding="utf-8"));
 if m["status"]!="PASS" or m["truth_class"]!="PHYSICS_SIM" or m["cross_split_family_overlap"] or sha256_file(d)!=m["sha256"]:raise RuntimeError("DATA_GATE")
 if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
 device=torch.device(a.device);props=torch.cuda.get_device_properties(device);raw=np.load(d,allow_pickle=False);xr=raw["x"].astype(np.float32);y=raw["y"].astype(np.float32);base=raw["baseline_pred"].astype(np.float32);split=raw["split"].astype(np.int8);train,val,test=(np.flatnonzero(split==c) for c in (0,1,2));mean,std=xr[train].mean(0),xr[train].std(0);std[std<1e-7]=1.;x=np.clip((xr-mean)/std,-12,12).astype(np.float32);yt=np.log1p(y).astype(np.float32);out=a.artifact_root.resolve()/"CAND-P-110";out.mkdir(parents=True,exist_ok=True);hb=out/"heartbeat.json";started=time.perf_counter()
 class MLP(nn.Module):
  def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(x.shape[1],96),nn.GELU(),nn.Linear(96,48),nn.GELU(),nn.Linear(48,1))
  def forward(self,v):return self.net(v)
 params=sum(p.numel() for p in MLP().parameters());
 if params>44000:raise RuntimeError(f"PARAMETER_CAP:{params}")
 bh=np.quantile(np.abs(base[val]-y[val]),.8);bv=metrics(y[val],base[val],bh);bt=metrics(y[test],base[test],bh);reports=[];states={};cal={};data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(yt[train,None]))
 for seed in SEEDS:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=MLP().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=3e-4);loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));ck=out/f"train_seed_{seed}"/"best.pt";ck.parent.mkdir(parents=True,exist_ok=True);best=-1e30;pat=0
  for epoch in range(a.max_epochs):
   model.train()
   for bx,by in loader:opt.zero_grad(set_to_none=True);pr=model(bx.to(device));loss=nn.functional.smooth_l1_loss(pr,by.to(device),beta=.1);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
   model.eval()
   with torch.no_grad():vp=np.maximum(np.expm1(model(torch.from_numpy(x[val]).to(device)).cpu().numpy()[:,0]),0)
   half=np.quantile(np.abs(vp-y[val]),.8);score=metrics(y[val],vp,half)["primary_composite"]
   if score>best+1e-5:best=score;pat=0;torch.save(model.state_dict(),ck)
   else:pat+=1
   heartbeat(hb,"CAND-P-110","TRAIN_PHYSICS_P110_EXACT",seed,epoch)
   if epoch+1>=a.min_epochs and pat>=a.early_stop_patience:break
  model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval();
  with torch.no_grad():vp=np.maximum(np.expm1(model(torch.from_numpy(x[val]).to(device)).cpu().numpy()[:,0]),0);tp=np.maximum(np.expm1(model(torch.from_numpy(x[test]).to(device)).cpu().numpy()[:,0]),0)
  half=np.quantile(np.abs(vp-y[val]),.8);cal[seed]=half;mm=metrics(y[test],tp,half);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":mm,"beats_baseline":mm["primary_composite"]>bt["primary_composite"]+1e-4});states[seed]={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 comp=np.asarray([r["test"]["primary_composite"] for r in reports]);agg={"mean":float(comp.mean()),"variance":float(comp.var()),"std":float(comp.std()),"worst":float(comp.min())};aggpass=agg["mean"]>bt["primary_composite"]+1e-4;bs=int(max(reports,key=lambda r:r["validation_primary_composite"])["seed"]);model=MLP().to(device);model.load_state_dict(states[bs]);model.eval();
 with torch.no_grad():fp=np.maximum(np.expm1(model(torch.from_numpy(x[test]).to(device)).cpu().numpy()[:,0]),0)
 fm=metrics(y[test],fp,cal[bs]);q,s=quantize_state(states[bs]);buf=io.BytesIO();np.savez_compressed(buf,**q,**{f"scale::{k}":v for k,v in s.items()});payload=buf.getvalue();
 if len(payload)>56*1024:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
 model.load_state_dict(dequantized_state(torch,q,s));
 with torch.no_grad():qp=np.maximum(np.expm1(model(torch.from_numpy(x[test]).to(device)).cpu().numpy()[:,0]),0)
 qm=metrics(y[test],qp,cal[bs]);qd=fm["primary_composite"]-qm["primary_composite"];qpass=qm["primary_composite"]>bt["primary_composite"]+1e-4 and qd<=.03;passed=aggpass and qpass;gold=out/"golden_vectors.npz";np.savez_compressed(gold,x=x[test[:64]],y=y[test[:64]],fp32=fp[:64],quantized=qp[:64]);model.load_state_dict(states[bs]);onnx_path=out/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["tim_thickness_conductivity_pressure_void_temperature"],output_names=["log1p_thermal_interface_resistance"],dynamic_axes={"tim_thickness_conductivity_pressure_void_temperature":{0:"batch"},"log1p_thermal_interface_resistance":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));schema={"task_kind":"regression","shape":[None,1],"semantics":"thermal_interface_resistance_K_per_W","postprocess":"max(expm1(value),0)","authority":0,"public_claim_scope":"SIM_ONLY"};release=hashlib.sha256(canonical_bytes({"candidate":"CAND-P-110","dataset":m["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(gold),"mean":agg["mean"]})).hexdigest();package=build_package(out,"CAND-P-110",payload,sha256_file(gold),release,hashlib.sha256(canonical_bytes(schema)).hexdigest());status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE_SIM_ONLY";audit={"schema":"cimc.forge200.physics-p110-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-P-110","truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","baseline":{"kind":m["baseline_execution"],"validation":bv,"test":bt},"seed_reports":reports,"aggregate":agg,"g3_aggregate_mean_gate":aggpass,"quantized_best_seed":{"seed":bs,"test":qm,"metric_delta":qd,"gate":qpass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False};write_json(out/"contract_exact_audit.json",audit);write_json(out/"source_manifest.json",m);write_json(out/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist(),"interval_halfwidth_validation_q80":float(cal[bs])});write_json(out/"output_schema.json",schema);write_json(out/"quantization_parity.json",{"primary_composite_delta":qd,"gate":qpass});(out/"model_card.md").write_text(f"# CAND-P-110 SIM_ONLY\n\n- Status `{status}`; no experimental claim.\n- Mean `{agg['mean']:.6f}` vs thickness/k baseline `{bt['primary_composite']:.6f}`.\n- Parameters `{params}`; W8 `{len(payload)}` bytes; authority `0`; board pending.\n",encoding="utf-8");promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-P-110","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"PHYSICS_SIM","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(gold),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}};write_json(out/"promotion_receipt.json",promotion);manifest(out);heartbeat(hb,"CAND-P-110","COMPLETE");write_json(root/"evidence"/"physics_p110_exact_closure.v1.json",{"schema":"cimc.forge200.physics-p110-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":"CAND-P-110","status":status,"mean_composite":agg["mean"],"baseline_composite":bt["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--artifact-root",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=256);p.add_argument("--max-epochs",type=int,default=100);p.add_argument("--min-epochs",type=int,default=30);p.add_argument("--early-stop-patience",type=int,default=14);p.add_argument("--learning-rate",type=float,default=8e-4);a=p.parse_args();r=run(a);return 0 if r["host_contract_pass"] else 2
if __name__=="__main__":raise SystemExit(main())
