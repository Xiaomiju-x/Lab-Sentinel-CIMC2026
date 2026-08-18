#!/usr/bin/env python3
"""Train/package SIM_ONLY P102 gel point model."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state
def f1(y,p):
 tp=np.sum((y==1)&(p==1));fp=np.sum((y==0)&(p==1));fn=np.sum((y==1)&(p==0));return float(2*tp/max(2*tp+fp+fn,1))
def metrics(y,pred,event):
 pos=event==1;tmae=float(np.mean(np.abs(pred[pos,0]-y[pos,0])));temae=float(np.mean(np.abs(pred[pos,1]-y[pos,1])));score=f1(event,(pred[:,3]>=.5).astype(int));ts=max(float(np.std(y[pos,0])),1e-6);temps=max(float(np.std(y[pos,1])),1e-6);r={"gel_time_MAE_min":tmae,"gel_temperature_MAE_C":temae,"gel_event_F1":score,"time_skill":1.-tmae/ts,"temperature_skill":1.-temae/temps};r["primary_composite"]=float(np.mean([r["time_skill"],r["temperature_skill"],score]));return r
def run(a):
 import onnx,torch
 from torch import nn
 from torch.utils.data import DataLoader,TensorDataset
 root=a.root.resolve();d=root/"data"/"staged_physics_p102_exact_v1"/"CAND-P-102.npz";m=json.loads(d.with_suffix(".metadata.json").read_text(encoding="utf-8"));
 if m["status"]!="PASS" or m["truth_class"]!="PHYSICS_SIM" or m["cross_split_family_overlap"] or sha256_file(d)!=m["sha256"]:raise RuntimeError("DATA_GATE")
 if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
 device=torch.device(a.device);props=torch.cuda.get_device_properties(device);raw=np.load(d,allow_pickle=False);xr=raw["x"].astype(np.float32);y=raw["y"].astype(np.float32);event=raw["event"].astype(np.int8);base=raw["baseline_pred"].astype(np.float32);split=raw["split"].astype(np.int8);train,val,test=(np.flatnonzero(split==c) for c in (0,1,2));mean,std=xr[train].mean(0),xr[train].std(0);std[std<1e-7]=1.;x=np.clip((xr-mean)/std,-12,12).astype(np.float32);ym=y[train].mean(0);ys=y[train].std(0);target=np.column_stack(((y-ym)/ys,event)).astype(np.float32);out=a.artifact_root.resolve()/"CAND-P-102";out.mkdir(parents=True,exist_ok=True);hb=out/"heartbeat.json";started=time.perf_counter()
 class MLP(nn.Module):
  def __init__(self):super().__init__();self.body=nn.Sequential(nn.Linear(x.shape[1],96),nn.GELU(),nn.Linear(96,48),nn.GELU());self.head=nn.Linear(48,4)
  def forward(self,v):z=self.head(self.body(v));return torch.cat((z[:,:3],torch.sigmoid(z[:,3:4])),1)
 params=sum(p.numel() for p in MLP().parameters());
 if params>42000:raise RuntimeError(f"PARAMETER_CAP:{params}")
 bv=metrics(y[val],base[val],event[val]);bt=metrics(y[test],base[test],event[test]);reports=[];states={};data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(target[train]))
 for seed in SEEDS:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=MLP().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=3e-4);loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));ck=out/f"train_seed_{seed}"/"best.pt";ck.parent.mkdir(parents=True,exist_ok=True);best=-1e30;pat=0
  for epoch in range(a.max_epochs):
   model.train()
   for bx,by in loader:opt.zero_grad(set_to_none=True);pr=model(bx.to(device));mask=by.to(device)[:,3]>0.5;reg=nn.functional.smooth_l1_loss(pr[mask,:3],by.to(device)[mask,:3],beta=.1);loss=reg+nn.functional.binary_cross_entropy(pr[:,3],by.to(device)[:,3]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
   model.eval()
   with torch.no_grad():vp=model(torch.from_numpy(x[val]).to(device)).cpu().numpy();vp[:,:3]=vp[:,:3]*ys+ym
   score=metrics(y[val],vp,event[val])["primary_composite"]
   if score>best+1e-5:best=score;pat=0;torch.save(model.state_dict(),ck)
   else:pat+=1
   heartbeat(hb,"CAND-P-102","TRAIN_PHYSICS_P102_EXACT",seed,epoch)
   if epoch+1>=a.min_epochs and pat>=a.early_stop_patience:break
  model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval();
  with torch.no_grad():tp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy();tp[:,:3]=tp[:,:3]*ys+ym
  mm=metrics(y[test],tp,event[test]);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":mm,"beats_baseline":mm["primary_composite"]>bt["primary_composite"]+1e-4});states[seed]={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 comp=np.asarray([r["test"]["primary_composite"] for r in reports]);agg={"mean":float(comp.mean()),"variance":float(comp.var()),"std":float(comp.std()),"worst":float(comp.min())};aggpass=agg["mean"]>bt["primary_composite"]+1e-4;bs=int(max(reports,key=lambda r:r["validation_primary_composite"])["seed"]);model=MLP().to(device);model.load_state_dict(states[bs]);model.eval();
 with torch.no_grad():fp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy();fp[:,:3]=fp[:,:3]*ys+ym
 fpm=metrics(y[test],fp,event[test]);q,s=quantize_state(states[bs]);buf=io.BytesIO();np.savez_compressed(buf,**q,**{f"scale::{k}":v for k,v in s.items()});payload=buf.getvalue();
 if len(payload)>52*1024:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
 model.load_state_dict(dequantized_state(torch,q,s));
 with torch.no_grad():qp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy();qp[:,:3]=qp[:,:3]*ys+ym
 qm=metrics(y[test],qp,event[test]);qd=fpm["primary_composite"]-qm["primary_composite"];qpass=qm["primary_composite"]>bt["primary_composite"]+1e-4 and qd<=.03;passed=aggpass and qpass;gold=out/"golden_vectors.npz";np.savez_compressed(gold,x=x[test[:64]],y=y[test[:64]],event=event[test[:64]],fp32=fp[:64],quantized=qp[:64]);model.load_state_dict(states[bs]);onnx_path=out/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["viscosity_frequency_temperature_cure_history"],output_names=["normalized_gel_targets_and_event_probability"],dynamic_axes={"viscosity_frequency_temperature_cure_history":{0:"batch"},"normalized_gel_targets_and_event_probability":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));schema={"task_kind":"gelpoint_multitask","shape":[None,4],"semantics":["gel_time_min","gel_temperature_C","gel_conversion","gel_event_probability"],"postprocess":"first_three_multiply_train_std_add_mean;probability_identity","authority":0,"public_claim_scope":"SIM_ONLY"};release=hashlib.sha256(canonical_bytes({"candidate":"CAND-P-102","dataset":m["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(gold),"mean":agg["mean"]})).hexdigest();package=build_package(out,"CAND-P-102",payload,sha256_file(gold),release,hashlib.sha256(canonical_bytes(schema)).hexdigest());status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE_SIM_ONLY";audit={"schema":"cimc.forge200.physics-p102-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-P-102","truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","baseline":{"kind":m["baseline_execution"],"validation":bv,"test":bt},"seed_reports":reports,"aggregate":agg,"g3_aggregate_mean_gate":aggpass,"quantized_best_seed":{"seed":bs,"test":qm,"metric_delta":qd,"gate":qpass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False};write_json(out/"contract_exact_audit.json",audit);write_json(out/"source_manifest.json",m);write_json(out/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist(),"target_mean":ym.tolist(),"target_std":ys.tolist()});write_json(out/"output_schema.json",schema);write_json(out/"quantization_parity.json",{"primary_composite_delta":qd,"gate":qpass});(out/"model_card.md").write_text(f"# CAND-P-102 SIM_ONLY\n\n- Status `{status}`; no experimental claim.\n- Mean `{agg['mean']:.6f}` vs fixed-viscosity baseline `{bt['primary_composite']:.6f}`.\n- Parameters `{params}`; W8 `{len(payload)}` bytes; authority `0`; board pending.\n",encoding="utf-8");promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-P-102","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"PHYSICS_SIM","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(gold),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}};write_json(out/"promotion_receipt.json",promotion);manifest(out);heartbeat(hb,"CAND-P-102","COMPLETE");write_json(root/"evidence"/"physics_p102_exact_closure.v1.json",{"schema":"cimc.forge200.physics-p102-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":"CAND-P-102","status":status,"mean_composite":agg["mean"],"baseline_composite":bt["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--artifact-root",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=256);p.add_argument("--max-epochs",type=int,default=100);p.add_argument("--min-epochs",type=int,default=30);p.add_argument("--early-stop-patience",type=int,default=14);p.add_argument("--learning-rate",type=float,default=8e-4);a=p.parse_args();r=run(a);return 0 if r["host_contract_pass"] else 2
if __name__=="__main__":raise SystemExit(main())
