#!/usr/bin/env python3
"""Train/package the exact curated-unit S036 classifier."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state
def metrics(y,p):
 f=[]
 for k in range(3):
  tp=np.sum((y==k)&(p==k));fp=np.sum((y!=k)&(p==k));fn=np.sum((y==k)&(p!=k));f.append(float(2*tp/max(2*tp+fp+fn,1)))
 invalid=float(np.sum((y==2)&(p==2))/max(np.sum(y==2),1));false=float(np.sum((y!=2)&(p==2))/max(np.sum(y!=2),1));r={"macro_F1":float(np.mean(f)),"invalid_recall":invalid,"false_reject_rate":false,"per_class_F1":f};r["primary_composite"]=float(np.mean([r["macro_F1"],invalid,1.-false]));return r
def run(a):
 import onnx,torch
 from torch import nn
 from torch.utils.data import DataLoader,TensorDataset
 root=a.root.resolve();d=root/"data"/"staged_support_s036_exact_v1"/"CAND-S-036.npz";m=json.loads(d.with_suffix(".metadata.json").read_text(encoding="utf-8"))
 if m["status"]!="PASS" or m["source_gate_match"]!="CURATED_SI_UNIT_CASES" or m["cross_split_template_overlap"] or sha256_file(d)!=m["sha256"]:raise RuntimeError("DATA_GATE")
 if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
 device=torch.device(a.device);props=torch.cuda.get_device_properties(device);raw=np.load(d,allow_pickle=False);xr=raw["x"].astype(np.float32);y=raw["y"].astype(np.int64);base=raw["baseline_pred"].astype(np.int64);split=raw["split"].astype(np.int8);train,val,test=(np.flatnonzero(split==c) for c in range(3));mean=xr[train].mean(0);std=xr[train].std(0);std[std<1e-7]=1.;x=np.clip((xr-mean)/std,-12,12).astype(np.float32);bv=metrics(y[val],base[val]);bt=metrics(y[test],base[test]);out=a.artifact_root.resolve()/"CAND-S-036";out.mkdir(parents=True,exist_ok=True);hb=out/"heartbeat.json";started=time.perf_counter()
 class M(nn.Module):
  def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(x.shape[1],72),nn.GELU(),nn.Linear(72,36),nn.GELU(),nn.Linear(36,3))
  def forward(self,v):return self.net(v)
 params=sum(p.numel() for p in M().parameters())
 if params>48000:raise RuntimeError(f"PARAMETER_CAP:{params}")
 data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(y[train]));reports=[];states={}
 for seed in SEEDS:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=M().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=4e-4);loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));ck=out/f"train_seed_{seed}"/"best.pt";ck.parent.mkdir(parents=True,exist_ok=True);best=-1e30;pat=0
  for epoch in range(a.max_epochs):
   model.train()
   for bx,by in loader:opt.zero_grad(set_to_none=True);logits=model(bx.to(device));loss=nn.functional.cross_entropy(logits,by.to(device));loss.backward();opt.step()
   model.eval()
   with torch.no_grad():vp=model(torch.from_numpy(x[val]).to(device)).argmax(1).cpu().numpy()
   score=metrics(y[val],vp)["primary_composite"]
   if score>best+1e-6:best=score;pat=0;torch.save(model.state_dict(),ck)
   else:pat+=1
   heartbeat(hb,"CAND-S-036","TRAIN_SUPPORT_EXACT",seed,epoch)
   if epoch+1>=a.min_epochs and pat>=a.early_stop_patience:break
  model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval()
  with torch.no_grad():tp=model(torch.from_numpy(x[test]).to(device)).argmax(1).cpu().numpy()
  mm=metrics(y[test],tp);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":mm,"beats_baseline":mm["primary_composite"]>bt["primary_composite"]+1e-4});states[seed]={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 scores=np.asarray([r["test"]["primary_composite"] for r in reports]);agg={"mean":float(scores.mean()),"variance":float(scores.var()),"std":float(scores.std()),"worst":float(scores.min())};aggpass=agg["mean"]>bt["primary_composite"]+1e-4;bs=int(max(reports,key=lambda r:r["validation_primary_composite"])["seed"]);model=M().to(device);model.load_state_dict(states[bs]);model.eval()
 with torch.no_grad():fp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()
 fm=metrics(y[test],fp.argmax(1));q,s=quantize_state(states[bs]);buf=io.BytesIO();np.savez_compressed(buf,**q,**{f"scale::{k}":v for k,v in s.items()});payload=buf.getvalue()
 if len(payload)>48*1024:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
 model.load_state_dict(dequantized_state(torch,q,s))
 with torch.no_grad():qp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()
 qm=metrics(y[test],qp.argmax(1));delta=fm["primary_composite"]-qm["primary_composite"];qpass=qm["primary_composite"]>bt["primary_composite"]+1e-4 and delta<=.03;passed=aggpass and qpass;gold=out/"golden_vectors.npz";np.savez_compressed(gold,x=x[test[:128]],y=y[test[:128]],fp32=fp[:128],quantized=qp[:128]);model.load_state_dict(states[bs]);onnx_path=out/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["equation_unit_dimension_context_features"],output_names=["unit_class_logits"],dynamic_axes={"equation_unit_dimension_context_features":{0:"batch"},"unit_class_logits":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));schema={"task_kind":"classification","shape":[None,3],"semantics":["unit_consistent","convertible","dimensionally_invalid"],"authority":0};release=hashlib.sha256(canonical_bytes({"candidate":"CAND-S-036","dataset":m["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(gold),"mean":agg["mean"]})).hexdigest();package=build_package(out,"CAND-S-036",payload,sha256_file(gold),release,hashlib.sha256(canonical_bytes(schema)).hexdigest());status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE";audit={"schema":"cimc.forge200.support-s036-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-S-036","truth_class":"CONTROLLED_FIXTURE","baseline":{"kind":m["baseline_execution"],"validation":bv,"test":bt},"seed_reports":reports,"aggregate":agg,"g3_aggregate_mean_gate":aggpass,"quantized_best_seed":{"seed":bs,"test":qm,"metric_delta":delta,"gate":qpass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False};write_json(out/"contract_exact_audit.json",audit);write_json(out/"source_manifest.json",m);write_json(out/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist()});write_json(out/"output_schema.json",schema);write_json(out/"quantization_parity.json",{"primary_composite_delta":delta,"gate":qpass});(out/"model_card.md").write_text(f"# CAND-S-036\n\n- Status `{status}`; curated SI fixture, not experimental data.\n- Three-seed mean `{agg['mean']:.6f}` vs context-blind parser `{bt['primary_composite']:.6f}`.\n- Authority `0`; board pending.\n",encoding="utf-8");promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-S-036","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"CONTROLLED_FIXTURE","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(gold),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}};write_json(out/"promotion_receipt.json",promotion);manifest(out);heartbeat(hb,"CAND-S-036","COMPLETE");write_json(root/"evidence"/"support_s036_exact_closure.v1.json",{"schema":"cimc.forge200.support-s036-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":"CAND-S-036","status":status,"mean_composite":agg["mean"],"baseline_composite":bt["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--artifact-root",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=256);p.add_argument("--max-epochs",type=int,default=100);p.add_argument("--min-epochs",type=int,default=25);p.add_argument("--early-stop-patience",type=int,default=14);p.add_argument("--learning-rate",type=float,default=8e-4);a=p.parse_args();r=run(a);return 0 if r["host_contract_pass"] else 2
if __name__=="__main__":raise SystemExit(main())
