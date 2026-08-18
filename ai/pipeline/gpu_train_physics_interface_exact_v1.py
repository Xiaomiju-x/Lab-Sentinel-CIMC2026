#!/usr/bin/env python3
"""Train/package SIM_ONLY P108 and P111 interface models."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state
SPECS={"CAND-P-108":{"cap":52000,"weight":64*1024,"outputs":5},"CAND-P-111":{"cap":50000,"weight":64*1024,"outputs":2}}
def ranks(v):o=np.argsort(v,kind="mergesort");r=np.empty(len(v));r[o]=np.arange(1,len(v)+1);return r
def auroc(y,s):p=y==1;n=y==0;return float((ranks(s)[p].sum()-p.sum()*(p.sum()+1)/2)/max(p.sum()*n.sum(),1))
def auprc(y,s):
 o=np.argsort(-s);yy=y[o];tp=np.cumsum(yy);precision=tp/np.arange(1,len(y)+1);return float(np.sum(precision*yy)/max(np.sum(yy),1))
def macro_f1(y,p,n):
 vals=[]
 for c in range(n):
  tp=np.sum((y==c)&(p==c));fp=np.sum((y!=c)&(p==c));fn=np.sum((y==c)&(p!=c));vals.append(2*tp/max(2*tp+fp+fn,1))
 return float(np.mean(vals))
def ece_binary(y,p):
 bins=np.minimum((p*10).astype(int),9);return float(sum(np.sum(bins==b)/len(p)*abs(float(np.mean(p[bins==b]))-float(np.mean(y[bins==b]))) for b in range(10) if np.any(bins==b)))
def metric(cid,raw,pred):
 if cid.endswith("108"):
  risk=raw["risk_label"];loc=raw["location_label"];prob=np.clip(pred[:,0],0,1);lp=pred[:,1:];auc=auroc(risk,prob);brier=float(np.mean((prob-risk)**2));f1=macro_f1(loc,np.argmax(lp,1),4);ece=ece_binary(risk,prob);r={"AUROC":auc,"Brier_score":brier,"location_macro_F1":f1,"ECE":ece};r["primary_composite"]=float(np.mean([auc,1-brier,f1,1-ece]));return r
 strength=raw["y_strength"];weak=raw["weak_label"];sp=pred[:,0];prob=np.clip(pred[:,1],0,1);mae=float(np.mean(np.abs(sp-strength)));skill=1-mae/max(float(np.std(strength)),1e-6);ap=auprc(weak,prob);ece=ece_binary(weak,prob);r={"MAE_MPa":mae,"weak_bond_AUPRC":ap,"ECE":ece,"strength_skill":skill};r["primary_composite"]=float(np.mean([skill,ap,1-ece]));return r
def run(a):
 import onnx,torch
 from torch import nn
 from torch.utils.data import DataLoader,TensorDataset
 cid=a.candidate_id;spec=SPECS[cid];root=a.root.resolve();d=root/"data"/"staged_physics_p108_p111_exact_v1"/f"{cid}.npz";m=json.loads(d.with_suffix(".metadata.json").read_text(encoding="utf-8"));
 if m["status"]!="PASS" or m["truth_class"]!="PHYSICS_SIM" or m["cross_split_family_overlap"] or sha256_file(d)!=m["sha256"]:raise RuntimeError("DATA_GATE")
 if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
 device=torch.device(a.device);props=torch.cuda.get_device_properties(device);z=np.load(d,allow_pickle=False);allraw={k:z[k] for k in z.files};xr=z["x"].astype(np.float32);split=z["split"].astype(np.int8);train,val,test=(np.flatnonzero(split==c) for c in (0,1,2));mean,std=xr[train].mean(0),xr[train].std(0);std[std<1e-7]=1.;x=np.clip((xr-mean)/std,-12,12).astype(np.float32);out=a.artifact_root.resolve()/cid;out.mkdir(parents=True,exist_ok=True);hb=out/"heartbeat.json";started=time.perf_counter()
 if cid.endswith("108"):
  target=np.column_stack((z["y_probability"],z["location_label"])).astype(np.float32);base=np.column_stack((z["baseline_probability"],np.eye(4,dtype=np.float32)[z["baseline_location"]]));strength_mean=strength_std=None
 else:
  strength_mean=float(z["y_strength"][train].mean());strength_std=float(z["y_strength"][train].std());target=np.column_stack(((z["y_strength"]-strength_mean)/strength_std,z["weak_label"])).astype(np.float32);base=np.column_stack((z["baseline_strength"],z["baseline_weak_probability"])).astype(np.float32)
 class MLP(nn.Module):
  def __init__(self):super().__init__();self.body=nn.Sequential(nn.Linear(x.shape[1],112),nn.GELU(),nn.Linear(112,56),nn.GELU());self.head=nn.Linear(56,spec["outputs"])
  def forward(self,v):
   q=self.head(self.body(v))
   if cid.endswith("108"):return torch.cat((torch.sigmoid(q[:,:1]),torch.softmax(q[:,1:],1)),1)
   return torch.cat((q[:,:1],torch.sigmoid(q[:,1:2])),1)
 params=sum(p.numel() for p in MLP().parameters());
 if params>spec["cap"]:raise RuntimeError(f"PARAMETER_CAP:{params}")
 def raw_slice(idx):return {k:v[idx] for k,v in allraw.items() if len(v.shape)>0 and len(v)==len(x)}
 bv=metric(cid,raw_slice(val),base[val]);bt=metric(cid,raw_slice(test),base[test]);reports=[];states={};data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(target[train]))
 for seed in SEEDS:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=MLP().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=3e-4);loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));ck=out/f"train_seed_{seed}"/"best.pt";ck.parent.mkdir(parents=True,exist_ok=True);best=-1e30;pat=0
  for epoch in range(a.max_epochs):
   model.train()
   for bx,by in loader:
    opt.zero_grad(set_to_none=True);pr=model(bx.to(device));yy=by.to(device)
    if cid.endswith("108"):loss=nn.functional.smooth_l1_loss(pr[:,0],yy[:,0],beta=.05)+nn.functional.nll_loss(torch.log(pr[:,1:].clamp_min(1e-7)),yy[:,1].long())
    else:loss=nn.functional.smooth_l1_loss(pr[:,0],yy[:,0],beta=.1)+nn.functional.binary_cross_entropy(pr[:,1],yy[:,1])
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
   model.eval()
   with torch.no_grad():vp=model(torch.from_numpy(x[val]).to(device)).cpu().numpy()
   if not cid.endswith("108"):vp[:,0]=vp[:,0]*strength_std+strength_mean
   score=metric(cid,raw_slice(val),vp)["primary_composite"]
   if score>best+1e-5:best=score;pat=0;torch.save(model.state_dict(),ck)
   else:pat+=1
   heartbeat(hb,cid,"TRAIN_PHYSICS_INTERFACE_EXACT",seed,epoch)
   if epoch+1>=a.min_epochs and pat>=a.early_stop_patience:break
  model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval();
  with torch.no_grad():tp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()
  if not cid.endswith("108"):tp[:,0]=tp[:,0]*strength_std+strength_mean
  mm=metric(cid,raw_slice(test),tp);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":mm,"beats_baseline":mm["primary_composite"]>bt["primary_composite"]+1e-4});states[seed]={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 comp=np.asarray([r["test"]["primary_composite"] for r in reports]);agg={"mean":float(comp.mean()),"variance":float(comp.var()),"std":float(comp.std()),"worst":float(comp.min())};aggpass=agg["mean"]>bt["primary_composite"]+1e-4;bs=int(max(reports,key=lambda r:r["validation_primary_composite"])["seed"]);model=MLP().to(device);model.load_state_dict(states[bs]);model.eval();
 with torch.no_grad():fp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()
 if not cid.endswith("108"):fp[:,0]=fp[:,0]*strength_std+strength_mean
 fm=metric(cid,raw_slice(test),fp);q,s=quantize_state(states[bs]);buf=io.BytesIO();np.savez_compressed(buf,**q,**{f"scale::{k}":v for k,v in s.items()});payload=buf.getvalue();
 if len(payload)>spec["weight"]:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
 model.load_state_dict(dequantized_state(torch,q,s));
 with torch.no_grad():qp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy()
 if not cid.endswith("108"):qp[:,0]=qp[:,0]*strength_std+strength_mean
 qm=metric(cid,raw_slice(test),qp);qd=fm["primary_composite"]-qm["primary_composite"];qpass=qm["primary_composite"]>bt["primary_composite"]+1e-4 and qd<=.03;passed=aggpass and qpass;gold=out/"golden_vectors.npz";np.savez_compressed(gold,x=x[test[:64]],fp32=fp[:64],quantized=qp[:64]);model.load_state_dict(states[bs]);onnx_path=out/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["interface_features"],output_names=["normalized_interface_outputs"],dynamic_axes={"interface_features":{0:"batch"},"normalized_interface_outputs":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));schema={"task_kind":"multitask","shape":[None,spec["outputs"]],"semantics":"delamination_probability_plus_location_probabilities" if cid.endswith("108") else "bond_strength_MPa_plus_weak_probability","postprocess":"identity" if cid.endswith("108") else "strength_multiply_train_std_add_mean;probability_identity","authority":0,"public_claim_scope":"SIM_ONLY"};release=hashlib.sha256(canonical_bytes({"candidate":cid,"dataset":m["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(gold),"mean":agg["mean"]})).hexdigest();package=build_package(out,cid,payload,sha256_file(gold),release,hashlib.sha256(canonical_bytes(schema)).hexdigest());status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE_SIM_ONLY";audit={"schema":"cimc.forge200.physics-interface-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":cid,"truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY","baseline":{"kind":m["baseline_execution"],"validation":bv,"test":bt},"seed_reports":reports,"aggregate":agg,"g3_aggregate_mean_gate":aggpass,"quantized_best_seed":{"seed":bs,"test":qm,"metric_delta":qd,"gate":qpass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False};write_json(out/"contract_exact_audit.json",audit);write_json(out/"source_manifest.json",m);write_json(out/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist(),"strength_mean":strength_mean,"strength_std":strength_std});write_json(out/"output_schema.json",schema);write_json(out/"quantization_parity.json",{"primary_composite_delta":qd,"gate":qpass});(out/"model_card.md").write_text(f"# {cid} SIM_ONLY\n\n- Status `{status}`; no experimental claim.\n- Mean `{agg['mean']:.6f}` vs baseline `{bt['primary_composite']:.6f}`.\n- Parameters `{params}`; W8 `{len(payload)}` bytes; authority `0`; board pending.\n",encoding="utf-8");promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":cid,"authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"PHYSICS_SIM","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(gold),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}};write_json(out/"promotion_receipt.json",promotion);manifest(out);heartbeat(hb,cid,"COMPLETE");write_json(root/"evidence"/f"physics_{cid[-4:].lower()}_exact_closure.v1.json",{"schema":"cimc.forge200.physics-interface-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":cid,"status":status,"mean_composite":agg["mean"],"baseline_composite":bt["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--artifact-root",type=Path,required=True);p.add_argument("--candidate-id",choices=sorted(SPECS),required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=256);p.add_argument("--max-epochs",type=int,default=100);p.add_argument("--min-epochs",type=int,default=30);p.add_argument("--early-stop-patience",type=int,default=14);p.add_argument("--learning-rate",type=float,default=8e-4);a=p.parse_args();r=run(a);return 0 if r["host_contract_pass"] else 2
if __name__=="__main__":raise SystemExit(main())
