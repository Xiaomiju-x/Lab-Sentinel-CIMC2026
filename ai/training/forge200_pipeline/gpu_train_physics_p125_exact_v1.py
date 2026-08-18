#!/usr/bin/env python3
"""Train/package the SIM_ONLY FEA P125 warpage-risk model."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state

def ranks(v):
    o=np.argsort(v,kind="mergesort");r=np.empty(len(v),float);r[o]=np.arange(1,len(v)+1);return r
def auroc(label,score):
    p=label==1;n=label==0;return float((ranks(score)[p].sum()-p.sum()*(p.sum()+1)/2)/max(p.sum()*n.sum(),1))
def metrics(y,pred,label):
    war=pred[:,0];prob=np.clip(pred[:,1],0,1);mae=float(np.mean(np.abs(war-y[:,0])));scale=max(float(np.std(y[:,0])),1e-6);auc=auroc(label,prob);bins=np.minimum((prob*10).astype(int),9);ece=sum(np.sum(bins==b)/len(prob)*abs(float(np.mean(prob[bins==b]))-float(np.mean(label[bins==b]))) for b in range(10) if np.any(bins==b));res={"warpage_MAE_um":mae,"delamination_AUROC":auc,"calibration_ECE":float(ece),"warpage_skill":1.-mae/scale};res["primary_composite"]=float(np.mean([res["warpage_skill"],auc,1.-ece]));return res
def run(a)->dict[str,Any]:
    import onnx,torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    root=a.root.resolve();d=root/"data"/"staged_physics_p125_exact_v1"/"CAND-P-125.npz";m=json.loads(d.with_suffix(".metadata.json").read_text(encoding="utf-8"));
    if m["status"]!="PASS" or m["truth_class"]!="PHYSICS_SIM_FEA" or m["cross_split_family_overlap"] or sha256_file(d)!=m["sha256"]:raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
    device=torch.device(a.device);props=torch.cuda.get_device_properties(device);raw=np.load(d,allow_pickle=False);xr=raw["x"].astype(np.float32);y=raw["y"].astype(np.float32);label=raw["risk_label"].astype(np.int8);base=raw["baseline_pred"].astype(np.float32);split=raw["split"].astype(np.int8);train,val,test=(np.flatnonzero(split==c) for c in (0,1,2));mean,std=xr[train].mean(0),xr[train].std(0);std[std<1e-7]=1.;x=np.clip((xr-mean)/std,-12,12).astype(np.float32);wymean=float(y[train,0].mean());wystd=float(y[train,0].std());target=np.column_stack(((y[:,0]-wymean)/wystd,y[:,1])).astype(np.float32);out=a.artifact_root.resolve()/"CAND-P-125";out.mkdir(parents=True,exist_ok=True);hb=out/"heartbeat.json";started=time.perf_counter()
    class MLP(nn.Module):
        def __init__(self):super().__init__();self.body=nn.Sequential(nn.Linear(x.shape[1],128),nn.GELU(),nn.Linear(128,64),nn.GELU());self.head=nn.Linear(64,2)
        def forward(self,v):z=self.head(self.body(v));return torch.cat((z[:,:1],torch.sigmoid(z[:,1:2])),dim=1)
    params=sum(p.numel() for p in MLP().parameters());
    if params>96000:raise RuntimeError(f"PARAMETER_CAP:{params}")
    bv=metrics(y[val],base[val],label[val]);bt=metrics(y[test],base[test],label[test]);reports=[];states={};data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(target[train]))
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=MLP().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=3e-4);loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));ck=out/f"train_seed_{seed}"/"best.pt";ck.parent.mkdir(parents=True,exist_ok=True);best=-1e30;pat=0
        for epoch in range(a.max_epochs):
            model.train()
            for bx,by in loader:opt.zero_grad(set_to_none=True);pr=model(bx.to(device));loss=nn.functional.smooth_l1_loss(pr[:,:1],by.to(device)[:,:1],beta=.1)+nn.functional.binary_cross_entropy(pr[:,1],by.to(device)[:,1]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);opt.step()
            model.eval()
            with torch.no_grad():vp=model(torch.from_numpy(x[val]).to(device)).cpu().numpy();vp[:,0]=vp[:,0]*wystd+wymean
            score=metrics(y[val],vp,label[val])["primary_composite"]
            if score>best+1e-5:best=score;pat=0;torch.save(model.state_dict(),ck)
            else:pat+=1
            heartbeat(hb,"CAND-P-125","TRAIN_PHYSICS_P125_EXACT",seed,epoch)
            if epoch+1>=a.min_epochs and pat>=a.early_stop_patience:break
        model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval();
        with torch.no_grad():tp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy();tp[:,0]=tp[:,0]*wystd+wymean
        mm=metrics(y[test],tp,label[test]);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":mm,"beats_baseline":mm["primary_composite"]>bt["primary_composite"]+1e-4});states[seed]={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
    comps=np.asarray([r["test"]["primary_composite"] for r in reports]);agg={"mean":float(comps.mean()),"variance":float(comps.var()),"std":float(comps.std()),"worst":float(comps.min())};aggpass=agg["mean"]>bt["primary_composite"]+1e-4;bs=int(max(reports,key=lambda r:r["validation_primary_composite"])["seed"]);model=MLP().to(device);model.load_state_dict(states[bs]);model.eval();
    with torch.no_grad():fp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy();fp[:,0]=fp[:,0]*wystd+wymean
    fpm=metrics(y[test],fp,label[test]);q,s=quantize_state(states[bs]);buf=io.BytesIO();np.savez_compressed(buf,**q,**{f"scale::{k}":v for k,v in s.items()});payload=buf.getvalue();
    if len(payload)>96*1024:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch,q,s));
    with torch.no_grad():qp=model(torch.from_numpy(x[test]).to(device)).cpu().numpy();qp[:,0]=qp[:,0]*wystd+wymean
    qm=metrics(y[test],qp,label[test]);qd=fpm["primary_composite"]-qm["primary_composite"];qpass=qm["primary_composite"]>bt["primary_composite"]+1e-4 and qd<=.03;passed=aggpass and qpass;gold=out/"golden_vectors.npz";np.savez_compressed(gold,x=x[test[:64]],y=y[test[:64]],risk_label=label[test[:64]],fp32=fp[:64],quantized=qp[:64]);model.load_state_dict(states[bs]);onnx_path=out/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["stack_thermal_cycle_features"],output_names=["normalized_warpage_and_risk_probability"],dynamic_axes={"stack_thermal_cycle_features":{0:"batch"},"normalized_warpage_and_risk_probability":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));schema={"task_kind":"multitask_regression_classification","shape":[None,2],"semantics":["warpage_um","delamination_risk_probability"],"postprocess":"warpage_multiply_train_std_add_mean;probability_identity","authority":0,"public_claim_scope":"SIM_ONLY"};release=hashlib.sha256(canonical_bytes({"candidate":"CAND-P-125","dataset":m["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(gold),"mean":agg["mean"]})).hexdigest();package=build_package(out,"CAND-P-125",payload,sha256_file(gold),release,hashlib.sha256(canonical_bytes(schema)).hexdigest());status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE_SIM_ONLY";audit={"schema":"cimc.forge200.physics-p125-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-P-125","truth_class":"PHYSICS_SIM_FEA","public_claim_scope":"SIM_ONLY","baseline":{"kind":m["baseline_execution"],"validation":bv,"test":bt},"seed_reports":reports,"aggregate":agg,"g3_aggregate_mean_gate":aggpass,"quantized_best_seed":{"seed":bs,"test":qm,"metric_delta":qd,"gate":qpass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False};write_json(out/"contract_exact_audit.json",audit);write_json(out/"source_manifest.json",m);write_json(out/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist(),"warpage_mean":wymean,"warpage_std":wystd});write_json(out/"output_schema.json",schema);write_json(out/"quantization_parity.json",{"primary_composite_delta":qd,"gate":qpass});(out/"model_card.md").write_text(f"# CAND-P-125 SIM_ONLY FEA\n\n- Status `{status}`; no experimental claim.\n- Mean `{agg['mean']:.6f}` vs laminated-beam baseline `{bt['primary_composite']:.6f}`.\n- Parameters `{params}`; W8 `{len(payload)}` bytes; authority `0`; board pending.\n",encoding="utf-8");promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-P-125","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"PHYSICS_SIM_FEA","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(gold),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}};write_json(out/"promotion_receipt.json",promotion);manifest(out);heartbeat(hb,"CAND-P-125","COMPLETE");write_json(root/"evidence"/"physics_p125_exact_closure.v1.json",{"schema":"cimc.forge200.physics-p125-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":"CAND-P-125","status":status,"mean_composite":agg["mean"],"baseline_composite":bt["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion
def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--artifact-root",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=256);p.add_argument("--max-epochs",type=int,default=100);p.add_argument("--min-epochs",type=int,default=30);p.add_argument("--early-stop-patience",type=int,default=14);p.add_argument("--learning-rate",type=float,default=8e-4);a=p.parse_args();r=run(a);return 0 if r["host_contract_pass"] else 2
if __name__=="__main__":raise SystemExit(main())
