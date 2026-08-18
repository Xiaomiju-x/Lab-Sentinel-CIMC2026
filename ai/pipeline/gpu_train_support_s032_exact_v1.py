#!/usr/bin/env python3
"""Train/package source-gate-exact S032 query reformulator."""
from __future__ import annotations
import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state
def normalize(value):return value/np.maximum(np.linalg.norm(value,axis=1,keepdims=True),1e-8)
def metrics(query,target,group,passage,passage_group):
 score=normalize(query)@normalize(passage).T;recall=[];mrr=[];bad=[];entity=[]
 for index in range(len(group)):
  relevant=np.flatnonzero(passage_group==group[index]);order=np.argsort(-score[index])[:20];hits=np.intersect1d(order,relevant);recall.append(len(hits)/max(len(relevant),1));ranks=np.flatnonzero(np.isin(order,relevant));mrr.append(1./(1+int(ranks[0])) if len(ranks) else 0.);bad.append(float(not len(hits)));target_terms=set(np.flatnonzero(target[index]>0));pred_terms=set(np.argsort(-query[index])[:min(8,max(len(target_terms),1))]);tp=len(target_terms&pred_terms);entity.append(2*tp/max(len(target_terms)+len(pred_terms),1))
 r={"recall_at_20":float(np.mean(recall)),"MRR_at_20":float(np.mean(mrr)),"entity_preservation_F1":float(np.mean(entity)),"bad_rewrite_rate":float(np.mean(bad))};r["primary_composite"]=float(np.mean([r["recall_at_20"],r["entity_preservation_F1"],1.-r["bad_rewrite_rate"]]));return r
def run(a):
 import onnx,torch
 from torch import nn
 from torch.utils.data import DataLoader,TensorDataset
 root=a.root.resolve();d=root/"data"/"staged_support_s032_exact_v1"/"CAND-S-032.npz";m=json.loads(d.with_suffix(".metadata.json").read_text(encoding="utf-8"))
 if m["status"]!="PASS" or m["source_gate_match"]!="LICENSED_QUERY_RELEVANCE_PAIRS_WITH_SESSION_LEVEL_SPLIT" or m["cross_split_document_family_overlap"] or sha256_file(d)!=m["sha256"]:raise RuntimeError("DATA_GATE")
 if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
 device=torch.device(a.device);props=torch.cuda.get_device_properties(device);raw=np.load(d,allow_pickle=False);tx=raw["train_x"].astype(np.float32);ty=raw["train_target"].astype(np.float32);vx=raw["validation_x"].astype(np.float32);vy=raw["validation_target"].astype(np.float32);xx=raw["test_x"].astype(np.float32);xy=raw["test_target"].astype(np.float32);mean=tx.mean(0);std=tx.std(0);std[std<1e-7]=1.;tx=np.clip((tx-mean)/std,-12,12);vx=np.clip((vx-mean)/std,-12,12);xx=np.clip((xx-mean)/std,-12,12);bv=metrics(raw["validation_raw_query"],vy,raw["validation_group"],raw["validation_passage"],raw["validation_passage_group"]);bt=metrics(raw["test_raw_query"],xy,raw["test_group"],raw["test_passage"],raw["test_passage_group"]);out=a.artifact_root.resolve()/"CAND-S-032";out.mkdir(parents=True,exist_ok=True);hb=out/"heartbeat.json";started=time.perf_counter()
 class M(nn.Module):
  def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(tx.shape[1],96),nn.GELU(),nn.Linear(96,256))
  def forward(self,v):return self.net(v)
 params=sum(p.numel() for p in M().parameters())
 if params>96000:raise RuntimeError(f"PARAMETER_CAP:{params}")
 data=TensorDataset(torch.from_numpy(tx.astype(np.float32)),torch.from_numpy(ty));reports=[];states={}
 for seed in SEEDS:
  torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=M().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=4e-4);loader=DataLoader(data,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));ck=out/f"train_seed_{seed}"/"best.pt";ck.parent.mkdir(parents=True,exist_ok=True);best=-1e30;pat=0
  for epoch in range(a.max_epochs):
   model.train()
   for bx,by in loader:opt.zero_grad(set_to_none=True);pred=nn.functional.normalize(model(bx.to(device)),dim=1);loss=1.-torch.mean(torch.sum(pred*by.to(device),dim=1));loss.backward();opt.step()
   model.eval()
   with torch.no_grad():vp=model(torch.from_numpy(vx.astype(np.float32)).to(device)).cpu().numpy()
   score=metrics(vp,vy,raw["validation_group"],raw["validation_passage"],raw["validation_passage_group"])["primary_composite"]
   if score>best+1e-6:best=score;pat=0;torch.save(model.state_dict(),ck)
   else:pat+=1
   heartbeat(hb,"CAND-S-032","TRAIN_SUPPORT_EXACT",seed,epoch)
   if epoch+1>=a.min_epochs and pat>=a.early_stop_patience:break
  model.load_state_dict(torch.load(ck,map_location=device,weights_only=True));model.eval()
  with torch.no_grad():tp=model(torch.from_numpy(xx.astype(np.float32)).to(device)).cpu().numpy()
  mm=metrics(tp,xy,raw["test_group"],raw["test_passage"],raw["test_passage_group"]);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":mm,"beats_baseline":mm["primary_composite"]>bt["primary_composite"]+1e-4});states[seed]={n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
 scores=np.asarray([r["test"]["primary_composite"] for r in reports]);agg={"mean":float(scores.mean()),"variance":float(scores.var()),"std":float(scores.std()),"worst":float(scores.min())};aggpass=agg["mean"]>bt["primary_composite"]+1e-4;bs=int(max(reports,key=lambda r:r["validation_primary_composite"])["seed"]);model=M().to(device);model.load_state_dict(states[bs]);model.eval()
 with torch.no_grad():fp=model(torch.from_numpy(xx.astype(np.float32)).to(device)).cpu().numpy()
 fm=metrics(fp,xy,raw["test_group"],raw["test_passage"],raw["test_passage_group"]);q,s=quantize_state(states[bs]);buf=io.BytesIO();np.savez_compressed(buf,**q,**{f"scale::{k}":v for k,v in s.items()});payload=buf.getvalue()
 if len(payload)>96*1024:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
 model.load_state_dict(dequantized_state(torch,q,s))
 with torch.no_grad():qp=model(torch.from_numpy(xx.astype(np.float32)).to(device)).cpu().numpy()
 qm=metrics(qp,xy,raw["test_group"],raw["test_passage"],raw["test_passage_group"]);delta=fm["primary_composite"]-qm["primary_composite"];qpass=qm["primary_composite"]>bt["primary_composite"]+1e-4 and delta<=.03;passed=aggpass and qpass;gold=out/"golden_vectors.npz";np.savez_compressed(gold,x=xx[:64],target=xy[:64],fp32=fp[:64],quantized=qp[:64]);model.load_state_dict(states[bs]);onnx_path=out/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(xx[:1].astype(np.float32)).to(device),onnx_path,input_names=["raw_query_context_domain_entity_features"],output_names=["retrieval_query_term_weights"],dynamic_axes={"raw_query_context_domain_entity_features":{0:"batch"},"retrieval_query_term_weights":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));schema={"task_kind":"query_reformulation","shape":[None,256],"semantics":"train_vocabulary_retrieval_query_term_weights","authority":0};release=hashlib.sha256(canonical_bytes({"candidate":"CAND-S-032","dataset":m["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(gold),"mean":agg["mean"]})).hexdigest();package=build_package(out,"CAND-S-032",payload,sha256_file(gold),release,hashlib.sha256(canonical_bytes(schema)).hexdigest());status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE";audit={"schema":"cimc.forge200.support-s032-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-S-032","truth_class":"STRUCTURE_DERIVED","claim_state":m["claim_state"],"baseline":{"kind":m["baseline_execution"],"validation":bv,"test":bt},"seed_reports":reports,"aggregate":agg,"g3_aggregate_mean_gate":aggpass,"quantized_best_seed":{"seed":bs,"test":qm,"metric_delta":delta,"gate":qpass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False};write_json(out/"contract_exact_audit.json",audit);write_json(out/"source_manifest.json",m);write_json(out/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist()});write_json(out/"output_schema.json",schema);write_json(out/"quantization_parity.json",{"primary_composite_delta":delta,"gate":qpass});(out/"model_card.md").write_text(f"# CAND-S-032\n\n- Status `{status}`. Same-document relevance is structure-derived, not expert judgment.\n- Three-seed mean `{agg['mean']:.6f}` vs raw query `{bt['primary_composite']:.6f}`.\n- Authority `0`; board pending.\n",encoding="utf-8");promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-S-032","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"STRUCTURE_DERIVED","claim_state":m["claim_state"],"three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(gold),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}};write_json(out/"promotion_receipt.json",promotion);manifest(out);heartbeat(hb,"CAND-S-032","COMPLETE");write_json(root/"evidence"/"support_s032_exact_closure.v1.json",{"schema":"cimc.forge200.support-s032-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":"CAND-S-032","status":status,"mean_composite":agg["mean"],"baseline_composite":bt["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument("--artifact-root",type=Path,required=True);p.add_argument("--device",default="cuda:0");p.add_argument("--batch-size",type=int,default=256);p.add_argument("--max-epochs",type=int,default=120);p.add_argument("--min-epochs",type=int,default=30);p.add_argument("--early-stop-patience",type=int,default=16);p.add_argument("--learning-rate",type=float,default=8e-4);a=p.parse_args();r=run(a);return 0 if r["host_contract_pass"] else 2
if __name__=="__main__":raise SystemExit(main())
