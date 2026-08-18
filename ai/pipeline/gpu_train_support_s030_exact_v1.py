#!/usr/bin/env python3
"""Train/package source-gate-exact S030 multidomain reranker."""
from __future__ import annotations

import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state

def grouped(grade,query,domain,score):
    ndcgs=[];mrrs=[];by_domain={value:[] for value in range(6)}
    for qid in np.unique(query):
        mask=query==qid;g=grade[mask];s=score[mask];order=np.argsort(-s)[:10];ideal=np.argsort(-g)[:10];discount=1./np.log2(np.arange(2,12));dcg=float(np.sum((2.**g[order]-1.)*discount[:len(order)]));idcg=float(np.sum((2.**g[ideal]-1.)*discount[:len(ideal)]));ndcg=dcg/max(idcg,1e-12);hits=np.flatnonzero(g[order]==2);mrr=1./(1+int(hits[0])) if len(hits) else 0.;ndcgs.append(ndcg);mrrs.append(mrr);by_domain[int(domain[mask][0])].append(ndcg)
    worst=min(float(np.mean(value)) for value in by_domain.values() if value);result={"NDCG_at_10":float(np.mean(ndcgs)),"MRR":float(np.mean(mrrs)),"worst_domain_NDCG":worst,"queries":len(ndcgs)};result["primary_composite"]=float(np.mean([result["NDCG_at_10"],result["MRR"],worst]));return result

def run(args):
    import onnx,torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    root=args.root.resolve();dataset=root/"data"/"staged_support_s030_exact_v1"/"CAND-S-030.npz";meta=json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta["status"]!="PASS" or meta["source_gate_match"]!="LICENSED_RAG_JUDGMENTS_WITH_DOCUMENT_FAMILY_SPLIT" or meta["cross_split_document_family_overlap"] or sha256_file(dataset)!=meta["sha256"]:raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
    device=torch.device(args.device);props=torch.cuda.get_device_properties(device);data=np.load(dataset,allow_pickle=False);train_x=data["train_x"].astype(np.float32);mean=train_x.mean(0);std=train_x.std(0);std[std<1e-7]=1.
    def split(name):return np.clip((data[f"{name}_x"].astype(np.float32)-mean)/std,-12,12),data[f"{name}_grade"].astype(np.int64),data[f"{name}_query"],data[f"{name}_domain"],data[f"{name}_bm25"].astype(np.float32)
    tx,ty,tq,td,tb=split("train");vx,vy,vq,vd,vb=split("validation");xx,xy,xq,xd,xb=split("test");baseline_validation=grouped(vy,vq,vd,vb);baseline_test=grouped(xy,xq,xd,xb);output=args.artifact_root.resolve()/"CAND-S-030";output.mkdir(parents=True,exist_ok=True);hb=output/"heartbeat.json";started=time.perf_counter()
    class Model(nn.Module):
        def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(tx.shape[1],64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        def forward(self,value):return self.net(value)
    params=sum(value.numel() for value in Model().parameters())
    if params>192000:raise RuntimeError(f"PARAMETER_CAP:{params}")
    train_data=TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty.astype(np.float32)[:,None]));reports=[];states={}
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Model().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=4e-4);loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));checkpoint=output/f"train_seed_{seed}"/"best.pt";checkpoint.parent.mkdir(parents=True,exist_ok=True);best=-1e30;patience=0
        for epoch in range(args.max_epochs):
            model.train()
            for bx,by in loader:optimizer.zero_grad(set_to_none=True);prediction=model(bx.to(device));loss=nn.functional.smooth_l1_loss(prediction,by.to(device),beta=.2);loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():validation=model(torch.from_numpy(vx).to(device)).cpu().numpy()[:,0]
            score=grouped(vy,vq,vd,validation)["primary_composite"]
            if score>best+1e-6:best=score;patience=0;torch.save(model.state_dict(),checkpoint)
            else:patience+=1
            heartbeat(hb,"CAND-S-030","TRAIN_SUPPORT_EXACT",seed,epoch)
            if epoch+1>=args.min_epochs and patience>=args.early_stop_patience:break
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
        with torch.no_grad():test_score=model(torch.from_numpy(xx).to(device)).cpu().numpy()[:,0]
        report=grouped(xy,xq,xd,test_score);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":report,"beats_baseline":report["primary_composite"]>baseline_test["primary_composite"]+1e-4});states[seed]={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    values=np.asarray([record["test"]["primary_composite"] for record in reports]);aggregate={"mean":float(values.mean()),"variance":float(values.var()),"std":float(values.std()),"worst":float(values.min())};aggregate_pass=aggregate["mean"]>baseline_test["primary_composite"]+1e-4;best_seed=int(max(reports,key=lambda record:record["validation_primary_composite"])["seed"]);model=Model().to(device);model.load_state_dict(states[best_seed]);model.eval()
    with torch.no_grad():fp32=model(torch.from_numpy(xx).to(device)).cpu().numpy()[:,0]
    fp32_metric=grouped(xy,xq,xd,fp32);quantized,scales=quantize_state(states[best_seed]);buffer=io.BytesIO();np.savez_compressed(buffer,**quantized,**{f"scale::{name}":value for name,value in scales.items()});payload=buffer.getvalue()
    if len(payload)>192*1024:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch,quantized,scales))
    with torch.no_grad():quant_score=model(torch.from_numpy(xx).to(device)).cpu().numpy()[:,0]
    quant_metric=grouped(xy,xq,xd,quant_score);delta=fp32_metric["primary_composite"]-quant_metric["primary_composite"];quant_pass=quant_metric["primary_composite"]>baseline_test["primary_composite"]+1e-4 and delta<=.03;passed=aggregate_pass and quant_pass;golden=output/"golden_vectors.npz";np.savez_compressed(golden,x=xx[:128],grade=xy[:128],fp32=fp32[:128],quantized=quant_score[:128]);model.load_state_dict(states[best_seed]);onnx_path=output/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(xx[:1]).to(device),onnx_path,input_names=["query_chunk_domain_and_retrieval_features"],output_names=["relevance_grade_score"],dynamic_axes={"query_chunk_domain_and_retrieval_features":{0:"batch"},"relevance_grade_score":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));schema={"task_kind":"cross_domain_reranking","shape":[None,1],"semantics":"shared_cross_domain_chunk_relevance_grade_score","authority":0};release=hashlib.sha256(canonical_bytes({"candidate":"CAND-S-030","dataset":meta["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(golden),"mean":aggregate["mean"]})).hexdigest();package=build_package(output,"CAND-S-030",payload,sha256_file(golden),release,hashlib.sha256(canonical_bytes(schema)).hexdigest());status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE";audit={"schema":"cimc.forge200.support-s030-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-S-030","truth_class":"STRUCTURE_DERIVED","claim_state":meta["claim_state"],"baseline":{"kind":meta["baseline_execution"],"validation":baseline_validation,"test":baseline_test},"seed_reports":reports,"aggregate":aggregate,"g3_aggregate_mean_gate":aggregate_pass,"quantized_best_seed":{"seed":best_seed,"test":quant_metric,"metric_delta":delta,"gate":quant_pass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False}
    write_json(output/"contract_exact_audit.json",audit);write_json(output/"source_manifest.json",meta);write_json(output/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist()});write_json(output/"output_schema.json",schema);write_json(output/"quantization_parity.json",{"primary_composite_delta":delta,"gate":quant_pass});(output/"model_card.md").write_text(f"# CAND-S-030\n\n- Status `{status}`.\n- Labels are same-document-family structure-derived relevance, not expert judgments.\n- Three-seed mean `{aggregate['mean']:.6f}` vs BM25 `{baseline_test['primary_composite']:.6f}`.\n- Authority `0`; board pending.\n",encoding="utf-8");promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-S-030","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"STRUCTURE_DERIVED","claim_state":meta["claim_state"],"three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(golden),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}};write_json(output/"promotion_receipt.json",promotion);manifest(output);heartbeat(hb,"CAND-S-030","COMPLETE");write_json(root/"evidence"/"support_s030_exact_closure.v1.json",{"schema":"cimc.forge200.support-s030-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":"CAND-S-030","status":status,"mean_composite":aggregate["mean"],"baseline_composite":baseline_test["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument("--artifact-root",type=Path,required=True);parser.add_argument("--device",default="cuda:0");parser.add_argument("--batch-size",type=int,default=512);parser.add_argument("--max-epochs",type=int,default=100);parser.add_argument("--min-epochs",type=int,default=25);parser.add_argument("--early-stop-patience",type=int,default=14);parser.add_argument("--learning-rate",type=float,default=8e-4);args=parser.parse_args();result=run(args);return 0 if result["host_contract_pass"] else 2

if __name__=="__main__":raise SystemExit(main())
