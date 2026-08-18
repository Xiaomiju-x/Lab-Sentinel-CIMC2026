#!/usr/bin/env python3
"""Train/package source-gate-exact S042 evidence freshness classifier."""
from __future__ import annotations

import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state

def metrics(y,p):
    classes=4;f1=[]
    for label in range(classes):
        tp=np.sum((y==label)&(p==label));fp=np.sum((y!=label)&(p==label));fn=np.sum((y==label)&(p!=label));f1.append(float(2*tp/max(2*tp+fp+fn,1)))
    stale_recall=float(np.sum((y==1)&(p==1))/max(np.sum(y==1),1));false_stale=float(np.sum((y!=1)&(p==1))/max(np.sum(y!=1),1));result={"macro_F1":float(np.mean(f1)),"stale_recall":stale_recall,"false_stale_rate":false_stale,"per_class_F1":f1};result["primary_composite"]=float(np.mean([result["macro_F1"],stale_recall,1.-false_stale]));return result

def run(args):
    import onnx,torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    root=args.root.resolve();dataset=root/"data"/"staged_support_s042_exact_v1"/"CAND-S-042.npz";meta=json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta["status"]!="PASS" or meta["source_gate_match"]!="TEAM_LEDGER_PLUS_LICENSED_REVISION_CASES_WITH_SOURCE_SPLIT" or meta["cross_split_source_family_overlap"] or sha256_file(dataset)!=meta["sha256"]:raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
    device=torch.device(args.device);props=torch.cuda.get_device_properties(device);raw=np.load(dataset,allow_pickle=False);xr=raw["x"].astype(np.float32);y=raw["y"].astype(np.int64);base=raw["baseline_pred"].astype(np.int64);split=raw["split"].astype(np.int8);train,val,test=(np.flatnonzero(split==code) for code in range(3));mean=xr[train].mean(0);std=xr[train].std(0);std[std<1e-7]=1.;x=np.clip((xr-mean)/std,-12,12).astype(np.float32);baseline_validation=metrics(y[val],base[val]);baseline_test=metrics(y[test],base[test]);output=args.artifact_root.resolve()/"CAND-S-042";output.mkdir(parents=True,exist_ok=True);hb=output/"heartbeat.json";started=time.perf_counter()
    class Model(nn.Module):
        def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(x.shape[1],64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,4))
        def forward(self,value):return self.net(value)
    params=sum(value.numel() for value in Model().parameters())
    if params>48000:raise RuntimeError(f"PARAMETER_CAP:{params}")
    data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(y[train]));reports=[];states={}
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Model().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=4e-4);loader=DataLoader(data,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));checkpoint=output/f"train_seed_{seed}"/"best.pt";checkpoint.parent.mkdir(parents=True,exist_ok=True);best=-1e30;patience=0
        for epoch in range(args.max_epochs):
            model.train()
            for bx,by in loader:optimizer.zero_grad(set_to_none=True);logits=model(bx.to(device));loss=nn.functional.cross_entropy(logits,by.to(device));loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():prediction=model(torch.from_numpy(x[val]).to(device)).argmax(1).cpu().numpy()
            score=metrics(y[val],prediction)["primary_composite"]
            if score>best+1e-6:best=score;patience=0;torch.save(model.state_dict(),checkpoint)
            else:patience+=1
            heartbeat(hb,"CAND-S-042","TRAIN_SUPPORT_EXACT",seed,epoch)
            if epoch+1>=args.min_epochs and patience>=args.early_stop_patience:break
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
        with torch.no_grad():prediction=model(torch.from_numpy(x[test]).to(device)).argmax(1).cpu().numpy()
        report=metrics(y[test],prediction);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":report,"beats_baseline":report["primary_composite"]>baseline_test["primary_composite"]+1e-4});states[seed]={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    scores=np.asarray([record["test"]["primary_composite"] for record in reports]);aggregate={"mean":float(scores.mean()),"variance":float(scores.var()),"std":float(scores.std()),"worst":float(scores.min())};aggregate_pass=aggregate["mean"]>baseline_test["primary_composite"]+1e-4;best_seed=int(max(reports,key=lambda record:record["validation_primary_composite"])["seed"]);model=Model().to(device);model.load_state_dict(states[best_seed]);model.eval()
    with torch.no_grad():fp32_logits=model(torch.from_numpy(x[test]).to(device)).cpu().numpy();fp32_prediction=fp32_logits.argmax(1)
    fp32_metric=metrics(y[test],fp32_prediction);quantized,scales=quantize_state(states[best_seed]);buffer=io.BytesIO();np.savez_compressed(buffer,**quantized,**{f"scale::{name}":value for name,value in scales.items()});payload=buffer.getvalue()
    if len(payload)>48*1024:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch,quantized,scales))
    with torch.no_grad():quant_logits=model(torch.from_numpy(x[test]).to(device)).cpu().numpy();quant_prediction=quant_logits.argmax(1)
    quant_metric=metrics(y[test],quant_prediction);delta=fp32_metric["primary_composite"]-quant_metric["primary_composite"];quant_pass=quant_metric["primary_composite"]>baseline_test["primary_composite"]+1e-4 and delta<=.03;passed=aggregate_pass and quant_pass;golden=output/"golden_vectors.npz";np.savez_compressed(golden,x=x[test[:128]],y=y[test[:128]],fp32=fp32_logits[:128],quantized=quant_logits[:128]);model.load_state_dict(states[best_seed]);onnx_path=output/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["evidence_revision_temporal_features"],output_names=["freshness_class_logits"],dynamic_axes={"evidence_revision_temporal_features":{0:"batch"},"freshness_class_logits":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path));schema={"task_kind":"classification","shape":[None,4],"semantics":["fresh","stale","superseded","time_irrelevant"],"postprocess":"argmax_or_softmax","authority":0};release=hashlib.sha256(canonical_bytes({"candidate":"CAND-S-042","dataset":meta["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(golden),"mean":aggregate["mean"]})).hexdigest();package=build_package(output,"CAND-S-042",payload,sha256_file(golden),release,hashlib.sha256(canonical_bytes(schema)).hexdigest());status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE";audit={"schema":"cimc.forge200.support-s042-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-S-042","truth_class":"STRUCTURE_DERIVED","baseline":{"kind":meta["baseline_execution"],"validation":baseline_validation,"test":baseline_test},"seed_reports":reports,"aggregate":aggregate,"g3_aggregate_mean_gate":aggregate_pass,"quantized_best_seed":{"seed":best_seed,"test":quant_metric,"metric_delta":delta,"gate":quant_pass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False}
    write_json(output/"contract_exact_audit.json",audit);write_json(output/"source_manifest.json",meta);write_json(output/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist()});write_json(output/"output_schema.json",schema);write_json(output/"quantization_parity.json",{"primary_composite_delta":delta,"gate":quant_pass});(output/"model_card.md").write_text(f"# CAND-S-042\n\n- Status `{status}`. Labels derive from frozen revision order, timestamps, and query scope.\n- Three-seed mean `{aggregate['mean']:.6f}` vs fixed-age baseline `{baseline_test['primary_composite']:.6f}`.\n- Authority `0`; board pending.\n",encoding="utf-8");promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-S-042","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"STRUCTURE_DERIVED","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(golden),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}};write_json(output/"promotion_receipt.json",promotion);manifest(output);heartbeat(hb,"CAND-S-042","COMPLETE");write_json(root/"evidence"/"support_s042_exact_closure.v1.json",{"schema":"cimc.forge200.support-s042-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":"CAND-S-042","status":status,"mean_composite":aggregate["mean"],"baseline_composite":baseline_test["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument("--artifact-root",type=Path,required=True);parser.add_argument("--device",default="cuda:0");parser.add_argument("--batch-size",type=int,default=256);parser.add_argument("--max-epochs",type=int,default=100);parser.add_argument("--min-epochs",type=int,default=25);parser.add_argument("--early-stop-patience",type=int,default=14);parser.add_argument("--learning-rate",type=float,default=8e-4);args=parser.parse_args();result=run(args);return 0 if result["host_contract_pass"] else 2

if __name__=="__main__":raise SystemExit(main())
