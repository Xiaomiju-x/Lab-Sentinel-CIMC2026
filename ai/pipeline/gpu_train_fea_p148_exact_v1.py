#!/usr/bin/env python3
"""Train, quantize, export, and package source-gate-exact P148 FEA model."""
from __future__ import annotations

import argparse,hashlib,io,json,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from gpu_train_job import SEEDS,build_package,canonical_bytes,heartbeat,sha256_file,write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state,manifest,quantize_state

def concordance(y:np.ndarray,p:np.ndarray)->float:
    dy=y[:,None]-y[None,:];dp=p[:,None]-p[None,:];mask=np.triu(np.abs(dy)>1e-8,1)
    return float(np.mean((dy[mask]*dp[mask])>0)) if np.any(mask) else .5

def metrics(y:np.ndarray,p:np.ndarray)->dict:
    damage_mae=float(np.mean(np.abs(p[:,0]-y[:,0])));life_mape=float(np.mean(np.abs(p[:,1]-y[:,1])/np.maximum(y[:,1],35.))*100.);ci=concordance(y[:,1],p[:,1])
    result={"damage_MAE":damage_mae,"RUL_MAPE_percent":life_mape,"concordance_index":ci,"damage_score":1./(1.+5.*damage_mae),"RUL_score":1./(1.+life_mape/100.),"concordance_score":ci}
    result["primary_composite"]=float(np.mean([result["damage_score"],result["RUL_score"],result["concordance_score"]]));return result

def run(args):
    import onnx,torch
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    root=args.root.resolve();dataset=root/"data"/"staged_fea_p148_exact_v1"/"CAND-P-148.npz";meta=json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta["status"]!="PASS" or meta["truth_class"]!="PHYSICS_SIM" or meta["source_gate_match"]!="FEA_DATA" or meta["cross_split_family_overlap"] or sha256_file(dataset)!=meta["sha256"]:raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUIRED")
    device=torch.device(args.device);props=torch.cuda.get_device_properties(device);raw=np.load(dataset,allow_pickle=False);xr=raw["x"].astype(np.float32);y=raw["y"].astype(np.float32);base=raw["baseline_pred"].astype(np.float32);split=raw["split"].astype(np.int8);train,val,test=(np.flatnonzero(split==code) for code in range(3))
    mean,std=xr[train].mean(0),xr[train].std(0);std[std<1e-7]=1.;x=np.clip((xr-mean)/std,-12,12).astype(np.float32);target=np.column_stack((y[:,0],np.log1p(y[:,1]))).astype(np.float32);tmean=target[train].mean(0);tstd=target[train].std(0);tstd[tstd<1e-7]=1.;target_norm=((target-tmean)/tstd).astype(np.float32)
    output=args.artifact_root.resolve()/"CAND-P-148";output.mkdir(parents=True,exist_ok=True);hb=output/"heartbeat.json";started=time.perf_counter()
    class Model(nn.Module):
        def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(x.shape[1],112),nn.GELU(),nn.Linear(112,64),nn.GELU(),nn.Linear(64,2))
        def forward(self,value):return self.net(value)
    params=sum(value.numel() for value in Model().parameters())
    if params>96000:raise RuntimeError(f"PARAMETER_CAP:{params}")
    def decode(value):
        raw_target=value*tstd+tmean
        return np.column_stack((np.maximum(raw_target[:,0],0.),np.maximum(np.expm1(raw_target[:,1]),35.))).astype(np.float32)
    baseline_validation=metrics(y[val],base[val]);baseline_test=metrics(y[test],base[test]);reports=[];states={};data=TensorDataset(torch.from_numpy(x[train]),torch.from_numpy(target_norm[train]))
    for seed in SEEDS:
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);model=Model().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=3e-4);loader=DataLoader(data,batch_size=args.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed));checkpoint=output/f"train_seed_{seed}"/"best.pt";checkpoint.parent.mkdir(parents=True,exist_ok=True);best=-1e30;patience=0
        for epoch in range(args.max_epochs):
            model.train()
            for bx,by in loader:
                optimizer.zero_grad(set_to_none=True);prediction=model(bx.to(device));loss=nn.functional.smooth_l1_loss(prediction,by.to(device),beta=.15);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            model.eval()
            with torch.no_grad():validation=decode(model(torch.from_numpy(x[val]).to(device)).cpu().numpy())
            score=metrics(y[val],validation)["primary_composite"]
            if score>best+1e-5:best=score;patience=0;torch.save(model.state_dict(),checkpoint)
            else:patience+=1
            heartbeat(hb,"CAND-P-148","TRAIN_FEA_EXACT",seed,epoch)
            if epoch+1>=args.min_epochs and patience>=args.early_stop_patience:break
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
        with torch.no_grad():test_prediction=decode(model(torch.from_numpy(x[test]).to(device)).cpu().numpy())
        report=metrics(y[test],test_prediction);reports.append({"seed":seed,"epochs":epoch+1,"validation_primary_composite":best,"test":report,"beats_baseline":report["primary_composite"]>baseline_test["primary_composite"]+1e-4});states[seed]={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
    scores=np.asarray([record["test"]["primary_composite"] for record in reports]);aggregate={"mean":float(scores.mean()),"variance":float(scores.var()),"std":float(scores.std()),"worst":float(scores.min())};aggregate_pass=aggregate["mean"]>baseline_test["primary_composite"]+1e-4;best_seed=int(max(reports,key=lambda record:record["validation_primary_composite"])["seed"]);model=Model().to(device);model.load_state_dict(states[best_seed]);model.eval()
    with torch.no_grad():fp32=decode(model(torch.from_numpy(x[test]).to(device)).cpu().numpy())
    fp32_metrics=metrics(y[test],fp32);quantized,scales=quantize_state(states[best_seed]);buffer=io.BytesIO();np.savez_compressed(buffer,**quantized,**{f"scale::{name}":value for name,value in scales.items()});payload=buffer.getvalue()
    if len(payload)>96*1024:raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch,quantized,scales))
    with torch.no_grad():quant_prediction=decode(model(torch.from_numpy(x[test]).to(device)).cpu().numpy())
    quant_metrics=metrics(y[test],quant_prediction);quant_delta=fp32_metrics["primary_composite"]-quant_metrics["primary_composite"];quant_pass=quant_metrics["primary_composite"]>baseline_test["primary_composite"]+1e-4 and quant_delta<=.03;passed=aggregate_pass and quant_pass
    golden=output/"golden_vectors.npz";np.savez_compressed(golden,x=x[test[:64]],y=y[test[:64]],fp32=fp32[:64],quantized=quant_prediction[:64]);model.load_state_dict(states[best_seed]);onnx_path=output/"fp32.onnx";torch.onnx.export(model,torch.from_numpy(x[test[:1]]).to(device),onnx_path,input_names=["thermal_cycle_package_and_damage_history"],output_names=["normalized_damage_and_log_life"],dynamic_axes={"thermal_cycle_package_and_damage_history":{0:"batch"},"normalized_damage_and_log_life":{0:"batch"}},opset_version=17,dynamo=False);onnx.checker.check_model(onnx.load(onnx_path))
    schema={"task_kind":"multitask_regression","shape":[None,2],"semantics":["package_fatigue_damage_fraction","cycles_to_failure"],"postprocess":"multiply_train_target_std_add_mean;life_expm1;nonnegative_clamp","authority":0,"public_claim_scope":"SIM_ONLY_FEA"};release=hashlib.sha256(canonical_bytes({"candidate":"CAND-P-148","dataset":meta["sha256"],"onnx":sha256_file(onnx_path),"golden":sha256_file(golden),"mean":aggregate["mean"]})).hexdigest();package=build_package(output,"CAND-P-148",payload,sha256_file(golden),release,hashlib.sha256(canonical_bytes(schema)).hexdigest())
    status="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE";audit={"schema":"cimc.forge200.fea-p148-contract-exact-audit.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"candidate_id":"CAND-P-148","truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY_FEA","source_gate_match":"FEA_DATA","baseline":{"kind":meta["baseline_execution"],"validation":baseline_validation,"test":baseline_test},"seed_reports":reports,"aggregate":aggregate,"g3_aggregate_mean_gate":aggregate_pass,"quantized_best_seed":{"seed":best_seed,"test":quant_metrics,"metric_delta":quant_delta,"gate":quant_pass},"parameter_count":params,"w8_payload_bytes":len(payload),"authority":0,"board_accepted":False,"countable_model":False}
    write_json(output/"contract_exact_audit.json",audit);write_json(output/"source_manifest.json",meta);write_json(output/"preprocessing_train_only.json",{"mean":mean.tolist(),"std":std.tolist(),"target_mean":tmean.tolist(),"target_std":tstd.tolist()});write_json(output/"output_schema.json",schema);write_json(output/"quantization_parity.json",{"primary_composite_delta":quant_delta,"gate":quant_pass});(output/"model_card.md").write_text(f"# CAND-P-148 FEA SIM_ONLY\n\n- Status `{status}`; no experimental lifetime claim.\n- Three-seed mean `{aggregate['mean']:.6f}` vs Miner+Coffin-Manson baseline `{baseline_test['primary_composite']:.6f}`.\n- Parameters `{params}`; W8 `{len(payload)}` bytes; authority `0`; board pending.\n",encoding="utf-8")
    promotion={"schema":"cimc.forge200.promotion-receipt.v3","status":status,"candidate_id":"CAND-P-148","authority":0,"board_accepted":False,"countable_model":False,"host_contract_pass":passed,"truth_class":"PHYSICS_SIM","public_claim_scope":"SIM_ONLY_FEA","source_gate_match":"FEA_DATA","three_seed_count":3,"release_root":release,"package":package,"onnx_sha256":sha256_file(onnx_path),"golden_sha256":sha256_file(golden),"runtime_seconds":time.perf_counter()-started,"gpu":{"name":props.name,"vram_gib":props.total_memory/1024**3}}
    write_json(output/"promotion_receipt.json",promotion);manifest(output);heartbeat(hb,"CAND-P-148","COMPLETE");write_json(root/"evidence"/"fea_p148_exact_closure.v1.json",{"schema":"cimc.forge200.fea-p148-exact-closure.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS" if passed else "PARTIAL","record":audit,"authority_nonzero":0,"board_actions":0});print(json.dumps({"candidate_id":"CAND-P-148","status":status,"mean_composite":aggregate["mean"],"baseline_composite":baseline_test["primary_composite"],"runtime_seconds":promotion["runtime_seconds"]},sort_keys=True));return promotion

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument("--artifact-root",type=Path,required=True);parser.add_argument("--device",default="cuda:0");parser.add_argument("--batch-size",type=int,default=256);parser.add_argument("--max-epochs",type=int,default=120);parser.add_argument("--min-epochs",type=int,default=35);parser.add_argument("--early-stop-patience",type=int,default=16);parser.add_argument("--learning-rate",type=float,default=8e-4);args=parser.parse_args();result=run(args);return 0 if result["host_contract_pass"] else 2

if __name__=="__main__":raise SystemExit(main())
