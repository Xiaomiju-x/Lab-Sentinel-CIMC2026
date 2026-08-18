#!/usr/bin/env python3
"""Evaluate S033/S035/S036/S038/S044 against frozen baselines."""

from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
from typing import Any
import numpy as np
from evaluate_support_exact_v3 import average_precision, binary_auroc, canonical_bytes, ece, hard_probability, macro_f1, reciprocal_rank, sha256_file, write_json

CANDIDATES=("CAND-S-033","CAND-S-035","CAND-S-036","CAND-S-038","CAND-S-044");SEEDS=(20260801,20260802,20260803)

def brier(y:np.ndarray,p:np.ndarray)->float:return float(np.mean(np.sum((p-np.eye(p.shape[1])[y])**2,axis=1)))

def metrics(cid:str,y:np.ndarray,out:np.ndarray,evidence:Any,baseline:bool=False)->dict[str,float]:
    if baseline:
        p=out if out.ndim==2 else hard_probability(out.astype(int),int(np.max(y))+1)
    else:p=out
    pred=p.argmax(1)
    if cid=="CAND-S-033":
        ood=evidence["ood_label"].astype(bool);domains=evidence["domain_id"].astype(int);worst=max(ece(y[domains==d],p[domains==d]) for d in np.unique(domains))
        r={"ece":ece(y,p),"brier":brier(y,p),"abstention_auroc":binary_auroc(ood,p[:,6]),"worst_domain_ece":worst};r["composite"]=((1-r["ece"])+(1-min(r["brier"],1))+r["abstention_auroc"]+(1-r["worst_domain_ece"]))/4;return r
    if cid=="CAND-S-035":
        contra=y==1;r={"macro_f1":macro_f1(y,pred,3),"contradiction_recall":float(np.mean(pred[contra]==1)),"numeric_false_accept_rate":float(np.mean(pred[y!=0]==0))};r["composite"]=(r["macro_f1"]+r["contradiction_recall"]+(1-r["numeric_false_accept_rate"]))/3;return r
    if cid=="CAND-S-036":
        invalid=y==2;r={"macro_f1":macro_f1(y,pred,3),"invalid_recall":float(np.mean(pred[invalid]==2)),"false_reject_rate":float(np.mean(pred[y!=2]==2))};r["composite"]=(r["macro_f1"]+r["invalid_recall"]+(1-r["false_reject_rate"]))/3;return r
    if cid=="CAND-S-038":
        r={"brier":brier(y,p),"ece":ece(y,p),"tier_macro_f1":macro_f1(y,pred,3),"high_trust_false_accept_rate":float(np.mean(pred[y!=0]==0))};r["composite"]=((1-min(r["brier"],1))+(1-r["ece"])+r["tier_macro_f1"]+(1-r["high_trust_false_accept_rate"]))/4;return r
    if cid=="CAND-S-044":
        unresolved=y==7;tp=np.sum((pred==7)&unresolved);fp=np.sum((pred==7)&~unresolved);fn=np.sum((pred!=7)&unresolved);uf1=float(2*tp/max(2*tp+fp+fn,1));r={"link_accuracy":float(np.mean(pred==y)),"mrr":reciprocal_rank(y,p),"unresolved_f1":uf1};r["composite"]=(r["link_accuracy"]+r["mrr"]+uf1)/3;return r
    raise KeyError(cid)

def rebuild(output:Path)->str:
    rec=[{"path":str(p.relative_to(output)).replace("\\","/"),"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in sorted(x for x in output.rglob("*") if x.is_file() and x.name!="artifact_manifest.json")];h=hashlib.sha256(canonical_bytes(rec)).hexdigest();write_json(output/"artifact_manifest.json",{"schema":"cimc.forge200.artifact-manifest.v1","records":rec,"content_root_sha256":h});return h

def one(root:Path,artifacts:Path,cid:str)->dict[str,Any]:
    out=artifacts/cid;e=np.load(out/"three_seed_test_predictions.npz",allow_pickle=False);y=e["y"].astype(int);base_input=e["baseline_probability"] if "baseline_probability" in e else e["baseline_prediction"];base=metrics(cid,y,base_input,e,True);seeds=[{"seed":s,**metrics(cid,y,e[f"seed_{s}"],e)} for s in SEEDS];q=metrics(cid,y,e["quantized_best_seed"],e);vals=np.asarray([x["composite"] for x in seeds]);grouped=json.loads((out/"eval_grouped.json").read_text(encoding="utf-8"));best=next(x for x in seeds if x["seed"]==int(grouped["best_seed"]));delta=best["composite"]-q["composite"];mean_pass=float(vals.mean())>base["composite"]+1e-6;quant_pass=delta<=.02;status="PASS_CONTRACT_BASELINE_BOARD_PENDING" if mean_pass and quant_pass else "FAIL_CONTRACT_BASELINE"
    r={"schema":"cimc.forge200.support-spare-exact-evaluation.v1","status":status,"candidate_id":cid,"baseline":base,"seed_reports":seeds,"quantized_best_seed":q,"three_seed_mean_composite":float(vals.mean()),"three_seed_variance_composite":float(vals.var()),"three_seed_worst_composite":float(vals.min()),"aggregate_mean_beats_preregistered_baseline":mean_pass,"individual_seed_baseline_results_reported_not_release_gate":[bool(v>base["composite"]+1e-6) for v in vals],"quantized_best_seed_metric_delta":delta,"quantization_pass":quant_pass,"dataset_sha256":sha256_file(root/"data"/"staged_support_spares_exact_v1"/f"{cid}.npz"),"prediction_evidence_sha256":sha256_file(out/"three_seed_test_predictions.npz"),"authority":0,"board_accepted":False,"countable_model":False};write_json(out/"contract_exact_evaluation.v1.json",r);pp=out/"promotion_receipt.json";pr=json.loads(pp.read_text(encoding="utf-8"));pr.update({"status":"HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if status.startswith("PASS") else "HOST_GPU_REJECTED_CONTRACT_BASELINE","exact_contract_baseline_pending":not status.startswith("PASS"),"contract_evaluation_sha256":sha256_file(out/"contract_exact_evaluation.v1.json"),"authority":0,"board_accepted":False,"countable_model":False});write_json(pp,pr)
    with (out/"model_card.md").open("a",encoding="utf-8") as h:h.write(f"\n- Exact contract evaluation: `{status}`; three-seed mean `{vals.mean():.6f}` vs baseline `{base['composite']:.6f}`.\n")
    r["artifact_content_root_sha256"]=rebuild(out);return r

def main()->int:
    a=argparse.ArgumentParser();a.add_argument("--root",type=Path,required=True);a.add_argument("--artifact-root",type=Path,required=True);a.add_argument("--output",type=Path,required=True);args=a.parse_args();records=[one(args.root.resolve(),args.artifact_root.resolve(),c) for c in CANDIDATES];report={"schema":"cimc.forge200.support-spares-exact-closure.v1","status":"PASS" if all(x["status"].startswith("PASS") for x in records) else "PARTIAL","candidate_count":len(records),"contract_pass":sum(x["status"].startswith("PASS") for x in records),"contract_fail":sum(not x["status"].startswith("PASS") for x in records),"authority_nonzero":0,"board_accepted":0,"countable_models":0,"records":records,"content_root_sha256":hashlib.sha256(canonical_bytes(records)).hexdigest()};write_json(args.output,report);print(json.dumps({k:report[k] for k in ("status","candidate_count","contract_pass","contract_fail","content_root_sha256")},sort_keys=True));return 0 if report["status"]=="PASS" else 2
if __name__=="__main__":raise SystemExit(main())
