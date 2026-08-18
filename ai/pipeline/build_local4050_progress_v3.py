#!/usr/bin/env python3
"""Build the authoritative D-only local4050 progress snapshot."""
from __future__ import annotations
import argparse,collections,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

PASS="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING"
TARGET={"P":112,"G":30,"S":28}
def canonical(v:Any)->bytes:return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
def sha(p:Path)->str:
 d=hashlib.sha256()
 with p.open("rb") as h:
  for b in iter(lambda:h.read(1<<20),b""):d.update(b)
 return d.hexdigest()
def write(p:Path,v:Any)->None:p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def main()->int:
 a=argparse.ArgumentParser();a.add_argument("--root",type=Path,required=True);a.add_argument("--output",type=Path,required=True);args=a.parse_args();root=args.root.resolve();receipts=collections.defaultdict(list)
 for path in root.glob("artifacts/**/promotion_receipt.json"):
  try:r=json.loads(path.read_text(encoding="utf-8"))
  except Exception:continue
  if r.get("status")==PASS:receipts[r["candidate_id"]].append((path,r))
 ids=sorted(receipts);counts=collections.Counter(cid.split("-")[1] for cid in ids);records=[]
 for cid in ids:
  choices=sorted(receipts[cid],key=lambda item:("local4050" not in str(item[0]),str(item[0])))
  path,r=choices[0];records.append({"candidate_id":cid,"category":cid.split("-")[1],"promotion_receipt":str(path.relative_to(root)).replace("\\","/"),"promotion_receipt_sha256":sha(path),"package_sha256":r.get("package",{}).get("sha256"),"authority":r.get("authority"),"board_accepted":r.get("board_accepted"),"countable_model":r.get("countable_model")})
 package_hashes=[x["package_sha256"] for x in records if x["package_sha256"]]
 evidence_names=("support_exact_closure.v3.json","nli_exact_closure.v1.json","reranker_exact_closure.v1.json","reranker_exact_closure.v2_s019_s020.json","support_spares_exact_closure.v1.json","s037_exact_closure.v1.json","ipop_exact_closure.v1.json","matbench_experimental_exact_closure.v1.json")
 evidence=[{"path":f"evidence/{name}","sha256":sha(root/"evidence"/name)} for name in evidence_names if (root/"evidence"/name).is_file()]
 result={"schema":"cimc.forge200.local4050-progress.v3","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"LOCAL_HOST_EXACT_PARTIAL_SUPPORT_TARGET_FILLED_BOARD_PENDING","workspace_root":str(root),"gpu":"NVIDIA GeForce RTX 4050 Laptop GPU","authorization":"USER_EXPLICITLY_AUTHORIZED_LOCAL4050_MAIN_TRAINING_2026_08_03","new_release_target":{"assets":170,"by_category":TARGET},"host_exact_pass":{"unique_candidates":len(ids),"by_category":{"predictive":counts["P"],"generative":counts["G"],"support":counts["S"]},"shortfall":{"predictive":TARGET["P"]-counts["P"],"generative":TARGET["G"]-counts["G"],"support":TARGET["S"]-counts["S"],"total":170-len(ids)},"candidate_ids":ids},"initial_board_baseline":{"assets":30,"logical_models":28},"combined_if_all_new_host_passes_later_board_accept":{"current_host_exact_assets_plus_initial_assets":30+len(ids),"current_host_exact_logical_plus_initial_logical":28+len(ids),"not_a_board_acceptance_claim":True},"new_models_board_accepted":0,"new_models_countable_publicly":0,"support_release_slots_filled":counts["S"]==TARGET["S"],"package_hashes":{"present":len(package_hashes),"unique":len(set(package_hashes)),"collisions":len(package_hashes)-len(set(package_hashes))},"trained_nanolm_exact_contract_pending":26,"cloud_no_card_download_required":False,"old_cloud_instances_may_remain_closed":True,"authority_nonzero":sum(x["authority"]!=0 for x in records),"records":records,"evidence":evidence}
 result["content_root_sha256"]=hashlib.sha256(canonical({"records":records,"evidence":evidence})).hexdigest();write(args.output,result);print(json.dumps({k:result[k] for k in ("status","host_exact_pass","new_models_board_accepted","package_hashes","content_root_sha256")},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
