#!/usr/bin/env python3
"""Build a deduplicated host closure separating exact and SIM_ONLY extension assets."""
from __future__ import annotations
import argparse,collections,hashlib,json,struct
from datetime import datetime,timezone
from pathlib import Path
from typing import Any

EXACT_STATUS="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING"
SIM_STATUS="HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY"
def canonical(value:Any)->bytes:return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
def sha(path:Path)->str:
 digest=hashlib.sha256()
 with path.open("rb") as handle:
  for block in iter(lambda:handle.read(1<<20),b""):digest.update(block)
 return digest.hexdigest()
def write(path:Path,value:Any):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def validate_package(path:Path,receipt:dict,allowed_engines:set[int])->dict:
 raw=path.read_bytes()
 if len(raw)<256:raise RuntimeError(f"PACKAGE_SHORT:{path}")
 magic,schema,header,engine,opset,authority,flags,tensors,generation,payload_bytes,scratch,arena,kv=struct.unpack_from("<4sHHHHBBHQQIII",raw,0)
 model_id=raw[44:76].split(b"\0",1)[0].decode("utf-8");payload_hash=raw[76:108].hex();golden_hash=raw[108:140].hex();release_root=raw[140:172].hex();output_schema_hash=raw[172:204].hex();reserved=raw[204:256]
 actual_payload=hashlib.sha256(raw[header:]).hexdigest();actual_package=hashlib.sha256(raw).hexdigest();expected_package=receipt.get("package",{}).get("sha256")
 errors=[]
 if magic!=b"ICMF" or schema!=1 or header!=256:errors.append("HEADER_SCHEMA")
 if engine not in allowed_engines or opset!=1:errors.append("ENGINE_OPSET")
 if authority!=0 or any(reserved):errors.append("AUTHORITY_OR_RESERVED")
 if model_id!=receipt["candidate_id"]:errors.append("MODEL_ID")
 if payload_bytes!=len(raw)-header or payload_hash!=actual_payload:errors.append("PAYLOAD")
 if expected_package and expected_package!=actual_package:errors.append("PACKAGE_SHA")
 if receipt.get("release_root") and receipt["release_root"]!=release_root:errors.append("RELEASE_ROOT")
 if errors:raise RuntimeError(f"PACKAGE_GATE:{path}:{errors}")
 return {"path":str(path),"bytes":len(raw),"sha256":actual_package,"payload_bytes":payload_bytes,"payload_sha256":payload_hash,"golden_sha256_header":golden_hash,"release_root_header":release_root,"output_schema_sha256_header":output_schema_hash,"engine_id":engine,"opset":opset,"tensor_count":tensors,"generation_counter":generation,"scratch_bytes":scratch,"arena_bytes":arena,"kv_bytes":kv,"flags":flags}
def main()->int:
 parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);a=parser.parse_args();root=a.root.resolve();abi=json.loads((root/"contracts"/"model_package_abi.v1.json").read_text(encoding="utf-8"));engines={item["engine_id"] for item in abi["engines"]};choices=collections.defaultdict(list);rejected=collections.defaultdict(set)
 for path in root.glob("artifacts/**/promotion_receipt.json"):
  try:receipt=json.loads(path.read_text(encoding="utf-8"))
  except Exception:continue
  cid=receipt.get("candidate_id");status=receipt.get("status","")
  if not cid:continue
  if status in {EXACT_STATUS,SIM_STATUS} and receipt.get("host_contract_pass") is not False:choices[(cid,status)].append((path,receipt))
  elif "REJECTED" in status:rejected[cid].add(status)
 exact=[];extensions=[]
 for (cid,status),items in sorted(choices.items()):
  items.sort(key=lambda item:("local4050" not in str(item[0]).lower(),-item[0].stat().st_mtime,str(item[0])))
  path,receipt=items[0]
  if receipt.get("authority")!=0 or receipt.get("board_accepted") is not False or receipt.get("countable_model") is not False:raise RuntimeError(f"AUTHORITY_BOARD_GATE:{cid}")
  package_path=path.parent/receipt["package"]["path"];package=validate_package(package_path,receipt,engines);golden=path.parent/"golden_vectors.npz";output_schema=path.parent/"output_schema.json";model_card=path.parent/"model_card.md"
  if not golden.is_file() or sha(golden)!=(receipt.get("golden_sha256") or package["golden_sha256_header"]):raise RuntimeError(f"GOLDEN_GATE:{cid}")
  if package["golden_sha256_header"]!=sha(golden):raise RuntimeError(f"HEADER_GOLDEN_GATE:{cid}")
  record={"candidate_id":cid,"category":cid.split("-")[1],"status":status,"truth_class":receipt.get("truth_class"),"claim_state":receipt.get("claim_state"),"public_claim_scope":receipt.get("public_claim_scope"),"promotion_receipt":str(path.relative_to(root)).replace('\\','/'),"promotion_receipt_sha256":sha(path),"package":{**package,"path":str(package_path.relative_to(root)).replace('\\','/')},"golden":{"path":str(golden.relative_to(root)).replace('\\','/'),"sha256":sha(golden)},"output_schema":{"path":str(output_schema.relative_to(root)).replace('\\','/'),"sha256":sha(output_schema)} if output_schema.is_file() else None,"model_card":{"path":str(model_card.relative_to(root)).replace('\\','/'),"sha256":sha(model_card)} if model_card.is_file() else None,"authority":0,"board_accepted":False,"countable_model":False}
  (exact if status==EXACT_STATUS else extensions).append(record)
 exact_ids={record["candidate_id"] for record in exact};extensions=[record for record in extensions if record["candidate_id"] not in exact_ids];all_records=exact+extensions;package_hashes=[record["package"]["sha256"] for record in all_records];payload_hashes=[record["package"]["payload_sha256"] for record in all_records];counts=lambda rows:dict(collections.Counter(record["category"] for record in rows));result={"schema":"cimc.forge200.host-closure.v4","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"HOST_CLOSURE_PARTIAL_RELEASE_FLOOR_NOT_MET_BOARD_PENDING","exact_contract":{"unique_candidates":len(exact),"by_category":counts(exact),"records":exact},"sim_only_extensions":{"unique_candidates":len(extensions),"by_category":counts(extensions),"records":extensions,"not_substitutes_for_frozen_source_gates":True},"host_qualified_total_including_extensions":len(all_records),"initial_board_baseline":{"assets":30,"logical_models":28},"combined_assets_if_all_host_assets_later_board_pass":30+len(all_records),"release_floor":{"total_assets":150,"new_assets_required":120,"exact_new_shortfall":max(120-len(exact),0),"including_sim_extension_shortfall":max(120-len(all_records),0),"met":30+len(exact)>=150},"integrity":{"package_hashes":len(package_hashes),"unique_package_hashes":len(set(package_hashes)),"package_collisions":len(package_hashes)-len(set(package_hashes)),"payload_hashes":len(payload_hashes),"unique_payload_hashes":len(set(payload_hashes)),"payload_collisions":len(payload_hashes)-len(set(payload_hashes))},"rejected_candidates":sorted({cid for cid in rejected if cid not in exact_ids}),"new_models_board_accepted":0,"new_models_countable_publicly":0,"authority_nonzero":0}
 result["content_root_sha256"]=hashlib.sha256(canonical({"exact":exact,"extensions":extensions,"integrity":result["integrity"]})).hexdigest();write(a.output,result);print(json.dumps({"status":result["status"],"exact":len(exact),"extensions":len(extensions),"by_category_exact":counts(exact),"combined_assets":result["combined_assets_if_all_host_assets_later_board_pass"],"integrity":result["integrity"],"content_root_sha256":result["content_root_sha256"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
