#!/usr/bin/env python3
"""Materialize a content-verified host ModelBank from host_closure.v4."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
def canonical(value:Any)->bytes:return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
def sha(path:Path)->str:
 digest=hashlib.sha256()
 with path.open("rb") as handle:
  for block in iter(lambda:handle.read(1<<20),b""):digest.update(block)
 return digest.hexdigest()
def write(path:Path,value:Any):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);a=parser.parse_args();root=a.root.resolve();out=a.output.resolve()
 releases=(root/"releases").resolve()
 if releases not in out.parents or out==releases:raise RuntimeError("OUTPUT_SCOPE_GATE")
 if out.exists():raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{out}")
 closure_path=root/"evidence"/"host_closure.v4.json";closure=json.loads(closure_path.read_text(encoding="utf-8"));records=[]
 for tier,key in (("EXACT_CONTRACT","exact_contract"),("SIM_ONLY_EXTENSION","sim_only_extensions")):
  for source_record in closure[key]["records"]:
   cid=source_record["candidate_id"];destination=out/"packages"/cid;destination.mkdir(parents=True,exist_ok=True);files={}
   mappings=((source_record["package"],"model.icmf"),(source_record["golden"],"golden_vectors.npz"),(source_record.get("output_schema"),"output_schema.json"),(source_record.get("model_card"),"model_card.md"))
   for spec,name in mappings:
    if not spec:continue
    source=root/spec["path"];target=destination/name;shutil.copy2(source,target)
    if sha(target)!=spec["sha256"]:raise RuntimeError(f"COPY_HASH_GATE:{cid}:{name}")
    files[name]={"path":str(target.relative_to(out)).replace('\\','/'),"bytes":target.stat().st_size,"sha256":sha(target)}
   receipt_source=root/source_record["promotion_receipt"];receipt_target=destination/"promotion_receipt.json";shutil.copy2(receipt_source,receipt_target);files["promotion_receipt.json"]={"path":str(receipt_target.relative_to(out)).replace('\\','/'),"bytes":receipt_target.stat().st_size,"sha256":sha(receipt_target)}
   records.append({"candidate_id":cid,"category":source_record["category"],"tier":tier,"truth_class":source_record.get("truth_class"),"claim_state":source_record.get("claim_state"),"public_claim_scope":source_record.get("public_claim_scope"),"engine_id":source_record["package"]["engine_id"],"opset":source_record["package"]["opset"],"payload_bytes":source_record["package"]["payload_bytes"],"scratch_bytes":source_record["package"]["scratch_bytes"],"arena_bytes":source_record["package"]["arena_bytes"],"kv_bytes":source_record["package"]["kv_bytes"],"release_root":source_record["package"]["release_root_header"],"files":files,"authority":0,"board_accepted":False,"countable_model":False})
 records.sort(key=lambda record:record["candidate_id"]);catalog_common={"schema":"cimc.forge200.modelbank-catalog.v4","status":"HOST_STAGING_BOARD_PENDING","created_at_utc":datetime.now(timezone.utc).isoformat(),"abi":"contracts/model_package_abi.v1.json","closure":{"path":"evidence/host_closure.v4.json","sha256":sha(closure_path),"content_root_sha256":closure["content_root_sha256"]},"model_count":len(records),"exact_count":sum(record["tier"]=="EXACT_CONTRACT" for record in records),"sim_only_extension_count":sum(record["tier"]=="SIM_ONLY_EXTENSION" for record in records),"models":records,"authority":0,"board_accepted":False,"countable_models":0}
 catalogs=[]
 for slot in ("A","B"):
  catalog={**catalog_common,"slot":slot,"generation_counter":0,"commit_state":"PREPARED_NOT_BOARD_COMMITTED"};catalog["content_root_sha256"]=hashlib.sha256(canonical({"slot":slot,"generation_counter":0,"models":records})).hexdigest();path=out/f"catalog_{slot}.json";write(path,catalog);catalogs.append({"slot":slot,"path":path.name,"bytes":path.stat().st_size,"sha256":sha(path),"content_root_sha256":catalog["content_root_sha256"]})
 restore={"schema":"cimc.forge200.modelbank-restore.v1","failure_action":"REFUSE_CANDIDATE_AND_RETURN_TO_INITIAL_30_ASSET_BASELINE","generation_counter_rule":"advance only after schema, bounds, engine/opset, payload SHA, golden, and output schema validations pass","catalog_selection":"highest fully valid committed generation; equal uncommitted A/B catalogs are staging only","board_actions":0};write(out/"RESTORE.json",restore)
 manifest={"schema":"cimc.forge200.modelbank-build.v4","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"HOST_MODELBANK_BUILT_BOARD_PENDING","root":str(out),"closure_sha256":sha(closure_path),"model_count":len(records),"exact_count":catalog_common["exact_count"],"sim_only_extension_count":catalog_common["sim_only_extension_count"],"catalogs":catalogs,"files":[],"authority_nonzero":0,"board_actions":0}
 for path in sorted(value for value in out.rglob("*") if value.is_file()):manifest["files"].append({"path":str(path.relative_to(out)).replace('\\','/'),"bytes":path.stat().st_size,"sha256":sha(path)})
 manifest["file_count"]=len(manifest["files"]);manifest["bytes"]=sum(item["bytes"] for item in manifest["files"]);manifest["content_root_sha256"]=hashlib.sha256(canonical(manifest["files"])).hexdigest();write(out/"MANIFEST.v4.json",manifest);receipt={**manifest,"manifest_path":str((out/"MANIFEST.v4.json").relative_to(root)).replace('\\','/'),"manifest_sha256":sha(out/"MANIFEST.v4.json")};write(root/"evidence"/"modelbank_build.v4.json",receipt);print(json.dumps({"status":manifest["status"],"model_count":len(records),"exact_count":catalog_common["exact_count"],"sim_only_extension_count":catalog_common["sim_only_extension_count"],"file_count":manifest["file_count"],"bytes":manifest["bytes"],"content_root_sha256":manifest["content_root_sha256"]},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
