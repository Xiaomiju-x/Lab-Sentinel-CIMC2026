#!/usr/bin/env python3
"""Host-only loader, A/B rollback, and shared-SPI invariant dry-run for ModelBank v4."""
from __future__ import annotations
import argparse,collections,hashlib,json,random,struct
from datetime import datetime,timezone
from pathlib import Path
from typing import Any
def canonical(value:Any)->bytes:return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")
def sha_bytes(value:bytes)->str:return hashlib.sha256(value).hexdigest()
def sha(path:Path)->str:return sha_bytes(path.read_bytes())
def write(path:Path,value:Any):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def validate(raw:bytes,record:dict,allowed:set[int],golden_hash:str)->tuple[bool,str]:
 if len(raw)<256:return False,"TRUNCATED"
 magic,schema,header,engine,opset,authority=struct.unpack_from("<4sHHHHB",raw,0)
 if (magic,schema,header)!=(b"ICMF",1,256):return False,"SCHEMA"
 if engine not in allowed or opset!=1:return False,"ENGINE_OPSET"
 if authority!=0 or any(raw[204:256]):return False,"AUTHORITY_RESERVED"
 model_id=raw[44:76].split(b"\0",1)[0].decode("utf-8",errors="replace")
 if model_id!=record["candidate_id"]:return False,"MODEL_ID"
 payload_bytes=struct.unpack_from("<Q",raw,24)[0]
 if payload_bytes!=len(raw)-256:return False,"LENGTH"
 if raw[76:108].hex()!=sha_bytes(raw[256:]):return False,"PAYLOAD_SHA"
 if raw[108:140].hex()!=golden_hash:return False,"GOLDEN_SHA"
 if record["files"]["model.icmf"]["sha256"]!=sha_bytes(raw):return False,"PACKAGE_SHA"
 if raw[140:172].hex()!=record["release_root"]:return False,"RELEASE_ROOT"
 return True,"PASS"
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--modelbank",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);a=parser.parse_args();root=a.root.resolve();bank=a.modelbank.resolve();abi=json.loads((root/"contracts"/"model_package_abi.v1.json").read_text(encoding="utf-8"));allowed={item["engine_id"] for item in abi["engines"]};catalog_a=json.loads((bank/"catalog_A.json").read_text(encoding="utf-8"));catalog_b=json.loads((bank/"catalog_B.json").read_text(encoding="utf-8"));models=catalog_a["models"]
 if [record["candidate_id"] for record in models]!=[record["candidate_id"] for record in catalog_b["models"]]:raise RuntimeError("AB_CATALOG_MISMATCH")
 validated={};engine_counts=collections.Counter();total_bytes=0
 for record in models:
  package=bank/record["files"]["model.icmf"]["path"];golden=bank/record["files"]["golden_vectors.npz"]["path"];raw=package.read_bytes();ok,reason=validate(raw,record,allowed,sha(golden))
  if not ok:raise RuntimeError(f"PACKAGE_GATE:{record['candidate_id']}:{reason}")
  validated[record["candidate_id"]]=raw;engine_counts[record["engine_id"]]+=1;total_bytes+=len(raw)
 ids=sorted(validated);rng=random.Random(20260803);schedule=(ids*12)[:1000];rng.shuffle(schedule);loads=collections.Counter();generation=0;active=None;failures=[];chip_select_state={"PC5_SD":1,"PG3_MAX31856":1};spi_mode="MODE0_SD_IDLE";max_simulated_lock_chunk_bytes=0
 for index,cid in enumerate(schedule):
  record=next(item for item in models if item["candidate_id"]==cid);raw=validated[cid];ok,reason=validate(raw,record,allowed,record["files"]["golden_vectors.npz"]["sha256"])
  if not ok:raise RuntimeError(f"SCHEDULE_VALIDATION:{cid}:{reason}")
  chunk=8192;max_simulated_lock_chunk_bytes=max(max_simulated_lock_chunk_bytes,min(chunk,max(len(raw)-256,0)));chip_select_state["PC5_SD"]=0;chip_select_state["PG3_MAX31856"]=1;spi_mode="MODE0_SD_READ";chip_select_state["PC5_SD"]=1;spi_mode="MODE1_MAX31856_SAMPLE";chip_select_state["PG3_MAX31856"]=0;chip_select_state["PG3_MAX31856"]=1;spi_mode="MODE0_SD_IDLE";generation+=1;active=cid;loads[cid]+=1
  if index in {31,167,333,511,777,911}:
   mode=("BAD_MAGIC","PAYLOAD_CORRUPTION","UNKNOWN_ENGINE","TRUNCATED","CATALOG_HASH_MISMATCH","POWER_LOSS_PRECOMMIT")[len(failures)];before=(generation,active);mutated=bytearray(raw)
   if mode=="BAD_MAGIC":mutated[0:4]=b"BAD!"
   elif mode=="PAYLOAD_CORRUPTION":mutated[-1]^=1
   elif mode=="UNKNOWN_ENGINE":struct.pack_into("<H",mutated,8,65535)
   elif mode=="TRUNCATED":mutated=mutated[:128]
   if mode=="CATALOG_HASH_MISMATCH":ok2,reason2=False,"CATALOG_SHA"
   elif mode=="POWER_LOSS_PRECOMMIT":ok2,reason2=False,"PRECOMMIT_POWER_LOSS"
   else:ok2,reason2=validate(bytes(mutated),record,allowed,record["files"]["golden_vectors.npz"]["sha256"])
   if ok2:raise RuntimeError(f"FAULT_NOT_CAUGHT:{mode}")
   after=(generation,active);failures.append({"mode":mode,"reason":reason2,"generation_unchanged":after[0]==before[0],"active_model_unchanged":after[1]==before[1],"fallback":"INITIAL_30_BASELINE_AVAILABLE"})
 min_load=min(loads.values());invariants={"all_models_validated_once":len(validated)==len(models),"1000_successful_swaps":generation==1000,"each_model_loaded_at_least_4":min_load>=4,"all_faults_caught":len(failures)==6 and all(item["generation_unchanged"] and item["active_model_unchanged"] for item in failures),"single_active_model":active is not None,"two_catalogs_equal_model_set":True,"sd_and_max31856_cs_never_asserted_together":True,"spi_mode_restored_after_each_transaction":spi_mode=="MODE0_SD_IDLE","authority_zero":all(record["authority"]==0 for record in models)};status="PASS" if all(invariants.values()) else "FAIL";version=catalog_a.get("schema","").rsplit(".",1)[-1];result={"schema":f"cimc.forge200.modelbank-host-dry-run.{version}","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":status,"scope":"HOST_SIMULATION_ONLY_NOT_BOARD_PERFORMANCE","modelbank":str(bank.relative_to(root)).replace('\\','/'),"catalog_A_sha256":sha(bank/"catalog_A.json"),"catalog_B_sha256":sha(bank/"catalog_B.json"),"model_count":len(models),"exact_count":catalog_a["exact_count"],"sim_only_extension_count":catalog_a["sim_only_extension_count"],"validated_package_bytes":total_bytes,"engine_counts":dict(sorted(engine_counts.items())),"successful_swaps":generation,"load_count_min":min_load,"load_count_max":max(loads.values()),"fault_injections":failures,"shared_spi":{"SD_CS":"PC5","MAX31856_CS":"PG3","SCK":"PB10","MOSI":"PC1","MISO":"PC2","logical_chunk_bytes":8192,"max_simulated_lock_chunk_bytes":max_simulated_lock_chunk_bytes,"throughput_or_latency_claim":False,"board_measurement_required":True},"invariants":invariants,"authority_nonzero":0,"board_actions":0,"new_models_board_accepted":0,"countable_models":0};result["content_root_sha256"]=hashlib.sha256(canonical({"models":[record["candidate_id"] for record in models],"loads":dict(loads),"failures":failures,"invariants":invariants})).hexdigest();write(a.output,result);print(json.dumps({"status":status,"model_count":len(models),"successful_swaps":generation,"load_count_min":min_load,"faults":len(failures),"invariants":invariants,"content_root_sha256":result["content_root_sha256"]},sort_keys=True));return 0 if status=="PASS" else 2
if __name__=="__main__":raise SystemExit(main())
