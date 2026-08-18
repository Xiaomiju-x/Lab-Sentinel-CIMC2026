#!/usr/bin/env python3
"""Stage S032 query reformulation from licensed document-family relevance pairs."""
from __future__ import annotations
import argparse,csv,hashlib,json,re
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json
TOKEN_RE=re.compile(r"[a-z][a-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]",re.I|re.UNICODE)
DOMAINS=("PHOSPHOR","FURNACE","SEMIMAT","METROLOGY","PACKAGING","FABQUALITY")
REWRITES=((r"\bphotoluminescence\b|\bluminescence\b","light emission"),(r"\bphosphor\b","emissive ceramic"),(r"\bdop(?:ed|ing|ant)?\b","activator substitution"),(r"\bsinter(?:ed|ing)?\b","thermal consolidation"),(r"\bfurnace\b","thermal chamber"),(r"\btemperature\b","thermal condition"),(r"\bsemiconductor\b","electronic material"),(r"\bband\s*gap\b","electronic energy gap"),(r"\bx[- ]?ray diffraction\b|\bxrd\b","crystal pattern measurement"),(r"\bmetrology\b|\bcharacteri[sz]ation\b","measurement analysis"),(r"\bpackaging\b","device assembly"),(r"\breliability\b","durability"),(r"\bsolder\b","metal joint"),(r"\bwafer\b","substrate"),(r"\byield\b","production quality"),(r"\bfabrication\b","manufacturing"))
def tokens(text):return TOKEN_RE.findall(text.lower())
def rewrite(text):
 for pattern,replacement in REWRITES:text=re.sub(pattern,replacement,text,flags=re.I)
 return text
def hashvec(text,bins=128):
 value=np.zeros(bins,np.float32)
 for term in tokens(text):
  digest=hashlib.sha256(term.encode()).digest();value[int.from_bytes(digest[:2],"little")%bins]+=1. if digest[2]&1 else -1.
 norm=float(np.linalg.norm(value));return value/norm if norm else value
def bow(text,vocab):
 value=np.zeros(len(vocab),np.float32)
 for term in tokens(text):
  if term in vocab:value[vocab[term]]+=1.
 norm=float(np.linalg.norm(value));return value/norm if norm else value
def main():
 p=argparse.ArgumentParser();p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);a=p.parse_args();root=a.root.resolve();source=root/"data"/"corpora"/"ccby_multidomain_v2.jsonl";rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()];counter=Counter()
 for row in rows:
  if row["split"]=="train":counter.update(tokens(row["title"]+" "+row["section"]+" "+row["text"]))
 ordered=[term for term,count in sorted(counter.items(),key=lambda item:(-item[1],item[0])) if count>=3][:256];vocab={term:index for index,term in enumerate(ordered)}
 if len(vocab)!=256:raise RuntimeError("VOCAB_GATE")
 arrays={};receipts=[]
 for code,name in enumerate(("train","validation","test")):
  split_rows=sorted([row for row in rows if row["split"]==name],key=lambda row:row["chunk_id"]);queries=split_rows if name=="train" else sorted(split_rows,key=lambda row:hashlib.sha256((row["chunk_id"]+":s032").encode()).digest())[:min(240,len(split_rows))];x=[];target=[];raw_target=[];group=[];domain=[]
  for row in queries:
   original=f"find evidence about {row['title']} {row['section']}";raw=rewrite(original);hint=[1. if row["domain"]==value else 0. for value in DOMAINS];x.append(np.concatenate((hashvec(raw),hint)));target.append(bow(original,vocab));raw_target.append(bow(raw,vocab));group.append(row["pmcid"]);domain.append(DOMAINS.index(row["domain"]))
  passages=np.asarray([bow(row["title"]+" "+row["section"]+" "+row["text"],vocab) for row in split_rows],np.float32);passage_group=np.asarray([row["pmcid"] for row in split_rows]);arrays[f"{name}_x"]=np.asarray(x,np.float32);arrays[f"{name}_target"]=np.asarray(target,np.float32);arrays[f"{name}_raw_query"]=np.asarray(raw_target,np.float32);arrays[f"{name}_group"]=np.asarray(group);arrays[f"{name}_domain"]=np.asarray(domain,np.int8);arrays[f"{name}_passage"]=passages;arrays[f"{name}_passage_group"]=passage_group;receipts.append({"split":name,"queries":len(queries),"passages":len(split_rows),"document_families":len(set(group))})
 sets={name:set(arrays[f"{name}_group"].tolist()) for name in ("train","validation","test")};overlap=sum(len(sets[i]&sets[j]) for i,j in (("train","validation"),("train","test"),("validation","test")))
 if overlap or min(item["queries"] for item in receipts)<100:raise RuntimeError(f"DATA_GATE:{overlap}:{receipts}")
 with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as h:contracts={r["candidate_id"]:r for r in csv.DictReader(h,delimiter="\t")}
 cid="CAND-S-032";contract=contracts[cid];out=root/"data"/"staged_support_s032_exact_v1"/f"{cid}.npz";out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**arrays);meta={"schema":"cimc.forge200.support-s032-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"STRUCTURE_DERIVED","claim_state":"SAME_DOCUMENT_FAMILY_RELEVANCE_NOT_INDEPENDENT_EXPERT_JUDGMENT","source_gate":contract["source_gate"],"source_gate_match":"LICENSED_QUERY_RELEVANCE_PAIRS_WITH_SESSION_LEVEL_SPLIT","split_receipts":receipts,"cross_split_document_family_overlap":overlap,"label_derivation":"original_title_section_terms_as_entity_preserving_rewrite_target","raw_query_derivation":"FROZEN_DOMAIN_SYNONYM_PARAPHRASE_MAP","baseline_execution":"RAW_QUERY_WITHOUT_REWRITE_FULL_SPLIT_RETRIEVAL","output_vocabulary_fit":"TRAIN_ONLY","output_vocabulary_terms":len(vocab),"output_vocabulary_sha256":hashlib.sha256(canonical_bytes(ordered)).hexdigest(),"input_contract":contract["input_contract"],"target_label":contract["target_label"],"primary_metric":contract["primary_metric"],"parameter_cap":contract["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(contract)).hexdigest(),"source_path":str(source.relative_to(root)).replace('\\','/'),"source_sha256":sha256_file(source),"license":"CC_BY_PER_METADATA_AND_JATS_VERIFIED","authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(out)};write_json(out.with_suffix(".metadata.json"),meta);write_json(root/"evidence"/"support_s032_exact_staging.v1.json",{"schema":"cimc.forge200.support-s032-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":meta,"authority_nonzero":0,"board_actions":0});print(json.dumps({"status":"PASS","candidate_id":cid,"splits":receipts},sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
