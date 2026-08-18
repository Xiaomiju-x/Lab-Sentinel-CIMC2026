#!/usr/bin/env python3
"""Stage S030 cross-domain reranking judgments from the licensed CC-BY corpus."""
from __future__ import annotations

import argparse,csv,hashlib,json,math,re
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from stage_matbench_experimental_v1 import canonical_bytes,sha256_file,write_json

TOKEN_RE=re.compile(r"[a-z][a-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]",re.I|re.UNICODE)
SPLITS=("train","validation","test")
DOMAINS=("PHOSPHOR","FURNACE","SEMIMAT","METROLOGY","PACKAGING","FABQUALITY")

def tokens(text:str)->list[str]:return TOKEN_RE.findall(text.lower())

def bm25(queries:list[str],passages:list[str])->np.ndarray:
    pts=[tokens(text) for text in passages];df=Counter();[df.update(set(value)) for value in pts];n=len(pts);avg=max(float(np.mean([len(value) for value in pts])),1.);result=np.zeros((len(queries),n),np.float32)
    for qi,query in enumerate(queries):
        for term in set(tokens(query)):
            frequency=df.get(term,0)
            if not frequency:continue
            inverse=math.log(1.+(n-frequency+.5)/(frequency+.5))
            for pi,terms in enumerate(pts):
                count=terms.count(term)
                if count:result[qi,pi]+=inverse*count*2.5/(count+1.5*(.25+.75*len(terms)/avg))
    return result

def vector(text:str,vocab:dict[str,int])->np.ndarray:
    value=np.zeros(len(vocab),np.float32)
    for term in tokens(text):
        if term in vocab:value[vocab[term]]+=1.
    norm=float(np.linalg.norm(value));return value/norm if norm else value

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]);args=parser.parse_args();root=args.root.resolve();source=root/"data"/"corpora"/"ccby_multidomain_v2.jsonl"
    rows=[json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()];counts=Counter()
    for row in rows:
        if row["split"]=="train":counts.update(tokens(row["title"]+" "+row["section"]+" "+row["text"]))
    ordered=[term for term,count in sorted(counts.items(),key=lambda item:(-item[1],item[0])) if count>=3][:1024];vocab={term:index for index,term in enumerate(ordered)}
    if len(vocab)<512:raise RuntimeError("TRAIN_VOCAB_GATE")
    arrays={};split_receipts=[]
    for split_code,split_name in enumerate(SPLITS):
        passages=sorted([row for row in rows if row["split"]==split_name],key=lambda row:row["chunk_id"]);queries=[]
        for domain in DOMAINS:
            domain_rows=[row for row in passages if row["domain"]==domain]
            queries.extend(sorted(domain_rows,key=lambda row:hashlib.sha256((row["chunk_id"]+":s030").encode()).digest())[:min(24,len(domain_rows))])
        query_text=[f"find evidence about {row['title']} {row['section']}" for row in queries];passage_text=[row["title"]+" "+row["section"]+" "+row["text"] for row in passages];bm=bm25(query_text,passage_text);qv=np.asarray([vector(text,vocab) for text in query_text]);pv=np.asarray([vector(text,vocab) for text in passage_text]);cos=qv@pv.T;features=[];grades=[];query_ids=[];domains=[];baseline=[];groups=[]
        for qi,query in enumerate(queries):
            selected=list(np.argsort(-bm[qi])[:50]);same=[pi for pi,p in enumerate(passages) if p["pmcid"]==query["pmcid"]]
            if same and not any(pi in same for pi in selected):selected[-1]=max(same,key=lambda pi:float(bm[qi,pi]))
            maximum=max(float(np.max(bm[qi])),1e-8);q_terms=set(tokens(query_text[qi]));domain_id=DOMAINS.index(query["domain"])
            for pi in selected:
                passage=passages[pi];p_terms=set(tokens(passage_text[pi]));overlap=len(q_terms&p_terms)/max(len(q_terms|p_terms),1);title_terms=set(tokens(query["title"]));title_overlap=len(title_terms&p_terms)/max(len(title_terms),1);same_domain=float(passage["domain"]==query["domain"]);section_match=float(passage["section"]==query["section"]);onehot=[1. if domain_id==value else 0. for value in range(6)]
                features.append([float(bm[qi,pi])/maximum,float(cos[qi,pi]),overlap,title_overlap,same_domain,section_match,min(len(tokens(passage["text"]))/400.,1.),*onehot]);grades.append(2 if passage["pmcid"]==query["pmcid"] else 1 if same_domain else 0);query_ids.append(qi);domains.append(domain_id);baseline.append(float(bm[qi,pi]));groups.append(query["pmcid"])
        prefix=split_name;arrays[f"{prefix}_x"]=np.asarray(features,np.float32);arrays[f"{prefix}_grade"]=np.asarray(grades,np.int64);arrays[f"{prefix}_query"]=np.asarray(query_ids,np.int32);arrays[f"{prefix}_domain"]=np.asarray(domains,np.int8);arrays[f"{prefix}_bm25"]=np.asarray(baseline,np.float32);arrays[f"{prefix}_group"]=np.asarray(groups)
        positives=int(np.sum(arrays[f"{prefix}_grade"]==2));split_receipts.append({"split":split_name,"queries":len(queries),"candidate_pairs":len(grades),"exact_document_positives":positives,"document_families":len(set(groups))})
        if len(queries)<24 or positives<24:raise RuntimeError(f"SPLIT_LABEL_GATE:{split_name}:{len(queries)}:{positives}")
    group_sets={name:set(arrays[f"{name}_group"].tolist()) for name in SPLITS};overlap=sum(len(group_sets[a]&group_sets[b]) for a,b in (("train","validation"),("train","test"),("validation","test")))
    if overlap:raise RuntimeError(f"DOCUMENT_FAMILY_LEAKAGE:{overlap}")
    with (root/"contracts"/"candidate_task_contracts_244.v1.tsv").open("r",encoding="utf-8-sig",newline="") as handle:contracts={row["candidate_id"]:row for row in csv.DictReader(handle,delimiter="\t")}
    cid="CAND-S-030";contract=contracts[cid];output=root/"data"/"staged_support_s030_exact_v1"/f"{cid}.npz";output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    metadata={"schema":"cimc.forge200.support-s030-exact-dataset.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","candidate_id":cid,"truth_class":"STRUCTURE_DERIVED","claim_state":"SAME_DOCUMENT_FAMILY_RELEVANCE_NOT_INDEPENDENT_EXPERT_JUDGMENT","source_gate":contract["source_gate"],"source_gate_match":"LICENSED_RAG_JUDGMENTS_WITH_DOCUMENT_FAMILY_SPLIT","label_derivation_rule":"grade_2_same_PMCI_document_family_grade_1_same_domain_else_0","split_receipts":split_receipts,"cross_split_document_family_overlap":overlap,"vocabulary_fit":"TRAIN_ONLY","vocabulary_terms":len(vocab),"vocabulary_sha256":hashlib.sha256(canonical_bytes(ordered)).hexdigest(),"baseline_execution":"BM25_SCORE_ORDER_FULL_FROZEN_TOP50","input_contract":contract["input_contract"],"target_label":contract["target_label"],"primary_metric":contract["primary_metric"],"parameter_cap":contract["parameter_cap"],"task_contract_sha256":hashlib.sha256(canonical_bytes(contract)).hexdigest(),"source_path":str(source.relative_to(root)).replace('\\','/'),"source_sha256":sha256_file(source),"license":"CC_BY_PER_METADATA_AND_JATS_VERIFIED","authority":0,"board_accepted":False,"countable_model":False,"generator_sha256":sha256_file(Path(__file__)),"sha256":sha256_file(output)}
    write_json(output.with_suffix(".metadata.json"),metadata);write_json(root/"evidence"/"support_s030_exact_staging.v1.json",{"schema":"cimc.forge200.support-s030-exact-staging.v1","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"PASS","record":metadata,"authority_nonzero":0,"board_actions":0});print(json.dumps({"status":"PASS","candidate_id":cid,"splits":split_receipts},sort_keys=True));return 0

if __name__=="__main__":raise SystemExit(main())
