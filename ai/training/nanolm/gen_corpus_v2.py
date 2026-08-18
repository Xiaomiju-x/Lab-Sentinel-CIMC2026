"""
gen_corpus_v2.py — richer distillation corpus for the FLAGSHIP edge LM.

Same 12-slot CONTRACT as gen_corpus.py (imported, single source of truth, so the
firmware build_context stays byte-identical) — only the TEACHER PROMPT, sample
count, phrasing diversity and length budget are upgraded, to give a larger
student (d160/d192) something worth its capacity:

  * longer, more actionable one-liners (cause -> concrete action, 18-42 chars)
  * varied temperature per call so identical states get diverse phrasings
  * the API model name is read from DEEPSEEK_MODEL; the current example/default
    is deepseek-v4-flash because the legacy aliases were retired in July 2026

Run (PC, no GPU needed — pure API):
  python gen_corpus_v2.py --n 8000 --workers 24 --out corpus_v2.jsonl
  python gen_corpus_v2.py --smoke
"""
import argparse
import json
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# reuse the EXACT slot schema + state builder (firmware contract — do not fork)
from gen_corpus import (SLOTS, SLOT_ORDER, describe, build_states,  # noqa: F401
                        DEEPSEEK_KEY, DEEPSEEK_URL, MODEL)

SYS_PROMPT = (
    "你是部署在近红外荧光粉(Cr3+掺杂石榴石)烧结炉旁的边缘AI诊断助手,运行在国产单片机上。"
    "根据给定的实时炉况,用一句话给出最关键的诊断与可执行操作建议。"
    "要求:18到42个汉字;先点明根因或关键现象,再给出具体动作(涉及参数时给数值);"
    "专业、具体、可落地;同类状态尽量换不同措辞,避免千篇一律;"
    "只输出这一句,不要解释推理过程,不要换行,不要引号,不要除化学式与温度单位外的英文。"
)


def call_deepseek(desc, temperature, retries=4):
    if not DEEPSEEK_KEY:
        return "__ERR__DEEPSEEK_API_KEY is not set"
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS_PROMPT},
                     {"role": "user", "content": "炉况状态:" + desc + "\n请给出一句话诊断与建议。"}],
        "temperature": temperature, "max_tokens": 120, "stream": False,
    }).encode("utf-8")
    for i in range(retries):
        try:
            req = urllib.request.Request(DEEPSEEK_URL, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_KEY}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            txt = d["choices"][0]["message"]["content"].strip()
            txt = txt.replace("\n", "").replace("\r", "").strip().strip('"“”')
            return txt
        except Exception as ex:
            if i == retries - 1:
                return f"__ERR__{ex}"
            time.sleep(1.5 * (i + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="corpus_v2.jsonl")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        rng = random.Random(7)
        for st in build_states(8):
            desc = describe(st)
            out = call_deepseek(desc, rng.uniform(0.5, 0.95))
            print("STATE:", {k: st[k] for k in ["stage", "temp", "risk", "tc", "gas"]})
            print("DIAG :", out, f"  ({len(out)}字)")
            print("-" * 70)
        return

    states = build_states(args.n)
    rng = random.Random(20260604)
    temps = [rng.uniform(0.5, 0.95) for _ in states]   # per-call phrasing diversity
    print(f"built {len(states)} states; querying {MODEL} "
          f"with {args.workers} workers...")
    t0 = time.time()
    rows = [None] * len(states)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_deepseek, describe(st), temps[i]): i
                for i, st in enumerate(states)}
        for fut in as_completed(futs):
            i = futs[fut]
            rows[i] = (states[i], fut.result())
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(states)}  ({time.time()-t0:.0f}s)")

    n_ok = n_err = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for st, txt in rows:
            if not txt or txt.startswith("__ERR__") or len(txt) > 72:
                n_err += 1
                continue
            f.write(json.dumps({"slots": st, "text": txt}, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"done: {n_ok} ok, {n_err} dropped -> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
