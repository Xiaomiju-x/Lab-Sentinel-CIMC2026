"""
x5_coteacher.py — RUN ON THE TEACHER-MODEL SERVER (set the API host below).
Generates the E2(recipe) and E4(qc) co-teacher samples from an offline domain
SFT model (Qwen2.5-1.5B NIR-SFT, served over a local OpenAI-style API), the
genuine "distil our own offline model to the edge" contribution. The SFT model
tends to elaborate, so we force one short sentence and trim to <=28 chars.

Needs gen_corpus.py alongside (for build_states/describe). Writes
corpus_e2_x5.jsonl / corpus_e4_x5.jsonl to merge back on the PC.

Run:  python3 x5_coteacher.py --n 260
"""
import argparse
import json
import re
import time
import urllib.request

from gen_corpus import build_states, describe

URL = "http://127.0.0.1:9001/v1/chat/completions"
SYS = {
    "e2": ("你是近红外荧光粉烧结工艺专家。只输出一句话(不超过28个汉字)的烧结工艺/配方优化建议"
           "(升温速率/保温时长/气氛/目标温度其一)。不要列条目,不要换行,不要解释,不要加引号。"),
    "e4": ("你是近红外荧光粉质量判读员。只输出一句话(不超过28个汉字)的批次质量判读"
           "(良品/疑似废品/废品)及主因。不要列条目,不要换行,不要解释,不要加引号。"),
}
ASK = {"e2": "请给出一句话工艺/配方优化建议。", "e4": "请给出一句话批次质量判读。"}
HEAD = {"e2": "炉况与配方:", "e4": "烧结批次状态:"}


def trim_one(txt):
    """NIR-SFT is verbose -> first sentence, drop list markers, <=28 chars."""
    txt = txt.strip().strip('"“”').replace("\r", "")
    txt = txt.split("\n")[0]
    txt = re.sub(r"^[\-\*0-9\.、)\s]+", "", txt)          # leading bullet/number
    m = re.split(r"[。;；]", txt)
    s = (m[0] if m and m[0] else txt).strip()
    if len(s) > 28:
        s = s[:28]
    if s and not re.search(r"[。!?！？]$", s):
        s += "。"
    return s


def call(role, desc, retries=3):
    body = json.dumps({
        "model": "nir", "temperature": 0.4, "max_tokens": 64, "stream": False,
        "messages": [{"role": "system", "content": SYS[role]},
                     {"role": "user", "content": HEAD[role] + desc + "\n" + ASK[role]}],
    }).encode("utf-8")
    for i in range(retries):
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode("utf-8"))
            return trim_one(d["choices"][0]["message"]["content"])
        except Exception as ex:
            if i == retries - 1:
                return f"__ERR__{ex}"
            time.sleep(2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=260)
    args = ap.parse_args()
    print(f"x5_coteacher start n={args.n}", flush=True)
    for role in ["e2", "e4"]:
        states = build_states(args.n, seed=77 + (0 if role == "e2" else 1))
        out = f"corpus_{role}_x5.jsonl"
        t0 = time.time(); n_ok = 0
        print(f"[{role}] begin {len(states)} states", flush=True)
        with open(out, "w", encoding="utf-8") as f:
            for i, s in enumerate(states):
                txt = call(role, describe(s))
                if txt and not txt.startswith("__ERR__") and 3 <= len(txt) <= 40:
                    f.write(json.dumps({"slots": s, "text": txt, "role": role}, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                if (i + 1) % 10 == 0:
                    print(f"[{role}] {i+1}/{len(states)} ok={n_ok} ({time.time()-t0:.0f}s)", flush=True)
        print(f"[{role}] DONE {n_ok}/{len(states)} -> {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
