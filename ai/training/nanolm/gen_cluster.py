"""
gen_cluster.py — build 5 role-specialized distillation corpora for the GD32
EDGE LLM CLUSTER (swap-loaded experts). One shared 12-slot furnace context
(reused from gen_corpus); each role gets a DIFFERENT system prompt so the
teacher produces a genuinely different KIND of sentence -> 5 distinct experts.

Roles (differentiated by TASK, not by teacher-architecture — that's the honest
way to get a real cluster; 5 same-task clones would be theatre):
  E1 diag    故障诊断      : why + what-to-do            (DeepSeek)
  E2 recipe  工艺/配方建议  : sintering profile tweak     (DeepSeek + offline NIR-SFT co-teacher)
  E3 energy  能耗/碳排效率  : efficiency note + saving     (DeepSeek)
  E4 qc      质量/批次判读  : good/suspect/scrap + cause  (DeepSeek + offline NIR-SFT co-teacher)
  E5 brief   操作/安全简报  : operator floor brief        (DeepSeek)

E2/E4 are co-taught by an offline Qwen2.5-1.5B NIR-SFT model (genuine "distil our
own offline model to the edge", done correctly = teacher generates training text,
not weight-cloning). That portion is produced by x5_coteacher.py run on the
teacher-model server and merged here.

Run:
  python gen_cluster.py --n 2400 --workers 24            # all 5 via DeepSeek
  python gen_cluster.py --role e3 --n 2400               # one role
  python gen_cluster.py --smoke                          # 2 calls/role, print
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from gen_corpus import build_states, describe, SLOT_ORDER, DEEPSEEK_KEY, DEEPSEEK_URL, MODEL

# ── 5 roles: system prompt + user-instruction tail. Same describe(state) input. ──
ROLES = {
    "e1": {
        "name": "diag", "cn": "故障诊断",
        "sys": ("你是部署在近红外荧光粉烧结炉旁的边缘AI诊断助手(运行在单片机上)。"
                "根据给定的实时炉况状态,用一句话给出最关键的诊断结论和操作建议。"
                "要求:不超过28个汉字;专业、具体、可执行;直接给结论,不解释推理;"
                "不要除标点外的英文字母和数字单位以外的英文;不要换行;不要加引号。"),
        "ask": "请给出一句话诊断与建议。", "teacher": "deepseek",
    },
    "e2": {
        "name": "recipe", "cn": "工艺配方建议",
        "sys": ("你是近红外荧光粉烧结工艺专家(部署在单片机边缘端)。根据炉况和配方,"
                "用一句话给出最关键的烧结工艺/配方优化建议(如升温速率、保温时长、"
                "烧结气氛、目标温度的调整)。要求:不超过28个汉字;面向该配方体系具体可执行;"
                "直接给建议;不要除标点外多余英文;不要换行;不要加引号;不要列条目。"),
        "ask": "请给出一句话工艺/配方优化建议。", "teacher": "both",
    },
    "e3": {
        "name": "energy", "cn": "能耗碳排",
        "sys": ("你是工业炉能效与碳排分析助手(部署在单片机边缘端)。根据炉况、加热占空比"
                "与能耗,用一句话给出能效/碳排评估与节能建议。要求:不超过28个汉字;"
                "具体可执行;直接给结论;不要除标点外多余英文;不要换行;不要加引号。"),
        "ask": "请给出一句话能效评估与节能建议。", "teacher": "deepseek",
    },
    "e4": {
        "name": "qc", "cn": "质量批次判读",
        "sys": ("你是近红外荧光粉质量判读助手(部署在单片机边缘端)。根据烧结过程与结果,"
                "用一句话给出该批次的质量判读(良品/疑似废品/废品)及主要原因。"
                "要求:不超过28个汉字;直接给判读结论;不要除标点外多余英文;不要换行;不要加引号。"),
        "ask": "请给出一句话批次质量判读。", "teacher": "both",
    },
    "e5": {
        "name": "brief", "cn": "操作简报",
        "sys": ("你是炉前操作简报助手(部署在单片机边缘端)。根据当前炉况,用一句话给出"
                "面向操作员的现场简报(当前该做什么、需要注意什么)。要求:不超过28个汉字;"
                "口吻像现场提示;直接给简报;不要除标点外多余英文;不要换行;不要加引号。"),
        "ask": "请给出一句话现场操作简报。", "teacher": "deepseek",
    },
    "e6": {
        "name": "chem", "cn": "配方化学",
        "sys": ("你是近红外荧光粉配方化学专家(部署在单片机边缘端)。根据当前基质体系与炉况,"
                "用一句话给出最关键的配方化学建议(如激活离子价态与格位、基质取代与电荷补偿、"
                "预期近红外发射特征),只谈配方化学不谈烧结工艺参数。要求:不超过28个汉字;"
                "全部用中文表述,价态写成'三价铬''二价镍'这类中文,不要元素符号、上标、加号等特殊字符;"
                "面向该基质体系具体可执行;直接给建议;不要除标点外的任何英文字母数字;不要换行;不要加引号;不要列条目。"),
        "ask": "请给出一句话配方化学建议。", "teacher": "deepseek",
    },
    "e7": {
        "name": "maint", "cn": "设备维护",
        "sys": ("你是工业烧结炉设备维护与预测性维护助手(部署在单片机边缘端)。只关注设备健康"
                "(加热元件、热电偶、传动振动、能耗),用一句话给出最关键的设备健康判读与检修建议"
                "(如加热元件更换、热电偶校验、传动紧固、停炉检修时机);若故障与设备无关则回复设备正常无需检修。"
                "要求:不超过28个汉字;全部用中文表述;不要除标点外的任何英文字母数字与特殊符号;"
                "具体可执行;直接给结论;不要换行;不要加引号。"),
        "ask": "请给出一句话设备维护与检修建议。", "teacher": "deepseek",
    },
}
ROLE_ORDER = ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
USER_HEAD = {"e1": "炉况状态:", "e2": "炉况与配方:", "e3": "炉况与能耗:",
             "e4": "烧结批次状态:", "e5": "当前炉况:", "e6": "基质与炉况:", "e7": "设备状态:"}


def call_deepseek_role(desc, sys_prompt, ask, temperature=0.5, retries=4, head="炉况状态:"):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": head + desc + "\n" + ask}],
        "temperature": temperature, "max_tokens": 80, "stream": False,
    }).encode("utf-8")
    for i in range(retries):
        try:
            req = urllib.request.Request(DEEPSEEK_URL, data=body, headers={
                "Content-Type": "application/json", "Authorization": f"Bearer {DEEPSEEK_KEY}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            txt = d["choices"][0]["message"]["content"].strip()
            return txt.replace("\n", "").replace("\r", "").strip().strip('"“”')
        except Exception as ex:
            if i == retries - 1:
                return f"__ERR__{ex}"
            time.sleep(1.5 * (i + 1))


_SUP_BAD = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼ºª"   # superscripts / special that bloat vocab & font


def _is_clean_cn(txt):
    """Drop sentences carrying ASCII letters or superscript/special chars so the
    shared cluster vocab + CJK font stay small and LCD-renderable (pure Chinese +
    standard punctuation + plain digits, matching E1-E5)."""
    for c in txt:
        if ("a" <= c <= "z") or ("A" <= c <= "Z"):
            return False
        if c in _SUP_BAD:
            return False
    return True


def gen_role(role, n, workers, out):
    r = ROLES[role]
    states = build_states(n, seed=20260604 + ROLE_ORDER.index(role))
    print(f"[{role}/{r['name']}] {len(states)} states -> DeepSeek x{workers}")
    rows = [None] * len(states)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(call_deepseek_role, describe(s), r["sys"], r["ask"],
                          0.5, 4, USER_HEAD[role]): i for i, s in enumerate(states)}
        for fut in as_completed(futs):
            i = futs[fut]; rows[i] = (states[i], fut.result()); done += 1
            if done % 300 == 0:
                print(f"  [{role}] {done}/{len(states)} ({time.time()-t0:.0f}s)")
    n_ok = 0
    with open(out, "w", encoding="utf-8") as f:
        for st, txt in rows:
            if not txt or txt.startswith("__ERR__") or len(txt) > 60 or not _is_clean_cn(txt):
                continue
            f.write(json.dumps({"slots": st, "text": txt, "role": role}, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[{role}] {n_ok}/{len(states)} ok -> {out} ({time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--role", default="all")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        for role in ROLE_ORDER:
            r = ROLES[role]; sts = build_states(2, seed=ROLE_ORDER.index(role))
            print(f"\n#### {role} {r['cn']} (teacher={r['teacher']}) ####")
            for s in sts:
                d = describe(s)
                print("  ", call_deepseek_role(d, r["sys"], r["ask"], head=USER_HEAD[role]))
        return

    roles = ROLE_ORDER if args.role == "all" else [args.role]
    for role in roles:
        gen_role(role, args.n, args.workers, f"corpus_{role}.jsonl")


if __name__ == "__main__":
    main()
