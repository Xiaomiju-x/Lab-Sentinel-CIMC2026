"""
train_cluster.py — train the 5 EDGE LLM CLUSTER experts on ONE SHARED VOCAB.

Why shared vocab: the firmware then has a single tokenizer + a single CJK font
subset, and every swap-loaded expert blob has the identical layout (only the
weights differ) — that's what makes clean swap-load possible. Each expert is the
same ~0.6M nano-LM arch (decoder GPT d128/3L/4H/FF512/seq96), trained on its own
role corpus but indexing the shared vocab built from the UNION of all 5 corpora.

E2/E4 merge the offline NIR-SFT co-teacher rows (corpus_e{2,4}_x5.jsonl) if present.

Outputs (-> cluster/):
  expert_e1.pt .. expert_e5.pt   per-role weights (shared V)
  cluster_vocab.json             the single shared vocab (source of truth)
  cluster_samples.txt            held-out generations per role (eyeball)

Run (RTX 4060):  python train_cluster.py --epochs 60
"""
import argparse
import json
import math
import os
import random

import torch
import torch.nn.functional as F

from train_nanolm import NanoLM, build_vocab, encode, DS, generate, MAX_SEQ, DEVICE
from gen_cluster import ROLE_ORDER, ROLES

OUT = "cluster"


def load_role_rows(role):
    rows = []
    for fn in (f"corpus_{role}.jsonl", f"corpus_{role}_x5.jsonl"):
        if os.path.exists(fn):
            for line in open(fn, encoding="utf-8"):
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def train_one(role, rows, tok2id, id2tok, control, V, epochs, bs, lr):
    random.Random(0).shuffle(rows)
    n_val = max(30, len(rows) // 12)
    val_rows, tr_rows = rows[:n_val], rows[n_val:]
    tr = torch.utils.data.DataLoader(DS(tr_rows, tok2id, control), batch_size=bs, shuffle=True)
    va = torch.utils.data.DataLoader(DS(val_rows, tok2id, control), batch_size=bs)

    model = NanoLM(V).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * max(1, len(tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)

    best = 1e9
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        model.eval(); vtot = 0.0; vn = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                l = F.cross_entropy(model(x).view(-1, V), y.view(-1), ignore_index=-100)
                vtot += l.item() * x.size(0); vn += x.size(0)
        vl = vtot / max(1, vn)
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "vocab": V}, os.path.join(OUT, f"expert_{role}.pt"))
    # samples
    model.load_state_dict(
        torch.load(
            os.path.join(OUT, f"expert_{role}.pt"),
            weights_only=True,
        )["model"]
    ); model.eval()
    lines = [f"==== {role} {ROLES[role]['cn']}  best_val {best:.4f} ppl {math.exp(best):.2f} ===="]
    for r in val_rows[:8]:
        ids, _ = encode(r, tok2id, control)
        ctx = ids[:ids.index(tok2id["<sep>"]) + 1]
        gen = generate(model, ctx, tok2id, id2tok)
        slot_str = " ".join(f"{k}={r['slots'][k]}" for k in ["stage", "risk", "gas", "host"])
        lines.append(f"[{slot_str}]\n  teacher: {r['text']}\n  expert : {gen}")
    return best, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    role_rows = {r: load_role_rows(r) for r in ROLE_ORDER}
    all_rows = [row for r in ROLE_ORDER for row in role_rows[r]]
    if not all_rows:
        raise SystemExit("no corpora found - run gen_cluster.py first")
    # ONE shared vocab from the union of all 5 corpora
    tok2id, id2tok, control = build_vocab(all_rows)
    V = len(id2tok)
    nctrl = sum(len(c) for c in control.values())
    print(f"shared vocab V={V} (specials 4 + control {nctrl} + chars {V-4-nctrl})")
    for r in ROLE_ORDER:
        print(f"  {r} {ROLES[r]['cn']}: {len(role_rows[r])} rows")

    # persist shared vocab (single source of truth for export + firmware)
    json.dump({"id2tok": id2tok, "tok2id": tok2id, "control": control,
               "slot_order": __import__("gen_corpus").SLOT_ORDER,
               "specials": {s: tok2id[s] for s in ["<pad>", "<bos>", "<sep>", "<eos>"]},
               "roles": ROLE_ORDER, "vocab": V},
              open(os.path.join(OUT, "cluster_vocab.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    all_lines = []
    summary = []
    for role in ROLE_ORDER:
        rows = role_rows[role]
        if len(rows) < 60:
            print(f"!! {role}: only {len(rows)} rows, skipping"); continue
        best, lines = train_one(role, rows, tok2id, id2tok, control, V, args.epochs, args.bs, args.lr)
        all_lines += lines + [""]
        summary.append((role, best, math.exp(best), len(rows)))
        print(f"[{role}] best_val {best:.4f} ppl {math.exp(best):.2f}")

    open(os.path.join(OUT, "cluster_samples.txt"), "w", encoding="utf-8").write("\n".join(all_lines))
    print("\n=== CLUSTER TRAIN SUMMARY ===")
    for role, bv, ppl, n in summary:
        print(f"  {role} {ROLES[role]['cn']:<8} rows={n:<5} val_loss {bv:.4f}  ppl {ppl:.2f}")
    print(f"shared vocab V={V}  -> {OUT}/expert_e*.pt + cluster_vocab.json")


if __name__ == "__main__":
    main()
