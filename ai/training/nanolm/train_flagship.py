"""
train_flagship.py — parametrized trainer for the FLAGSHIP edge LM size sweep.

Same architecture/contract as train_nanolm.py (decoder-only GPT, slot control
tokens -> Chinese diagnosis, loss masked to post-<sep>), but the size is passed
on the CLI so we can sweep configs on ONE corpus and pick — measure-first — the
biggest one that still (a) beats the 0.6M baseline's val ppl and (b) fits the
M7 latency/flash budget. The per-config (params, val ppl, est latency) rows ARE
the "hardware-ceiling curve" deliverable.

The C engine ai_nanolm.c is config-driven, so the winning config just changes a
few macros + the INT8 weight header — no firmware rewrite.

Run (RTX 4060, mace_env torch):
  # baseline reproduce
  python train_flagship.py --corpus corpus_v2.jsonl --tag s0p6 \
        --d_model 128 --n_layers 3 --n_heads 4 --d_ff 512 --max_seq 96
  # safe flagship
  python train_flagship.py --corpus corpus_v2.jsonl --tag m1p35 \
        --d_model 160 --n_layers 4 --n_heads 5 --d_ff 640 --max_seq 128
  # ceiling flagship
  python train_flagship.py --corpus corpus_v2.jsonl --tag x1p9 \
        --d_model 192 --n_layers 4 --n_heads 6 --d_ff 768 --max_seq 128
"""
import argparse
import json
import math
import random
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_corpus import SLOTS, SLOT_ORDER

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


def parse_tag(value: str) -> str:
    """Accept a filename-safe experiment tag, never a path."""
    if TAG_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "tag must be 1-64 ASCII letters, digits, dots, underscores, or hyphens"
        )
    return value


# ── vocab (identical scheme to train_nanolm) ────────────────────────────────
def build_vocab(rows):
    tok2id, id2tok = {}, []

    def add(t):
        if t not in tok2id:
            tok2id[t] = len(id2tok)
            id2tok.append(t)

    for s in ["<pad>", "<bos>", "<sep>", "<eos>"]:
        add(s)
    control = {}
    for slot in SLOT_ORDER:
        control[slot] = {}
        for opt in SLOTS[slot].keys():
            add(f"@{slot}={opt}")
            control[slot][opt] = tok2id[f"@{slot}={opt}"]
    chars = set()
    for r in rows:
        chars.update(r["text"])
    for ch in sorted(chars):
        add(ch)
    return tok2id, id2tok, control


def encode(row, tok2id, control):
    ids = [tok2id["<bos>"]]
    for slot in SLOT_ORDER:
        ids.append(control[slot][row["slots"][slot]])
    sep_pos = len(ids)
    ids.append(tok2id["<sep>"])
    for ch in row["text"]:
        ids.append(tok2id[ch])
    ids.append(tok2id["<eos>"])
    return ids, sep_pos


# ── model (config passed in, not module constants) ──────────────────────────
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, h, ff = cfg["d_model"], cfg["n_heads"], cfg["d_ff"]
        self.h = h
        self.ln1 = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, ff); self.ff2 = nn.Linear(ff, d)
        self.drop = nn.Dropout(cfg["dropout"])
        self.dh = d // h

    def forward(self, x, mask):
        B, T, D = x.shape
        z = self.ln1(x)
        q = self.q(z).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(z).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(z).view(B, T, self.h, self.dh).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.dh))
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.drop(self.o(y))
        z = self.ln2(x)
        x = x + self.drop(self.ff2(F.gelu(self.ff1(z))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        d, L, S = cfg["d_model"], cfg["n_layers"], cfg["max_seq"]
        self.cfg = cfg
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(S, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.drop = nn.Dropout(cfg["dropout"])
        self.register_buffer("mask", torch.tril(torch.ones(S, S)).view(1, 1, S, S))
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight   # tied

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        for b in self.blocks:
            x = b(x, self.mask)
        return self.head(self.lnf(x))


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, tok2id, control, S):
        self.samples = []
        pad = tok2id["<pad>"]
        for r in rows:
            ids, sep = encode(r, tok2id, control)
            if len(ids) > S:
                ids = ids[:S]
            x, y = ids[:-1], ids[1:]
            y = [(-100 if i < sep else t) for i, t in enumerate(y)]
            n = len(x)
            x = x + [pad] * (S - n)
            y = y + [-100] * (S - n)
            self.samples.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


@torch.no_grad()
def generate(model, ctx, tok2id, id2tok, S, max_new=48):
    model.eval()
    ids = list(ctx)
    eos = tok2id["<eos>"]
    for _ in range(max_new):
        x = torch.tensor([ids[-S:]], device=DEVICE)
        nxt = int(torch.argmax(model(x)[0, -1]))
        if nxt == eos:
            break
        ids.append(nxt)
    out = ids[ctx.index(tok2id["<sep>"]) + 1:]
    skip = ("<pad>", "<bos>", "<sep>", "<eos>")
    return "".join(id2tok[i] for i in out if id2tok[i] not in skip)


def latency_factor(vocab, cfg):
    """Per-token compute relative to the 0.6M baseline (d128/3L/512)."""
    def cost(v, c):
        d, L, ff = c["d_model"], c["n_layers"], c["d_ff"]
        return L * (4 * d * d + 2 * d * ff) + v * d   # attn proj + ffn + tied head
    base = cost(452, {"d_model": 128, "n_layers": 3, "d_ff": 512})
    return cost(vocab, cfg) / base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus_v2.jsonl")
    ap.add_argument("--tag", required=True, type=parse_tag)
    ap.add_argument("--d_model", type=int, default=192)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=768)
    ap.add_argument("--max_seq", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    cfg = {"d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads,
           "d_ff": args.d_ff, "max_seq": args.max_seq, "dropout": args.dropout}

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    n_val = max(60, len(rows) // 12)
    val_rows, tr_rows = rows[:n_val], rows[n_val:]
    tok2id, id2tok, control = build_vocab(rows)
    V = len(id2tok)

    tr = torch.utils.data.DataLoader(DS(tr_rows, tok2id, control, args.max_seq),
                                     batch_size=args.bs, shuffle=True)
    va = torch.utils.data.DataLoader(DS(val_rows, tok2id, control, args.max_seq),
                                     batch_size=args.bs)

    model = GPT(V, cfg).to(DEVICE)
    n_uniq = sum(p.numel() for p in model.parameters()) - model.head.weight.numel()
    lf = latency_factor(V, cfg)
    print(f"[{args.tag}] vocab {V}  cfg {cfg}")
    print(f"[{args.tag}] params {n_uniq/1e6:.3f}M  INT8 ~{n_uniq/1e6:.2f}MB  "
          f"est latency x{lf:.2f} vs 0.6M baseline")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    best = 1e9
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * x.size(0)
        model.eval(); vtot = vn = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                vtot += F.cross_entropy(model(x).view(-1, V), y.view(-1),
                                        ignore_index=-100).item() * x.size(0)
                vn += x.size(0)
        vl = vtot / vn
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[{args.tag}] ep {ep+1:3d}  train {tot/len(tr_rows):.4f}  "
                  f"val {vl:.4f}  ppl {math.exp(vl):.2f}")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "vocab": V, "cfg": cfg,
                        "tok2id": tok2id, "id2tok": id2tok, "control": control},
                       f"flagship_{args.tag}.pt")

    # held-out sample generations (eyeball richness)
    model.load_state_dict(
        torch.load(
            f"flagship_{args.tag}.pt",
            map_location=DEVICE,
            weights_only=True,
        )["model"]
    )
    model.eval()
    lines = []
    for r in val_rows[:16]:
        ids, _ = encode(r, tok2id, control)
        ctx = ids[:ids.index(tok2id["<sep>"]) + 1]
        gen = generate(model, ctx, tok2id, id2tok, args.max_seq)
        sl = " ".join(f"{k}={r['slots'][k]}" for k in ["stage", "risk", "gas", "tc", "ramp"])
        lines.append(f"[{sl}]\n  teacher: {r['text']}\n  flagship: {gen}")
    open(f"samples_{args.tag}.txt", "w", encoding="utf-8").write("\n".join(lines))
    print(f"\n=== [{args.tag}] held-out samples ===")
    print("\n".join(lines[:8]))
    print(f"\n[{args.tag}] RESULT  params {n_uniq/1e6:.3f}M  best_val {best:.4f}  "
          f"ppl {math.exp(best):.2f}  est_latency x{lf:.2f}")


if __name__ == "__main__":
    main()
