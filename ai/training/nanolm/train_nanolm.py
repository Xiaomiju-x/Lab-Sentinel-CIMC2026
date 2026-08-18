"""
train_nanolm.py — train the GD32 edge nano-LM (decoder-only GPT) on the
DeepSeek distillation corpus.

Architecture (sized to run INT8 on a Cortex-M7 @600MHz, no NPU):
  decoder-only GPT, pre-LayerNorm, learned absolute pos, tied embeddings, GELU.
  d_model=128, n_layers=3, n_heads=4, d_ff=512, max_seq=96  -> ~0.6M params
  INT8 weight-only ~0.6MB Flash; ~0.7-1.3s per short Chinese diagnosis on-chip.

Tokenization (contract with firmware ai_nanolm.build_context):
  vocab = [<pad><bos><sep><eos>] + control tokens (one per slot/option, from
  gen_corpus.SLOTS) + output chars (every char seen in teacher sentences).
  sequence = <bos> [12 control tokens, SLOT_ORDER] <sep> [diagnosis chars] <eos>
  loss masked to the generated (post-<sep>) part = standard SFT.

Outputs (downloaded + fed to export_nanolm_to_c.py):
  nanolm.pt        trained weights + config
  vocab.json       full vocab + control-token ids per slot (single source of truth)
  samples.txt      greedy generations on held-out states (eyeball coherence)

Run (5090):  python3 train_nanolm.py --corpus corpus.jsonl --epochs 60
"""
import argparse
import json
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from gen_corpus import SLOTS, SLOT_ORDER

D_MODEL, N_LAYERS, N_HEADS, D_FF, MAX_SEQ, DROPOUT = 128, 3, 4, 512, 96, 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── vocab ──────────────────────────────────────────────────────────────────
def build_vocab(rows):
    tok2id, id2tok = {}, []

    def add(t):
        if t not in tok2id:
            tok2id[t] = len(id2tok)
            id2tok.append(t)

    for s in ["<pad>", "<bos>", "<sep>", "<eos>"]:
        add(s)
    # control tokens: one per slot/option, in SLOT_ORDER then option order
    control = {}
    for slot in SLOT_ORDER:
        control[slot] = {}
        for opt in SLOTS[slot].keys():
            name = f"@{slot}={opt}"
            add(name)
            control[slot][opt] = tok2id[name]
    # output chars (sorted for determinism)
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
    sep_pos = len(ids)          # position of <sep>
    ids.append(tok2id["<sep>"])
    for ch in row["text"]:
        ids.append(tok2id[ch])
    ids.append(tok2id["<eos>"])
    return ids, sep_pos


# ── model ──────────────────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.q = nn.Linear(D_MODEL, D_MODEL)
        self.k = nn.Linear(D_MODEL, D_MODEL)
        self.v = nn.Linear(D_MODEL, D_MODEL)
        self.o = nn.Linear(D_MODEL, D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ff1 = nn.Linear(D_MODEL, D_FF)
        self.ff2 = nn.Linear(D_FF, D_MODEL)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x, mask):
        B, T, _ = x.shape
        h = self.ln1(x)
        q = self.q(h).view(B, T, N_HEADS, D_MODEL // N_HEADS).transpose(1, 2)
        k = self.k(h).view(B, T, N_HEADS, D_MODEL // N_HEADS).transpose(1, 2)
        v = self.v(h).view(B, T, N_HEADS, D_MODEL // N_HEADS).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(D_MODEL // N_HEADS))
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D_MODEL)
        x = x + self.drop(self.o(y))
        h = self.ln2(x)
        x = x + self.drop(self.ff2(F.gelu(self.ff1(h))))
        return x


class NanoLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.tok = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Embedding(MAX_SEQ, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.drop = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(MAX_SEQ, MAX_SEQ)).view(1, 1, MAX_SEQ, MAX_SEQ)
        self.register_buffer("mask", mask)
        # tied head
        self.head = nn.Linear(D_MODEL, vocab, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        for b in self.blocks:
            x = b(x, self.mask)
        return self.head(self.lnf(x))


# ── data ───────────────────────────────────────────────────────────────────
class DS(torch.utils.data.Dataset):
    def __init__(self, rows, tok2id, control):
        self.samples = []
        pad = tok2id["<pad>"]
        for r in rows:
            ids, sep = encode(r, tok2id, control)
            if len(ids) > MAX_SEQ:
                ids = ids[:MAX_SEQ]
            x = ids[:-1]
            y = ids[1:]
            # mask everything up to and including <sep> prediction target
            y = [(-100 if i < sep else t) for i, t in enumerate(y)]
            n = len(x)
            x = x + [pad] * (MAX_SEQ - n)
            y = y + [-100] * (MAX_SEQ - n)
            self.samples.append((torch.tensor(x), torch.tensor(y)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


@torch.no_grad()
def generate(model, ctx_ids, tok2id, id2tok, max_new=40):
    model.eval()
    ids = list(ctx_ids)
    eos = tok2id["<eos>"]
    for _ in range(max_new):
        x = torch.tensor([ids[-MAX_SEQ:]], device=DEVICE)
        logits = model(x)[0, -1]
        nxt = int(torch.argmax(logits))
        if nxt == eos:
            break
        ids.append(nxt)
    out = ids[ctx_ids.index(tok2id["<sep>"]) + 1:]
    return "".join(id2tok[i] for i in out if id2tok[i] not in ("<pad>", "<bos>", "<sep>", "<eos>"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.corpus, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    n_val = max(40, len(rows) // 12)
    val_rows, tr_rows = rows[:n_val], rows[n_val:]
    tok2id, id2tok, control = build_vocab(rows)
    V = len(id2tok)
    print(f"corpus {len(rows)} (train {len(tr_rows)} / val {len(val_rows)})  vocab {V}  "
          f"(specials 4 + control {sum(len(c) for c in control.values())} + chars {V-4-sum(len(c) for c in control.values())})")

    tr = torch.utils.data.DataLoader(DS(tr_rows, tok2id, control), batch_size=args.bs, shuffle=True)
    va = torch.utils.data.DataLoader(DS(val_rows, tok2id, control), batch_size=args.bs)

    model = NanoLM(V).to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # tied head shares tok weight -> count once
    n_uniq = n_param - model.head.weight.numel()
    print(f"params: {n_uniq/1e6:.3f}M unique  (INT8 ~{n_uniq/1e6:.2f}MB Flash)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=-100)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * x.size(0)
        # val
        model.eval(); vtot = 0.0; vn = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                l = F.cross_entropy(model(x).view(-1, V), y.view(-1), ignore_index=-100)
                vtot += l.item() * x.size(0); vn += x.size(0)
        vl = vtot / vn
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"ep {ep+1:3d}  train {tot/len(tr_rows):.4f}  val {vl:.4f}  ppl {math.exp(vl):.2f}")
        if vl < best:
            best = vl
            torch.save({"model": model.state_dict(), "vocab": V,
                        "cfg": {"d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                                "d_ff": D_FF, "max_seq": MAX_SEQ}}, "nanolm.pt")

    # save vocab.json (single source of truth)
    json.dump({
        "id2tok": id2tok,
        "tok2id": tok2id,
        "control": control,                 # slot -> opt -> id
        "slot_order": SLOT_ORDER,
        "specials": {s: tok2id[s] for s in ["<pad>", "<bos>", "<sep>", "<eos>"]},
        "cfg": {"d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                "d_ff": D_FF, "max_seq": MAX_SEQ, "vocab": V},
    }, open("vocab.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # reload best + generate samples on held-out states
    model.load_state_dict(
        torch.load("nanolm.pt", map_location=DEVICE, weights_only=True)["model"]
    )
    model.eval()
    lines = []
    for r in val_rows[:40]:
        ids, _ = encode(r, tok2id, control)
        ctx = ids[:ids.index(tok2id["<sep>"]) + 1]
        gen = generate(model, ctx, tok2id, id2tok)
        slot_str = " ".join(f"{k}={r['slots'][k]}" for k in ["stage", "risk", "gas", "tc", "ramp"])
        lines.append(f"[{slot_str}]\n  teacher: {r['text']}\n  nanolm : {gen}")
    open("samples.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n=== sample generations (held-out) ===")
    print("\n".join(lines[:12]))
    print(f"\nbest val loss {best:.4f}  ppl {math.exp(best):.2f}  -> nanolm.pt / vocab.json / samples.txt")


if __name__ == "__main__":
    main()
