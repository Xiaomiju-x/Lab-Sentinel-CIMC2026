"""
export_nanolm_to_c.py — quantize the trained nano-LM to INT8 and emit the C
headers for the GD32 firmware, plus a golden test vector.

Why INT8 here (and only here): fp32 weights would be ~2.4MB Flash (won't fit
alongside 20 models + LVGL); INT8 weight-only is ~0.7MB and FITS. This is the
"INT8 belongs where it pays" decision (vs blanket-quantizing the 20 working
models, which buys nothing on this M7). Scheme matches the project's existing
B1 `nn_linear_int8`: per-output-row symmetric INT8 weights, fp32 bias, fp32
compute (M7 has no INT8 SIMD -> the win is Flash/D-cache, honestly labelled).

Golden contract: we dequantize the INT8 weights back to fp32, load them into a
torch model (= the EXACT deployed model), and (a) greedy-generate demo diagnoses
and (b) dump last-position logits for a fixed prefix. The C engine must
reproduce both. So the host test proves "C runs the deployed INT8 model", not
just "C compiles".

Outputs (-> ../../firmware/ai_models_c/):
  nanolm_weights.h   INT8 weights + fp32 scales/biases/LayerNorm/pos
  nanolm_vocab.h     id->UTF8 detok table + control-token ids + dims
  nanolm_golden.h    demo contexts, expected greedy token ids, ref logits

Run (5090, after train):  python3 export_nanolm_to_c.py
"""
import json
from pathlib import Path

import numpy as np
import torch

from train_nanolm import NanoLM, D_MODEL, N_LAYERS, N_HEADS, D_FF, MAX_SEQ

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "firmware" / "ai_models_c"
OUT.mkdir(parents=True, exist_ok=True)


def quant_rows(W):
    """Per-output-row symmetric INT8. W[out,in] -> (q int8[out,in], scale f32[out])."""
    W = W.detach().cpu().numpy().astype(np.float32)
    amax = np.maximum(np.abs(W).max(axis=1), 1e-12)
    scale = amax / 127.0
    q = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
    return q, scale.astype(np.float32)


# ── C emit helpers ───────────────────────────────────────────────────────────
def ci8(name, arr):
    a = np.asarray(arr, dtype=np.int8).reshape(-1)
    return f"static const signed char {name}[{a.size}] = {{{','.join(str(int(v)) for v in a)}}};\n"


def cf32(name, arr):
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    return f"static const float {name}[{a.size}] = {{{','.join(f'{float(v):.7e}f' for v in a)}}};\n"


def ci8_2d(name, mats):
    rows = ",\n".join("{" + ",".join(str(int(v)) for v in np.asarray(m, np.int8).reshape(-1)) + "}" for m in mats)
    n = np.asarray(mats[0], np.int8).size
    return f"static const signed char {name}[{len(mats)}][{n}] = {{\n{rows}\n}};\n"


def cf32_2d(name, mats):
    rows = ",\n".join("{" + ",".join(f"{float(v):.7e}f" for v in np.asarray(m, np.float32).reshape(-1)) + "}" for m in mats)
    n = np.asarray(mats[0], np.float32).size
    return f"static const float {name}[{len(mats)}][{n}] = {{\n{rows}\n}};\n"


def main():
    voc = json.load(open(HERE / "vocab.json", encoding="utf-8"))
    V = voc["cfg"]["vocab"]
    id2tok = voc["id2tok"]
    ck = torch.load(HERE / "nanolm.pt", map_location="cpu", weights_only=True)
    model = NanoLM(V)
    model.load_state_dict(ck["model"])
    model.eval()

    sd = model.state_dict()

    # ── quantize: tok emb (per-row) + every Linear (per-out-row) ──────────────
    tok_q, tok_s = quant_rows(model.tok.weight)               # [V,D]
    pos = model.pos.weight.detach().cpu().numpy().astype(np.float32)  # [MAXSEQ,D] fp32

    def lin(name):
        return quant_rows(sd[name + ".weight"]), sd[name + ".bias"].detach().cpu().numpy().astype(np.float32)

    layers = {k: [] for k in ["ln1g", "ln1b", "qq", "qs", "qb", "kq", "ks", "kb",
                              "vq", "vs", "vb", "oq", "os", "ob", "ln2g", "ln2b",
                              "f1q", "f1s", "f1b", "f2q", "f2s", "f2b"]}
    for l in range(N_LAYERS):
        p = f"blocks.{l}."
        layers["ln1g"].append(sd[p + "ln1.weight"].numpy()); layers["ln1b"].append(sd[p + "ln1.bias"].numpy())
        for src, dst in [("q", "q"), ("k", "k"), ("v", "v"), ("o", "o")]:
            (q, s), b = lin(p + src)
            layers[dst + "q"].append(q); layers[dst + "s"].append(s); layers[dst + "b"].append(b)
        layers["ln2g"].append(sd[p + "ln2.weight"].numpy()); layers["ln2b"].append(sd[p + "ln2.bias"].numpy())
        (q, s), b = lin(p + "ff1"); layers["f1q"].append(q); layers["f1s"].append(s); layers["f1b"].append(b)
        (q, s), b = lin(p + "ff2"); layers["f2q"].append(q); layers["f2s"].append(s); layers["f2b"].append(b)
    lnf_g = sd["lnf.weight"].numpy(); lnf_b = sd["lnf.bias"].numpy()

    # ── build the DEPLOYED model (dequantized) for golden references ──────────
    deq = NanoLM(V)
    dsd = deq.state_dict()
    dsd["tok.weight"] = torch.tensor(tok_q.astype(np.float32) * tok_s[:, None])
    dsd["pos.weight"] = torch.tensor(pos)
    for l in range(N_LAYERS):
        p = f"blocks.{l}."
        dsd[p + "ln1.weight"] = torch.tensor(layers["ln1g"][l]); dsd[p + "ln1.bias"] = torch.tensor(layers["ln1b"][l])
        for d in ["q", "k", "v", "o"]:
            dsd[p + d + ".weight"] = torch.tensor(layers[d + "q"][l].astype(np.float32) * layers[d + "s"][l][:, None])
            dsd[p + d + ".bias"] = torch.tensor(layers[d + "b"][l])
        dsd[p + "ln2.weight"] = torch.tensor(layers["ln2g"][l]); dsd[p + "ln2.bias"] = torch.tensor(layers["ln2b"][l])
        dsd[p + "ff1.weight"] = torch.tensor(layers["f1q"][l].astype(np.float32) * layers["f1s"][l][:, None])
        dsd[p + "ff1.bias"] = torch.tensor(layers["f1b"][l])
        dsd[p + "ff2.weight"] = torch.tensor(layers["f2q"][l].astype(np.float32) * layers["f2s"][l][:, None])
        dsd[p + "ff2.bias"] = torch.tensor(layers["f2b"][l])
    dsd["lnf.weight"] = torch.tensor(lnf_g); dsd["lnf.bias"] = torch.tensor(lnf_b)
    deq.load_state_dict(dsd); deq.eval()

    bos, sep, eos = voc["specials"]["<bos>"], voc["specials"]["<sep>"], voc["specials"]["<eos>"]
    control = voc["control"]
    SLOT_ORDER = voc["slot_order"]

    def ctx_for(state):
        ids = [bos] + [control[s][state[s]] for s in SLOT_ORDER] + [sep]
        return ids

    @torch.no_grad()
    def greedy(state, max_new=40):
        ids = ctx_for(state)
        for _ in range(max_new):
            x = torch.tensor([ids[-MAX_SEQ:]])
            nxt = int(torch.argmax(deq(x)[0, -1]))
            if nxt == eos:
                break
            ids.append(nxt)
        return ids

    # demo states for golden + on-chip self-test (cover normal + 2 faults)
    demos = [
        {"stage": "sinter", "temp": "over", "risk": "crit", "ramp": "ok", "drift": "hi",
         "tc": "ok", "gas": "cr6", "ae": "anom", "vib": "ok", "energy": "ok", "host": "yag", "elem": "ok"},
        {"stage": "sinter", "temp": "rt", "risk": "crit", "ramp": "ok", "drift": "ok",
         "tc": "open", "gas": "ok", "ae": "ok", "vib": "ok", "energy": "ok", "host": "gagg", "elem": "ok"},
        {"stage": "idle", "temp": "rt", "risk": "good", "ramp": "ok", "drift": "ok",
         "tc": "ok", "gas": "ok", "ae": "ok", "vib": "ok", "energy": "ok", "host": "yag", "elem": "ok"},
    ]
    gens = [greedy(d) for d in demos]
    # numeric golden: last-position logits for demo[0] full greedy prefix
    with torch.no_grad():
        gx = torch.tensor([gens[0][:MAX_SEQ]])
        glog = deq(gx)[0, -1].numpy().astype(np.float32)

    print("=== deployed (INT8-dequant) greedy demos ===")
    for d, g in zip(demos, gens):
        txt = "".join(id2tok[i] for i in g[g.index(sep) + 1:] if i not in (bos, sep, eos))
        print(f"  {d['stage']}/{d['risk']}/{d.get('gas')}/{d.get('tc')}: {txt}")

    # ── emit nanolm_weights.h ─────────────────────────────────────────────────
    w = ("/* nanolm_weights.h - AUTO-GENERATED by export_nanolm_to_c.py. Do not edit.\n"
         " * GD32 edge nano-LM: INT8 weight-only (per-row symmetric) + fp32 scales/bias/LN/pos.\n"
         " * Distilled from DeepSeek. Weights stored in Flash (.rodata). */\n"
         "#ifndef NANOLM_WEIGHTS_H\n#define NANOLM_WEIGHTS_H\n\n")
    w += ci8("nlm_tok_q", tok_q) + cf32("nlm_tok_s", tok_s)
    w += cf32("nlm_pos", pos)
    w += ci8_2d("nlm_q_q", layers["qq"]) + cf32_2d("nlm_q_s", layers["qs"]) + cf32_2d("nlm_q_b", layers["qb"])
    w += ci8_2d("nlm_k_q", layers["kq"]) + cf32_2d("nlm_k_s", layers["ks"]) + cf32_2d("nlm_k_b", layers["kb"])
    w += ci8_2d("nlm_v_q", layers["vq"]) + cf32_2d("nlm_v_s", layers["vs"]) + cf32_2d("nlm_v_b", layers["vb"])
    w += ci8_2d("nlm_o_q", layers["oq"]) + cf32_2d("nlm_o_s", layers["os"]) + cf32_2d("nlm_o_b", layers["ob"])
    w += ci8_2d("nlm_f1_q", layers["f1q"]) + cf32_2d("nlm_f1_s", layers["f1s"]) + cf32_2d("nlm_f1_b", layers["f1b"])
    w += ci8_2d("nlm_f2_q", layers["f2q"]) + cf32_2d("nlm_f2_s", layers["f2s"]) + cf32_2d("nlm_f2_b", layers["f2b"])
    w += cf32_2d("nlm_ln1_g", layers["ln1g"]) + cf32_2d("nlm_ln1_b", layers["ln1b"])
    w += cf32_2d("nlm_ln2_g", layers["ln2g"]) + cf32_2d("nlm_ln2_b", layers["ln2b"])
    w += cf32("nlm_lnf_g", lnf_g) + cf32("nlm_lnf_b", lnf_b)
    w += "\n#endif\n"
    (OUT / "nanolm_weights.h").write_text(w, encoding="utf-8")

    # ── emit nanolm_vocab.h ───────────────────────────────────────────────────
    # id -> UTF8 bytes (only output chars need real bytes; specials/control empty)
    first_char = 4 + sum(len(c) for c in control.values())
    v = ("/* nanolm_vocab.h - AUTO-GENERATED. id->UTF8 detok + control-token ids + dims. */\n"
         "#ifndef NANOLM_VOCAB_H\n#define NANOLM_VOCAB_H\n\n")
    v += f"#define NLM_VOCAB {V}\n#define NLM_DMODEL {D_MODEL}\n#define NLM_NLAYER {N_LAYERS}\n"
    v += f"#define NLM_NHEAD {N_HEADS}\n#define NLM_DHEAD {D_MODEL // N_HEADS}\n#define NLM_DFF {D_FF}\n"
    v += f"#define NLM_MAXSEQ {MAX_SEQ}\n#define NLM_BOS {bos}\n#define NLM_SEP {sep}\n#define NLM_EOS {eos}\n"
    v += f"#define NLM_NCTX {len(SLOT_ORDER)}\n#define NLM_FIRST_CHAR {first_char}\n\n"
    # UTF8 byte table: offsets + bytes (variable length)
    blob = bytearray(); offs = []
    for t in id2tok:
        offs.append(len(blob))
        if isinstance(t, str) and not t.startswith("@") and t not in ("<pad>", "<bos>", "<sep>", "<eos>"):
            blob += t.encode("utf-8")
    offs.append(len(blob))
    v += f"static const unsigned short nlm_tok_off[{len(offs)}] = {{{','.join(str(o) for o in offs)}}};\n"
    v += f"static const unsigned char nlm_tok_utf8[{max(1,len(blob))}] = {{{','.join(str(b) for b in blob) if blob else '0'}}};\n\n"
    # control-token ids per slot as named macros + a flat per-slot lookup the C build_context uses
    for slot in SLOT_ORDER:
        for opt, tid in control[slot].items():
            v += f"#define NLM_CTX_{slot.upper()}_{opt.upper()} {tid}\n"
    v += "\n#endif\n"
    (OUT / "nanolm_vocab.h").write_text(v, encoding="utf-8")

    # ── emit nanolm_golden.h ──────────────────────────────────────────────────
    g = ("/* nanolm_golden.h - AUTO-GENERATED. Demo contexts + expected greedy ids + ref logits.\n"
         " * C engine must reproduce the deployed INT8 model. */\n"
         "#ifndef NANOLM_GOLDEN_H\n#define NANOLM_GOLDEN_H\n\n")
    g += f"#define NLM_NDEMO {len(demos)}\n"
    # each demo: context ids (NCTX+2 incl bos/sep) and expected full greedy ids
    ctxs = [ctx_for(d) for d in demos]
    cn = len(ctxs[0])
    g += f"#define NLM_CTXLEN {cn}\n"
    g += "static const short nlm_demo_ctx[NLM_NDEMO][NLM_CTXLEN] = {\n"
    g += ",\n".join("{" + ",".join(str(i) for i in c) + "}" for c in ctxs) + "\n};\n"
    maxg = max(len(x) for x in gens)
    g += f"#define NLM_MAXGEN {maxg}\n"
    g += "static const short nlm_demo_gen_len[NLM_NDEMO] = {" + ",".join(str(len(x)) for x in gens) + "};\n"
    g += "static const short nlm_demo_gen[NLM_NDEMO][NLM_MAXGEN] = {\n"
    g += ",\n".join("{" + ",".join(str(i) for i in (x + [0] * (maxg - len(x)))) + "}" for x in gens) + "\n};\n\n"
    g += cf32("nlm_golden_logits", glog)   # last-pos logits for demo0 full greedy prefix
    g += f"#define NLM_GOLDEN_PREFIX_LEN {len(gens[0])}\n"
    g += "static const short nlm_golden_prefix[NLM_GOLDEN_PREFIX_LEN] = {" + ",".join(str(i) for i in gens[0]) + "};\n"
    g += "\n#endif\n"
    (OUT / "nanolm_golden.h").write_text(g, encoding="utf-8")

    sz = sum((OUT / f).stat().st_size for f in ["nanolm_weights.h", "nanolm_vocab.h", "nanolm_golden.h"])
    int8_bytes = tok_q.size + sum(np.asarray(layers[k]).size for k in ["qq", "kq", "vq", "oq", "f1q", "f2q"])
    print(f"\nINT8 weight bytes ~{int8_bytes/1024:.0f}KB + pos fp32 {pos.size*4/1024:.0f}KB; headers {sz/1024:.0f}KB")
    print("wrote nanolm_weights.h / nanolm_vocab.h / nanolm_golden.h ->", OUT)


if __name__ == "__main__":
    main()
