"""
export_lmbank.py — pack the SMALLER swept flagship sizes (s0p6, m1p35) into ONE
SPI-flash "LM bank" image + emit firmware headers + golden, so they can be
swap-loaded into SDRAM and switched at runtime against the always-on internal
x1p9. The user wanted 3 size-variants switchable (one resident at a time) — this
is the hardware-ceiling curve made LIVE on the board.

Why a bank (not internal flash): all 3 sizes' INT8 weights (0.6+1.26+1.8MB) blow
the 3.84MB internal flash. x1p9 stays internal (the verified default); s0p6 +
m1p35 live in the 8MB SPI flash and load on demand — exactly the cluster's
swap-load mechanism, but the bank holds DIFFERENT-dim models (so the engine binds
runtime dims, vs the cluster's fixed d128).

KEY: s0p6/m1p35/x1p9 were trained on the SAME corpus_v2 -> IDENTICAL vocab (V,
token ids, detok, control macros). So the bank reuses nanolm_vocab.h (x1p9's) for
detok/control — it carries WEIGHTS only. (Asserted below.)

Layout per model blob (little-endian, 4-byte aligned, SAME section order as
export_cluster.serialize):
  tok_q i8[V*d] | tok_s f32[V] | pos f32[MS*d]
  L x { ln1_g/b f32[d]; {q,k,v,o}: q i8[d*d] s f32[d] b f32[d]; ln2_g/b f32[d];
        f1: q i8[ff*d] s f32[ff] b f32[ff]; f2: q i8[d*ff] s f32[d] b f32[d] }
  lnf_g/b f32[d]
Image: model k at a sector-aligned SPI offset (per-model, sizes differ).

Outputs (-> ../../firmware/ai_models_c/ + here):
  nlm_bank.h       per-model descriptor (dims + section offsets + spi off) + labels + golden
  lmbank_image.bin raw bank image (UART provisioning into SPI flash)

Run:  python export_lmbank.py
"""
import json
import struct
from pathlib import Path

import numpy as np
import torch

from gen_corpus import SLOT_ORDER
from train_flagship import GPT
from export_nanolm_to_c import quant_rows

HERE = Path(__file__).parent
OUT = HERE.parent.parent / "firmware" / "ai_models_c"
SECTOR = 4096

# bank members (the two smaller sizes; x1p9 stays internal). label/ppl/latency for HMI.
BANK = [
    {"tag": "s0p6",  "label": "0.6M",  "ppl": 3.00, "latx": 1.10},
    {"tag": "m1p35", "label": "1.26M", "ppl": 2.69, "latx": 2.13},
]
# the internal default x1p9, for the HMI roster (idx 0 = internal, not in the SPI bank)
INTERNAL = {"tag": "x1p9", "label": "1.8M", "ppl": 2.56, "latx": 3.01}

# demo states for golden (same 3 as export_flagship: idle + 2 faults)
DEMOS = [
    {"stage": "sinter", "temp": "over", "risk": "crit", "ramp": "ok", "drift": "hi",
     "tc": "ok", "gas": "cr6", "ae": "anom", "vib": "ok", "energy": "ok", "host": "yag", "elem": "ok"},
    {"stage": "sinter", "temp": "rt", "risk": "crit", "ramp": "ok", "drift": "ok",
     "tc": "open", "gas": "ok", "ae": "ok", "vib": "ok", "energy": "ok", "host": "gagg", "elem": "ok"},
    {"stage": "idle", "temp": "rt", "risk": "good", "ramp": "ok", "drift": "ok",
     "tc": "ok", "gas": "ok", "ae": "ok", "vib": "ok", "energy": "ok", "host": "yag", "elem": "ok"},
]


def serialize(model, cfg):
    """blob bytes + (offset dict) + dequant state, fixed section order, this model's dims."""
    sd = model.state_dict()
    d, nl, ff = cfg["d_model"], cfg["n_layers"], cfg["d_ff"]
    V = model.tok.weight.shape[0]
    buf = bytearray(); deq = {}; off = {}

    def mark(name):
        off[name] = len(buf)

    def put_i8(a):
        buf.extend(np.asarray(a, np.int8).tobytes())

    def put_f32(a):
        buf.extend(np.asarray(a, np.float32).tobytes())

    mark("TOK_Q"); tq, ts = quant_rows(model.tok.weight); put_i8(tq)
    mark("TOK_S"); put_f32(ts); deq["tok"] = tq.astype(np.float32) * ts[:, None]
    mark("POS"); pos = model.pos.weight.detach().cpu().numpy().astype(np.float32); put_f32(pos); deq["pos"] = pos

    mark("LAYER0")
    layer_stride = None
    for l in range(nl):
        p = f"blocks.{l}."
        lbase = len(buf); loff = {}

        def ladd(name, fn):
            loff[name] = len(buf) - lbase; fn()

        ladd("LN1_G", lambda: put_f32(sd[p + "ln1.weight"].numpy()))
        ladd("LN1_B", lambda: put_f32(sd[p + "ln1.bias"].numpy()))
        deq[p + "ln1.weight"] = sd[p + "ln1.weight"].numpy(); deq[p + "ln1.bias"] = sd[p + "ln1.bias"].numpy()
        for nm in ("q", "k", "v", "o"):
            q, s = quant_rows(sd[p + nm + ".weight"]); bb = sd[p + nm + ".bias"].numpy().astype(np.float32)
            ladd(nm.upper() + "_Q", (lambda qq=q: put_i8(qq)))
            ladd(nm.upper() + "_S", (lambda ss=s: put_f32(ss)))
            ladd(nm.upper() + "_B", (lambda b=bb: put_f32(b)))
            deq[p + nm + ".weight"] = q.astype(np.float32) * s[:, None]; deq[p + nm + ".bias"] = bb
        ladd("LN2_G", lambda: put_f32(sd[p + "ln2.weight"].numpy()))
        ladd("LN2_B", lambda: put_f32(sd[p + "ln2.bias"].numpy()))
        deq[p + "ln2.weight"] = sd[p + "ln2.weight"].numpy(); deq[p + "ln2.bias"] = sd[p + "ln2.bias"].numpy()
        for nm in ("ff1", "ff2"):
            q, s = quant_rows(sd[p + nm + ".weight"]); bb = sd[p + nm + ".bias"].numpy().astype(np.float32)
            tag = "F1" if nm == "ff1" else "F2"
            ladd(tag + "_Q", (lambda qq=q: put_i8(qq)))
            ladd(tag + "_S", (lambda ss=s: put_f32(ss)))
            ladd(tag + "_B", (lambda b=bb: put_f32(b)))
            deq[p + nm + ".weight"] = q.astype(np.float32) * s[:, None]; deq[p + nm + ".bias"] = bb
        if layer_stride is None:
            layer_stride = len(buf) - lbase
            off["LOFF"] = loff
    off["LAYER_STRIDE"] = layer_stride
    mark("LNF_G"); put_f32(sd["lnf.weight"].numpy()); deq["lnf.weight"] = sd["lnf.weight"].numpy()
    mark("LNF_B"); put_f32(sd["lnf.bias"].numpy()); deq["lnf.bias"] = sd["lnf.bias"].numpy()
    return bytes(buf), off, deq


def deq_model(cfg, V, deq):
    m = GPT(V, cfg); msd = m.state_dict()
    msd["tok.weight"] = torch.tensor(deq["tok"]); msd["pos.weight"] = torch.tensor(deq["pos"])
    for l in range(cfg["n_layers"]):
        p = f"blocks.{l}."
        for k in ("ln1.weight", "ln1.bias", "q.weight", "q.bias", "k.weight", "k.bias",
                  "v.weight", "v.bias", "o.weight", "o.bias", "ln2.weight", "ln2.bias",
                  "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            msd[p + k] = torch.tensor(deq[p + k])
    msd["lnf.weight"] = torch.tensor(deq["lnf.weight"]); msd["lnf.bias"] = torch.tensor(deq["lnf.bias"])
    m.load_state_dict(msd); m.eval(); return m


def main():
    # reference vocab = x1p9 (== s0p6 == m1p35, asserted). carries detok/control.
    ref = torch.load(HERE / "flagship_x1p9.pt", map_location="cpu", weights_only=True)
    V = ref["vocab"]; control = ref["control"]; tok2id = ref["tok2id"]
    bos, sep, eos = tok2id["<bos>"], tok2id["<sep>"], tok2id["<eos>"]

    models = {}
    blobs = {}; offs = {}; deqs = {}
    for m in BANK:
        ck = torch.load(
            HERE / f"flagship_{m['tag']}.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert ck["vocab"] == V, f"{m['tag']} vocab {ck['vocab']} != x1p9 {V} (must share corpus_v2 vocab)"
        assert ck["tok2id"] == tok2id, f"{m['tag']} tok2id differs from x1p9 — bank cannot reuse nanolm_vocab.h"
        cfg = ck["cfg"]
        gpt = GPT(V, cfg); gpt.load_state_dict(ck["model"]); gpt.eval()
        blob, off, deq = serialize(gpt, cfg)
        models[m["tag"]] = cfg; blobs[m["tag"]] = blob; offs[m["tag"]] = off; deqs[m["tag"]] = deq

    # ── lmbank_image.bin: model k at sector-aligned cumulative offset ─────────
    spi_off = {}; img = bytearray(); cur = 0
    for m in BANK:
        cur = (cur + SECTOR - 1) // SECTOR * SECTOR
        spi_off[m["tag"]] = cur
        img.extend(b"\x00" * (cur - len(img)))
        img.extend(blobs[m["tag"]]); cur = len(img)
    # pad image tail to a whole sector so UART provisioning sends clean 4KB sectors
    pad = (SECTOR - (len(img) % SECTOR)) % SECTOR
    img.extend(b"\x00" * pad)
    (HERE / "lmbank_image.bin").write_bytes(img)
    (OUT / "lmbank_image.bin").write_bytes(img)
    max_blob = max(len(blobs[m["tag"]]) for m in BANK)

    # SPI flash placement: bank sits after the cluster image, sector-aligned.
    # Cluster grew 5->7 experts (782336B stride * 7 = 5.22MB, ends 0x539000), so the
    # bank moved up from 0x400000 to 0x540000 (still clear of bank magic @ 0x7E0000).
    BANK_SPI_BASE_ADDR = 0x540000
    # standalone provisioning header (NO vocab include -> flash_provision.c can use it
    # alongside nlm_cluster_vocab.h without the NLM_* dim-macro clash)
    ph = ("/* nlm_bank_prov.h - AUTO-GENERATED by export_lmbank.py. SPI-flash placement of\n"
          " * the LM size bank, for both the swap-load engine and UART provisioning.\n"
          " * Standalone (no vocab) so flash_provision.c stays clear of NLM_* dim macros. */\n"
          "#ifndef NLM_BANK_PROV_H\n#define NLM_BANK_PROV_H\n"
          f"#define BANK_PROV_BASE  0x{BANK_SPI_BASE_ADDR:06X}u   /* SPI flash byte offset of the bank image */\n"
          f"#define BANK_PROV_BYTES {len(img)}u        /* padded image size = UART provisioning total */\n"
          "#endif\n")
    (OUT / "nlm_bank_prov.h").write_text(ph, encoding="utf-8")

    # ── golden per model (greedy on dequant model) ───────────────────────────
    def ctx_for(state):
        return [bos] + [control[s][state[s]] for s in SLOT_ORDER] + [sep]

    golden = {}
    for m in BANK:
        tag = m["tag"]; cfg = models[tag]; MS = cfg["max_seq"]
        dm = deq_model(cfg, V, deqs[tag])
        outs = []
        for st in DEMOS:
            ids = ctx_for(st)
            with torch.no_grad():
                for _ in range(48):
                    x = torch.tensor([ids[-MS:]])
                    nxt = int(torch.argmax(dm(x)[0, -1]))
                    if nxt == eos:
                        break
                    ids.append(nxt)
            outs.append(ids)
        with torch.no_grad():
            gx = torch.tensor([outs[0][:MS]])
            glog = dm(gx)[0, -1].numpy().astype(np.float32)
        golden[tag] = (outs, glog)
        txt = "".join(ref["id2tok"][i] for i in outs[2][outs[2].index(sep) + 1:] if i not in (bos, sep, eos))
        print(f"  [{tag} {m['label']}] idle demo: {txt}")

    # bank-wide buffer maxima (engine sizes static scratch to these)
    DMAX = max(c["d_model"] for c in models.values())
    FFMAX = max(c["d_ff"] for c in models.values())
    NLMAX = max(c["n_layers"] for c in models.values())
    MSMAX = max(c["max_seq"] for c in models.values())
    CTXLEN = len(ctx_for(DEMOS[0]))
    MAXGEN = max(len(o) for tag in blobs for o in golden[tag][0])

    # ── emit nlm_bank.h ───────────────────────────────────────────────────────
    h = ("/* nlm_bank.h - AUTO-GENERATED by export_lmbank.py. Do not edit.\n"
         " * SPI-flash swap-load bank of smaller flagship sizes (s0p6,m1p35); shares\n"
         " * nanolm_vocab.h (V/detok/control/BOS/SEP/EOS) with the internal x1p9. */\n"
         "#ifndef NLM_BANK_H\n#define NLM_BANK_H\n#include \"nanolm_vocab.h\"\n#include \"nlm_bank_prov.h\"\n\n")
    h += f"#define BANK_N {len(BANK)}\n"
    h += f"#define BANK_DMAX {DMAX}\n#define BANK_FFMAX {FFMAX}\n#define BANK_NLMAX {NLMAX}\n#define BANK_MSMAX {MSMAX}\n"
    h += f"#define BANK_CTXLEN {CTXLEN}\n#define BANK_MAXGEN {MAXGEN}\n"
    h += f"#define BANK_BYTES {len(img)}u        /* padded image size = UART provisioning total */\n"
    h += f"#define BANK_MAX_BLOB {max_blob}u   /* largest single blob = SDRAM working-blob region */\n\n"
    h += ("typedef struct { int d, nl, nh, dh, ff, v, ms;\n"
          "  unsigned int spi_off, blob_bytes;\n"
          "  unsigned int off_tok_q, off_tok_s, off_pos, off_layer0, layer_stride, off_lnf_g, off_lnf_b;\n"
          "  unsigned int lo_ln1g, lo_ln1b, lo_qq, lo_qs, lo_qb, lo_kq, lo_ks, lo_kb,\n"
          "               lo_vq, lo_vs, lo_vb, lo_oq, lo_os, lo_ob, lo_ln2g, lo_ln2b,\n"
          "               lo_f1q, lo_f1s, lo_f1b, lo_f2q, lo_f2s, lo_f2b;\n"
          "} bank_model_t;\n\n")
    h += "static const bank_model_t g_bank[BANK_N] = {\n"
    for m in BANK:
        tag = m["tag"]; c = models[tag]; o = offs[tag]; lo = o["LOFF"]
        h += ("  { .d=%d, .nl=%d, .nh=%d, .dh=%d, .ff=%d, .v=%d, .ms=%d,\n"
              "    .spi_off=%uu, .blob_bytes=%uu,\n"
              "    .off_tok_q=%uu, .off_tok_s=%uu, .off_pos=%uu, .off_layer0=%uu, .layer_stride=%uu, .off_lnf_g=%uu, .off_lnf_b=%uu,\n"
              "    .lo_ln1g=%uu,.lo_ln1b=%uu,.lo_qq=%uu,.lo_qs=%uu,.lo_qb=%uu,.lo_kq=%uu,.lo_ks=%uu,.lo_kb=%uu,\n"
              "    .lo_vq=%uu,.lo_vs=%uu,.lo_vb=%uu,.lo_oq=%uu,.lo_os=%uu,.lo_ob=%uu,.lo_ln2g=%uu,.lo_ln2b=%uu,\n"
              "    .lo_f1q=%uu,.lo_f1s=%uu,.lo_f1b=%uu,.lo_f2q=%uu,.lo_f2s=%uu,.lo_f2b=%uu },\n") % (
            c["d_model"], c["n_layers"], c["n_heads"], c["d_model"] // c["n_heads"], c["d_ff"], V, c["max_seq"],
            spi_off[tag], len(blobs[tag]),
            o["TOK_Q"], o["TOK_S"], o["POS"], o["LAYER0"], o["LAYER_STRIDE"], o["LNF_G"], o["LNF_B"],
            lo["LN1_G"], lo["LN1_B"], lo["Q_Q"], lo["Q_S"], lo["Q_B"], lo["K_Q"], lo["K_S"], lo["K_B"],
            lo["V_Q"], lo["V_S"], lo["V_B"], lo["O_Q"], lo["O_S"], lo["O_B"], lo["LN2_G"], lo["LN2_B"],
            lo["F1_Q"], lo["F1_S"], lo["F1_B"], lo["F2_Q"], lo["F2_S"], lo["F2_B"])
    h += "};\n\n"
    # HMI roster: index 0 = internal x1p9, 1..N = bank
    labels = [INTERNAL] + BANK
    h += f"#define LM_ROSTER_N {len(labels)}\n"
    h += "static const char *const lm_roster_tag[LM_ROSTER_N] = {" + ",".join('"%s"' % x["tag"] for x in labels) + "};\n"
    h += "static const char *const lm_roster_lab[LM_ROSTER_N] = {" + ",".join('"%s"' % x["label"] for x in labels) + "};\n"
    h += "static const short lm_roster_pplx100[LM_ROSTER_N] = {" + ",".join(str(int(round(x["ppl"]*100))) for x in labels) + "};\n"
    h += "static const short lm_roster_latx10[LM_ROSTER_N]  = {" + ",".join(str(int(round(x["latx"]*10))) for x in labels) + "};\n\n"
    # golden
    h += "static const short bank_demo_ctx[BANK_N][3][BANK_CTXLEN] = {\n"
    for m in BANK:
        rows = ["{" + ",".join(str(i) for i in ctx_for(st)) + "}" for st in DEMOS]
        h += " {" + ",".join(rows) + "},\n"
    h += "};\n"
    h += "static const short bank_gen_len[BANK_N][3] = {\n"
    for m in BANK:
        h += " {" + ",".join(str(len(o)) for o in golden[m["tag"]][0]) + "},\n"
    h += "};\n"
    h += "static const short bank_gen[BANK_N][3][BANK_MAXGEN] = {\n"
    for m in BANK:
        outs = golden[m["tag"]][0]
        rows = ["{" + ",".join(str(i) for i in (o + [0] * (MAXGEN - len(o)))) + "}" for o in outs]
        h += " {" + ",".join(rows) + "},\n"
    h += "};\n"
    h += f"static const float bank_golden_logits[BANK_N][{V}] = {{\n"
    for m in BANK:
        h += " {" + ",".join(f"{float(x):.6e}f" for x in golden[m["tag"]][1]) + "},\n"
    h += "};\n"
    h += "static const short bank_golden_prefix_len[BANK_N] = {" + ",".join(str(len(golden[m["tag"]][0][0])) for m in BANK) + "};\n"
    h += f"static const short bank_golden_prefix[BANK_N][{MAXGEN}] = {{\n"
    for m in BANK:
        o = golden[m["tag"]][0][0]
        h += " {" + ",".join(str(i) for i in (o + [0] * (MAXGEN - len(o)))) + "},\n"
    h += "};\n\n#endif\n"
    (OUT / "nlm_bank.h").write_text(h, encoding="utf-8")

    for m in BANK:
        print(f"  {m['tag']}: blob {len(blobs[m['tag']])/1024:.0f}KB @ SPI 0x{spi_off[m['tag']]:06X}")
    print(f"\nlmbank_image.bin {len(img)/1024/1024:.2f}MB ({len(BANK)} models)  DMAX={DMAX} FFMAX={FFMAX} -> nlm_bank.h")


if __name__ == "__main__":
    main()
