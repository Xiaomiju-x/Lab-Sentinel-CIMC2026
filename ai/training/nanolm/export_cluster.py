"""
export_cluster.py — quantize the 5 cluster experts to INT8, pack them into ONE
SPI-flash image with a fixed shared layout, and emit the firmware headers + a
raw provisioning binary + per-expert golden vectors.

Layout (shared dims => every expert blob is identical in structure, only weights
differ; the runtime engine computes section pointers from the offset macros):
  per expert blob (contiguous, little-endian, all sections 4-byte aligned):
    tok_q int8[V*D] | tok_s f32[V] | pos f32[MAXSEQ*D]
    L x { ln1_g f32[D] ln1_b f32[D]
          q_q i8[D*D] q_s f32[D] q_b f32[D]   (q,k,v,o)
          ln2_g f32[D] ln2_b f32[D]
          f1_q i8[F*D] f1_s f32[F] f1_b f32[F]
          f2_q i8[D*F] f2_s f32[D] f2_b f32[D] }
    lnf_g f32[D] lnf_b f32[D]
  image = expert[k] placed at SPI_BASE + k*STRIDE  (STRIDE = blob rounded to 4KB)

Outputs (-> ../../firmware/ai_models_c/ and here):
  nlm_cluster_vocab.h    dims + detok + control macros + offsets + SPI map + roles
  nlm_cluster_golden.h   per-expert demo ctx, expected greedy ids, ref logits
  cluster_image.bin      raw 5-blob image (host fread + SPI-flash provisioning)

Run:  python export_cluster.py
"""
import json
import struct
from pathlib import Path

import numpy as np
import torch

from train_nanolm import NanoLM, D_MODEL, N_LAYERS, N_HEADS, D_FF, MAX_SEQ
from export_nanolm_to_c import quant_rows
from gen_cluster import ROLE_ORDER, ROLES

HERE = Path(__file__).parent
CL = HERE / "cluster"
OUT = HERE.parent.parent / "firmware" / "ai_models_c"
SECTOR = 4096


def serialize_expert(model):
    """Return (blob_bytes, dequant_state_dict) in the fixed shared layout."""
    sd = model.state_dict()
    buf = bytearray()
    deq = {}

    def put_i8(a):
        a = np.asarray(a, np.int8); buf.extend(a.tobytes())

    def put_f32(a):
        a = np.asarray(a, np.float32); buf.extend(a.tobytes())

    # tok emb (per-row int8) + tok scale
    tq, ts = quant_rows(model.tok.weight)
    put_i8(tq); put_f32(ts)
    deq["tok"] = tq.astype(np.float32) * ts[:, None]
    pos = model.pos.weight.detach().cpu().numpy().astype(np.float32)
    put_f32(pos); deq["pos"] = pos

    for l in range(N_LAYERS):
        p = f"blocks.{l}."
        g = sd[p + "ln1.weight"].numpy(); b = sd[p + "ln1.bias"].numpy()
        put_f32(g); put_f32(b); deq[p + "ln1.weight"] = g; deq[p + "ln1.bias"] = b
        for nm in ("q", "k", "v", "o"):
            q, s = quant_rows(sd[p + nm + ".weight"]); bb = sd[p + nm + ".bias"].numpy().astype(np.float32)
            put_i8(q); put_f32(s); put_f32(bb)
            deq[p + nm + ".weight"] = q.astype(np.float32) * s[:, None]; deq[p + nm + ".bias"] = bb
        g = sd[p + "ln2.weight"].numpy(); b = sd[p + "ln2.bias"].numpy()
        put_f32(g); put_f32(b); deq[p + "ln2.weight"] = g; deq[p + "ln2.bias"] = b
        for nm in ("ff1", "ff2"):
            q, s = quant_rows(sd[p + nm + ".weight"]); bb = sd[p + nm + ".bias"].numpy().astype(np.float32)
            put_i8(q); put_f32(s); put_f32(bb)
            deq[p + nm + ".weight"] = q.astype(np.float32) * s[:, None]; deq[p + nm + ".bias"] = bb
    g = sd["lnf.weight"].numpy(); b = sd["lnf.bias"].numpy()
    put_f32(g); put_f32(b); deq["lnf.weight"] = g; deq["lnf.bias"] = b
    return bytes(buf), deq


def deq_model(V, deq):
    m = NanoLM(V); msd = m.state_dict()
    msd["tok.weight"] = torch.tensor(deq["tok"]); msd["pos.weight"] = torch.tensor(deq["pos"])
    for l in range(N_LAYERS):
        p = f"blocks.{l}."
        for k in ("ln1.weight", "ln1.bias", "q.weight", "q.bias", "k.weight", "k.bias",
                  "v.weight", "v.bias", "o.weight", "o.bias", "ln2.weight", "ln2.bias",
                  "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"):
            msd[p + k] = torch.tensor(deq[p + k])
    msd["lnf.weight"] = torch.tensor(deq["lnf.weight"]); msd["lnf.bias"] = torch.tensor(deq["lnf.bias"])
    m.load_state_dict(msd); m.eval(); return m


# role-appropriate demo contexts for golden (cover the role's signature states)
DEMO = {
    "e1": [{"stage": "sinter", "temp": "over", "risk": "crit", "gas": "cr6", "drift": "hi", "ae": "anom"},
           {"stage": "ramp", "temp": "rt", "risk": "crit", "tc": "open"}],
    "e2": [{"stage": "ramp", "temp": "t1200", "risk": "bad", "ramp": "fast", "host": "yag"},
           {"stage": "sinter", "temp": "t1500", "risk": "good", "host": "gagg"}],
    "e3": [{"stage": "sinter", "temp": "t1500", "risk": "warn", "energy": "high", "elem": "warn"},
           {"stage": "idle", "temp": "rt", "risk": "good"}],
    "e4": [{"stage": "done", "temp": "rt", "risk": "good", "host": "yag"},
           {"stage": "sinter", "temp": "t1500", "risk": "bad", "gas": "cr6", "host": "sygo"}],
    "e5": [{"stage": "idle", "temp": "rt", "risk": "good", "host": "yag"},
           {"stage": "calcine", "temp": "t900", "risk": "warn", "gas": "co2"}],
    "e6": [{"stage": "sinter", "temp": "t1500", "risk": "warn", "host": "gagg"},
           {"stage": "calcine", "temp": "t900", "risk": "good", "host": "sygo"}],
    "e7": [{"stage": "sinter", "temp": "t1500", "risk": "bad", "elem": "alarm", "energy": "high"},
           {"stage": "calcine", "temp": "t900", "risk": "warn", "vib": "abn"}],
}
BASE = {"stage": "idle", "temp": "rt", "risk": "good", "ramp": "ok", "drift": "ok", "tc": "ok",
        "gas": "ok", "ae": "ok", "vib": "ok", "energy": "ok", "host": "yag", "elem": "ok"}


def main():
    voc = json.load(open(CL / "cluster_vocab.json", encoding="utf-8"))
    V = voc["vocab"]; id2tok = voc["id2tok"]; control = voc["control"]; SLOT_ORDER = voc["slot_order"]
    bos, sep, eos = voc["specials"]["<bos>"], voc["specials"]["<sep>"], voc["specials"]["<eos>"]

    blobs = {}; deqs = {}
    for role in ROLE_ORDER:
        pt = CL / f"expert_{role}.pt"
        if not pt.exists():
            raise SystemExit(f"missing {pt} - run train_cluster.py")
        m = NanoLM(V); m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True)["model"]); m.eval()
        blob, deq = serialize_expert(m); blobs[role] = blob; deqs[role] = deq
    blob_bytes = len(blobs[ROLE_ORDER[0]])
    assert all(len(blobs[r]) == blob_bytes for r in ROLE_ORDER), "blob size mismatch"
    stride = (blob_bytes + SECTOR - 1) // SECTOR * SECTOR

    # ── cluster_image.bin (expert k at k*stride, sector-aligned) ──────────────
    img = bytearray(stride * len(ROLE_ORDER))
    for k, role in enumerate(ROLE_ORDER):
        img[k * stride:k * stride + blob_bytes] = blobs[role]
    (HERE / "cluster_image.bin").write_bytes(img)
    (OUT / "cluster_image.bin").write_bytes(img)

    # ── golden per expert (greedy on deq model) ──────────────────────────────
    def ctx_for(state):
        st = dict(BASE); st.update(state)
        return [bos] + [control[s][st[s]] for s in SLOT_ORDER] + [sep]

    golden = {}
    for role in ROLE_ORDER:
        dm = deq_model(V, deqs[role])
        outs = []
        for st in DEMO[role]:
            ids = ctx_for(st)
            with torch.no_grad():
                for _ in range(48):
                    x = torch.tensor([ids[-MAX_SEQ:]])
                    nxt = int(torch.argmax(dm(x)[0, -1]))
                    if nxt == eos:
                        break
                    ids.append(nxt)
            outs.append(ids)
        with torch.no_grad():
            gx = torch.tensor([outs[0][:MAX_SEQ]])
            glog = dm(gx)[0, -1].numpy().astype(np.float32)
        golden[role] = (outs, glog)
        txt = "".join(id2tok[i] for i in outs[0][outs[0].index(sep) + 1:] if i not in (bos, sep, eos))
        print(f"  [{role} {ROLES[role]['cn']}] demo0: {txt}")

    # ── section offsets (shared; engine binds blob+off) ───────────────────────
    D, F, L = D_MODEL, D_FF, N_LAYERS
    off = {}; cur = 0
    def add(name, nbytes):
        nonlocal cur; off[name] = cur; cur += nbytes
    add("TOK_Q", V * D); add("TOK_S", V * 4); add("POS", MAX_SEQ * D * 4)
    off["LAYER0"] = cur
    lcur = 0; loff = {}
    def ladd(name, nbytes):
        nonlocal lcur; loff[name] = lcur; lcur += nbytes
    ladd("LN1_G", D * 4); ladd("LN1_B", D * 4)
    for nm in ("Q", "K", "V", "O"):
        ladd(nm + "_Q", D * D); ladd(nm + "_S", D * 4); ladd(nm + "_B", D * 4)
    ladd("LN2_G", D * 4); ladd("LN2_B", D * 4)
    ladd("F1_Q", F * D); ladd("F1_S", F * 4); ladd("F1_B", F * 4)
    ladd("F2_Q", D * F); ladd("F2_S", D * 4); ladd("F2_B", D * 4)
    layer_stride = lcur
    cur = off["LAYER0"] + L * layer_stride
    add_base = cur
    off["LNF_G"] = add_base; off["LNF_B"] = add_base + D * 4
    total = off["LNF_B"] + D * 4
    assert total == blob_bytes, f"offset calc {total} != blob {blob_bytes}"

    # ── emit nlm_cluster_vocab.h ──────────────────────────────────────────────
    first_char = 4 + sum(len(c) for c in control.values())
    v = ("/* nlm_cluster_vocab.h - AUTO-GENERATED by export_cluster.py. Do not edit.\n"
         f" * Shared vocab for the {len(ROLE_ORDER)} swap-loaded edge-LLM experts. */\n"
         "#ifndef NLM_CLUSTER_VOCAB_H\n#define NLM_CLUSTER_VOCAB_H\n\n")
    v += f"#define NLM_CL_VOCAB {V}\n#define NLM_DMODEL {D}\n#define NLM_NLAYER {L}\n"
    v += f"#define NLM_NHEAD {N_HEADS}\n#define NLM_DHEAD {D//N_HEADS}\n#define NLM_DFF {F}\n"
    v += f"#define NLM_MAXSEQ {MAX_SEQ}\n#define NLM_BOS {bos}\n#define NLM_SEP {sep}\n#define NLM_EOS {eos}\n"
    v += f"#define NLM_NCTX {len(SLOT_ORDER)}\n#define NLM_FIRST_CHAR {first_char}\n\n"
    v += f"#define NLM_CL_NEXPERT {len(ROLE_ORDER)}\n"
    v += f"#define NLM_CL_BLOB_BYTES {blob_bytes}u\n#define NLM_CL_STRIDE {stride}u\n"
    v += "#define NLM_CL_SPI_BASE 0x000000u   /* expert k @ SPI_BASE + k*STRIDE */\n"
    v += f"#define NLM_CL_LAYER_STRIDE {layer_stride}u\n\n"
    for k in ("TOK_Q", "TOK_S", "POS", "LAYER0", "LNF_G", "LNF_B"):
        v += f"#define NLM_CL_OFF_{k} {off[k]}u\n"
    for k in loff:
        v += f"#define NLM_CL_LOFF_{k} {loff[k]}u\n"
    v += "\n"
    names = ",".join('"%s"' % ROLES[r]["name"] for r in ROLE_ORDER)
    cns = ",".join('"%s"' % r for r in ROLE_ORDER)
    v += f"static const char *const nlm_cl_role[NLM_CL_NEXPERT] = {{{names}}};\n"
    v += f"static const char *const nlm_cl_key[NLM_CL_NEXPERT]  = {{{cns}}};\n\n"
    # detok table
    blob2 = bytearray(); offs = []
    for t in id2tok:
        offs.append(len(blob2))
        if isinstance(t, str) and not t.startswith("@") and t not in ("<pad>", "<bos>", "<sep>", "<eos>"):
            blob2 += t.encode("utf-8")
    offs.append(len(blob2))
    v += f"static const unsigned short nlm_cl_tok_off[{len(offs)}] = {{{','.join(str(o) for o in offs)}}};\n"
    v += f"static const unsigned char nlm_cl_tok_utf8[{max(1,len(blob2))}] = {{{','.join(str(b) for b in blob2) if blob2 else '0'}}};\n\n"
    for slot in SLOT_ORDER:
        for opt, tid in control[slot].items():
            v += f"#define NLM_CTX_{slot.upper()}_{opt.upper()} {tid}\n"
    v += "\n#endif\n"
    (OUT / "nlm_cluster_vocab.h").write_text(v, encoding="utf-8")

    # ── emit nlm_cluster_golden.h ─────────────────────────────────────────────
    cn = len(ctx_for(DEMO["e1"][0]))
    g = ("/* nlm_cluster_golden.h - AUTO-GENERATED. Per-expert demo ctx + greedy ids + ref logits. */\n"
         "#ifndef NLM_CLUSTER_GOLDEN_H\n#define NLM_CLUSTER_GOLDEN_H\n\n")
    g += f"#define NLM_CL_NDEMO 2\n#define NLM_CTXLEN {cn}\n"
    maxg = max(len(o) for role in ROLE_ORDER for o in golden[role][0])
    g += f"#define NLM_CL_MAXGEN {maxg}\n\n"
    # ctx[expert][demo][ctxlen]
    g += "static const short nlm_cl_demo_ctx[NLM_CL_NEXPERT][NLM_CL_NDEMO][NLM_CTXLEN] = {\n"
    for role in ROLE_ORDER:
        rows = []
        for st in DEMO[role]:
            rows.append("{" + ",".join(str(i) for i in ctx_for(st)) + "}")
        g += " {" + ",".join(rows) + "},\n"
    g += "};\n"
    g += "static const short nlm_cl_gen_len[NLM_CL_NEXPERT][NLM_CL_NDEMO] = {\n"
    for role in ROLE_ORDER:
        g += " {" + ",".join(str(len(o)) for o in golden[role][0]) + "},\n"
    g += "};\n"
    g += "static const short nlm_cl_gen[NLM_CL_NEXPERT][NLM_CL_NDEMO][NLM_CL_MAXGEN] = {\n"
    for role in ROLE_ORDER:
        outs = golden[role][0]
        rows = ["{" + ",".join(str(i) for i in (o + [0] * (maxg - len(o)))) + "}" for o in outs]
        g += " {" + ",".join(rows) + "},\n"
    g += "};\n"
    # ref logits: demo0 of each expert
    g += f"static const float nlm_cl_golden_logits[NLM_CL_NEXPERT][{V}] = {{\n"
    for role in ROLE_ORDER:
        gl = golden[role][1]
        g += " {" + ",".join(f"{float(x):.6e}f" for x in gl) + "},\n"
    g += "};\n"
    g += "static const short nlm_cl_golden_prefix_len[NLM_CL_NEXPERT] = {" + \
         ",".join(str(len(golden[r][0][0])) for r in ROLE_ORDER) + "};\n"
    g += f"static const short nlm_cl_golden_prefix[NLM_CL_NEXPERT][{maxg}] = {{\n"
    for role in ROLE_ORDER:
        o = golden[role][0][0]
        g += " {" + ",".join(str(i) for i in (o + [0] * (maxg - len(o)))) + "},\n"
    g += "};\n\n#endif\n"
    (OUT / "nlm_cluster_golden.h").write_text(g, encoding="utf-8")

    print(f"\nblob/expert {blob_bytes/1024:.0f}KB  stride {stride/1024:.0f}KB  image {len(img)/1024/1024:.2f}MB ({len(ROLE_ORDER)} experts)")
    print(f"vocab V={V}  -> nlm_cluster_vocab.h / nlm_cluster_golden.h / cluster_image.bin")


if __name__ == "__main__":
    main()
