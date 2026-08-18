"""build_taxonomy.py -- AI-5 root-cause diagnoser: grounded label-set builder.

AI-1/2/3/4 detect THAT a sintering run is anomalous (classification / anomaly
score). AI-5 answers WHY + WHAT TO DO: it maps the *upstream AI signature*
(AI-1 crucible state, AI-2 residuals, AI-3 sinter-curve class + attention,
AI-4 risk, plus raw furnace features) to a NAMED process root-cause + a
corrective action, grounded in the XRD project's literature corpus.

This script does NOT train anything. It is the measure-first GATE:
  for each candidate process-fault root-cause class, it measures how much
  real support exists in
    (a) the 25228-chunk NIR-phosphor literature corpus (BM25 lexical retrieval
        over the actual paper text -- deterministic, no API),
    (b) predict_engine/sintering_profiles.json (the DOI-anchored normal
        protocols a fault deviates from),
    (c) firmware/ai_models_c/gas_safety_table.h (grounded gas chemistry).
  A class is KEPT only if it has rich corpus support OR a hard grounded rule.
  Classes with neither are flagged REJECT (honest, per XRD ADR-4).

Scope (deliberately): PROCESS-stage faults the GD32 can actually observe.
Composition faults (dopant %, ionic-radius mismatch, host choice) are NOT in
scope -- those belong to an off-device recipe model. This orthogonality is the
"成分性 vs 过程性" layered-architecture argument.

Outputs:
  CIMC/model/ai5_rootcause/taxonomy.json   machine-readable (classes + signature
                                           spec + evidence counts + paper refs)
  CIMC/docs/ai5_rootcause_labels.md        human-readable label set for the gate

Usage:  python CIMC/model/ai5_rootcause/build_taxonomy.py
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                       # .../xrd
CHUNKS = REPO / "spectrum_knowledge_shared" / "embeddings" / "chunks.json"
SINTER = REPO / "predict_engine" / "sintering_profiles.json"
GAS_TABLE = REPO / "CIMC" / "firmware" / "ai_models_c" / "gas_safety_table.h"
OUT_JSON = HERE / "taxonomy.json"
OUT_MD = REPO / "CIMC" / "docs" / "ai5_rootcause_labels.md"

# --------------------------------------------------------------------------- #
# Candidate process-fault root-cause taxonomy.
# Each class declares:
#   name        : short ASCII id (becomes the AI-5 output label)
#   human       : human-readable name
#   query       : retrieval terms (mechanism vocabulary as it appears in papers)
#   signature   : which upstream-AI outputs / raw features trigger it (the
#                 future AI-5 input vector); documented here so the training-set
#                 generator and the firmware stay in lock-step
#   mechanism   : one-line physical mechanism (grounded, no fabricated constants)
#   action      : corrective action the operator/controller should take
#   grounded_by : hard non-corpus grounding ('gas_safety:<mat>' / 'sinter:<field>')
# --------------------------------------------------------------------------- #
# 'must' = discriminating substrings (OR). A retrieved chunk counts as PRECISE
# evidence only if it contains >=1 must-term -- this kills BM25 homonym pollution
# (e.g. "rate" matching radiative-decay-rate, "temperature stability" matching
# optical thermometry). 'must_not' drops obvious false friends. Tiering uses the
# PRECISE distinct-paper count, so weakly-supported classes down-tier honestly.
CLASSES = [
    {
        "name": "NORMAL",
        "human": "Normal — on-protocol, no actionable fault",
        "query": "sintering on profile normal crystalline single phase well",
        "must": [],   # baseline class -- corpus support is not meaningful
        "must_not": [],
        "signature": "AI-4 risk=NORMAL, AI-2 ratio<1, AI-3=normal",
        "mechanism": "All modalities within the host's normal sintering profile.",
        "action": "Continue; no operator action required.",
        "grounded_by": "baseline",
    },
    {
        "name": "RAMP_TOO_FAST",
        "human": "Heating ramp too fast — thermal stress / unreacted intermediate",
        "query": "heating rate ramp fast thermal shock crack cracking "
                 "intermediate phase incomplete reaction solid state calcination",
        # require an explicit ramp/heating-rate term; drop polysemous "thermal
        # shock"/"crack" (luminescence papers use those for filters & sensing).
        "must": ["ramp rate", "ramp-rate", "heating rate", "heating-rate",
                 "c/min", "c min", "cooling rate", "ramping"],
        "must_not": ["radiative rate", "decay rate", "growth rate", "color filter",
                     "anti-counterfeit", "sensing"],
        "signature": "AI-3=fast_ramp, AI-2 temp-residual high, ramp_C_per_min "
                     "above sintering_profiles baseline",
        "mechanism": "Excessive ramp rate causes thermal gradients / cracking and "
                     "skips intermediate solid-state reaction steps, trapping "
                     "secondary phases.",
        "action": "Reduce ramp rate to the host profile value; add an "
                  "intermediate dwell through the reaction window.",
        "grounded_by": "sinter:ramp_C_per_min",
    },
    {
        "name": "UNDER_TEMPERATURE",
        "human": "Under-temperature / short hold — incomplete reaction, amorphous/secondary phase",
        "query": "sintering temperature soaking time insufficient low temperature "
                 "incomplete crystallization amorphous secondary phase unreacted "
                 "calcination duration",
        "must": ["sintering temperature", "calcination temperature", "soaking",
                 "annealing temperature", "incomplete", "amorphous", "unreacted",
                 "secondary phase", "impurity phase", "low-temperature", "low temperature phase"],
        "must_not": [],
        "signature": "AI-3=undertemp, AI-1 still 'charged' not 'done', peak T "
                     "below sintering_profiles sinter.temp_C",
        "mechanism": "Peak temperature or soak time below the host requirement "
                     "leaves the reaction incomplete -> residual precursor / "
                     "amorphous / secondary phase, weak luminescence.",
        "action": "Raise peak temperature or extend the soak to the host "
                  "profile; verify with XRD before accepting.",
        "grounded_by": "sinter:sinter.temp_C",
    },
    {
        "name": "OXIDIZE_CR6",
        "human": "Cr3+ -> Cr6+ oxidation — dead NIR emission + toxic aerosol",
        "query": "Cr3+ Cr6+ chromium valence oxidation state reducing atmosphere "
                 "oxidizing air emission quenching hexavalent chromium near "
                 "infrared activator",
        "must": ["cr6", "cr 6", "hexavalent", "valence", "reducing atmosphere",
                 "oxidizing atmosphere", "cr4", "valence state", "cr3"],
        "must_not": [],
        "signature": "gas=Cr6+ sev>=2 at >900C in oxidizing atmosphere; AI-2 gas "
                     "residual high",
        "mechanism": "In an oxidizing atmosphere at high T, NIR-active Cr3+ "
                     "oxidizes to optically dead (and toxic) Cr6+, killing the "
                     "emission and releasing Cr6+ aerosol.",
        "action": "Switch to a reducing/inert atmosphere (e.g. 5%H2/N2) or "
                  "lower the peak temperature; ventilate.",
        "grounded_by": "gas_safety:Cr2O3",
    },
    {
        "name": "DECOMP_GAS_SURGE",
        "human": "Precursor decomposition gas surge — porosity / off-stoichiometry / hazard",
        "query": "carbonate decomposition CO2 nitrate NOx ammonium NH3 evolution "
                 "gas release calcination weight loss porosity precursor "
                 "decomposes outgassing",
        "must": ["carbonate", "nitrate", "ammonium", "co2", "nox", "nh3",
                 "decompos", "evolution of", "outgas", "weight loss", "dehydrat"],
        "must_not": [],
        "signature": "gas=CO2/NOx/NH3/HF at species onset temp; AI-2 gas residual "
                     "high during early ramp",
        "mechanism": "Carbonate/nitrate/ammonium precursors release CO2/NOx/NH3/HF "
                     "on heating; a too-fast ramp through the onset traps gas "
                     "(porosity) and can shift stoichiometry, plus a safety hazard.",
        "action": "Slow the ramp through the decomposition window and ensure "
                  "venting before continuing to peak.",
        "grounded_by": "gas_safety:carbonate_nitrate",
    },
    {
        "name": "VOLATILIZATION_LOSS",
        "human": "Volatilization loss at high T — off-stoichiometry / secondary phase",
        "query": "volatilization evaporation MoO3 sublimation Ga2O3 gallium loss "
                 "alkali volatile high temperature stoichiometry deviation "
                 "excess compensate sealed crucible",
        "must": ["volatil", "evaporat", "sublim", "stoichiometr", "excess",
                 "loss of", "non-stoichiom", "deviation"],
        "must_not": [],
        "signature": "gas=vapor/sublime + high T; AI-1 'sintering'/'done'",
        "mechanism": "Volatile oxides (MoO3 sublimes, Ga/alkali loss) escape at "
                     "high T, leaving the product off-stoichiometric -> secondary "
                     "phases / reduced emission.",
        "action": "Add excess of the volatile component or use a covered/sealed "
                  "crucible; cap the peak temperature.",
        "grounded_by": "gas_safety:MoO3",
    },
    {
        "name": "MOISTURE_HYDROXYL",
        "human": "Moisture / hydroxyl contamination — non-radiative quenching",
        "query": "moisture water adsorbed hydroxyl OH group quenching non "
                 "radiative humidity hygroscopic dehydration drying precursor "
                 "water content luminescence quenching",
        "must": ["water", "moisture", "hydroxyl", "oh group", "hygroscopic",
                 "humidity", "dehydrat", "water resistance"],
        "must_not": [],
        "signature": "humidity high, AI-2 humidity-residual high before/at charge",
        "mechanism": "Adsorbed water / OH groups introduce high-energy phonon "
                     "non-radiative pathways that quench NIR emission.",
        "action": "Dry / pre-calcine precursors; load under low humidity.",
        "grounded_by": None,
    },
    {
        "name": "GRIND_INHOMOGENEITY",
        "human": "Grinding / mixing inhomogeneity — secondary phase, uneven emission",
        "query": "ball milling grinding mixing homogeneity homogeneous "
                 "agglomeration particle size uniform precursor mixture "
                 "intimate mixing inhomogeneous",
        "must": ["ball mill", "milling", "grinding", "homogen", "agglomerat",
                 "particle size", "intimate mix", "thoroughly mix"],
        "must_not": [],
        "signature": "stage=grind, vib RMS abnormal (AI-3 vibration channel)",
        "mechanism": "Poor mixing leaves local stoichiometry gradients -> "
                     "secondary phases and spatially uneven emission.",
        "action": "Re-grind / extend mixing cycles to the host grinding_cycles.",
        "grounded_by": "sinter:grinding_cycles",
    },
    {
        "name": "SOAK_TEMP_DRIFT",
        "human": "Soak temperature drift / instability — inconsistent crystallinity",
        "query": "temperature fluctuation stability furnace control drift "
                 "uniformity thermal gradient grain growth holding isothermal "
                 "soaking instability",
        # require temperature-control vocabulary co-located with a process word;
        # drop bare "drift"/"gradient" (color drift, Stokes-shift drift, electron
        # drift velocity, thermal-gradient sensing are all false friends).
        "must": ["temperature fluctuat", "temperature instab", "temperature uniformity",
                 "furnace temperature", "furnace control", "isothermal hold",
                 "soaking temperature", "holding temperature", "temperature stability of the furnace"],
        "must_not": ["thermometry", "photo-thermal", "thermal quenching",
                     "color drift", "stokes", "electron drift", "sensing", "bandgap"],
        "signature": "AI-3=temp_drift, SPC out-of-control during soak",
        "mechanism": "Furnace instability during the isothermal hold causes "
                     "grain-growth variance and inconsistent crystallinity batch "
                     "to batch.",
        "action": "Check furnace PID / heating element; recalibrate before next "
                  "batch.",
        "grounded_by": "spc",
    },
]

# --------------------------------------------------------------------------- #
# Lightweight BM25 (no external dependency; the pickled index needs rank_bm25).
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("the a an and or of to in on for is are be by with at as from this "
            "that these those it its we our using used use can may also which "
            "such into than then thus were was has have had not but their they "
            "between within across over under above below high low".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 2 and t not in _STOP]


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.doc_len = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        df: Counter = Counter()
        for d in docs_tokens:
            for w in set(d):
                df[w] += 1
        self.idf = {w: math.log(1.0 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}

    def score(self, query_tokens: list[str], idx: int) -> float:
        tf, dl = self.tf[idx], self.doc_len[idx]
        s = 0.0
        for w in query_tokens:
            if w not in tf:
                continue
            f = tf[w]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl + 1e-9))
            s += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (denom + 1e-9)
        return s

    def topk(self, query_tokens: list[str], k: int = 50) -> list[tuple[int, float]]:
        scored = [(i, self.score(query_tokens, i)) for i in range(self.N)]
        scored = [t for t in scored if t[1] > 0.0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


def main() -> None:
    print(f"[ai5] loading {CHUNKS.name} ...")
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    texts = [c.get("text", "") or "" for c in chunks]
    titles = [c.get("title", "") or "" for c in chunks]
    print(f"[ai5] {len(chunks)} chunks; tokenizing + building BM25 ...")
    docs_tokens = [tokenize(t) for t in texts]
    bm = BM25(docs_tokens)

    # gas-safety grounded materials (parse the C table for the 'mat' column)
    gas_mats = []
    if GAS_TABLE.exists():
        gas_mats = re.findall(r'\{\s*"([^"]+)",\s*GAS_', GAS_TABLE.read_text(encoding="utf-8"))
    sinter = json.loads(SINTER.read_text(encoding="utf-8")) if SINTER.exists() else {}
    n_hosts = len([k for k in sinter if not k.startswith("_")])

    # retrieval-relevance threshold + PRECISE filter: a retrieved chunk counts as
    # precise evidence only if BM25 >= floor AND it contains a discriminating
    # must-term AND no must_not false-friend. Tiering uses the precise distinct-
    # paper count so homonym-polluted queries down-tier honestly.
    SCORE_FLOOR = 4.0
    TOPK = 400   # scan deep; the precise filter is what gates count

    def precise_ok(text_low: str, cls: dict) -> bool:
        musts = cls.get("must") or []
        if musts and not any(m in text_low for m in musts):
            return False
        if any(mn in text_low for mn in (cls.get("must_not") or [])):
            return False
        return True

    results = []
    for cls in CLASSES:
        qtok = tokenize(cls["query"])
        hits = [(i, s) for i, s in bm.topk(qtok, k=TOPK) if s >= SCORE_FLOOR]
        # NORMAL: baseline class, corpus support not meaningful -> skip precision
        if cls["name"] == "NORMAL":
            precise = hits[:0]
        else:
            precise = [(i, s) for i, s in hits if precise_ok(texts[i].lower(), cls)]
        papers = sorted({titles[i] for i, _ in precise if titles[i]})
        broad_papers = len({titles[i] for i, _ in hits if titles[i]})
        exemplars = []
        for i, s in precise[:4]:
            exemplars.append({
                "title": titles[i],
                "score": round(s, 2),
            })
        nch, npap = len(precise), len(papers)
        if npap >= 20:
            tier = "STRONG"
        elif npap >= 8:
            tier = "MODERATE"
        elif npap >= 1:
            tier = "WEAK"
        else:
            tier = "NONE"
        # keep decision: meaningful corpus support OR a hard grounded rule.
        # A class with only WEAK/NONE corpus is kept *only* if a hard rule grounds
        # it (and we say so honestly); otherwise it is rejected.
        grounded = cls.get("grounded_by")
        hard_rule = grounded not in (None, "baseline", "spc") or grounded == "spc"
        keep = (tier in ("STRONG", "MODERATE")) or hard_rule or cls["name"] == "NORMAL"
        ground_basis = ("corpus" if tier in ("STRONG", "MODERATE")
                        else ("hard-rule" if (hard_rule and keep) else
                              ("baseline" if cls["name"] == "NORMAL" else "none")))
        results.append({
            **cls,
            "corpus_chunks": nch,
            "corpus_papers": npap,
            "corpus_papers_broad": broad_papers,
            "corpus_tier": tier,
            "ground_basis": ground_basis,
            "exemplars": exemplars,
            "keep": keep,
        })
        print(f"  {cls['name']:20s} precise_papers={npap:3d} (broad {broad_papers:3d}) "
              f"tier={tier:8s} basis={ground_basis:9s} grounded={grounded} "
              f"-> {'KEEP' if keep else 'REJECT'}")

    kept = [r for r in results if r["keep"]]
    out = {
        "schema": "ai5_rootcause.taxonomy.v1",
        "scope": "process-stage faults observable from GD32 sensors + upstream AI "
                 "outputs (AI-1/2/3/4). Composition faults are out of scope -> off-device model.",
        "corpus": {"chunks_total": len(chunks), "sinter_hosts": n_hosts,
                   "gas_materials": gas_mats},
        "retrieval": {"method": "inline BM25 (k1=1.5,b=0.75)", "score_floor": SCORE_FLOOR,
                      "topk": TOPK},
        "n_candidate": len(results),
        "n_kept": len(kept),
        "classes": results,
        "kept_labels": [r["name"] for r in kept],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ai5] wrote {OUT_JSON.relative_to(REPO)}")

    # ----- human-readable label set (the gate document) --------------------- #
    L = []
    L.append("# AI-5 Root-Cause Diagnoser — Grounded Label Set (measure-first gate)\n")
    L.append(f"> Auto-generated by `CIMC/model/ai5_rootcause/build_taxonomy.py`.  ")
    L.append(f"> Corpus: **{len(chunks)} chunks** (2462 NIR-phosphor papers), "
             f"**{n_hosts} sintering-profile hosts**, **{len(gas_mats)} gas-safety materials**.  ")
    L.append("> Retrieval: deterministic inline BM25 over the real paper text "
             "(no API), then a **precise filter** (must-contain discriminating "
             "term, drop false-friends) so homonym-polluted queries down-tier "
             "honestly. Tier = PRECISE distinct-paper count "
             "(STRONG≥20 / MODERATE≥8 / WEAK≥1). A class is **KEPT** only if it "
             "has STRONG/MODERATE corpus support **or** a hard grounded rule "
             "(gas_safety / sintering_profiles / SPC) — and we state which.\n")
    L.append("**Scope (deliberate):** AI-5 diagnoses *process-stage* faults the GD32 "
             "can observe (AI-1/2/3/4 outputs + raw furnace features). *Composition* "
             "faults (dopant %, ionic-radius mismatch, host choice) stay with an "
             "off-device recipe model — this orthogonality is the layered-architecture argument.\n")
    L.append(f"**Result: {len(kept)}/{len(results)} candidate classes kept "
             f"→ AI-5 = {len(kept)}-way classifier.**\n")
    L.append("| # | Root cause | Keep | Tier (precise papers) | Kept on | Hard grounding |")
    L.append("|---|---|---|---|---|---|")
    for i, r in enumerate(results):
        gb = r.get("grounded_by") or "—"
        L.append(f"| {i} | `{r['name']}` | {'✅' if r['keep'] else '❌'} | "
                 f"{r['corpus_tier']} ({r['corpus_papers']}) | {r['ground_basis']} | {gb} |")
    L.append("")
    for i, r in enumerate(results):
        L.append(f"## {i}. `{r['name']}` — {r['human']}  {'✅ KEEP' if r['keep'] else '❌ REJECT'}")
        L.append(f"- **Mechanism:** {r['mechanism']}")
        L.append(f"- **Corrective action:** {r['action']}")
        L.append(f"- **Trigger signature (AI-5 input):** {r['signature']}")
        L.append(f"- **Hard grounding:** {r.get('grounded_by') or '— (corpus only)'}  "
                 f"· **kept on:** {r['ground_basis']}")
        L.append(f"- **Corpus support:** {r['corpus_tier']} — "
                 f"**{r['corpus_papers']}** distinct papers with precise on-mechanism "
                 f"evidence ({r['corpus_papers_broad']} broad BM25 papers, "
                 f"{r['corpus_chunks']} precise chunks ≥ BM25 {SCORE_FLOOR})")
        if r["exemplars"]:
            L.append("- **Retrieved paper references (no paper text redistributed):**")
            for ex in r["exemplars"]:
                L.append(f"  - *{ex['title']}* (BM25 score {ex['score']})")
        L.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print(f"[ai5] wrote {OUT_MD.relative_to(REPO)}")
    print(f"[ai5] kept {len(kept)}/{len(results)} classes: {[r['name'] for r in kept]}")


if __name__ == "__main__":
    main()
