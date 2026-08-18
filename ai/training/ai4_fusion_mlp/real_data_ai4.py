"""
real_data_ai4.py — Ground AI-4 fusion risk classifier in REAL lab outcomes
==========================================================================
CIMC Lab-Sentinel — answers "did we actually use the XRD project's real data?"

Source of truth
---------------
exp_ground_truth/observed_pl.csv  — 67 real NIR-phosphor synthesis records from
the lab group (cofire_exploration / reported_garnets / expanded_candidates xlsx).
Each row has the REAL batch outcome (`xrd_result` ∈ pure/mixed/amorphous/unknown),
REAL process conditions (sinter_temp_C / sinter_hours / atmosphere) and REAL quality
metrics (thermal_stability_pct_at_150C / quantum_yield_pct / fwhm / lambda_em).

predict_engine/sintering_profiles.json — 13 host-family ideal profiles (4 real DOI)
used as the REFERENCE the real conditions are compared against (the same profiles
AI-2/AI-3 are trained on). The deviation (real conditions − ideal profile) is a
genuine, computed quantity — not invented.

How this is honest (no fabrication, ADR-4)
------------------------------------------
* LABEL comes ONLY from the measured XRD phase result:
      pure → good(0) | mixed → bad(2) | amorphous → critical(3)
  `unknown`-phase rows are EXCLUDED (no phase ground truth). The intermediate
  `suspected(1)` class has no single-phase analogue and is left to the synthetic
  generator (a designed early-warning interpolation class).
* FEATURES use real measurements only:
    - AI-2 anomaly ratio / temp&gas residual  ← real (conditions vs profile + atm match)
    - quality residual                         ← real thermal_stability / quantum_yield
    - AI-1 (vision) / AI-3 (temp-curve) / DOA (audio) sub-signals are MARGINALISED to
      fixed in-class-neutral priors, because historical paper/xlsx records contain NO
      per-batch camera frame, temperature time-series, or microphone trace. This is
      flagged per-feature in `_REAL_MASK` and stated in the report — we do not pretend
      to have data we don't.
* Label is derived from PHASE; the predictive features are PROCESS-DEVIATION + THERMAL
  QUENCHING — physically distinct quantities, so the validation is non-circular.

A real, honest finding (used in the report)
--------------------------------------------
Most historical failures are COMPOSITIONAL (mixed/amorphous at nominal conditions),
not process anomalies — which is exactly why the system separates a (heavier, off-device)
composition-screening stage from the on-chip GD32 process sentinel. We report this split
rather than hiding it.

Outputs (into this dir):
  X_fusion_real.npy / y_fusion_real.npy   real-anchored fusion vectors + phase labels
  ai4_real_info.json                       provenance, feature mask, base rates, family map
  ../../docs/real_data_grounding.md        failure-mode analysis + data provenance (report §)

Run: cd CIMC/model/ai4_fusion_mlp && python real_data_ai4.py
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CSV_PATH      = HERE / ".." / ".." / ".." / "exp_ground_truth" / "observed_pl.csv"
PROFILES_PATH = HERE / ".." / ".." / ".." / "predict_engine" / "sintering_profiles.json"
DOC_PATH      = HERE / ".." / ".." / "docs" / "real_data_grounding.md"

N_FEAT = 16
CLASS_NAMES = ["good", "suspected", "bad", "critical"]

# phase outcome → risk label (label uses MEASURED PHASE ONLY — no quality leakage)
PHASE_TO_RISK = {"pure": 0, "mixed": 2, "amorphous": 3}   # unknown → excluded

# Which of the 16 fusion features are backed by a REAL measurement for historical rows.
# 1 = real-derived, 0 = marginalised (no per-batch trace in the records).
_REAL_MASK = np.array([
    0, 0, 0, 0,   # [0:4]  AI-1 vision probs       — no camera frame  → marginalised
    1,            # [4]    AI-2 anomaly ratio       — real (cond vs profile)
    1,            # [5]    AI-2 temp residual        — real (|Δtemp| vs profile)
    0,            # [6]    AI-2 vib  residual        — no vibration trace → marginalised
    1,            # [7]    AI-2 gas  residual        — real (atmosphere mismatch)
    0, 0, 0, 0, 0,# [8:13] AI-3 temp-curve probs    — no temperature time-series → marginalised
    0,            # [13]   DOA flag                  — no microphone trace → marginalised
    0,            # [14]   DOA intensity             — no microphone trace → marginalised
    1,            # [15]   sintering progress        — real (all records are COMPLETED batches → 1.0)
], dtype=np.int32)


# ── host-family inference (for rows whose host_family column is blank) ──────────
def infer_family(formula: str, declared: str) -> str:
    """Map a real formula to a sintering_profiles.json family key. Heuristic, documented."""
    d = (declared or "").strip().lower()
    if d in ("garnet",):
        # split Al-garnet vs Ga-garnet by which trivalent former dominates
        return "garnet_gallium" if formula.count("Ga") and "Al" not in formula else "garnet"
    if d in ("perovskite",):
        return "perovskite"
    if d in ("pyroxene",):
        return "default_unknown"          # no pyroxene profile; safe default
    f = formula
    if "O12" in f:                         # A3B5O12 garnet stoichiometry
        return "garnet_gallium" if ("Ga" in f and "Al" not in f) else "garnet"
    if "PO4" in f or "(PO4)" in f:
        return "phosphate"
    if "Ge" in f and "O12" not in f:
        return "germanate"
    if "Si2O6" in f:
        return "default_unknown"          # pyroxene-like, no profile
    if "WO6" in f or "NbO6" in f:
        return "perovskite"               # double perovskite B-site
    if f.endswith("O3") or "AlO3" in f or "GaO3" in f:
        return "perovskite"
    if "F" in f and "O" not in f:
        return "fluoride"
    return "default_unknown"


def _f(x):
    """Parse a CSV cell to float or None."""
    try:
        x = (x or "").strip()
        return float(x) if x not in ("", "+", "-") else None
    except ValueError:
        return None


# ── per-row real feature vector ─────────────────────────────────────────────────
def build_vector(row, profiles):
    """Return (vec16, label, meta) or (None,...) if row has no phase ground truth."""
    phase = (row.get("xrd_result") or "").strip().lower()
    if phase not in PHASE_TO_RISK:
        return None, None, None
    label = PHASE_TO_RISK[phase]

    formula  = (row.get("formula") or "").strip()
    family   = infer_family(formula, row.get("host_family"))
    prof     = profiles.get(family, profiles["default_unknown"])
    p_temp   = float(prof["sinter"]["temp_C"])
    p_hours  = float(prof["sinter"]["hours"])
    p_atm    = str(prof["sinter"]["atmosphere"]).lower()

    r_temp   = _f(row.get("sinter_temp_C"))
    r_hours  = _f(row.get("sinter_hours"))
    r_atm    = (row.get("atmosphere") or "").strip().lower()
    ts       = _f(row.get("thermal_stability_pct_at_150C"))
    qy       = _f(row.get("quantum_yield_pct"))

    # ── REAL process-deviation features (vs the host's reference profile) ──
    # temperature deviation: normalise by 100 °C; cold (under-temp) weighted more,
    # because under-firing is the process failure mode a furnace monitor must catch.
    dtemp = 0.0 if r_temp is None else (p_temp - r_temp) / 100.0       # +ve = under-fired
    temp_resid = min(abs(dtemp) / 2.0, 1.0)                             # [0,1]
    # hours deviation
    dhours = 0.0 if r_hours is None else (p_hours - r_hours) / 2.0      # +ve = under-soaked
    # atmosphere mismatch (air vs reducing etc.) → gas-channel residual
    gas_resid = 0.0 if (r_atm == "" or r_atm == p_atm) else 0.5
    # quality residual from REAL thermal quenching: low retention = bad sign.
    # (ts is a DIFFERENT physical quantity than phase purity → not a label leak.)
    qual = 0.0
    if ts is not None:
        qual = float(np.clip((60.0 - ts) / 60.0, 0.0, 1.0))            # ts<60 → >0, ts>=60 → 0
    if qy is not None:
        qual = max(qual, float(np.clip((40.0 - qy) / 40.0, 0.0, 1.0)))

    # AI-2 anomaly ratio = score/q_hat proxy: combine process deviation + quality
    ai2_ratio = 0.4 * temp_resid + 0.3 * abs(dhours) + 0.6 * gas_resid + 0.8 * qual
    ai2_ratio = float(np.clip(ai2_ratio, 0.0, 6.0))

    # ── marginalised sub-signals (documented; no per-batch trace exists) ──
    vec = np.zeros(N_FEAT, dtype=np.float32)
    vec[0:4] = [0.05, 0.10, 0.25, 0.60]      # AI-1: completed-batch neutral prior (done-leaning)
    vec[4]   = ai2_ratio                      # REAL
    vec[5]   = temp_resid                      # REAL
    vec[6]   = 0.10                            # vib residual neutral (no trace)
    vec[7]   = float(np.clip(gas_resid + 0.3 * qual, 0.0, 1.0))   # REAL (atm + quality)
    vec[8:13] = [0.70, 0.07, 0.13, 0.05, 0.05]  # AI-3: normal-leaning neutral prior
    vec[13]  = 0.0                             # DOA flag (no audio)
    vec[14]  = 0.0                             # DOA intensity (no audio)
    vec[15]  = 1.0                             # REAL: completed batch

    meta = {"formula": formula, "family": family, "phase": phase,
            "r_temp": r_temp, "p_temp": p_temp, "ts": ts, "qy": qy,
            "ai2_ratio": round(ai2_ratio, 3), "label": CLASS_NAMES[label]}
    return vec, label, meta


def load_profiles():
    return json.loads(PROFILES_PATH.read_text(encoding="utf-8"))


def parse_real():
    profiles = load_profiles()
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    X, y, metas = [], [], []
    excluded = 0
    for row in rows:
        vec, label, meta = build_vector(row, profiles)
        if vec is None:
            excluded += 1
            continue
        X.append(vec); y.append(label); metas.append(meta)
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, metas, len(rows), excluded


# ── failure-mode analysis + provenance doc ──────────────────────────────────────
def write_grounding_doc(X, y, metas, n_rows, excluded):
    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)}
    n_lab = len(y)
    # raw phase tally over ALL rows
    profiles = load_profiles()
    phase_tally = {"pure": 0, "mixed": 0, "amorphous": 0, "unknown": 0, "other": 0}
    atm_tally = {}
    ts_vals = []
    for row in csv.DictReader(CSV_PATH.open(encoding="utf-8")):
        ph = (row.get("xrd_result") or "").strip().lower()
        phase_tally[ph if ph in phase_tally else "other"] += 1
        atm = (row.get("atmosphere") or "").strip().lower() or "(blank)"
        atm_tally[atm] = atm_tally.get(atm, 0) + 1
        ts = _f(row.get("thermal_stability_pct_at_150C"))
        if ts is not None:
            ts_vals.append(ts)
    ts_arr = np.array(ts_vals) if ts_vals else np.array([0.0])

    lines = []
    lines.append("# AI-4 真实数据接地 — observed_pl.csv 失败模式分析与数据来源")
    lines.append("")
    lines.append("> 自动生成 by `model/ai4_fusion_mlp/real_data_ai4.py`。回答评委「数据集规范性」"
                 "与「实际测试」两项，并诚实标注哪些通道是真实测得、哪些因历史记录缺失而边缘化。")
    lines.append("")
    lines.append("## 1. 数据来源 (provenance)")
    lines.append("")
    lines.append(f"- **来源**: `exp_ground_truth/observed_pl.csv` — 课题组 NIR 荧光粉合成实测记录, "
                 f"共 **{n_rows} 行** (cofire_exploration / reported_garnets / expanded_candidates xlsx 汇总)。")
    lines.append(f"- **参考工艺**: `predict_engine/sintering_profiles.json` — 13 host-family 标准烧结 profile "
                 f"(含 4 条真实 DOI), 作为真实条件的偏差基准 (AI-2/AI-3 同源)。")
    lines.append(f"- **标注方式**: 标签直接取自实测 XRD 物相结果 `xrd_result` (pure/mixed/amorphous)，"
                 f"无人为编造。`unknown` 物相行无金标准 → 排除 ({excluded} 行)。")
    lines.append("")
    lines.append("## 2. 真实物相结果分布 (全 67 行)")
    lines.append("")
    lines.append("| xrd_result | 计数 | 含义 | → 风险标签 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| pure (单相) | {phase_tally['pure']} | 成相成功, 主要成功判据 | good |")
    lines.append(f"| mixed (杂相) | {phase_tally['mixed']} | 含杂质相, off-spec | bad |")
    lines.append(f"| amorphous (非晶) | {phase_tally['amorphous']} | 未结晶, 彻底失败 | critical |")
    lines.append(f"| unknown (未测) | {phase_tally['unknown']} | 无 XRD 金标准 | (排除) |")
    if phase_tally["other"]:
        lines.append(f"| other | {phase_tally['other']} | — | (排除) |")
    lines.append("")
    succ = phase_tally["pure"]
    fail = phase_tally["mixed"] + phase_tally["amorphous"]
    meas = succ + fail
    if meas:
        lines.append(f"**实测样本良率 = {succ}/{meas} = {100*succ/meas:.0f}%** "
                     f"(失败 {fail}: 杂相 {phase_tally['mixed']} + 非晶 {phase_tally['amorphous']})。")
    lines.append("")
    lines.append("## 3. 真实标注的 AI-4 验证集 (有金标准的行)")
    lines.append("")
    lines.append(f"映射出 **{n_lab} 条**真实标注 fusion 向量, 类别分布:")
    lines.append("")
    lines.append("| 风险类 | 计数 |")
    lines.append("|---|---|")
    for c in CLASS_NAMES:
        lines.append(f"| {c} | {counts[c]} |")
    lines.append("")
    lines.append("> 注: `suspected` 在单相/杂相二值物相结果里无对应 → 真实集只覆盖 good/bad/critical; "
                 "`suspected` 是合成生成器设计的早期预警过渡类 (见 synth_data_ai4.py)。"
                 "`critical` 仅 1 条真实非晶样本 (诚实标注, 不粉饰)。")
    lines.append("")
    lines.append("## 4. 特征真实性掩码 (16-D, 诚实标注)")
    lines.append("")
    lines.append("历史 xlsx/论文记录**没有逐批次的摄像头帧 / 温度时序 / 麦克风音轨**, 故对应通道边缘化到"
                 "类内中性先验; 真实测得的通道如下:")
    lines.append("")
    feat_names = ["AI-1 empty","AI-1 loaded","AI-1 sintering","AI-1 done",
                  "AI-2 anomaly_ratio","AI-2 temp_resid","AI-2 vib_resid","AI-2 gas_resid",
                  "AI-3 normal","AI-3 fast_ramp","AI-3 undertemp","AI-3 temp_drift","AI-3 slow_ramp",
                  "DOA flag","DOA intensity","progress"]
    lines.append("| # | 特征 | 真实来源? | 依据 |")
    lines.append("|---|---|---|---|")
    src = {4:"真实条件 vs profile + 热稳", 5:"|Δtemp| vs profile", 7:"气氛失配 + 热稳", 15:"已完成批 → 1.0"}
    for i, nm in enumerate(feat_names):
        real = "✅ 真实" if _REAL_MASK[i] else "○ 边缘化"
        why = src.get(i, "无逐批次记录")
        lines.append(f"| {i} | {nm} | {real} | {why} |")
    lines.append("")
    lines.append(f"真实测得通道占比: **{int(_REAL_MASK.sum())}/16**。标签 100% 来自实测物相。")
    lines.append("")
    lines.append("## 5. 关键诚实发现 (报告创新点)")
    lines.append("")
    lines.append("- **历史失败以「成分性」为主** (杂相/非晶发生在标称工艺条件下, 非升温曲线异常): "
                 "如 La3InGa4O12 (mixed) 与 Gd3InGa4O12 (pure) 同在 1400°C/5h, 失败源于配方本身能否成单相。")
    lines.append("- 这恰好论证**职责分层的必要性**: 成分筛选属上游配方模型 (离线, 不在本系统芯片上), "
                 "过程监工属 GD32 哨兵 (本系统), 二者正交互补 — 单一过程监测无法预判成分性失败。")
    lines.append(f"- 真实热稳定性分布: 均值 {ts_arr.mean():.0f}% / 中位 {np.median(ts_arr):.0f}% / "
                 f"范围 [{ts_arr.min():.0f}, {ts_arr.max():.0f}]% (n={len(ts_vals)}); "
                 "极端低值 (如 2.06%) 是真实「热猝灭废品」, 用作 AI-4 质量残差输入。")
    lines.append("")
    lines.append("## 6. 在 AI-4 训练/验证中的使用")
    lines.append("")
    lines.append("- **训练**: 合成 fusion 集 (类均衡, 见 ADR — 安全关键的 bad/critical 不可欠采) "
                 "+ 全部真实标注行混入 → 最终烧入固件的模型见过真实数据。")
    lines.append("- **真实验证 (留一交叉)**: 对每条真实行, 用 [合成 + 其余真实行] 训练、留出该行测试, "
                 "聚合得到真实留一准确率 (小样本统计最严谨做法)。结果见 `eval_report.txt`。")
    lines.append("")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {DOC_PATH}")


def main():
    X, y, metas, n_rows, excluded = parse_real()
    print(f"Parsed {n_rows} rows → {len(y)} labelled real fusion vectors ({excluded} excluded as unknown-phase)")
    print(f"  class counts: " + ", ".join(f"{CLASS_NAMES[i]}={int((y==i).sum())}" for i in range(4)))

    np.save(HERE / "X_fusion_real.npy", X)
    np.save(HERE / "y_fusion_real.npy", y)
    info = {
        "source_csv": "exp_ground_truth/observed_pl.csv",
        "reference_profiles": "predict_engine/sintering_profiles.json",
        "n_rows_total": n_rows,
        "n_labelled": int(len(y)),
        "n_excluded_unknown_phase": int(excluded),
        "phase_to_risk": PHASE_TO_RISK,
        "class_names": CLASS_NAMES,
        "class_counts": {CLASS_NAMES[i]: int((y == i).sum()) for i in range(4)},
        "real_feature_mask": _REAL_MASK.tolist(),
        "n_real_features": int(_REAL_MASK.sum()),
        "label_policy": "label from measured XRD phase ONLY (pure→good, mixed→bad, amorphous→critical); unknown excluded",
        "honesty_note": "vision/temp-curve/audio channels marginalised — no per-batch trace in historical records",
        "rows": metas,
    }
    (HERE / "ai4_real_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote X_fusion_real.npy {X.shape} / y_fusion_real.npy / ai4_real_info.json")

    write_grounding_doc(X, y, metas, n_rows, excluded)


if __name__ == "__main__":
    main()
