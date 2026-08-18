import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from predict_engine import ts_torch as T
from predict_engine.dqb_regressor import DqBRegressor, CKPT_PATH, predict_dqb

m = DqBRegressor(); sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
m.load_state_dict(sd["model_state_dict"]); m.eval()
print("ckpt test MAE:", sd.get("test_mae_Dq_cm1"), sd.get("test_mae_B_cm1"))

print("\n-- predict_dqb() public API on presets --")
for fm, site in [("Y3Al5O12","Al"),("Gd3Al2Ga3O12","Ga"),("Y3Ga5O12","Ga"),("Mg2SiO4","Mg")]:
    r = predict_dqb(fm, site, 1.0)
    print(f"{fm:16s} Dq={r['Dq_cm1']:.0f} B={r['B_cm1']:.0f}")

print("\n-- direct m(formula_descriptor) on presets --")
for fm, site in [("Y3Al5O12","Al"),("Gd3Al2Ga3O12","Ga"),("Y3Ga5O12","Ga"),("Mg2SiO4","Mg")]:
    d = T.formula_descriptor(fm, site, 1.0)
    with torch.no_grad():
        o = m(d[None])
    print(f"{fm:16s} Dq={float(o['Dq_cm1']):.0f} B={float(o['B_cm1']):.0f}")

print("\n-- real corpus Dq spread + teacher on corpus --")
corpus = json.load(open(ROOT/"predictions"/"dqb_train_data.json", encoding="utf-8"))["samples"]
realDq = np.array([s["Dq_cm1"] for s in corpus])
print(f"real Dq: min={realDq.min():.0f} max={realDq.max():.0f} mean={realDq.mean():.0f} std={realDq.std():.0f}")
preds = []
with torch.no_grad():
    for s in corpus:
        d = T.formula_descriptor(s["formula"], s.get("site") or "Al", float(s.get("pct") or 1.0))
        preds.append(float(m(d[None])["Dq_cm1"]))
preds = np.array(preds)
print(f"teacher pred Dq on corpus: min={preds.min():.0f} max={preds.max():.0f} std={preds.std():.0f} MAE={np.mean(np.abs(preds-realDq)):.0f}")
# show a few worst formulas
order = np.argsort(-np.abs(preds-realDq))[:5]
for i in order:
    print(f"   {corpus[i]['formula']:18s} real={realDq[i]:.0f} pred={preds[i]:.0f}")
