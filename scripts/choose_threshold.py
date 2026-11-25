"""Which gait_threshold to use for the clinic rule "come in if the Gait model predicts > threshold"
(needs the clinic = actual gait sub-score >= 4). Reads the Gait model's cached out-of-fold predictions (run gait_model.ipynb first);
nothing is retrained. Writes results/tables/threshold_*.csv.

AUROC does not depend on the threshold, and "fewest missed" alone is met by flagging everyone, so three trade-off rules:
  youden   : maximise sensitivity + specificity - 1
  errors   : minimise missed + unneeded visits
  sens80   : highest threshold whose sensitivity is >= 0.80 (fewest unneeded visits with at most 20% missed)
Honest estimate: in each of the 50 splits the threshold is chosen on the other four folds' out-of-fold predictions
(same repeat) and applied to the held-out fold. A sweep over fixed thresholds is written as description only.
"""
import glob, os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO); sys.path.insert(0, str(REPO))
import numpy as np
import pandas as pd
from edgeclinic import fusion, paths, evaluate

NEEDS_AT, FIXED = 4, 4.0
OUT = REPO / "results" / "tables"
oe = pd.read_parquet(glob.glob("cache/oof_gt_model3_sub_219_*.parquet")[0])
oc = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob("cache/oof_cnn_gait_moving_3723bdaeda/*.parquet")))
oof = fusion.average(oe, oc)
assert len(oof) == 970 and oof.groupby("repeat").size().eq(97).all()


def counts(y, p, t):
    need, flag = y >= NEEDS_AT, p > t
    return {"tp": (need & flag).sum(), "fn": (need & ~flag).sum(), "fp": (~need & flag).sum(), "tn": (~need & ~flag).sum()}


def metrics(y, p, t):
    c = counts(y, p, t)
    return {"Sensitivity": c["tp"] / (c["tp"] + c["fn"]), "Specificity": c["tn"] / (c["tn"] + c["fp"]),
            "PPV": c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else np.nan,
            "Missed": c["fn"], "Unneeded visits": c["fp"], "Patients flagged": c["tp"] + c["fp"],
            "Accuracy": (c["tp"] + c["tn"]) / len(y)}


def choose(y, p, rule):
    s = np.sort(np.unique(p))
    cand = np.r_[s[0] - 0.01, (s[:-1] + s[1:]) / 2, s[-1] + 0.01]
    rows = pd.DataFrame([metrics(y, p, t) | {"t": t} for t in cand])
    if rule == "youden":
        score = rows["Sensitivity"] + rows["Specificity"] - 1
        best = rows[score >= score.max() - 1e-12]
    elif rule == "errors":
        e = rows["Missed"] + rows["Unneeded visits"]
        best = rows[e <= e.min()]
    elif rule == "sens80":
        ok = rows[rows["Sensitivity"] >= 0.80]
        best = ok[ok["t"] >= ok["t"].max()]
    return float(best["t"].median())


# ---- honest: threshold chosen inside each split ----
RULES = {"youden": "Max sensitivity + specificity (Youden)", "errors": "Fewest missed + unneeded", "sens80": "Catch >= 80%, fewest unneeded"}
flags, chosen = [], []
for (r, k), test in oof.groupby(["repeat", "fold"]):
    other = oof[(oof["repeat"] == r) & (oof["fold"] != k)]
    for rule in RULES:
        t = choose(other["y"].to_numpy(), other["pred"].to_numpy(), rule)
        chosen.append({"rule": rule, "repeat": r, "fold": k, "threshold": t})
        flags.append(test[["repeat", "patient", "y", "pred"]].assign(rule=rule, t=t))
for t in [FIXED]:
    flags.append(oof[["repeat", "patient", "y", "pred"]].assign(rule="fixed4", t=t))
flags = pd.concat(flags, ignore_index=True)
chosen = pd.DataFrame(chosen)


def per_repeat(f):
    out = []
    for r, g in f.groupby("repeat"):
        need, flag = g["y"] >= NEEDS_AT, g["pred"] > g["t"]
        tp, fn, fp, tn = (need & flag).sum(), (need & ~flag).sum(), (~need & flag).sum(), (~need & ~flag).sum()
        out.append({"Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp), "PPV": tp / (tp + fp) if tp + fp else np.nan,
                    "Missed": fn, "Unneeded visits": fp, "Patients flagged": tp + fp, "Accuracy": (tp + tn) / len(g)})
    return pd.DataFrame(out).mean()


rng = np.random.default_rng(paths.SEED)
pats = oof["patient"].unique()
boots = [rng.choice(pats, len(pats)) for _ in range(1000)]
rows = []
for rule, f in flags.groupby("rule", sort=False):
    est = per_repeat(f)
    byp = {p: g for p, g in f.groupby("patient")}
    bs = pd.DataFrame([per_repeat(pd.concat([byp[p] for p in b])) for b in boots])
    thr = chosen.loc[chosen.rule == rule, "threshold"] if rule != "fixed4" else pd.Series([FIXED])
    for m in est.index:
        rows.append({"rule": RULES.get(rule, "Fixed gait_threshold = 4.0 (current)"), "metric": m, "estimate": est[m],
                     "ci_low": bs[m].quantile(0.025), "ci_high": bs[m].quantile(0.975)})
    rows.append({"rule": RULES.get(rule, "Fixed gait_threshold = 4.0 (current)"), "metric": "Threshold (median, range over splits)",
                 "estimate": thr.median(), "ci_low": thr.min(), "ci_high": thr.max()})
res = pd.DataFrame(rows)
res.to_csv(OUT / "threshold_nested.csv", index=False)
chosen.to_csv(OUT / "threshold_chosen.csv", index=False)
wide = res.pivot(index="metric", columns="rule", values="estimate")
print(f"AUROC (threshold-free): {evaluate._auroc((oof.y >= NEEDS_AT).to_numpy(), oof.pred.to_numpy()):.3f} pooled; per repeat mean "
      f"{np.mean([evaluate._auroc((g.y >= NEEDS_AT).to_numpy(), g.pred.to_numpy()) for _, g in oof.groupby('repeat')]):.3f}")
pd.set_option("display.width", 250)
print("\nThreshold chosen inside each split, applied to held-out patients (per-repeat averages; 95% CI by resampling patients):")
print(res.assign(v=res.apply(lambda x: f"{x.estimate:.2f} ({x.ci_low:.2f}-{x.ci_high:.2f})", axis=1))
      .pivot(index="metric", columns="rule", values="v").to_string())

# ---- descriptive sweep (not used to choose anything) ----
sweep = pd.DataFrame([{"threshold": t, **pd.DataFrame([metrics(g.y.to_numpy(), g.pred.to_numpy(), t)
                                                       for _, g in oof.groupby("repeat")]).mean()}
                      for t in np.round(np.arange(2.5, 6.01, 0.25), 2)])
sweep["Youden"] = sweep["Sensitivity"] + sweep["Specificity"] - 1
sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
print("\nFixed thresholds, all 50 splits (descriptive):")
print(sweep.round(2).to_string(index=False))
