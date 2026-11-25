"""Metrics, confidence intervals and descriptive tables.

Section 2 uses the descriptive tools (proportion_ci, cohort_table). From Section 3 on, every
model is scored with `summarise`: each accuracy metric is averaged over the cross-validation
repeats, with a 95% confidence interval from resampling patients (the patient-level bootstrap).
`error_by_range` shows where along the Part III scale the errors fall.

The models are judged on how close the predicted Part III is to the actual score. The recall
metrics for the 33-point threshold (sensitivity, specificity, AUROC, PR-AUC) are kept in
`recall_metrics` for the later call-back analysis, but are not part of the headline tables.
"""

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from . import paths


def proportion_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float, float]:
    """Share k/n with a Wilson confidence interval: (share, low, high)."""
    lo, hi = proportion_confint(k, n, alpha=1 - level, method="wilson")
    return k / n, lo, hi


def _mean_sd(s: pd.Series) -> str:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.mean():.1f} ± {s.std():.1f} (n={len(s)})" if len(s) else "-"


def _count_pct(mask: pd.Series) -> str:
    known = mask.dropna()
    return f"{int(known.sum())} ({known.mean():.0%}, n={len(known)})" if len(known) else "-"


def cohort_table(pd_df: pd.DataFrame, control_df: pd.DataFrame) -> pd.DataFrame:
    """Demographics of the PD and control groups side by side, with a test of the difference.

    Continuous variables: mean ± SD and Welch's t-test. Proportions: count (%) and Fisher's
    exact test. Each cell reports its own n because some fields are missing.
    """
    pd_age, hc_age = pd_df["Age (years)"], control_df["Age"]
    pd_male = pd_df["Sex"].str.lower().eq("male").where(pd_df["Sex"].notna())
    hc_male = control_df["Sex"].str.lower().eq("male").where(control_df["Sex"].notna())

    def welch(a, b):
        a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
        return stats.ttest_ind(a, b, equal_var=False).pvalue

    def fisher(a, b):
        a, b = a.dropna().astype(bool), b.dropna().astype(bool)
        return stats.fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]]).pvalue

    rows = [
        ("Participants", str(len(pd_df)), str(len(control_df)), np.nan),
        ("Age, years", _mean_sd(pd_age), _mean_sd(hc_age), welch(pd_age, hc_age)),
        ("Male", _count_pct(pd_male), _count_pct(hc_male), fisher(pd_male, hc_male)),
        ("Height, in", _mean_sd(pd_df["Height (in)"]), _mean_sd(control_df["Height (in)"]),
         welch(pd_df["Height (in)"], control_df["Height (in)"])),
        ("Weight, kg", _mean_sd(pd_df["Weight (kg)"]), _mean_sd(control_df["Weight (kg)"]),
         welch(pd_df["Weight (kg)"], control_df["Weight (kg)"])),
        ("Years since PD diagnosis", _mean_sd(pd_df["Years since PD diagnosis"]), "-", np.nan),
        ("Deep brain stimulation", _count_pct(pd_df["DBS?"].str.lower().eq("yes").where(pd_df["DBS?"].notna())),
         "-", np.nan),
    ]
    out = pd.DataFrame(rows, columns=["", "Parkinson's", "Controls", "p-value"]).set_index("")
    out["p-value"] = out["p-value"].map(lambda p: "" if pd.isna(p) else f"{p:.3f}" if p >= 0.001 else "<0.001")
    return out


# ---------------------------------------------------------------------------
# Model metrics (Section 3 onward)
# ---------------------------------------------------------------------------
METRICS = {
    "MAE": "Mean absolute error: average points off on the 0-132 scale (lower is better)",
    "RMSE": "Root mean squared error, in points: like MAE but large misses count more (lower is better)",
    "MSE": "Mean squared error, in points squared: RMSE before the square root; what the MSE-tuned models minimise (lower is better)",
    "R2": "Share of the variation in Part III explained, compared with predicting the average (1 is perfect, 0 is no better than the average, below 0 is worse)",
    "Spearman": "Rank correlation: does the model order patients from mild to severe? (-1 to 1, 0 is no ordering)",
    "Slope": "Calibration slope: actual Part III regressed on the prediction. 1 is ideal; above 1 means the predictions are squeezed towards the average, below 1 that they are too spread out",
}

# Kept for the later call-back analysis (the 33-point recall threshold); not in the headline tables.
RECALL_METRICS = {
    "Sensitivity": "Share of severe patients (Part III >= 33) whose prediction is >= 33",
    "Specificity": "Share of non-severe patients whose prediction is below 33",
    "AUROC": "Chance a random severe patient is scored above a random non-severe one (0.5 is a coin flip)",
    "PR-AUC": "Average precision for finding severe patients; the coin-flip level is the share severe",
}


def _auroc(flag: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum (Mann-Whitney) formula; ties count half."""
    n_pos = flag.sum()
    n_neg = len(flag) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(score)
    return (ranks[flag].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_precision(flag: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision, treating tied scores as one threshold (as scikit-learn does)."""
    n_pos = flag.sum()
    if n_pos == 0:
        return np.nan
    order = np.argsort(-score, kind="mergesort")
    s, f = score[order], flag[order]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]   # end of each block of tied scores
    tp = np.cumsum(f)[last]
    precision = tp / (last + 1)
    recall_gain = np.diff(np.r_[0, tp]) / n_pos
    return float((precision * recall_gain).sum())


def regression_metrics(y, pred) -> dict:
    """How close one set of predictions is to the actual Part III (the keys of METRICS).

    Spearman and the calibration slope are undefined for a constant prediction (Model 0 within
    one fold), so they are NaN there.
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    err = pred - y
    mse = (err ** 2).mean()
    ss_tot = ((y - y.mean()) ** 2).sum()
    varies = np.ptp(pred) > 0
    return {
        "MAE": np.abs(err).mean(),
        "RMSE": np.sqrt(mse),
        "MSE": mse,
        "R2": 1 - (err ** 2).sum() / ss_tot if ss_tot > 0 else np.nan,
        "Spearman": stats.spearmanr(y, pred).statistic if varies else np.nan,
        "Slope": np.cov(pred, y)[0, 1] / np.var(pred, ddof=1) if varies else np.nan,
    }


def recall_metrics(y, pred, threshold: float = paths.THRESHOLD) -> dict:
    """The recall decision at `threshold` (the keys of RECALL_METRICS). Kept for the later
    call-back analysis; the headline tables use regression_metrics only.

    A patient is flagged when the prediction is >= threshold; AUROC and PR-AUC use the
    predicted score itself, so they do not depend on where the threshold sits.
    """
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    flag, called = y >= threshold, pred >= threshold
    return {
        "Sensitivity": called[flag].mean() if flag.any() else np.nan,
        "Specificity": (~called[~flag]).mean() if (~flag).any() else np.nan,
        "AUROC": _auroc(flag, pred),
        "PR-AUC": _average_precision(flag, pred),
    }


def interval_metrics(y, lower, upper) -> dict:
    """Empirical coverage (share of patients inside their interval) and mean interval width."""
    y, lower, upper = (np.asarray(a, dtype=float) for a in (y, lower, upper))
    return {"Coverage": ((y >= lower) & (y <= upper)).mean(), "Mean width": (upper - lower).mean()}


def per_repeat(oof: pd.DataFrame) -> pd.DataFrame:
    """Metrics for each cross-validation repeat (each repeat predicts every patient once).

    `oof` needs one row per patient per repeat (use cv.by_patient for trial or window rows).
    """
    return pd.DataFrame({r: regression_metrics(g["y"], g["pred"])
                         for r, g in oof.groupby("repeat")}).T.rename_axis("repeat")


def summarise(oof: pd.DataFrame, n_boot: int = 1000, level: float = 0.95,
              seed: int = paths.SEED) -> pd.DataFrame:
    """The headline table for one model: estimate and patient-level bootstrap CI per metric.

    Estimate: the metric averaged over repeats. CI: resample patients with replacement
    n_boot times, recompute the repeat-averaged metric on each sample, and take percentiles.
    Resampling whole patients keeps the bootstrap honest about the sample being 97 people.
    Also reports the spread across repeats (min to max), which shows how much the folds alone
    move the result.
    """
    if oof.duplicated(["repeat", "patient"]).any():
        raise ValueError("summarise needs one row per patient per repeat; use cv.by_patient first")
    reps = per_repeat(oof)
    wide_y = oof.pivot(index="patient", columns="repeat", values="y")
    wide_p = oof.pivot(index="patient", columns="repeat", values="pred").loc[wide_y.index]
    Y, P = wide_y.to_numpy(), wide_p.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(Y)
    boot = np.empty((n_boot, len(METRICS)))
    with warnings.catch_warnings():   # a constant predictor has no Spearman: NaN, not an error
        warnings.simplefilter("ignore", RuntimeWarning)
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            boot[b] = np.nanmean([list(regression_metrics(Y[idx, r], P[idx, r]).values())
                                  for r in range(Y.shape[1])], axis=0)
        a = (1 - level) / 2
        lo, hi = np.nanquantile(boot, [a, 1 - a], axis=0)
    return pd.DataFrame({"estimate": reps.mean(), "ci_low": lo, "ci_high": hi,
                         "repeat_min": reps.min(), "repeat_max": reps.max(),
                         "patients": n, "repeats": Y.shape[1]}).rename_axis("metric")


def paired_difference(oof_a: pd.DataFrame, oof_b: pd.DataFrame, metric: str = "MAE",
                      n_boot: int = 1000, level: float = 0.95, seed: int = paths.SEED) -> dict:
    """Difference in a metric between two models scored on the same patients (b minus a).

    Both models are resampled with the same patients in each bootstrap draw, so the interval
    reflects only how differently the two models do, not which patients happened to be in the
    sample. An interval that excludes 0 means the difference is unlikely to be noise.
    """
    def wide(oof, col):
        return oof.pivot(index="patient", columns="repeat", values=col)

    ya, pa, pb = wide(oof_a, "y"), wide(oof_a, "pred"), wide(oof_b, "pred")
    if not (pa.index.equals(pb.index) and pa.columns.equals(pb.columns)):
        raise ValueError("the two models must be scored on the same patients and repeats")
    Y, A, B = ya.to_numpy(), pa.to_numpy(), pb.to_numpy()

    def diff(idx):
        return np.nanmean([regression_metrics(Y[idx, r], B[idx, r])[metric]
                           - regression_metrics(Y[idx, r], A[idx, r])[metric]
                           for r in range(Y.shape[1])])

    rng = np.random.default_rng(seed)
    n = len(Y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        est = diff(np.arange(n))
        boot = np.array([diff(rng.integers(0, n, n)) for _ in range(n_boot)])
        a = (1 - level) / 2
        lo, hi = np.nanquantile(boot, [a, 1 - a])
    return {"metric": metric, "difference": est, "ci_low": lo, "ci_high": hi, "patients": n}


def results_table(summaries: dict, metrics=("MAE", "RMSE", "R2", "Spearman", "Slope")) -> pd.DataFrame:
    """One row per model, one column per metric, each cell 'estimate (low to high)'.
    MSE, in points squared, is shown with no decimals."""
    def cell(s, m):
        r = s.loc[m]
        if pd.isna(r["estimate"]):
            return "n/a"
        f = ".0f" if m == "MSE" else ".2f"
        return f"{r['estimate']:{f}} ({r['ci_low']:{f}} to {r['ci_high']:{f}})"

    out = pd.DataFrame({name: {m: cell(s, m) for m in metrics} for name, s in summaries.items()}).T
    out.insert(0, "patients", [int(s["patients"].iloc[0]) for s in summaries.values()])
    return out.rename_axis("model")


def error_by_range(oof: pd.DataFrame, edges=None) -> pd.DataFrame:
    """Where along the Part III scale a model's errors fall.

    Patients are grouped by their actual Part III: by default into thirds of the cohort, or at
    the given `edges`. For each group: patients, actual range, MAE, and bias (mean of predicted
    minus actual; negative means the model predicts too low). Each is computed within each
    cross-validation repeat and averaged over the repeats.
    """
    if oof.duplicated(["repeat", "patient"]).any():
        raise ValueError("error_by_range needs one row per patient per repeat; use cv.by_patient first")
    actual = oof.groupby("patient")["y"].first()
    edges = np.quantile(actual, [1 / 3, 2 / 3]) if edges is None else np.asarray(edges, dtype=float)
    cuts = np.r_[-np.inf, edges, np.inf]
    names = ["low", "middle", "high"] if len(cuts) == 4 else [f"group {i + 1}" for i in range(len(cuts) - 1)]
    band_of = pd.Series(pd.cut(actual, cuts, labels=names, right=False), index=actual.index)

    d = oof.assign(band=oof["patient"].map(band_of), err=oof["pred"] - oof["y"])
    per = d.groupby(["band", "repeat"], observed=True)["err"].agg(
        MAE=lambda e: e.abs().mean(), bias="mean")
    out = per.groupby("band", observed=True).mean()
    rng = actual.groupby(band_of, observed=True).agg(["size", "min", "max"])
    out.insert(0, "patients", rng["size"].astype(int))
    out.insert(1, "actual Part III", [f"{lo:.0f} to {hi:.0f}" for lo, hi in zip(rng["min"], rng["max"])])
    return out.rename_axis("Part III range")
