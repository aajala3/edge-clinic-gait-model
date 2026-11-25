"""Patient-level cross-validation.

Every split in the project goes through this module, so all models are tested on the same
patients in the same folds. Rows can be patients, trials or windows: whatever they are,
`groups` holds the patient ID, and all rows of a patient land on the same side of a split.
By default folds are stratified on the recall flag (Part III >= THRESHOLD), so each test fold
gets its share of the 21 severe patients. With `n_bins`, they are stratified on quantile bins of
Part III instead (e.g. n_bins=5: quintiles), which balances the whole score range across folds
without any threshold.
"""

import hashlib
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection._search import BaseSearchCV

from . import paths


# Columns that carry the answer: the MDS-UPDRS items and parts, H&Y, the computed target and
# flag, the rater's dyskinesia notes and the video-annotated freezing / stagger episodes
# (ClinicalEvent). None of them may ever be a model input.
FORBIDDEN_INPUTS = re.compile(r"MDSUPDRS|UPDRS|Hoehn|Yahr|^part3|^recall$|Dyskinesia|ClinicalEvent|^fog_|^stagger_",
                              re.IGNORECASE)


def check_inputs(columns) -> list[str]:
    """Raise if any input column is target-derived (see FORBIDDEN_INPUTS); return the columns."""
    bad = [c for c in columns if FORBIDDEN_INPUTS.search(str(c))]
    if bad:
        raise ValueError(f"target-derived columns cannot be model inputs: {bad[:5]}")
    return list(columns)


def quantile_strata(y, n_bins: int) -> np.ndarray:
    """Label each value of y with its quantile bin (0 .. n_bins-1), computed on y itself."""
    y = np.asarray(y, dtype=float)
    inner_edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(inner_edges, y, side="right")


class PatientKFold:
    """K-fold split grouped by patient and stratified on the continuous Part III target.

    Strata: with `n_bins` (e.g. 5), the quantile bin of y, recomputed on whatever y the split
    is given (so the inner loop of a search uses its own training patients' quantiles);
    otherwise the recall flag y >= threshold. Usable anywhere scikit-learn accepts a `cv`
    splitter, e.g. as the inner loop of GridSearchCV (pass groups= to fit).
    """

    def __init__(self, n_splits: int = 5, threshold: float = paths.THRESHOLD,
                 random_state: int = paths.SEED, n_bins: int | None = None):
        self.n_splits, self.threshold, self.random_state = n_splits, threshold, random_state
        self.n_bins = n_bins

    def __repr__(self) -> str:
        return (f"PatientKFold(n_splits={self.n_splits}, threshold={self.threshold}, "
                f"random_state={self.random_state}, n_bins={self.n_bins})")

    def split(self, X, y, groups):
        if groups is None:
            raise ValueError("PatientKFold needs groups (the patient ID of every row)")
        if self.n_bins:
            strata = quantile_strata(y, self.n_bins)
        else:
            strata = (np.asarray(y, dtype=float) >= self.threshold).astype(int)
        skf = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.random_state)
        yield from skf.split(np.zeros(len(strata)), strata, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


class Fold(NamedTuple):
    repeat: int
    fold: int
    train: np.ndarray   # row positions
    test: np.ndarray


def repeated_folds(y, groups, n_splits: int = 5, n_repeats: int = 10,
                   seed: int = paths.SEED, n_bins: int | None = None) -> list[Fold]:
    """The outer cross-validation: n_repeats reshuffles of a patient-level n_splits-fold split.

    Repeat r uses random_state seed + r, so the folds are the same on every run and for every
    model. Each repeat tests every patient exactly once. `n_bins` stratifies on quantile bins
    of y instead of the recall flag (see PatientKFold).
    """
    folds = []
    for r in range(n_repeats):
        splitter = PatientKFold(n_splits, random_state=seed + r, n_bins=n_bins)
        for k, (tr, te) in enumerate(splitter.split(None, y, groups)):
            folds.append(Fold(r, k, tr, te))
    check_patient_split(folds, groups)
    return folds


def check_patient_split(folds: list[Fold], groups) -> None:
    """Raise if any patient is on both sides of a split, or a repeat misses or repeats a patient."""
    groups = np.asarray(groups)
    for f in folds:
        shared = set(groups[f.train]) & set(groups[f.test])
        if shared:
            raise ValueError(f"repeat {f.repeat} fold {f.fold}: patients in train and test: {sorted(shared)[:5]}")
    for r in sorted({f.repeat for f in folds}):
        tested = np.concatenate([f.test for f in folds if f.repeat == r])
        if len(tested) != len(groups) or len(np.unique(tested)) != len(groups):
            raise ValueError(f"repeat {r} does not test every row exactly once")


def fold_balance(folds: list[Fold], groups, y, threshold: float | None = paths.THRESHOLD) -> pd.DataFrame:
    """One row per outer test fold: patients, mean and SD of Part III, and (unless threshold is
    None) the number of patients at or above the threshold."""
    groups, y = np.asarray(groups), np.asarray(y, dtype=float)
    rows = []
    for f in folds:
        pat = pd.Series(y[f.test]).groupby(groups[f.test]).mean()   # one value per patient
        row = {"repeat": f.repeat, "fold": f.fold, "test patients": len(pat)}
        if threshold is not None:
            row["test severe"] = int((pat >= threshold).sum())
        row.update({"test mean Part III": pat.mean(), "test SD Part III": pat.std(ddof=1),
                    "train patients": len(np.unique(groups[f.train]))})
        rows.append(row)
    return pd.DataFrame(rows)


def subset_folds(folds: list[Fold], mask) -> list[Fold]:
    """The same folds restricted to the rows in `mask`, with positions renumbered for the subset.

    Used to compare two models on the same patients (e.g. only those with walkway data)
    without reshuffling: each patient keeps the fold it had in the full analysis.
    """
    mask = np.asarray(mask, dtype=bool)
    new_pos = np.full(len(mask), -1)
    new_pos[mask] = np.arange(mask.sum())
    return [Fold(f.repeat, f.fold, new_pos[f.train[mask[f.train]]], new_pos[f.test[mask[f.test]]])
            for f in folds]


def oof_predict(model, X, y, groups, folds: list[Fold], return_models: bool = False):
    """Out-of-fold predictions: a fresh copy of `model` is fitted on each training fold.

    Preprocessing must live inside `model` (a Pipeline) so it is refitted on training rows
    only. If `model` is a search (GridSearchCV etc.), it gets the training patients' IDs as
    groups, so tuning uses an inner split that is also by patient (nested CV).

    Returns one row per test row per repeat: repeat, fold, row, patient, y, pred (and, for a
    search, `params`: the settings the inner CV chose on that fold). With return_models=True,
    also returns the fitted models in fold order.
    """
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    y, groups = np.asarray(y, dtype=float), np.asarray(groups)
    out, fitted = [], []
    for f in folds:
        m = clone(model)
        fit_kw = {"groups": groups[f.train]} if isinstance(m, BaseSearchCV) else {}
        m.fit(X.iloc[f.train], y[f.train], **fit_kw)
        rows = pd.DataFrame({"repeat": f.repeat, "fold": f.fold, "row": f.test,
                             "patient": groups[f.test], "y": y[f.test],
                             "pred": m.predict(X.iloc[f.test])})
        if isinstance(m, BaseSearchCV):
            rows["params"] = repr({k.split("__")[-1]: (v.item() if hasattr(v, "item") else v)
                                   for k, v in m.best_params_.items()})
        out.append(rows)
        fitted.append(m)
    oof = pd.concat(out, ignore_index=True)
    return (oof, fitted) if return_models else oof


def by_patient(oof: pd.DataFrame) -> pd.DataFrame:
    """Average trial- or window-level predictions to one prediction per patient per repeat."""
    return (oof.groupby(["repeat", "patient"], as_index=False)
               .agg(fold=("fold", "first"), y=("y", "mean"), pred=("pred", "mean")))


def _fingerprint(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
    return h.hexdigest()[:10]


def cached_oof(name: str, model, X, y, groups, folds: list[Fold]) -> pd.DataFrame:
    """oof_predict, saved to cache/oof_<name>_<hash>.parquet and reused on the next run.

    The hash covers the model's settings, the code in models.py and cv.py, the inputs, the target and
    the folds, so changing any of them refits instead of reusing a stale result.
    """
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    code = b"".join((Path(__file__).parent / f).read_bytes() for f in ("models.py", "cv.py"))
    # repr(model) can contain memory addresses (e.g. '<function f_regression at 0x...>') that change
    # between Python sessions; strip them so the same model always gets the same key.
    key = _fingerprint(re.sub(r" at 0x[0-9a-fA-F]+", "", repr(model)), code, list(X.columns),
                       pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes(),
                       np.asarray(y, dtype=float).tobytes(), np.asarray(groups).tolist(),
                       [(f.repeat, f.fold, f.test.tolist()) for f in folds])
    path = paths.CACHE_DIR / f"oof_{name}_{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    oof = oof_predict(model, X, y, groups, folds)
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(path)
    return oof
