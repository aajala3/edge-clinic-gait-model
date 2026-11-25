"""Late fusion of the CNN and a tabular model (Section 7), as in Lecture 15.

- average: the mean of the two models' out-of-fold predictions. Nothing is fitted, so it
  cannot leak.
- stack: a linear regression on the two models' predictions ("stacked aggregation"). Its
  weights must come from predictions on *training* patients that the base models did not
  fit, so for each outer fold both base models are cross-fitted again inside the training
  patients (inner_oof_tabular, inner_oof_cnn), the stacker is fitted on those inner
  predictions, and then applied to the outer test predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import cv, paths

KEYS = ["repeat", "fold", "row", "patient", "y"]


def average(*oofs: pd.DataFrame) -> pd.DataFrame:
    """Mean of several models' out-of-fold predictions, matched by repeat, fold and patient."""
    merged = oofs[0][KEYS + ["pred"]]
    for i, o in enumerate(oofs[1:], 1):
        merged = merged.merge(o[KEYS + ["pred"]], on=KEYS, suffixes=("", f"_{i}"), validate="one_to_one")
    if len(merged) != len(oofs[0]):
        raise ValueError("the models were not scored on the same patients and folds")
    pred_cols = [c for c in merged if c.startswith("pred")]
    return merged[KEYS].assign(pred=merged[pred_cols].mean(axis=1))


def _inner_folds(y, patients, train: np.ndarray, seed: int) -> list[cv.Fold]:
    return cv.repeated_folds(np.asarray(y)[train], np.asarray(patients)[train], n_splits=5, n_repeats=1, seed=seed)


def inner_oof_tabular(model, X: pd.DataFrame, y, patients, train: np.ndarray, seed: int,
                      name: str | None = None) -> pd.Series:
    """Cross-fitted predictions for the training patients `train` (positions), indexed by row.
    With `name`, the result is cached (cv.cached_oof)."""
    folds = _inner_folds(y, patients, train, seed)
    args = (model, X.iloc[train], np.asarray(y)[train], np.asarray(patients)[train], folds)
    oof = cv.cached_oof(name, *args) if name else cv.oof_predict(*args)
    return pd.Series(oof["pred"].to_numpy(), index=train[oof["row"].to_numpy()])


def inner_oof_cnn(name: str, windows: dict, y, patients, train: np.ndarray, seed: int, **kw) -> pd.Series:
    """As inner_oof_tabular, for the CNN: only the training patients' windows are used."""
    from . import cnn

    keep = np.isin(windows["meta"]["record_id"].to_numpy(), np.asarray(patients)[train])
    sub = {k: (v[keep] if k != "meta" else v[keep].reset_index(drop=True)) for k, v in windows.items()}
    folds = _inner_folds(y, patients, train, seed)
    oof = cnn.oof_cnn(name, sub, np.asarray(y)[train], np.asarray(patients)[train], folds, **kw)
    return pd.Series(oof["pred"].to_numpy(), index=train[oof["row"].to_numpy()])


def stack(outer: dict[str, pd.DataFrame], inner: dict[tuple[int, int], pd.DataFrame], y) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stacked late fusion.

    `outer` maps model name -> out-of-fold predictions (cv.oof_predict format). `inner` maps
    (repeat, fold) -> a DataFrame indexed by training row with one column per model name: the
    cross-fitted predictions for that fold's training patients. For each outer fold a linear
    regression (non-negative weights, with intercept) is fitted on the inner predictions and
    applied to the outer test predictions. Returns (fused predictions, weights per fold).
    """
    names = list(outer)
    test = outer[names[0]][KEYS].copy()
    for n in names:
        test = test.merge(outer[n][KEYS + ["pred"]].rename(columns={"pred": n}), on=KEYS, validate="one_to_one")
    y = np.asarray(y, float)
    out, weights = [], []
    for (r, k), P in inner.items():
        lr = LinearRegression(positive=True).fit(P[names].to_numpy(), y[P.index.to_numpy()])
        t = test[(test["repeat"] == r) & (test["fold"] == k)]
        out.append(t[KEYS].assign(pred=lr.predict(t[names].to_numpy())))
        weights.append({"repeat": r, "fold": k, "intercept": lr.intercept_, **dict(zip(names, lr.coef_))})
    return pd.concat(out, ignore_index=True), pd.DataFrame(weights)
