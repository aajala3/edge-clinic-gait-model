"""Model inputs and model definitions.

Every model is a scikit-learn estimator whose preprocessing (imputation, scaling) lives inside
it, so cv.oof_predict refits all of it on each training fold. Section 4 holds the baselines:

- Model 0: the training-fold mean (DummyRegressor)
- Model 1: clinical facts only, elastic net
- Model 2: clinical facts + walkway (PKMAS) gait metrics, elastic net

Section 6 holds the tabular models on the wearable features: elastic net, random forest and
gradient boosting, each with feature selection and tuning inside the training fold.

Every tuned model takes `objective`: "MAE" (the default) tunes its settings to minimise the
mean absolute error in the inner cross-validation, "MSE" the mean squared error.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import cv, paths

# What the inner cross-validation minimises when it tunes a model's settings.
OBJECTIVES = {"MAE": "neg_mean_absolute_error", "MSE": "neg_mean_squared_error"}


def _scoring(objective: str) -> str:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {list(OBJECTIVES)}, not {objective!r}")
    return OBJECTIVES[objective]

# Clinical inputs (Model 1): what the clinic knows without any gait test. Missing stays NaN
# and is imputed inside the folds.
CLINICAL = {
    "age": "Age at the gait session, years",
    "male": "1 if male",
    "years_since_diagnosis": "Years since PD diagnosis",
    "dbs": "1 if the patient has deep brain stimulation",
    "med_on": "1 if rated ON medication (item 3b)",
    "med_off": "1 if rated OFF medication (item 3b); ON and OFF both 0 means another state",
    "minutes_since_dose": "Minutes since the last levodopa dose (from 3c1)",
    "height_cm": "Height, cm (from inches); gait measures such as step length scale with body size",
    "weight_kg": "Body weight, kg",
}

# Walkway inputs (Model 2): the standard metrics of a clinical gait-lab report, from the
# ProtoKinetics (PKMAS) walkway, self-paced task. The file's FAP column is empty for every
# task, so the composite score is not used.
WALKWAY = {
    "Velocity (cm./sec.)": "Walking speed",
    "Cadence (steps/min.)": "Steps per minute",
    "Step Length (cm.) | Mean": "Average step length",
    "Stride Length (cm.) | Mean": "Average stride length (two steps)",
    "Stride Time (sec.) | Mean": "Average stride duration",
    "Stride Width (cm.) | Mean": "Average distance between the feet, side to side",
    "Total D. Support % | Mean": "Share of the gait cycle with both feet on the ground",
    "Swing % | Mean": "Share of the gait cycle with the foot in the air",
    "Stride Time (sec.) | %CV": "Stride-to-stride variability of stride time",
    "Step Length (cm.) | %CV": "Step-to-step variability of step length",
    "Step Length (cm.) | Mean - ASI": "Left/right step-length asymmetry index",
    "Mean eGVI": "Enhanced Gait Variability Index, averaged over both feet",
}


def clinical_features(analysis: pd.DataFrame) -> pd.DataFrame:
    """Model 1 inputs, one row per patient, aligned with `analysis`."""
    def yes(s):
        return s.str.lower().eq("yes").astype(float).where(s.notna())

    med = analysis["med_state"]
    X = pd.DataFrame({
        "age": analysis["Age (years)"].astype(float),
        "male": analysis["Sex"].str.lower().eq("male").astype(float).where(analysis["Sex"].notna()),
        "years_since_diagnosis": analysis["Years since PD diagnosis"].astype(float),
        "dbs": yes(analysis["DBS?"]),
        "med_on": med.eq("ON").astype(float).where(med.notna()),
        "med_off": med.eq("OFF").astype(float).where(med.notna()),
        "minutes_since_dose": analysis["minutes_since_dose"].astype(float),
        "height_cm": analysis["Height (in)"].astype(float) * 2.54,
        "weight_kg": analysis["Weight (kg)"].astype(float),
    }, index=analysis.index)
    return X[cv.check_inputs(list(CLINICAL))]


def walkway_features(analysis: pd.DataFrame, pkmas: pd.DataFrame) -> pd.DataFrame:
    """Model 2's walkway inputs, aligned with `analysis` by patient; NaN where a patient has none.

    Uses the PD, session-1 rows of `pkmas` (io.load_pkmas('SelfPace')).
    """
    pk = pkmas[(pkmas["group"] == "pd") & (pkmas["session"] == 1)].set_index("subject_id")
    if pk.index.duplicated().any():
        raise ValueError("more than one walkway record per patient")
    X = pk.reindex(analysis["subject_id"])[list(WALKWAY)].astype(float)
    X.index = analysis.index
    return X[cv.check_inputs(list(WALKWAY))]


def model0(strategy: str = "mean") -> DummyRegressor:
    """Model 0: predict the training patients' average Part III for everyone.

    strategy="mean" (the default) or "median"; the median is the single number with the
    lowest mean absolute error on the training patients.
    """
    return DummyRegressor(strategy=strategy)


ALPHAS = np.logspace(-2, 1.5, 15)
L1_RATIOS = [0.05, 0.25, 0.5, 0.75, 1.0]   # near ridge ... pure LASSO (Lecture 4)


def _linear_pipe(model) -> Pipeline:
    """Median imputation with missing-value flags, scaling, then a penalised linear model."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


# Ridge's alpha is on a different scale from LASSO / elastic net (scikit-learn does not divide its
# squared-error term by the number of patients), so it gets its own, wider grid.
RIDGE_ALPHAS = np.logspace(-1, 4, 21)


def lasso(inner_splits: int = 5, seed: int = paths.SEED, objective: str = "MAE",
          n_bins: int | None = None) -> GridSearchCV:
    """As elastic_net, with the pure LASSO penalty (Lecture 4): shrinks coefficients and sets the
    least useful to exactly zero. Only the penalty strength is tuned."""
    return GridSearchCV(_linear_pipe(Lasso(max_iter=50_000)), {"model__alpha": ALPHAS},
                        cv=cv.PatientKFold(inner_splits, random_state=seed, n_bins=n_bins),
                        scoring=_scoring(objective), n_jobs=-1)


def ridge(inner_splits: int = 5, seed: int = paths.SEED, objective: str = "MAE",
          n_bins: int | None = None) -> GridSearchCV:
    """As elastic_net, with the pure ridge penalty (Lecture 4): shrinks every coefficient towards
    zero but keeps them all. Only the penalty strength is tuned (on RIDGE_ALPHAS)."""
    return GridSearchCV(_linear_pipe(Ridge()), {"model__alpha": RIDGE_ALPHAS},
                        cv=cv.PatientKFold(inner_splits, random_state=seed, n_bins=n_bins),
                        scoring=_scoring(objective), n_jobs=-1)


def elastic_net(inner_splits: int = 5, seed: int = paths.SEED, objective: str = "MAE",
                n_bins: int | None = None) -> GridSearchCV:
    """Median imputation (with missing-value flags), scaling and an elastic net, tuned by an
    inner patient-level CV on `objective` (mean absolute or mean squared error). `n_bins`
    stratifies the inner folds on quantile bins of Part III instead of the recall flag.

    The mix l1_ratio runs from near-ridge (0.05) to pure LASSO (1.0), so the inner CV chooses
    between the two Lecture 4 penalties and the blends in between.
    """
    pipe = _linear_pipe(ElasticNet(max_iter=50_000))
    grid = {"model__alpha": ALPHAS, "model__l1_ratio": L1_RATIOS}
    return GridSearchCV(pipe, grid, cv=cv.PatientKFold(inner_splits, random_state=seed, n_bins=n_bins),
                        scoring=_scoring(objective), n_jobs=-1)


def coefficients(search: GridSearchCV) -> pd.Series:
    """Standardised coefficients of a fitted elastic_net(), lasso() or ridge(), named, largest first.

    Missing-value flags appear as '<column> missing'. Coefficients are per standard deviation
    of each input, so their sizes can be compared.
    """
    pipe = search.best_estimator_
    names = list(pipe.named_steps["impute"].get_feature_names_out())
    names = [n.replace("missingindicator_", "") + " missing" if n.startswith("missingindicator_") else n
             for n in names]
    coef = pd.Series(pipe.named_steps["model"].coef_, index=names)
    return coef.reindex(coef.abs().sort_values(ascending=False).index)


class Clip(TransformerMixin, BaseEstimator):
    """Clip each column to its training-fold quantiles (winsorising), so a few extreme but real
    values (e.g. a very jerky wrist) cannot dominate a linear model. Fitted inside the fold."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_, self.high_ = np.nanquantile(X, self.lower, axis=0), np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


# Number of features kept by the in-fold univariate screen (correlation with Part III on the
# training patients only); "all" lets the model see every feature.
K_FEATURES = [10, 30, 60, "all"]


def _search(pipe: Pipeline, grid: dict, inner_splits: int, seed: int, objective: str) -> GridSearchCV:
    return GridSearchCV(pipe, grid, cv=cv.PatientKFold(inner_splits, random_state=seed),
                        scoring=_scoring(objective), n_jobs=-1)


def sensor_elastic_net(inner_splits: int = 5, seed: int = paths.SEED, objective: str = "MAE") -> GridSearchCV:
    """Elastic net for many correlated features: median imputation, clipping to the 1st-99th
    training percentiles, scaling, a univariate screen to the best k features, then the
    penalty. k, the penalty strength and the LASSO/ridge mix are all tuned in the inner CV."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", Clip()),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_regression)),
        ("model", ElasticNet(max_iter=50_000)),
    ])
    grid = {"select__k": K_FEATURES, "model__alpha": ALPHAS, "model__l1_ratio": L1_RATIOS}
    return _search(pipe, grid, inner_splits, seed, objective)


def random_forest(inner_splits: int = 5, seed: int = paths.SEED, objective: str = "MAE") -> GridSearchCV:
    """Random forest (Lecture 5) after the same in-fold screen. Tuned: k, the share of features
    tried at each split, and the minimum patients per leaf (larger = smoother)."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(f_regression)),
        ("model", RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1)),
    ])
    grid = {"select__k": K_FEATURES, "model__max_features": [0.1, 0.33, 1.0],
            "model__min_samples_leaf": [3, 8]}
    return _search(pipe, grid, inner_splits, seed, objective)


def gradient_boosting(inner_splits: int = 5, seed: int = paths.SEED, objective: str = "MAE") -> GridSearchCV:
    """Gradient boosting (scikit-learn HistGradientBoostingRegressor) after the same in-fold
    screen. Small trees and slow learning rates suit 77 training patients."""
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(f_regression)),
        ("model", HistGradientBoostingRegressor(early_stopping=False,
                                                random_state=seed)),
    ])
    grid = {"select__k": [10, 30, "all"], "model__learning_rate": [0.01, 0.03, 0.1],
            "model__max_depth": [2, 3], "model__max_iter": [100, 300],
            "model__min_samples_leaf": [10, 20]}
    return _search(pipe, grid, inner_splits, seed, objective)


def selected_features(search: GridSearchCV, columns) -> list[str]:
    """Names of the features a fitted sensor model kept after its in-fold screen."""
    keep = search.best_estimator_.named_steps["select"].get_support()
    return [c for c, k in zip(columns, keep) if k]
