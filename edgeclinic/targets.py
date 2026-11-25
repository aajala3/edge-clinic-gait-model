"""The prediction target and the recall flag.

The target is the MDS-UPDRS Part III total (33 clinician-rated motor items, each 0-4, total 0-132).
The file has no total column, so it is computed here from the items, prorating when a few are
missing (Goetz et al., Movement Disorders 2015, doi:10.1002/mds.26153).
"""

import numpy as np
import pandas as pd

from . import io, paths


def item_means(df: pd.DataFrame, exclude_zeros: bool = False, whole: bool = False) -> pd.Series:
    """Mean score of each Part III item across the records in `df` (missing and not-rated excluded).

    With exclude_zeros=True, only records that scored the item above 0 count: the mean severity
    among patients who show the sign at all. With whole=True the means are rounded to whole
    numbers, so a filled item looks like a real 0-4 rating.
    """
    x = df[io.part3_items(df)].astype(float)
    m = (x.where(x > 0) if exclude_zeros else x).mean()
    return m.round() if whole else m


def fill_not_rated(df: pd.DataFrame, means: pd.Series | None) -> tuple[pd.DataFrame, pd.Series]:
    """The Part III items with not-rated cells (listed in `part3_not_rated`) filled from `means`.

    Returns the items as floats and the number of cells filled per record. With means=None,
    nothing is filled and not-rated cells stay missing.
    """
    x = df[io.part3_items(df)].astype(float)
    filled = pd.Series(0, index=df.index)
    if means is not None and "part3_not_rated" in df:
        for i, cells in df["part3_not_rated"].fillna("").items():
            for c in [c for c in cells.split(";") if c]:
                x.loc[i, c] = means[c]
                filled[i] += 1
    return x, filled


def updrs3_total(df: pd.DataFrame, max_missing: int = paths.MAX_MISSING_PART3,
                 not_rated_means: pd.Series | None = None) -> pd.DataFrame:
    """Part III total per record, prorated when up to `max_missing` items are missing.

    Items entered as "could not be rated" (listed in `part3_not_rated`) are filled with the
    item's mean from `not_rated_means` when given (see item_means), and then count as rated.
    Any remaining missing items are prorated (Goetz et al. 2015):
    prorated total = (sum of the items present) x 33 / (number of items present).
    Records with more than `max_missing` items missing get NaN.

    Returns a DataFrame aligned with `df` with columns:
      part3          the total (complete or prorated), NaN if too many items are missing
      part3_missing  number of the 33 items missing
      part3_prorated True when the total was prorated (1 to max_missing items missing)
      part3_imputed  number of not-rated items filled with the item mean
    """
    items = io.part3_items(df)
    if len(items) != paths.N_PART3_ITEMS:
        raise ValueError(f"expected {paths.N_PART3_ITEMS} Part III items, found {len(items)}")
    x, imputed = fill_not_rated(df, not_rated_means)
    n_missing = x.isna().sum(axis=1)
    n_present = paths.N_PART3_ITEMS - n_missing
    total = x.sum(axis=1, min_count=1) * paths.N_PART3_ITEMS / n_present.replace(0, np.nan)
    total = total.where(n_missing <= max_missing)
    return pd.DataFrame({
        "part3": total,
        "part3_missing": n_missing,
        "part3_prorated": (n_missing > 0) & (n_missing <= max_missing),
        "part3_imputed": imputed,
    }, index=df.index)


def recall_flag(y: pd.Series | np.ndarray, threshold: float = paths.THRESHOLD) -> np.ndarray:
    """1 if the patient should be brought back sooner (Part III >= threshold), else 0.

    Missing values raise an error rather than being silently counted as 0.
    """
    y = np.asarray(y, dtype=float)
    if np.isnan(y).any():
        raise ValueError("recall_flag received missing Part III values")
    return (y >= threshold).astype(int)


def analysis_set(clinical: pd.DataFrame, max_missing: int = paths.MAX_MISSING_PART3,
                 exclude_zeros: bool = True, whole: bool = True) -> pd.DataFrame:
    """The Version 1 PD records with a usable Part III, with the target columns added.

    Not-rated items (a "9") are filled with that item's mean across the other PD patients in
    `clinical`, counting only patients who scored it above 0 and rounded to a whole number
    (exclude_zeros=True, whole=True, as agreed for NLS036); other missing items are prorated.
    The item columns hold the filled values, so for records that are not prorated the items
    sum to part3.
    Adds part3, part3_missing, part3_prorated, part3_imputed, recall (the >= THRESHOLD flag), med_state
    (ON / OFF / OTHER from '3b'), minutes_since_dose (from '3c1') and stale_rating (rating
    more than STALE_DAYS from the gait session).
    """
    means = item_means(clinical, exclude_zeros, whole)
    out = pd.concat([clinical, updrs3_total(clinical, max_missing, means)], axis=1)
    out[io.part3_items(clinical)] = fill_not_rated(clinical, means)[0]   # items as used in the total
    out = out[out["part3"].notna()].copy()
    out["recall"] = recall_flag(out["part3"])
    out["med_state"] = io.medication_state(out["3b"])
    out["minutes_since_dose"] = io.minutes_since_dose(out["3c1"])
    out["stale_rating"] = out["Days since Part III Clinical Evaluation"].abs() > paths.STALE_DAYS
    return out.reset_index(drop=True)


def pair_v2_sessions(v2: pd.DataFrame, v1: pd.DataFrame | None = None,
                     max_missing: int = paths.MAX_MISSING_PART3) -> pd.DataFrame:
    """One row per Version 2 patient with both sessions side by side.

    Columns: subject_id, part3_s1, part3_s2, change (s2 - s1), days_between (from 'Days since
    previous session' on the session-2 record; NaN where not recorded), both_usable (both
    totals present), and s1_is_v1_visit (session 1's Part III items and age are identical to
    the patient's Version 1 record, so it must not be counted twice). s1_is_v1_visit is NA
    when `v1` is not given or the patient is not in Version 1.
    """
    t = pd.concat([v2[["subject_id", "session"]], updrs3_total(v2, max_missing)["part3"]], axis=1)
    wide = t.pivot_table(index="subject_id", columns="session", values="part3", aggfunc="first", dropna=False)
    wide = wide.reindex(columns=[1.0, 2.0])
    out = pd.DataFrame({"part3_s1": wide[1.0], "part3_s2": wide[2.0]})
    out["change"] = out["part3_s2"] - out["part3_s1"]
    out["both_usable"] = out[["part3_s1", "part3_s2"]].notna().all(axis=1)

    s2 = v2[v2["session"] == 2].set_index("subject_id")
    out["days_between"] = s2["Days since previous session"].reindex(out.index).astype(float)

    out["s1_is_v1_visit"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if v1 is not None:
        items = io.part3_items(v2)
        s1 = v2[v2["session"] == 1].set_index("subject_id")
        ref = v1.set_index("subject_id")
        for sid in out.index.intersection(s1.index).intersection(ref.index):
            a = s1.loc[sid, items + ["Age (years)"]].astype(float).to_numpy()
            b = ref.loc[sid, items + ["Age (years)"]].astype(float).to_numpy()
            out.loc[sid, "s1_is_v1_visit"] = bool(np.array_equal(a, b, equal_nan=True))
    return out.reset_index()


# ---------------------------------------------------------------------------
# The cleaned analysis table, as shared with the team
# ---------------------------------------------------------------------------
EXPORT_COLUMNS = {
    "subject_id": "Patient ID (Version 1)",
    "site_prefix": "Letters of the ID (NLS, WPD); appears to track site or cohort (unverified)",
    "Age (years)": "Age at the gait session",
    "Sex": "Sex",
    "Height (in)": "Height, inches",
    "Weight (kg)": "Weight, kg",
    "Years since PD diagnosis": "Disease duration, years",
    "DBS?": "Deep brain stimulation (Yes / No)",
    "med_state": "Medication state from Part III item 3b, grouped: ON / OFF / OTHER",
    "minutes_since_dose": "Minutes since last levodopa dose, parsed from 3c1",
    "Days since Part III Clinical Evaluation": "Days between the clinical rating and the gait session",
    "stale_rating": "True if the rating is more than 30 days from the gait session",
    "Modified Hoehn & Yahr Score": "Hoehn & Yahr stage: descriptive only, never a target or an input",
    "part3": "TARGET: MDS-UPDRS Part III total (0-132), complete, filled or prorated",
    "part3_missing": "Part III items missing (after filling not-rated items)",
    "part3_imputed": "Not-rated items filled with the item mean (zeros excluded, rounded)",
    "part3_prorated": "True if the total was prorated (Goetz et al. 2015)",
    "recall": "Recall flag: 1 if Part III >= 33",
}


def export_table(analysis: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The cleaned analysis set for sharing, plus a dictionary of its columns.

    Returns (data, columns): one row per patient with the columns in EXPORT_COLUMNS followed by
    the 33 cleaned Part III items, and a two-column table describing every column.
    """
    a = analysis.copy()
    a["site_prefix"] = a["subject_id"].str.extract(r"^([A-Z]+)", expand=False)
    items = io.part3_items(a)
    data = a[list(EXPORT_COLUMNS) + items]
    desc = dict(EXPORT_COLUMNS)
    desc.update({c: "Part III item (0-4) as used in the total; missing = not rated and prorated" for c in items})
    columns = pd.DataFrame({"column": list(desc), "meaning": list(desc.values())})
    return data, columns
