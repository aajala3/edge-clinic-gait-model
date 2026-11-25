"""Reading the data: the integrity check, the clinical tables, the walkway metrics and the trial files.

Each loader fixes the quirks recorded in the data card, so nothing downstream has to:

- clinical tables have two header rows (section, then variable) and code missing values as "-";
- three control trial files are named like "NLS145 (control)_SelfPace.csv";
- Version 2 record IDs carry the session, e.g. "NLS005s1";
- the walkway table has two header rows (metric, then statistic), trailing spaces in task names,
  and second-session rows marked "_2" (e.g. "NLS103_2");
- the trial files store time as text, e.g. "0.01 sec".
"""

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

MISSING = {"-", "", "nan", "NaN"}
NOT_RATED_CODES = {"9"}   # an MDS-UPDRS item entered as 9 = could not be rated


# ---------------------------------------------------------------------------
# Integrity check
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, read in 1 MB chunks so large trials fit in memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_manifest(
    sample: int | None = 20,
    data_dir: Path = paths.DATA_DIR,
    manifest: Path = paths.MANIFEST,
    seed: int = paths.SEED,
) -> pd.DataFrame:
    """Check the downloaded files against MANIFEST.csv.

    Every file is checked for existence and size (cheap). A random sample of
    `sample` files, drawn within each version so both are covered, is also
    re-hashed with SHA-256 (slow: the data are about 7.5 GB). Pass
    sample=None to re-hash everything.

    Returns one row per manifest file with columns relative_path, version_dir,
    exists, size_ok, hashed and hash_ok (hash_ok is NA when not hashed).
    """
    m = pd.read_csv(manifest)
    m["version_dir"] = m["relative_path"].str.split("/").str[0]
    full = [data_dir / p for p in m["relative_path"]]
    m["exists"] = [p.exists() for p in full]
    m["size_ok"] = [p.exists() and p.stat().st_size == b for p, b in zip(full, m["bytes"])]

    if sample is None:
        to_hash = m.index[m["exists"]]
    else:
        # Spread the sample across the version folders in proportion to their size.
        rng = np.random.default_rng(seed)
        to_hash = []
        for _, g in m[m["exists"]].groupby("version_dir"):
            k = max(1, round(sample * len(g) / m["exists"].sum()))
            to_hash.extend(rng.choice(g.index, size=min(k, len(g)), replace=False))

    m["hashed"] = m.index.isin(to_hash)
    m["hash_ok"] = pd.Series(pd.NA, index=m.index, dtype="boolean")
    for i in to_hash:
        m.loc[i, "hash_ok"] = sha256_file(full[i]) == m.loc[i, "sha256"]

    return m[["relative_path", "version_dir", "exists", "size_ok", "hashed", "hash_ok"]]


def manifest_summary(check: pd.DataFrame) -> pd.DataFrame:
    """One row per version folder: files listed, found, right size, hashed, hash matches."""
    return (
        check.groupby("version_dir")
        .agg(
            files=("relative_path", "size"),
            found=("exists", "sum"),
            size_ok=("size_ok", "sum"),
            hashed=("hashed", "sum"),
            hash_ok=("hash_ok", lambda s: int(s.fillna(False).sum())),
        )
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace in every text cell and turn the missing codes into NA."""
    out = df.apply(lambda s: s.str.strip() if s.dtype == object or pd.api.types.is_string_dtype(s) else s)
    return out.mask(out.isin(MISSING))


def _to_numeric_where_possible(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a column to numbers only when every non-missing value parses as a number.

    Columns with any text (free-text medications, "ON"/"OFF", clock times) stay as text,
    so nothing is silently turned into NA.
    """
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.notna().any():
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() == s.notna().sum():
                out[c] = num
    return out


def _dedupe(names: list[str]) -> list[str]:
    """Make column names unique by appending .1, .2, ... to repeats."""
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def split_record_id(record_id: str) -> tuple[str, float]:
    """Split a Version 2 record ID like 'NLS005s1' into ('NLS005', 1).

    IDs without a session suffix (Version 1) return the ID unchanged and NaN.
    """
    m = re.fullmatch(r"(.+?)s(\d+)", record_id)
    return (m.group(1), float(m.group(2))) if m else (record_id, np.nan)


# ---------------------------------------------------------------------------
# Clinical tables
# ---------------------------------------------------------------------------
CLINICAL_FILES = {
    ("v1", "pd"): paths.V1_PD_CLINICAL,
    ("v1", "control"): paths.V1_CONTROL_CLINICAL,
    ("v2", "pd"): paths.V2_PD_CLINICAL,
}


def load_clinical(version: str = "v1", group: str = "pd", path: Path | None = None) -> pd.DataFrame:
    """Load a demographic + clinical table as one tidy row per record.

    version: "v1" (cross-sectional) or "v2" (longitudinal, PD only).
    group:   "pd" or "control".

    Columns are named by the second header row (e.g. "Age (years)", "MDSUPDRS_3-1", "3b").
    Added at the front: record_id (as written in the file), subject_id (session suffix removed),
    session (1, 2, ... in Version 2; NaN in Version 1), version and group. A column whose
    variable name is blank in the file is dropped.
    """
    path = path or CLINICAL_FILES[(version, group)]
    raw = pd.read_csv(path, header=[0, 1], dtype=str, keep_default_na=False)

    names = [str(var).strip() for _, var in raw.columns]
    keep = [not n.startswith("Unnamed") for n in names]
    df = raw.loc[:, keep]
    df.columns = _dedupe([n for n, k in zip(names, keep) if k])

    df = _clean_strings(df)
    # MDS-UPDRS items are scored 0-4; a rater's note can follow the score (NLS143 has
    # "2*parationia" on two rigidity items), so keep the leading score and drop the note.
    # A "9" means the item could not be rated (NLS036, postural tremor). It becomes NA like
    # any other missing item, but is listed in part3_not_rated so it can be imputed differently.
    item_cols = [c for c in df.columns if c.startswith("MDSUPDRS_")]
    p3_cols = [c for c in item_cols if c.startswith("MDSUPDRS_3-")]
    not_rated = df[p3_cols].apply(lambda s: s.astype("string").str.strip().isin(NOT_RATED_CODES))
    df["part3_not_rated"] = not_rated.apply(lambda r: ";".join(r.index[r.fillna(False).astype(bool)]), axis=1)
    for c in item_cols:
        df[c] = item_score(df[c])
    df = _to_numeric_where_possible(df)
    df = df[df["Subject ID"].notna()].reset_index(drop=True)

    split = df["Subject ID"].astype(str).map(split_record_id)
    ids = pd.DataFrame({
        "record_id": df["Subject ID"].astype(str),
        "subject_id": split.str[0],
        "session": split.str[1].astype(float) if version == "v2" else np.nan,
        "version": version,
        "group": group,
    })
    return pd.concat([ids, df.drop(columns="Subject ID")], axis=1)


def item_score(raw: pd.Series) -> pd.Series:
    """MDS-UPDRS item scores as numbers: the leading 0-4 digit of each answer, else NA.

    "2" -> 2, "2*parationia" -> 2, "-" or blank -> NA. Anything that does not start with a
    digit 0-4 is treated as missing rather than guessed.
    """
    return pd.to_numeric(raw.astype("string").str.extract(r"^\s*([0-4])(?!\d)", expand=False),
                         errors="coerce")


def minutes_since_dose(raw: pd.Series) -> pd.Series:
    """Parse the Part III '3c1' answer (minutes since the last levodopa dose) into minutes.

    "240" or "330 min" -> the number; "240+" -> 240 (a lower bound); "840, DBS off" -> 840;
    ">12 hours" -> 720; a continuous pump rate such as "6.0mL/hour", "n/a" or "-" -> NA.
    """
    def one(x):
        if pd.isna(x):
            return np.nan
        t = str(x).strip().lower()
        if "ml" in t:
            return np.nan                        # infusion pump: no "last dose"
        m = re.match(r"^>?\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|h)\b", t)
        if m:
            return float(m.group(1)) * 60
        m = re.match(r"^>?\s*(\d+(?:\.\d+)?)", t)
        return float(m.group(1)) if m else np.nan
    return raw.map(one).astype(float)


def part3_items(df: pd.DataFrame) -> list[str]:
    """The 33 scored MDS-UPDRS Part III item columns, in file order."""
    return [c for c in df.columns if c.startswith("MDSUPDRS_3-")]


def medication_state(raw: pd.Series) -> pd.Series:
    """Collapse the free-text Part III '3b' answer into ON, OFF, OTHER or NA.

    'ON', 'On', 'ON Meds, ON DBS' -> ON; 'OFF', 'Off', 'MED OFF/DBS ON', 'OFF Meds, ON DBS' -> OFF
    (the medication state, whatever the stimulator is doing); anything in between, such as
    'Wearing-Off' or 'Partial Kick In', -> OTHER.
    """
    def one(x):
        if pd.isna(x):
            return pd.NA
        t = str(x).strip().upper()
        if t in {"ON", "ON MEDS, ON DBS"}:
            return "ON"
        if t in {"OFF", "MED OFF/DBS ON", "OFF MEDS, ON DBS", "MED OFF/STIM ON"}:
            return "OFF"
        return "OTHER"
    return raw.map(one).astype("string")


# ---------------------------------------------------------------------------
# Walkway (PKMAS) metrics
# ---------------------------------------------------------------------------
def load_pkmas(task: str | None = "SelfPace", path: Path = paths.PKMAS) -> pd.DataFrame:
    """Load the ProtoKinetics walkway gait metrics.

    The two header rows are joined as "<metric> | <statistic>" (e.g. "Step Length (cm.) | Mean");
    metrics with a single value keep their own name (e.g. "Cadence (steps/min.)").
    Task names are stripped of trailing spaces and, if `task` is given, only that task is kept.
    Added at the front: record_id (as written), subject_id ("_2" removed), session (1, or 2 for
    "_2" rows) and group ("pd" / "control").
    """
    raw = pd.read_csv(path, header=[0, 1], dtype=str, keep_default_na=False)
    names = []
    for metric, stat in raw.columns:
        metric, stat = str(metric).strip(), str(stat).strip()
        names.append(metric if (stat.startswith("Unnamed") or stat == "") else f"{metric} | {stat}")
    df = raw.copy()
    df.columns = _dedupe(names)
    df = _to_numeric_where_possible(_clean_strings(df))
    df = df[df["Participant ID"].notna()]

    if task is not None:
        df = df[df["Task"] == task]

    rid = df["Participant ID"].astype(str)
    second = rid.str.endswith("_2")
    ids = pd.DataFrame({
        "record_id": rid,
        "subject_id": rid.str.replace(r"_2$", "", regex=True),
        "session": np.where(second, 2.0, 1.0),
        "group": df["PD vs Control"].str.lower().replace({"control": "control", "pd": "pd"}),
    }, index=df.index)
    out = pd.concat([ids, df.drop(columns=["Participant ID", "PD vs Control"])], axis=1)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trial files
# ---------------------------------------------------------------------------
TRIAL_DIRS = {
    ("v1", "pd"): paths.V1_DIR / "PD PARTICIPANTS" / "CSV files",
    ("v1", "control"): paths.V1_DIR / "CONTROL PARTICIPANTS" / "CSV files",
    ("v2", "pd"): paths.V2_DIR / "PD Participants" / "00-CSV files",
}
TRIAL_NAME = re.compile(r"^(?P<rid>.+?)(?: \(control\))?_(?P<trial>SelfPace(?:_mat)?)\.csv$")


def list_trials(dirs: dict | None = None) -> pd.DataFrame:
    """One row per trial file: version, group, record_id, subject_id, session, trial, path.

    The " (control)" tag in three control file names is removed so record_id matches the
    clinical table. Files that do not look like a trial are ignored.
    """
    rows = []
    for (version, group), d in (dirs or TRIAL_DIRS).items():
        for p in sorted(Path(d).glob("*.csv")):
            m = TRIAL_NAME.match(p.name)
            if not m:
                continue
            sid, session = split_record_id(m["rid"]) if version == "v2" else (m["rid"], np.nan)
            rows.append({"version": version, "group": group, "record_id": m["rid"],
                         "subject_id": sid, "session": session, "trial": m["trial"], "path": p})
    return pd.DataFrame(rows)


def trial_columns(path: Path) -> list[str]:
    """The column names of a trial file, read from its header only."""
    return pd.read_csv(path, nrows=0).columns.tolist()


def cached(name: str, params: dict, build, refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame from cache/<name>_<hash>.parquet, building and saving it if needed.

    The hash covers `params`, so changing a parameter builds a new file instead of reusing
    a stale one. Delete project1/cache/ at any time to rebuild everything.
    """
    key = hashlib.sha256(repr(sorted(params.items())).encode()).hexdigest()[:10]
    path = paths.CACHE_DIR / f"{name}_{key}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = build()
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def sensor_coverage(path: Path) -> dict:
    """Share of samples with data for each sensor unit in one trial (0 if the columns are absent).

    An IMU counts as present on a sample when its `_Acc_X` is not missing; the insoles when
    `LPressure1` or `RPressure1` is not missing; the walkway when `Walkway_X` is not missing.
    Also returns n_samples, n_walk (samples labelled `Walk`) and n_turn (labelled `Turn`).
    """
    have = set(trial_columns(path))
    probe = {u: f"{u}_Acc_X" for u in paths.IMUS}
    want = ["GeneralEvent", "Walkway_X", "LPressure1", "RPressure1"] + list(probe.values())
    df = load_trial(path, [c for c in want if c in have])
    n = len(df)
    out = {"n_samples": n,
           "n_walk": int((df["GeneralEvent"] == "Walk").sum()) if "GeneralEvent" in df else 0,
           "n_turn": int((df["GeneralEvent"] == "Turn").sum()) if "GeneralEvent" in df else 0}
    for u, c in probe.items():
        out[u] = float(df[c].notna().mean()) if c in df else 0.0
    ins = [c for c in ("LPressure1", "RPressure1") if c in df]
    out[paths.INSOLES] = float(df[ins].notna().any(axis=1).mean()) if ins else 0.0
    out["Walkway"] = float(df["Walkway_X"].notna().mean()) if "Walkway_X" in df else 0.0
    return out


def sensor_inventory(trials: pd.DataFrame) -> pd.DataFrame:
    """sensor_coverage() for every trial in `trials` (a list_trials() table), one row each."""
    rows = [sensor_coverage(p) for p in trials["path"]]
    return pd.concat([trials.drop(columns="path").reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def write_tables(sheets: dict[str, pd.DataFrame], name: str, folder: Path = paths.TABLES_DIR) -> Path:
    """Write several tables to one Excel workbook (one sheet each) and the first one also as CSV.

    Returns the path of the .xlsx file. Column widths are set from the content so the workbook
    is readable when opened.
    """
    folder.mkdir(parents=True, exist_ok=True)
    xlsx = folder / f"{name}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for sheet, df in sheets.items():
            df.to_excel(w, sheet_name=sheet, index=False)
            ws = w.sheets[sheet]
            for j, col in enumerate(df.columns, start=1):
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)])
                ws.column_dimensions[ws.cell(1, j).column_letter].width = min(max(8, width + 2), 80)
            ws.freeze_panes = "B2"
    next(iter(sheets.values())).to_csv(folder / f"{name}.csv", index=False)
    return xlsx


def load_trial(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load one trial file (100 Hz), optionally only some columns.

    `Time` is always included and converted from text like "0.01 sec" to seconds (float).
    Requested columns that the file does not have are returned as all-NaN columns, so trials
    with a missing sensor still line up; check with trial_columns() if that matters.
    """
    have = trial_columns(path)
    want = None if columns is None else ["Time"] + [c for c in columns if c != "Time"]
    use = None if want is None else [c for c in want if c in have]
    df = pd.read_csv(path, usecols=use, low_memory=False)
    df["Time"] = pd.to_numeric(df["Time"].astype(str).str.replace("sec", "", regex=False).str.strip(),
                               errors="coerce")
    if want is not None:
        for c in want:
            if c not in df.columns:
                df[c] = np.nan
        df = df[want]
    return df
