"""Tremor features from the Standing parts of each walking trial.

Parkinsonian rest and postural tremor sits at about 3-7 Hz (rest tremor typically 4-6 Hz), a band
that the walking features in features.py do not look at. While the patient stands still before or
after a walk, little else moves the limbs, so power in that band is mostly tremor.

For every Standing bout of at least MIN_STAND_S seconds (TRIM_S cut from each end to drop the
transition from or into walking), on each sensor in UNITS:
- gyroscope and accelerometer spectra (Welch, 2 s segments), summed over the three axes so the
  result does not depend on how the sensor sits on the body;
- `tremor_gyr`: log10 RMS angular velocity in 3-7 Hz (deg/s);
- `tremor_acc`: log10 RMS acceleration in 3-7 Hz (m/s²);
- `tremor_ratio`: share of 0.5-15 Hz gyroscope power that falls in 3-7 Hz.
Bouts are averaged with duration as the weight. At the patient level, tremor is often one-sided,
so for each left/right pair the worse side (`tremor__<pair>_max_*`) is added.

No MDS-UPDRS item is used: the Standing label comes from the video annotation in `GeneralEvent`.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from . import features, io, paths, segments

UNITS = ("Forehead", "L_Wrist", "R_Wrist", "L_Ankle", "R_Ankle", "L_DorsalFoot", "R_DorsalFoot")
PAIRS = {"wrist": ("L_Wrist", "R_Wrist"), "ankle": ("L_Ankle", "R_Ankle"), "foot": ("L_DorsalFoot", "R_DorsalFoot")}
BAND = (3.0, 7.0)
BROAD = (0.5, 15.0)
MIN_STAND_S = 3.0
TRIM_S = 0.5
KINDS = ("tremor_gyr", "tremor_acc", "tremor_ratio")


def _psd(x: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch PSD summed over axes (columns), after removing each axis' linear trend."""
    f, p = signal.welch(signal.detrend(x, axis=0), fs=fs, nperseg=min(2 * fs, len(x)), axis=0)
    return f, p.sum(axis=1)


def _band(f, p, lo, hi) -> float:
    m = (f >= lo) & (f <= hi)
    return float(np.trapezoid(p[m], f[m]))


def bout_features(gyr: np.ndarray, acc: np.ndarray, fs: int = paths.FS) -> dict | None:
    """Tremor features of one Standing segment (rows = samples, columns = X, Y, Z)."""
    if np.isnan(gyr).any() or np.isnan(acc).any():
        return None
    f, pg = _psd(np.degrees(gyr), fs)
    _, pa = _psd(acc, fs)
    g_band, a_band, g_broad = _band(f, pg, *BAND), _band(f, pa, *BAND), _band(f, pg, *BROAD)
    return {"tremor_gyr": np.log10(np.sqrt(g_band) + 1e-6),
            "tremor_acc": np.log10(np.sqrt(a_band) + 1e-6),
            "tremor_ratio": g_band / g_broad if g_broad > 0 else np.nan}


def trial_tremor(trial: pd.DataFrame, fs: int = paths.FS) -> dict:
    """One row of tremor features for a trial: `<unit>__tremor_*` plus `stand_seconds`."""
    trial, _ = features.clean_imus(trial, fs)
    trim = int(TRIM_S * fs)
    stands = [(s + trim, e - trim) for s, e in segments.bouts(trial, "Standing", MIN_STAND_S, fs)]
    out = {"stand_seconds": sum(e - s for s, e in stands) / fs}
    for u in UNITS:
        gyr, acc = features._cols(trial, u, "Gyr", "XYZ"), features._cols(trial, u, "Acc", "XYZ")
        if gyr is None or acc is None:
            continue
        gyr, acc = features._fill(gyr), features._fill(acc)
        rows, w = [], []
        for s, e in stands:
            r = bout_features(gyr[s:e], acc[s:e], fs)
            if r is not None:
                rows.append(r)
                w.append(e - s)
        if rows:
            for k in KINDS:
                v = np.array([r[k] for r in rows], dtype=float)
                ok = ~np.isnan(v)
                if ok.any():
                    out[f"{u}__{k}"] = float(np.average(v[ok], weights=np.array(w)[ok]))
    return out


def columns() -> list[str]:
    cols = ["GeneralEvent"]
    for u in paths.IMUS:   # clean_imus checks every IMU for spikes
        cols += [f"{u}_{k}_{a}" for k, axes in (("Acc", "XYZ"), ("FreeAcc", "ENU"), ("Gyr", "XYZ")) for a in axes]
    return cols


def build(trials: pd.DataFrame) -> pd.DataFrame:
    """Tremor features for every trial in `trials` (from io.list_trials); one row per trial."""
    cols = columns()
    rows = [{"record_id": t["record_id"], "trial": t["trial"], **trial_tremor(io.load_trial(t["path"], cols))}
            for _, t in trials.iterrows()]
    return pd.DataFrame(rows)


def code_version() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:10]


def patient_tremor(trial_tremor_table: pd.DataFrame) -> pd.DataFrame:
    """Average over a person's trials, then add the worse side of each left/right pair."""
    cols = [c for c in trial_tremor_table if "__" in c]
    p = trial_tremor_table.groupby("record_id")[cols].mean()
    for pair, (left, right) in PAIRS.items():
        for k in KINDS:
            a, b = f"{left}__{k}", f"{right}__{k}"
            if a in p and b in p:
                p[f"tremor__{pair}_max_{k.split('_')[1]}"] = p[[a, b]].max(axis=1)
    return p.reset_index()
