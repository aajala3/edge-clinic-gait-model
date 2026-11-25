"""Raw-signal windows for the CNN (Section 7).

Each `Walk` pass is cut into fixed windows (default 4 s = 400 samples, 50% overlap). A window
holds, for each of the 13 IMUs, acceleration and angular velocity on the sensor's own axes
(`_Acc_XYZ`, `_Gyr_XYZ`: 6 channels), and from the insoles total force and centre of pressure
per foot (6 channels). `_FreeAcc_ENU` is not used: its East/North axes depend on which way
the person faced, not on how they walked.

Recording faults are handled as for the features: spikes removed (features.clean_imus), gaps of
up to 0.1 s filled, longer gaps left missing. A sensor counts as present in a window when at
least 90% of its samples are there; `present` records this, and absent or missing values
become 0 after standardisation (the training mean), so the network sees "no information".
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import features, io, paths, segments

IMU_CHANNELS = ("Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z")
INSOLE_CHANNELS = ("LTotalForce", "RTotalForce", "LCoP_X", "LCoP_Y", "RCoP_X", "RCoP_Y")
MIN_PRESENT = 0.9


def columns() -> list[str]:
    """Trial columns to load: what the windows use plus what clean_imus needs."""
    imu = [f"{u}_{k}_{a}" for u in paths.IMUS for k, axes in (("Acc", "XYZ"), ("FreeAcc", "ENU"), ("Gyr", "XYZ"))
           for a in axes]
    return ["GeneralEvent"] + imu + list(INSOLE_CHANNELS)


def trial_windows(trial: pd.DataFrame, length_s: float = 4.0, overlap: float = 0.5,
                  fs: int = paths.FS) -> dict:
    """Windows from one trial: imu (n, T, 13, 6), insole (n, T, 6), present (n, 14) and start (n,).

    The last column of `present` is the insoles. Values are raw (not standardised); NaN marks
    samples that are missing after cleaning.
    """
    trial, _ = features.clean_imus(trial, fs)
    gap = int(0.1 * fs)

    def get(col):
        x = trial[col].to_numpy(float) if col in trial else np.full(len(trial), np.nan)
        return features._fill(x, gap)

    imu = np.stack([np.stack([get(f"{u}_{c}") for c in IMU_CHANNELS], axis=1) for u in paths.IMUS], axis=1)
    ins = np.stack([get(c) for c in INSOLE_CHANNELS], axis=1)
    # centre of pressure is undefined while the foot is in the air; force is 0 then
    for force, cop in ((0, [2, 3]), (1, [4, 5])):
        ins[np.ix_(ins[:, force] <= 0, cop)] = 0.0

    spans = segments.windows(trial, length_s, overlap, fs=fs)
    n, T = len(spans), int(length_s * fs)
    out = {"imu": np.empty((n, T, len(paths.IMUS), len(IMU_CHANNELS)), np.float32),
           "insole": np.empty((n, T, len(INSOLE_CHANNELS)), np.float32),
           "present": np.zeros((n, len(paths.IMUS) + 1), bool),
           "start": np.array([s for s, _ in spans], int)}
    for i, (s, e) in enumerate(spans):
        w_imu, w_ins = imu[s:e], ins[s:e]
        ok_imu = (~np.isnan(w_imu).any(axis=2)).mean(axis=0) >= MIN_PRESENT
        ok_ins = (~np.isnan(w_ins[:, :2]).any(axis=1)).mean() >= MIN_PRESENT
        w_imu = np.where(ok_imu[None, :, None], w_imu, np.nan)
        out["imu"][i], out["insole"][i] = w_imu, w_ins if ok_ins else np.nan
        out["present"][i] = np.r_[ok_imu, ok_ins]
    return out


def build_windows(trials: pd.DataFrame, length_s: float = 4.0, overlap: float = 0.5) -> dict:
    """Windows from every trial in `trials` (from io.list_trials), stacked, with a `meta`
    table (record_id, trial, start) giving each window's person and source."""
    parts, meta = [], []
    for _, t in trials.iterrows():
        w = trial_windows(io.load_trial(t["path"], columns()), length_s, overlap)
        parts.append(w)
        meta.append(pd.DataFrame({"record_id": t["record_id"], "trial": t["trial"], "start": w["start"]}))
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("imu", "insole", "present")}
    out["meta"] = pd.concat(meta, ignore_index=True)
    return out


def code_version() -> str:
    """Fingerprint of the code that shapes the windows, for the cache key."""
    src = b"".join((Path(__file__).parent / f).read_bytes() for f in ("signals.py", "segments.py", "features.py"))
    return hashlib.sha256(src).hexdigest()[:10]


def cached_windows(name: str, trials: pd.DataFrame, length_s: float = 4.0, overlap: float = 0.5) -> dict:
    """build_windows, saved to cache/<name>_<hash>.npz and reused while the code and trials match."""
    key = hashlib.sha256(repr((code_version(), sorted(trials["path"].astype(str)), length_s, overlap))
                         .encode()).hexdigest()[:10]
    path = paths.CACHE_DIR / f"{name}_{key}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        out = {k: z[k] for k in ("imu", "insole", "present")}
        out["meta"] = pd.DataFrame({"record_id": z["record_id"], "trial": z["trial"], "start": z["start"]})
        return out
    out = build_windows(trials, length_s, overlap)
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    m = out["meta"]
    np.savez(path, imu=out["imu"], insole=out["insole"], present=out["present"],
             record_id=m["record_id"].to_numpy(str), trial=m["trial"].to_numpy(str), start=m["start"].to_numpy())
    return out


class Standardiser:
    """Per-channel mean and SD from training windows only; transform standardises and sets
    missing values to 0 (the training mean)."""

    def fit(self, x: np.ndarray):
        self.mean_ = np.nanmean(x, axis=(0, 1))   # over windows and time: one value per channel
        self.sd_ = np.nanstd(x, axis=(0, 1))
        self.sd_[~(self.sd_ > 0)] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean_) / self.sd_
        return np.nan_to_num(z, nan=0.0).astype(np.float32)
