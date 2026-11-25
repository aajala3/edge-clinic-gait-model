"""4 s CNN windows over walking *and* turning (gait sub-score notebook).

signals.py cuts windows from `Walk` runs only, so each pass along the walkway is cut off where
the person turns. Turning matters for the gait-and-balance items (3.10 gait includes turning;
freezing, 3.11, shows mostly at turns and starts), and steps per turn was the strongest input
of the gait elastic net. Here every sample labelled Walk or Turn is relabelled as one movement
label before windowing, so windows run through walk -> turn -> walk transitions and turns are
included. Everything else (channels, cleaning, gap filling, presence rule) is signals.py
unchanged; this module only changes which samples count as movement.

Note: features.clean_imus, called inside signals.trial_windows, checks for a frozen sensor over
the samples labelled Walk, which here means Walk + Turn. Kept as is (a frozen sensor is flat in
both).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import io, paths, signals

MOVING = ("Walk", "Turn")


def relabel(trial: pd.DataFrame) -> pd.DataFrame:
    """The trial with Walk and Turn samples labelled 'Walk' (one movement label)."""
    t = trial.copy()
    t["GeneralEvent"] = t["GeneralEvent"].where(~t["GeneralEvent"].isin(MOVING), "Walk")
    return t


def build_windows(trials: pd.DataFrame, length_s: float = 4.0, overlap: float = 0.5) -> dict:
    """As signals.build_windows, over Walk + Turn runs."""
    parts, meta = [], []
    for _, t in trials.iterrows():
        w = signals.trial_windows(relabel(io.load_trial(t["path"], signals.columns())), length_s, overlap)
        parts.append(w)
        meta.append(pd.DataFrame({"record_id": t["record_id"], "trial": t["trial"], "start": w["start"]}))
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("imu", "insole", "present")}
    out["meta"] = pd.concat(meta, ignore_index=True)
    return out


def cached_windows(name: str, trials: pd.DataFrame, length_s: float = 4.0, overlap: float = 0.5) -> dict:
    """build_windows, saved to cache/<name>_<hash>.npz and reused while the code and trials match."""
    code = signals.code_version() + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:10]
    key = hashlib.sha256(repr((code, sorted(trials["path"].astype(str)), length_s, overlap)).encode()).hexdigest()[:10]
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
