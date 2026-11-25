"""Gait features from the wearable sensors.

Units, checked on the data (Section 5): 100 Hz; `_Acc_XYZ` in m/s² including gravity (its
magnitude is 9.8 at rest); `_FreeAcc_ENU` in m/s² with gravity removed, east/north/up; `_Gyr_XYZ`
in rad/s. Sensor orientation on the body is unknown, so features use quantities that do not
depend on it: vector magnitudes, the vertical (from gravity or FreeAcc_U), and each sensor's own
main rotation axis.

One row per trial, columns named `<sensor>__<feature>` (e.g. `LowerBack__step_regularity`),
plus `turn__*`, `insole__*` and `asym__*` (left/right asymmetry). Walking features are computed
on each Walk bout and averaged with bout duration as weight; turn features on each Turn bout.
"""

import numpy as np
import pandas as pd
from scipy import ndimage, signal

from . import io, paths, segments

TRUNK = ["LowerBack", "Xiphoid", "Forehead"]
WRISTS = ["L_Wrist", "R_Wrist"]
LEGS = ["L_LatShank", "R_LatShank", "L_Ankle", "R_Ankle", "L_DorsalFoot", "R_DorsalFoot"]
# Insole cells by region, from the order in which they peak during stance (Section 5 check):
# cells 1-6 peak early (heel), 7-8 mid-stance (midfoot), 9-16 late (forefoot and toes).
INSOLE_REGIONS = {"heel": range(1, 7), "midfoot": range(7, 9), "forefoot": range(9, 17)}

GLOSSARY = {
    # every IMU, walking
    "vert_rms": "Vertical acceleration RMS (m/s²): how much the body part bounces",
    "horiz_rms": "Horizontal acceleration RMS (m/s²): forward and sideways movement",
    "jerk_rms": "RMS of the change in acceleration (m/s³): low = smooth movement",
    "gyro_rms": "Angular velocity RMS (deg/s): how much the body part rotates",
    "gyro_p95": "95th percentile of angular velocity (deg/s): the fastest rotations",
    "dom_freq": "Dominant frequency of the vertical acceleration (Hz): roughly step frequency",
    "loco_power": "Share of acceleration power in the 0.5-3 Hz walking band",
    "step_time": "Step time from the autocorrelation of vertical acceleration (s); trunk sensors only",
    "step_regularity": "Autocorrelation at one step: how alike consecutive steps are (1 = identical); trunk only",
    "stride_regularity": "Autocorrelation at one stride (two steps); trunk only",
    "step_asymmetry": "|Step regularity - stride regularity|: 0 = left and right steps alike; trunk only",
    "harmonic_ratio": "Even over odd harmonics of vertical acceleration: higher = smoother, more symmetric; trunk only",
    # every IMU, turning
    "turn_gyro_rms": "Angular velocity RMS during turns (deg/s)",
    "turn_acc_rms": "Free-acceleration RMS during turns (m/s²)",
    # trunk
    "tilt_sd": "Trunk sway: SD of the trunk's tilt from its average posture while walking (deg)",
    "rock_rms": "Trunk rocking: angular velocity about horizontal axes while walking (deg/s)",
    # wrists
    "arm_swing_rms": "Arm swing: angular velocity RMS about the wrist's main swing axis (deg/s)",
    "arm_swing_range": "Arm swing range: 5th to 95th percentile of that angular velocity (deg/s)",
    # legs
    "stride_time": "Stride time from swing peaks of the leg sensor (s)",
    "stride_cv": "Stride-to-stride variability of stride time (coefficient of variation, %)",
    "swing_peak": "Peak angular velocity in swing (deg/s): how fast the leg swings",
    # turns (lower back + foot sensors)
    "turn__duration": "Average turn duration (s)",
    "turn__peak_rate": "Peak turning speed of the lower back (deg/s)",
    "turn__mean_rate": "Average turning speed of the lower back (deg/s)",
    "turn__steps": "Steps per turn (swings of either foot)",
    # insoles
    "insole__stance_time": "Stance time per step (s)",
    "insole__stance_cv": "Stance time variability (%)",
    "insole__double_support": "Share of walking time with both feet loaded (%)",
    "insole__peak_load": "Peak force per step relative to the standing force of that foot",
    "insole__cop_ap": "Centre of pressure: heel-to-toe travel per step (insole units)",
    "insole__cop_ml_sd": "Centre of pressure: side-to-side wobble per step (SD, insole units)",
    "insole__heel_share": "Share of the stance pressure carried by the heel cells",
    "insole__fore_share": "Share of the stance pressure carried by the forefoot cells",
    "insole__heel_to_toe": "Time from heel peak to forefoot peak, as a share of stance",
    # asymmetry
    "asym__": "Left/right asymmetry |L - R| / (L + R) of the named feature",
}


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------
def _cols(trial, unit, kind, axes):
    cols = [f"{unit}_{kind}_{a}" for a in axes]
    if not all(c in trial for c in cols):
        return None
    x = trial[cols].to_numpy(dtype=float)
    return None if np.isnan(x).all() else x


MAX_GAP = 10   # samples (0.1 s): longer gaps stay missing, and passes or turns containing them are skipped


def _fill(x: np.ndarray, max_gap: int = MAX_GAP) -> np.ndarray:
    """Linear fill of interior gaps of at most `max_gap` samples; longer gaps and gaps at the
    ends stay NaN, so a sensor that stopped recording is never bridged with a made-up line."""
    x = np.asarray(x, dtype=float)
    flat = x.ndim == 1
    df = pd.DataFrame(x[:, None] if flat else x)
    filled = df.interpolate(limit_area="inside")
    for c in df:
        miss = df[c].isna().to_numpy()
        run_id = np.cumsum(np.r_[True, miss[1:] != miss[:-1]])
        run_len = pd.Series(run_id).map(pd.Series(run_id).value_counts()).to_numpy()
        filled.loc[miss & (run_len > max_gap), c] = np.nan
    out = filled.to_numpy()
    return out[:, 0] if flat else out


def _autocorr(v: np.ndarray) -> np.ndarray:
    v = v - v.mean()
    n = len(v)
    f = np.fft.rfft(v, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= np.arange(n, 0, -1)               # unbiased
    return ac / ac[0] if ac[0] > 0 else ac


def gait_autocorr(v: np.ndarray, fs: int = paths.FS) -> dict:
    """Step and stride regularity from the autocorrelation of a vertical acceleration signal.

    The first dominant peak (step lag, 0.35-0.9 s) and the peak near twice that lag (stride)
    give step regularity and stride regularity (Moe-Nilssen & Helbostad 2004). Their difference
    is the step asymmetry: 0 when left and right steps are alike. (The ratio form, step / stride
    regularity, explodes when stride regularity is near 0, so the bounded difference is used.)
    """
    ac = _autocorr(np.asarray(v, dtype=float))
    lo, hi = int(0.35 * fs), int(0.9 * fs)
    if len(ac) < 2 * hi + 10:
        return dict(step_time=np.nan, step_regularity=np.nan, stride_regularity=np.nan, step_asymmetry=np.nan)
    k1 = lo + int(np.argmax(ac[lo:hi]))
    a, b = int(1.6 * k1), min(int(2.4 * k1), len(ac) - 1)
    k2 = a + int(np.argmax(ac[a:b + 1]))
    d1, d2 = ac[k1], ac[k2]
    return dict(step_time=k1 / fs, step_regularity=d1, stride_regularity=d2, step_asymmetry=abs(d1 - d2))


def harmonic_ratio(v: np.ndarray, stride_time: float, fs: int = paths.FS, n_harmonics: int = 10) -> float:
    """Ratio of even to odd harmonic amplitudes of a vertical signal at the stride frequency."""
    if not np.isfinite(stride_time) or stride_time <= 0:
        return np.nan
    t = np.arange(len(v)) / fs
    v = v - v.mean()
    amp = np.array([abs(np.sum(v * np.exp(-2j * np.pi * k / stride_time * t))) for k in range(1, n_harmonics + 1)])
    odd, even = amp[0::2].sum(), amp[1::2].sum()
    return even / odd if odd > 0 else np.nan


def spectrum_features(v: np.ndarray, fs: int = paths.FS) -> dict:
    """Dominant frequency (0.5-4 Hz) and the share of power in the 0.5-3 Hz walking band."""
    f, p = signal.welch(v - v.mean(), fs=fs, nperseg=min(len(v), 256))
    band = (f >= 0.5) & (f <= 4)
    total = p[(f >= 0.5) & (f <= 15)].sum()
    return dict(dom_freq=f[band][np.argmax(p[band])] if band.any() else np.nan,
                loco_power=p[(f >= 0.5) & (f <= 3)].sum() / total if total > 0 else np.nan)


def vertical(acc: np.ndarray, smooth_s: float = 1.0, fs: int = paths.FS) -> np.ndarray:
    """Unit vector of gravity in the sensor frame, from the accelerometer averaged over smooth_s."""
    g = pd.DataFrame(acc).rolling(max(1, int(smooth_s * fs)), center=True, min_periods=1).mean().to_numpy()
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def vertical_rotation_rate(trial: pd.DataFrame, unit: str = "LowerBack", smooth_s: float = 1.0) -> pd.Series:
    """Rotation speed about the vertical axis (deg/s) for one IMU: positive and negative = turning.

    The vertical is estimated from gravity (see `vertical`) and the gyroscope (rad/s) is
    projected onto it.
    """
    acc = trial[[f"{unit}_Acc_{a}" for a in "XYZ"]].to_numpy(dtype=float)
    gyr = trial[[f"{unit}_Gyr_{a}" for a in "XYZ"]].to_numpy(dtype=float)
    g = vertical(acc, smooth_s)
    return pd.Series(np.degrees((gyr * g).sum(axis=1)), index=trial.index, name=f"{unit}_vertical_rate")


def main_axis(gyr: np.ndarray) -> np.ndarray:
    """Projection of angular velocity on its main axis (first principal component), sign set so
    the largest excursions are positive (for leg sensors: swing peaks)."""
    g = gyr - np.nanmean(gyr, axis=0)
    _, _, vt = np.linalg.svd(g[~np.isnan(g).any(axis=1)], full_matrices=False)
    p = gyr @ vt[0]
    return p if np.nanpercentile(p, 99) >= -np.nanpercentile(p, 1) else -p


def stride_period(p: np.ndarray, fs: int = paths.FS) -> float:
    """Stride period (s) of a leg signal: the autocorrelation peak between 0.7 and 2.0 s."""
    ac = _autocorr(np.asarray(p, dtype=float))
    lo, hi = int(0.7 * fs), min(int(2.0 * fs), len(ac) - 1)
    return (lo + int(np.argmax(ac[lo:hi]))) / fs if hi > lo else np.nan


def swing_peaks(p: np.ndarray, fs: int = paths.FS) -> np.ndarray:
    """Sample positions of swing peaks in a leg sensor's main-axis angular velocity (rad/s).

    Peaks must be at least 60% of the signal's own stride period apart, so the smaller
    secondary peak inside each stride (common in slow walkers) is not counted as a stride.
    """
    period = stride_period(p, fs)
    distance = int(0.6 * period * fs) if np.isfinite(period) else int(0.6 * fs)
    height = 0.4 * np.nanpercentile(p, 99)
    peaks, _ = signal.find_peaks(p, height=height, distance=distance, prominence=height)
    return peaks


def turn_steps(gyr: np.ndarray, fs: int = paths.FS) -> np.ndarray:
    """Sample positions of foot swings during turns. A turning foot pivots and lifts about
    changing axes, so peaks are found in the angular velocity magnitude (rad/s), not on the
    straight-walking swing axis."""
    peaks, _ = signal.find_peaks(np.linalg.norm(gyr, axis=1), height=1.5, distance=int(0.3 * fs), prominence=1.0)
    return peaks


def _weighted(rows: list[dict], weights: list[float]) -> dict:
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    w = np.asarray(weights, dtype=float)
    out = {}
    for c in df:
        ok = df[c].notna().to_numpy()
        out[c] = np.average(df[c][ok], weights=w[ok]) if ok.any() else np.nan
    return out


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
MAX_ACC = 200.0     # m/s², about 20 g: beyond any walking impact (largest genuine foot impact seen: about 17 g)
MAX_GYR = 35.0      # rad/s, about 2,000 deg/s
FLAT_ACC_SD = 0.05  # m/s²: an IMU whose acceleration varies less than this on every axis while walking is frozen


def clean_imus(trial: pd.DataFrame, fs: int = paths.FS) -> tuple[pd.DataFrame, dict]:
    """Remove recording faults before computing features.

    Spikes: samples where an IMU's |Acc|, |FreeAcc| or |Gyr| exceeds MAX_ACC / MAX_GYR are set to
    NaN (and later filled from their neighbours). Frozen sensors: an IMU whose acceleration SD
    during walking is below FLAT_ACC_SD on every axis is treated as absent in this trial.
    Returns the cleaned trial and a log: {'spike_samples': n, 'frozen': [sensor, ...]}.
    """
    trial = trial.copy()
    walking = trial["GeneralEvent"].eq("Walk").to_numpy()
    log = {"spike_samples": 0, "frozen": []}
    for u in paths.IMUS:
        cols = [c for c in trial if c.startswith(u + "_")]
        acc, free, gyr = _cols(trial, u, "Acc", "XYZ"), _cols(trial, u, "FreeAcc", "ENU"), _cols(trial, u, "Gyr", "XYZ")
        if acc is None or free is None or gyr is None:
            continue
        with np.errstate(invalid="ignore"):
            spike = ((np.linalg.norm(acc, axis=1) > MAX_ACC) | (np.linalg.norm(free, axis=1) > MAX_ACC)
                     | (np.linalg.norm(gyr, axis=1) > MAX_GYR))
        if spike.any():
            trial.loc[spike, cols] = np.nan
            log["spike_samples"] += int(spike.sum())
        if walking.any() and (np.nanstd(acc[walking & ~spike], axis=0) < FLAT_ACC_SD).all():
            trial[cols] = np.nan
            log["frozen"].append(u)
    return trial, log


# ---------------------------------------------------------------------------
# Per-sensor features
# ---------------------------------------------------------------------------
def imu_walk_features(trial: pd.DataFrame, unit: str, walk: list, fs: int = paths.FS) -> dict:
    """Generic walking features for one IMU, averaged over Walk bouts (duration-weighted)."""
    acc, free, gyr = _cols(trial, unit, "Acc", "XYZ"), _cols(trial, unit, "FreeAcc", "ENU"), _cols(trial, unit, "Gyr", "XYZ")
    if acc is None or free is None or gyr is None:
        return {}
    acc, free, gyr = _fill(acc), _fill(free), _fill(gyr)
    rows, w, strides = [], [], []
    for s, e in walk:
        fa, g, a = free[s:e], gyr[s:e], acc[s:e]
        if np.isnan(fa).any() or np.isnan(g).any() or np.isnan(a).any():
            continue                             # a gap longer than 0.1 s: skip this pass
        vert, gmag = fa[:, 2], np.degrees(np.linalg.norm(g, axis=1))
        r = dict(vert_rms=np.sqrt(np.mean((vert - vert.mean()) ** 2)),
                 horiz_rms=np.sqrt(np.mean(fa[:, 0] ** 2 + fa[:, 1] ** 2)),
                 jerk_rms=np.sqrt(np.mean(np.sum((np.diff(fa, axis=0) * fs) ** 2, axis=1))),
                 gyro_rms=np.sqrt(np.mean(gmag ** 2)), gyro_p95=np.percentile(gmag, 95))
        r.update(spectrum_features(vert, fs))
        if unit in TRUNK:
            # step rhythm measures assume a sensor that moves with every step (the trunk); a limb
            # moves once per stride, so on limbs they are unstable and are not computed
            r.update(gait_autocorr(vert, fs))
            r["harmonic_ratio"] = harmonic_ratio(vert, 2 * r["step_time"], fs)
            gv = vertical(a, 0.5, fs)
            mean_v = gv.mean(axis=0) / np.linalg.norm(gv.mean(axis=0))
            r["tilt_sd"] = np.std(np.degrees(np.arccos(np.clip(gv @ mean_v, -1, 1))))
            gvert = vertical(a, 1.0, fs)
            horiz = g - (g * gvert).sum(axis=1, keepdims=True) * gvert
            r["rock_rms"] = np.degrees(np.sqrt(np.mean(np.sum(horiz ** 2, axis=1))))
        if unit in WRISTS:
            p = np.degrees(main_axis(g))
            r["arm_swing_rms"] = np.sqrt(np.mean((p - p.mean()) ** 2))
            r["arm_swing_range"] = np.percentile(p, 95) - np.percentile(p, 5)
        if unit in LEGS:
            p = main_axis(g)
            pk = swing_peaks(p, fs)
            st = np.diff(pk) / fs
            if len(st) >= 4:                     # drop the first and last stride: starting and stopping
                st = st[1:-1]
            strides.extend(st[(st > 0.7) & (st < 2.0)])
            r["swing_peak"] = np.degrees(np.median(p[pk])) if len(pk) else np.nan
        rows.append(r)
        w.append(e - s)
    out = _weighted(rows, w)
    if unit in LEGS and out:                     # stride timing pooled over the trial's bouts
        st = np.asarray(strides)
        if len(st):                              # drop odd strides (missed or extra peaks)
            st = st[np.abs(st - np.median(st)) <= 0.35 * np.median(st)]
        out["stride_time"] = st.mean() if len(st) else np.nan
        out["stride_cv"] = 100 * st.std() / st.mean() if len(st) >= 4 else np.nan
    return out


def imu_turn_features(trial: pd.DataFrame, unit: str, turns: list) -> dict:
    free, gyr = _cols(trial, unit, "FreeAcc", "ENU"), _cols(trial, unit, "Gyr", "XYZ")
    if free is None or gyr is None:
        return {}
    free, gyr = _fill(free), _fill(gyr)
    rows, w = [], []
    for s, e in turns:
        if np.isnan(gyr[s:e]).any() or np.isnan(free[s:e]).any():
            continue
        rows.append(dict(turn_gyro_rms=np.degrees(np.sqrt(np.mean(np.sum(gyr[s:e] ** 2, axis=1)))),
                         turn_acc_rms=np.sqrt(np.mean(np.sum(free[s:e] ** 2, axis=1)))))
        w.append(e - s)
    return _weighted(rows, w)


def turn_features(trial: pd.DataFrame, turns: list, fs: int = paths.FS) -> dict:
    """Turn duration and speed (lower back) and steps per turn (foot sensors)."""
    out = {"turn__duration": np.mean([(e - s) / fs for s, e in turns]) if turns else np.nan}
    if _cols(trial, "LowerBack", "Gyr", "XYZ") is not None and turns:
        rate = vertical_rotation_rate(trial.assign(**{c: _fill(trial[c].to_numpy(float))
                                                      for c in trial if c.startswith("LowerBack_")}))
        rate = rate.rolling(int(0.2 * fs), center=True).mean().abs().to_numpy()
        ok = [(s, e) for s, e in turns if not np.isnan(rate[s:e]).any()]
        out["turn__peak_rate"] = np.mean([rate[s:e].max() for s, e in ok]) if ok else np.nan
        out["turn__mean_rate"] = np.mean([rate[s:e].mean() for s, e in ok]) if ok else np.nan
    feet = [u for u in ("L_DorsalFoot", "R_DorsalFoot") if _cols(trial, u, "Gyr", "XYZ") is not None]
    if len(feet) == 2 and turns:
        gyros = [_fill(_cols(trial, u, "Gyr", "XYZ")) for u in feet]
        ok = [(s, e) for s, e in turns if not any(np.isnan(g[s:e]).any() for g in gyros)]
        steps = np.zeros(len(ok))
        for g in gyros:
            pk = turn_steps(np.nan_to_num(g), fs)
            steps += [((pk >= s) & (pk < e)).sum() for s, e in ok]
        out["turn__steps"] = steps.mean() if ok else np.nan
    return out


def insole_features(trial: pd.DataFrame, walk: list, fs: int = paths.FS) -> dict:
    """Stance timing, loading and centre-of-pressure features from the two insoles."""
    if "LTotalForce" not in trial or trial["LTotalForce"].isna().all() or trial["RTotalForce"].isna().all():
        return {}
    standing = trial["GeneralEvent"].eq("Standing").to_numpy()
    per_foot, loaded = {}, {}
    for side in "LR":
        force = _fill(trial[f"{side}TotalForce"].to_numpy(float), int(0.1 * fs))
        cop_ap, cop_ml = trial[f"{side}CoP_X"].to_numpy(float), trial[f"{side}CoP_Y"].to_numpy(float)
        cells = {r: trial[[f"{side}Pressure{i}" for i in idx]].to_numpy(float).sum(axis=1)
                 for r, idx in INSOLE_REGIONS.items()}
        stand_force = np.nanmedian(force[standing]) if standing.any() else np.nan
        on = force > 0.1 * np.nanpercentile(force, 95)
        on = ndimage.binary_closing(on, structure=np.ones(int(0.1 * fs)))   # merge stances split by a flicker
        loaded[side] = on
        steps = []
        for s, e in walk:
            edges = np.diff(np.r_[0, on[s:e].astype(int), 0])
            starts, ends = np.flatnonzero(edges == 1) + s, np.flatnonzero(edges == -1) + s
            for a, b in list(zip(starts, ends))[1:-1]:      # skip the first and last step of a bout
                if not (0.25 * fs <= b - a <= 2.0 * fs) or np.isnan(force[a:b]).mean() > 0.2:
                    continue
                total = np.nansum([np.nansum(c[a:b]) for c in cells.values()])
                if np.isnan(cells["heel"][a:b]).all() or np.isnan(cells["forefoot"][a:b]).all():
                    continue
                hp, fp = np.nanargmax(cells["heel"][a:b]), np.nanargmax(cells["forefoot"][a:b])
                steps.append(dict(stance_time=(b - a) / fs,
                                  peak_load=np.nanmax(force[a:b]) / stand_force if stand_force > 0 else np.nan,
                                  cop_ap=np.nanmax(cop_ap[a:b]) - np.nanmin(cop_ap[a:b]),
                                  cop_ml_sd=np.nanstd(cop_ml[a:b]),
                                  heel_share=np.nansum(cells["heel"][a:b]) / total if total > 0 else np.nan,
                                  fore_share=np.nansum(cells["forefoot"][a:b]) / total if total > 0 else np.nan,
                                  heel_to_toe=(fp - hp) / (b - a)))
        if len(steps) >= 4:
            df = pd.DataFrame(steps)
            per_foot[side] = df.median().to_dict()
            per_foot[side]["stance_cv"] = 100 * df["stance_time"].std() / df["stance_time"].mean()
    if not per_foot:
        return {}
    out = {f"insole__{k}": np.nanmean([per_foot[s][k] for s in per_foot]) for k in next(iter(per_foot.values()))}
    if len(per_foot) == 2:
        in_walk = np.zeros(len(trial), bool)
        for s, e in walk:
            in_walk[s:e] = True
        out["insole__double_support"] = 100 * (loaded["L"] & loaded["R"] & in_walk).sum() / in_walk.sum()
        for k in ("stance_time", "peak_load"):
            out[f"asym__insole_{k}"] = _asym(per_foot["L"][k], per_foot["R"][k])
    return out


def _asym(left, right):
    return abs(left - right) / (left + right) if np.isfinite(left) and np.isfinite(right) and left + right > 0 else np.nan


def trial_features(trial: pd.DataFrame, fs: int = paths.FS) -> dict:
    """All features for one trial (see the module docstring for naming), after clean_imus."""
    trial, log = clean_imus(trial, fs)
    walk = segments.bouts(trial, "Walk", min_s=2.5, fs=fs)
    turns = segments.bouts(trial, "Turn", min_s=0.5, fs=fs)
    out = {}
    for unit in paths.IMUS:
        for k, v in imu_walk_features(trial, unit, walk, fs).items():
            out[f"{unit}__{k}"] = v
        for k, v in imu_turn_features(trial, unit, turns).items():
            out[f"{unit}__{k}"] = v
    out.update(turn_features(trial, turns, fs))
    out.update(insole_features(trial, walk, fs))
    for feat in ("arm_swing_rms", "arm_swing_range"):
        out[f"asym__wrist_{feat}"] = _asym(out.get(f"L_Wrist__{feat}", np.nan), out.get(f"R_Wrist__{feat}", np.nan))
    for u in ("LatShank", "Ankle", "DorsalFoot"):
        for feat in ("swing_peak",):            # not stride time: both feet share one stride period
            out[f"asym__{u}_{feat}"] = _asym(out.get(f"L_{u}__{feat}", np.nan), out.get(f"R_{u}__{feat}", np.nan))
    out["n_walk_bouts"], out["walk_seconds"] = len(walk), sum(e - s for s, e in walk) / fs
    out["n_turns"] = len(turns)
    out["spike_samples"], out["frozen_sensors"] = log["spike_samples"], ";".join(log["frozen"])
    return out


def code_version() -> str:
    """Fingerprint of the feature code, used as the cache key so a code change rebuilds the cache."""
    import hashlib
    from pathlib import Path
    src = b"".join(Path(m.__file__).read_bytes() for m in (segments,)) + Path(__file__).read_bytes()
    return hashlib.sha256(src).hexdigest()[:12]


def units_check(trial: pd.DataFrame, unit: str = "LowerBack") -> dict:
    """Signal scales that identify the units: at rest, |Acc| is 9.8 if it is m/s² with gravity and
    FreeAcc is 0 if gravity is removed; the gyroscope's turning peak shows rad/s vs deg/s."""
    acc, free, gyr = _cols(trial, unit, "Acc", "XYZ"), _cols(trial, unit, "FreeAcc", "ENU"), _cols(trial, unit, "Gyr", "XYZ")
    if acc is None or free is None or gyr is None:
        return {}
    still = trial["GeneralEvent"].eq("Standing").to_numpy()
    turning = trial["GeneralEvent"].eq("Turn").to_numpy()
    return {"Acc magnitude standing": np.nanmean(np.linalg.norm(acc[still], axis=1)),
            "FreeAcc magnitude standing": np.nanmean(np.linalg.norm(free[still], axis=1)),
            "Gyr magnitude turning, 99th pct": np.nanpercentile(np.linalg.norm(gyr[turning], axis=1), 99),
            "sample interval (s)": np.nanmedian(np.diff(trial["Time"].to_numpy(float)))}


def insole_cell_timing(trial: pd.DataFrame, side: str = "L", fs: int = paths.FS) -> pd.Series | None:
    """When in stance (0 = heel strike, 1 = toe off) each of the 16 pressure cells peaks, median
    over the trial's walking steps; used to check which cells are heel, midfoot and forefoot."""
    if f"{side}TotalForce" not in trial or trial[f"{side}TotalForce"].isna().all():
        return None
    force = _fill(trial[f"{side}TotalForce"].to_numpy(float), int(0.1 * fs))
    walk = np.zeros(len(trial), bool)
    for s, e in segments.bouts(trial, "Walk", 2.5, fs):
        walk[s:e] = True
    on = ndimage.binary_closing(force > 0.1 * np.nanpercentile(force, 95), structure=np.ones(int(0.1 * fs))) & walk
    edges = np.diff(np.r_[0, on.astype(int), 0])
    cells = trial[[f"{side}Pressure{i}" for i in range(1, 17)]].to_numpy(float)
    phases = [np.nanargmax(cells[a:b], axis=0) / (b - a)
              for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
              if 0.3 * fs <= b - a <= 2.0 * fs and not np.isnan(cells[a:b]).all(axis=0).any()]
    return pd.Series(np.median(phases, axis=0), index=range(1, 17)) if len(phases) >= 4 else None


def walkway_stride_time(trial: pd.DataFrame, side: str = "L", fs: int = paths.FS) -> tuple[float, int]:
    """Median stride time (s) from the walkway's foot contacts, as ground truth for checking the
    IMU stride detection. Only consecutive contacts of the same foot on one pass count.
    Returns (median stride time, number of strides); NaN if fewer than 3 strides."""
    col = f"{side} Foot Contact"
    if col not in trial:
        return np.nan, 0
    contact = trial[col].fillna(0).to_numpy() > 0
    onsets = np.flatnonzero(np.diff(np.r_[0, contact.astype(int)]) == 1)
    st = np.diff(onsets) / fs
    st = st[(st > 0.7) & (st < 2.0)]
    return (float(np.median(st)), len(st)) if len(st) >= 3 else (np.nan, len(st))


def build_trial_features(trials: pd.DataFrame) -> pd.DataFrame:
    """Features for every trial in `trials` (from io.list_trials); one row per trial."""
    rows = []
    for _, t in trials.iterrows():
        trial = io.load_trial(t["path"])
        f = trial_features(trial)
        check, n_check = walkway_stride_time(trial)
        events = trial["ClinicalEvent"].fillna("unlabeled") if "ClinicalEvent" in trial else pd.Series(dtype=str)
        rows.append({"record_id": t["record_id"], "subject_id": t.get("subject_id", t["record_id"]),
                     "version": t["version"], "group": t["group"], "trial": t["trial"],
                     "walkway_stride_time": check, "walkway_strides": n_check,
                     "fog_seconds": (events == "FoG").sum() / paths.FS,
                     "stagger_seconds": (events == "Stagger").sum() / paths.FS, **f})
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-input feature columns of a trial or patient feature table."""
    return [c for c in df if "__" in c]


def patient_features(trial_feats: pd.DataFrame) -> pd.DataFrame:
    """Average each feature over a person's trials (NaN-aware), one row per record_id."""
    cols = feature_columns(trial_feats)
    return trial_feats.groupby("record_id")[cols].mean().reset_index()


def sensor_of(feature: str) -> str:
    """The sensor a feature comes from: an IMU name, 'Insoles', or 'LowerBack' for turn metrics
    (turn steps also use the feet; see sensor_needs)."""
    head = feature.split("__")[0]
    if head == "insole" or feature.startswith("asym__insole"):
        return paths.INSOLES
    if head == "turn":
        return "LowerBack"
    if head == "asym":
        part = feature.split("__")[1].split("_")[0]
        return {"wrist": "Wrists", "LatShank": "LatShanks", "Ankle": "Ankles", "DorsalFoot": "DorsalFeet"}.get(part, part)
    return head


def sensor_needs(feature: str) -> set[str]:
    """Every sensor a feature needs, for selecting the features of a sensor kit (Section 8)."""
    if feature == "turn__steps":
        return {"L_DorsalFoot", "R_DorsalFoot"}
    if feature == "turn__duration":
        return set()
    head = feature.split("__")[0]
    if head == "insole" or feature.startswith("asym__insole"):
        return {paths.INSOLES}
    if head == "turn":
        return {"LowerBack"}
    if head == "asym":
        part = feature.split("__")[1]
        for u in ("Wrist", "LatShank", "Ankle", "DorsalFoot"):
            if part.lower().startswith(u.lower()):
                return {f"L_{u}", f"R_{u}"}
    return {head}


def icc_a1(a, b) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement (Shrout & Fleiss
    notation; Koo & Li 2016, Table 3), for two measurements per person. Pairs with a missing value
    are dropped. Koo & Li read < 0.5 as poor, 0.5-0.75 moderate, 0.75-0.9 good, > 0.9 excellent."""
    x = np.column_stack([np.asarray(a, float), np.asarray(b, float)])
    x = x[~np.isnan(x).any(axis=1)]
    n, k = x.shape
    if n < 3:
        return np.nan
    grand = x.mean()
    msr = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return (msr - mse) / denom if denom > 0 else np.nan
