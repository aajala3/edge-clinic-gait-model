"""The 1D-CNN on raw signal windows (Section 7), in keras.

Design: one small convolutional encoder is shared by all 13 IMUs (each sensor's 6 channels go
through the same filters, so 97 patients train one encoder, not 13), and the insoles get their
own encoder. Each sensor's summary is multiplied by its presence flag, so a missing sensor
contributes nothing, and the summaries are concatenated in a fixed order, so the network
knows which body segment each came from. A small dense head predicts Part III per window;
a patient's prediction is the average over their windows.

Keras runs on the PyTorch backend, which uses the Apple GPU (MPS) when available; set
KERAS_BACKEND=torch before keras is first imported.

Training, inside each outer training fold only:
- standardisation statistics from the training windows (signals.Standardiser);
- the target standardised with the training patients' mean and SD;
- early stopping on a validation split of training *patients* (never test patients);
- augmentation: small random rotations, amplitude scaling and noise, per window and sensor.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import cv, paths, signals


def build_cnn(window: int = 400, n_sensors: int = 13, n_channels: int = 6, n_insole: int = 6,
              filters: int = 32, dropout: float = 0.3):
    import keras
    from keras import layers

    def encoder(name, channels):
        return keras.Sequential([
            keras.Input((window, channels)),
            layers.Conv1D(filters, 7, padding="same", activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(filters, 5, padding="same", activation="relu"),
            layers.MaxPooling1D(2),
            layers.Conv1D(2 * filters, 5, padding="same", activation="relu"),
            layers.GlobalAveragePooling1D(),
        ], name=name)

    imu = keras.Input((window, n_sensors, n_channels), name="imu")
    insole = keras.Input((window, n_insole), name="insole")
    present = keras.Input((n_sensors + 1,), name="present")

    per_sensor = layers.Permute((2, 1, 3))(imu)                              # (sensors, time, channels)
    z_imu = layers.TimeDistributed(encoder("imu_encoder", n_channels))(per_sensor)   # (sensors, 2F)
    z_imu = z_imu * layers.Reshape((n_sensors, 1))(present[:, :n_sensors])
    z_ins = encoder("insole_encoder", n_insole)(insole) * present[:, n_sensors:]
    z = layers.Concatenate()([layers.Flatten()(z_imu), z_ins])
    z = layers.Dropout(dropout)(z)
    z = layers.Dense(32, activation="relu")(z)
    z = layers.Dropout(dropout)(z)
    out = layers.Dense(1)(z)
    return keras.Model([imu, insole, present], out, name="gait_cnn")


def _rotations(rng, n: int, max_deg: float) -> np.ndarray:
    """n random 3-D rotation matrices with angles up to max_deg about a random axis."""
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    theta = np.deg2rad(rng.uniform(-max_deg, max_deg, n))
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(0, 2, 1)
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3) + s * K + (1 - c) * K @ K


def augment(imu: np.ndarray, insole: np.ndarray, rng, max_deg: float = 10, scale: float = 0.1,
            noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Augment standardised windows: rotate each sensor's acc and gyro axes together by a small
    random rotation (sensor placement varies), scale amplitude, add noise. Zeros (missing)
    stay zero."""
    b, T, S, _ = imu.shape
    R = _rotations(rng, b * S, max_deg).reshape(b, S, 3, 3).astype(np.float32)
    acc = np.einsum("bsij,btsj->btsi", R, imu[..., :3])
    gyr = np.einsum("bsij,btsj->btsi", R, imu[..., 3:])
    out = np.concatenate([acc, gyr], axis=-1)
    out *= rng.uniform(1 - scale, 1 + scale, (b, 1, S, 1)).astype(np.float32)
    out += rng.normal(0, noise, out.shape).astype(np.float32)
    out[imu == 0] = 0
    ins = insole * rng.uniform(1 - scale, 1 + scale, (b, 1, 1)).astype(np.float32)
    ins += rng.normal(0, noise, ins.shape).astype(np.float32) * (insole != 0)
    return out.astype(np.float32), ins.astype(np.float32)


def _batches(imu, insole, present, y, batch_size, seed, train: bool):
    import keras

    class Batches(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(len(imu))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(imu) / batch_size))

        def __getitem__(self, i):
            idx = self.order[i * batch_size:(i + 1) * batch_size]
            a, b = imu[idx], insole[idx]
            if train:
                a, b = augment(a, b, self.rng)
            return (a, b, present[idx].astype(np.float32)), y[idx]

        def on_epoch_end(self):
            if train:
                self.rng.shuffle(self.order)

    return Batches()


def fit_predict(windows: dict, y_win: np.ndarray, groups_win: np.ndarray, train: np.ndarray,
                test: np.ndarray, seed: int = paths.SEED, epochs: int = 100, patience: int = 10,
                batch_size: int = 64, sensors: np.ndarray | None = None, filters: int = 16,
                dropout: float = 0.5, lr: float = 3e-4, weight_decay: float = 1e-2) -> tuple[np.ndarray, int]:
    """Train on windows `train`, predict windows `test`. Returns (predictions, best epoch).

    About a sixth of the training patients (stratified on the recall flag) are held out to
    pick the stopping epoch; their windows are not used for fitting. `sensors` (bool, 14)
    optionally keeps only some sensor units (the rest are treated as absent), for Section 8.
    """
    import keras

    keras.utils.set_random_seed(seed)
    present = windows["present"] if sensors is None else windows["present"] & sensors[None, :]
    inner = cv.PatientKFold(6, random_state=seed)
    fit_pos, val_pos = next(inner.split(np.zeros(len(train)), y_win[train], groups_win[train]))
    fit, val = train[fit_pos], train[val_pos]

    s_imu = signals.Standardiser().fit(windows["imu"][fit])
    s_ins = signals.Standardiser().fit(windows["insole"][fit])

    def prep(idx):
        imu = s_imu.transform(windows["imu"][idx]) * present[idx, None, :-1, None]
        ins = s_ins.transform(windows["insole"][idx]) * present[idx, None, -1:]
        return imu.astype(np.float32), ins.astype(np.float32), present[idx]

    # standardise the target with the fitting patients' mean and SD (one value per patient)
    per_patient = pd.Series(y_win[fit]).groupby(groups_win[fit]).first()
    mu, sd = per_patient.mean(), per_patient.std()
    z = ((y_win - mu) / sd).astype(np.float32)

    model = build_cnn(window=windows["imu"].shape[1], filters=filters, dropout=dropout)
    model.compile(optimizer=keras.optimizers.AdamW(lr, weight_decay=weight_decay), loss=keras.losses.Huber(delta=1.0))
    stop = keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True)
    hist = model.fit(_batches(*prep(fit), z[fit], batch_size, seed, train=True),
                     validation_data=_batches(*prep(val), z[val], 256, seed, train=False),
                     epochs=epochs, callbacks=[stop], verbose=0)
    pred = model.predict(_batches(*prep(test), z[test], 256, seed, train=False), verbose=0).ravel()
    best = int(np.argmin(hist.history["val_loss"])) + 1
    keras.backend.clear_session()
    return pred * sd + mu, best


def oof_cnn(name: str, windows: dict, y: np.ndarray, patients: np.ndarray, folds: list[cv.Fold],
            seeds=(paths.SEED,), sensors: np.ndarray | None = None, **fit_kw) -> pd.DataFrame:
    """Out-of-fold patient predictions from the CNN, in the same format as cv.oof_predict
    (repeat, fold, row, patient, y, pred) plus `epochs` (best epoch, averaged over seeds).

    `y` and `patients` are per patient (aligned with the folds); windows are mapped to patients
    by record_id. Each window prediction is averaged over the seed ensemble, then over the
    patient's windows. Each fold is cached in cache/<name>_<hash>/ as it finishes, so an
    interrupted run resumes where it stopped.
    """
    row_of = {p: i for i, p in enumerate(patients)}
    win_row = windows["meta"]["record_id"].map(row_of).to_numpy()
    if np.isnan(win_row.astype(float)).any():
        raise ValueError("some windows belong to patients outside the analysis set")
    win_row = win_row.astype(int)
    y_win, groups_win = np.asarray(y, float)[win_row], np.asarray(patients)[win_row]

    code = b"".join((Path(__file__).parent / f).read_bytes() for f in ("cnn.py", "signals.py"))
    key = hashlib.sha256(repr((code, windows["imu"].shape, list(windows["meta"]["record_id"]), np.asarray(y, float).tolist(),
                               list(patients), list(seeds),
                               None if sensors is None else sensors.tolist(), sorted(fit_kw.items()))).encode()).hexdigest()[:10]
    folder = paths.CACHE_DIR / f"{name}_{key}"
    folder.mkdir(parents=True, exist_ok=True)

    out = []
    for f in folds:
        fold_key = hashlib.sha256(repr(sorted(f.test.tolist())).encode()).hexdigest()[:8]
        path = folder / f"r{f.repeat}_f{f.fold}_{fold_key}.parquet"
        if path.exists():
            out.append(pd.read_parquet(path))
            continue
        train, test = np.flatnonzero(np.isin(win_row, f.train)), np.flatnonzero(np.isin(win_row, f.test))
        if set(groups_win[train]) & set(groups_win[test]):
            raise ValueError(f"repeat {f.repeat} fold {f.fold}: a patient's windows are on both sides")
        preds, best = zip(*[fit_predict(windows, y_win, groups_win, train, test, seed=s + 1000 * f.repeat + f.fold,
                                        sensors=sensors, **fit_kw) for s in seeds])
        w = pd.DataFrame({"row": win_row[test], "pred": np.mean(preds, axis=0)})
        res = w.groupby("row", as_index=False)["pred"].mean()
        res.insert(0, "repeat", f.repeat)
        res.insert(1, "fold", f.fold)
        res["patient"] = np.asarray(patients)[res["row"]]
        res["y"] = np.asarray(y, float)[res["row"]]
        res["epochs"] = float(np.mean(best))
        res = res[["repeat", "fold", "row", "patient", "y", "pred", "epochs"]]
        res.to_parquet(path)
        out.append(res)
    return pd.concat(out, ignore_index=True)
