"""Walking bouts, turns and windows, from the video annotations in `GeneralEvent`.

Each trial is labelled sample by sample as Walk, Turn or Standing. A bout is one unbroken run
of a label; features are computed per bout and windows for the CNN are cut from Walk bouts.
"""

import numpy as np
import pandas as pd

from . import paths


def runs(labels, value: str) -> list[tuple[int, int]]:
    """(start, end) row positions of each unbroken run of `value`; end is exclusive."""
    hit = np.asarray(pd.Series(labels).eq(value), dtype=int)
    edges = np.diff(np.r_[0, hit, 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def bouts(trial: pd.DataFrame, label: str = "Walk", min_s: float = 2.5,
          fs: int = paths.FS) -> list[tuple[int, int]]:
    """Runs of `label` lasting at least `min_s` seconds (shorter ones hold too few steps)."""
    return [(s, e) for s, e in runs(trial["GeneralEvent"], label) if e - s >= min_s * fs]


def windows(trial: pd.DataFrame, length_s: float = 4.0, overlap: float = 0.5, label: str = "Walk",
            fs: int = paths.FS) -> list[tuple[int, int]]:
    """Fixed-length windows inside each `label` bout, overlapping by `overlap` (a share of the length)."""
    n = int(length_s * fs)
    step = max(1, int(n * (1 - overlap)))
    out = []
    for s, e in runs(trial["GeneralEvent"], label):
        out += [(a, a + n) for a in range(s, e - n + 1, step)]
    return out
