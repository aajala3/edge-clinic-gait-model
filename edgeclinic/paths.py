"""All paths and constants in one place.

Every other module imports from here, so a path or a threshold is changed once.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. PROJECT_DIR is the repository root, found from where this file lives, so nothing depends on
# the current working directory. The raw WearGait-PD data (about 11 GB) is not in the repository: put it in
# data/weargait_pd/ (see README) or set the environment variable EDGECLINIC_DATA_DIR to its folder.
# ---------------------------------------------------------------------------
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("EDGECLINIC_DATA_DIR", PROJECT_DIR / "data" / "weargait_pd"))   # read-only: never write here
V1_DIR = DATA_DIR / "Version 1 (Cross-sectional)"
V2_DIR = DATA_DIR / "Version 2 (Longitudinal)"
MANIFEST = DATA_DIR / "MANIFEST.csv"

V1_PD_CLINICAL = V1_DIR / "PD - Demographic+Clinical - datasetV1.csv"
V1_CONTROL_CLINICAL = V1_DIR / "CONTROLS - Demographic+Clinical - datasetV1.csv"
V2_PD_CLINICAL = V2_DIR / "PD - Demographic+Clinical - datasetV2 (Longitudinal) - TEMP.csv"
PKMAS = V1_DIR / "Walkway-derived metrics" / "PKMAS Walkway Gait Metrics - HP+SP.csv"

CACHE_DIR = PROJECT_DIR / "cache"                 # derived data; safe to delete
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 15071          # one global seed, used everywhere
THRESHOLD = 33        # recall a patient sooner if Part III >= THRESHOLD
N_PART3_ITEMS = 33    # scored items in MDS-UPDRS Part III (total 0-132)
MAX_MISSING_PART3 = 7  # prorate Part III with up to this many items missing (Goetz et al. 2015)
STALE_DAYS = 30       # a rating more than this many days from the gait session is "stale"
FS = 100              # sampling rate of every trial file (Hz)
TRIALS = ("SelfPace", "SelfPace_mat")   # the two walking trials per person

# ---------------------------------------------------------------------------
# Sensors. The 13 Xsens IMUs are named by body segment in the trial files
# (e.g. "LowerBack_Acc_X"); the Moticon insoles use "LPressure1", "Linsole:Acc_X",
# "LCoP_X" and the same names with an R.
# ---------------------------------------------------------------------------
IMUS = (
    "Forehead", "Xiphoid", "LowerBack",
    "L_Wrist", "R_Wrist",
    "L_MidLatThigh", "R_MidLatThigh",
    "L_LatShank", "R_LatShank",
    "L_Ankle", "R_Ankle",
    "L_DorsalFoot", "R_DorsalFoot",
)
INSOLES = "Insoles"   # treated as one sensor unit (both feet together)

# The kits compared in Contribution 1, each a tuple of sensor units.
# "one_wrist" is resolved per patient (which side is decided in Section 8),
# so it is stored here as a placeholder rather than a fixed side.
SENSOR_SETS = {
    "one_wrist": ("ONE_WRIST",),
    "both_wrists": ("L_Wrist", "R_Wrist"),
    "lower_back": ("LowerBack",),
    "insoles": (INSOLES,),
    "ankles": ("L_Ankle", "R_Ankle"),
    "lower_back_one_wrist": ("LowerBack", "ONE_WRIST"),
    "all_imus": IMUS,
    "everything": IMUS + (INSOLES,),
}


def ensure_output_dirs() -> None:
    """Create the cache and results folders if they do not exist yet."""
    for d in (CACHE_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
