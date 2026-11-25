# Edge Movement Clinic: predicting gait and balance from a short sensor walk

**15.071 The AI Edge (MIT Sloan), Project 1.** A fictional movement-disorders clinic reviews all 1,500 of its
Parkinson's patients every 6 months, whether or not anything has changed. This repository predicts each patient's
**gait-and-balance sub-score** (MDS-UPDRS Part III items 3.9–3.14, range 0–24) from a short walk recorded with
wearable sensors, and turns the prediction into a decision: **keep the review if the predicted score is above 3.75,
otherwise defer it**, freeing specialist time for patients who need it.

## The Gait model

Two views of each walk, averaged with equal weights (late fusion):

| View | Input | Model |
|---|---|---|
| Engineered features | 9 clinical inputs, 180 gait features, 30 tremor features (219) | elastic net: a regression that blends Lasso and Ridge penalties, tuned by cross-validation |
| Raw signals | 4-second windows (50% overlap) from 13 body-worn IMUs and pressure insoles | 1D CNN with one encoder shared by the 13 IMUs (23,553 parameters, 3 seeds) |

**Validation:** 5-fold cross-validation by patient, repeated 10 times (50 splits); imputation, scaling, feature
selection, tuning, early stopping and the cut-off are all fit on training patients only. 95% confidence intervals
come from 1,000 bootstrap resamples of patients.

## Results (97 patients)

| | MAE (points) |
|---|---|
| Predict the average | 2.13 (95% CI 1.79–2.53) |
| Elastic net alone | 1.84 |
| CNN alone | 1.89 |
| **Gait model** | **1.80 (1.54–2.09)** |

At the 3.75 cut-off ("needs the clinic" = actual score ≥ 4, 50 of 97 patients): sensitivity 71%, specificity 70%,
NPV 69%, AUROC 0.74; with the cut-off chosen inside each split, sensitivity 68% and specificity 69%. For the
fictional clinic (planning assumptions: 2-hour reviews at $500 per specialist hour, so $1,000 per repeat review; a
first visit pays twice that) this defers about 1,460 of 3,000 yearly reviews and frees about 2,800 specialist hours,
enough for about 1,400 first visits: about $1.22M a year more income net of sensor walks and catch-up visits. About 450
deferred reviews are for patients who needed care. Main limits: scripted lab walks rather than home walks, uncontrolled medication state, and predictions pulled
toward the middle (the most impaired are predicted about 2.3 points too low).

## Repository

| Path | What |
|---|---|
| `gait_model.ipynb` | the full analysis: data, both models, fusion, results, figures, clinic decision and impact |
| `edgeclinic/` | the Python package the notebook uses (data loading, features, windows, models, CNN, evaluation, figure style) |
| `scripts/choose_threshold.py` | how the 3.75 cut-off was chosen (inside each split; run after the notebook) |
| `results/figures/`, `results/tables/` | the outputs the notebook writes (figures as 600 dpi PNG, PDF and SVG) |

## Reproduce

1. **Python 3.13** and the packages: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. **Data:** download WearGait-PD (Scientific Data, 2026; Synapse syn52540892; access follows the dataset's terms) and
   put it in `data/weargait_pd/`, keeping its folder names (`Version 1 (Cross-sectional)/`, …). Or set
   `EDGECLINIC_DATA_DIR` to the folder where it already is.
3. **Run** `gait_model.ipynb` from the repository root (Run All), or
   `jupyter nbconvert --to notebook --execute --inplace gait_model.ipynb`.
   The first run builds `cache/`: features and windows from the raw data, then the CNN for 50 splits × 3 seeds,
   about 2 hours on an Apple GPU (MPS) or a CUDA GPU. Later runs read the cache and take seconds. One global seed
   (15071) is used throughout.

## Team

Afis Ajala, Abby Wu, Maria Imbett, Felix Humphries.
