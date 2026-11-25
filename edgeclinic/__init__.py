"""Edge Clinic: predicting MDS-UPDRS Part III from WearGait-PD walking trials.

Functions are grouped by purpose so each step can be tested on its own:

- paths     all paths and constants (SEED, THRESHOLD, sensor sets)
- io        reading the clinical tables, the walkway metrics and the trial files
- targets   the Part III total (with prorating) and the recall flag
- segments  walking bouts, turns and windows
- features  per-sensor gait features
- cv        patient-level cross-validation
- models    baselines, tabular models and the 1D-CNN
- conformal prediction intervals
- decision  recall rules and the Edge Clinic impact model
- evaluate  metrics and bootstrap confidence intervals
- deck_style figure style matching the project deck (colours, Georgia/Arial, PNG+PDF+SVG export)
"""
