# Changelog

All notable changes to the **SUBIT-Ink** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [26.0.0] - 2026-10-27
### Added
- **Late-Epoch Ensemble Filter (`MIN_ENSEMBLE_EPOCH = 16`):** Prevents under-converged early models (such as Ep14) from entering the top-3 ensemble.
- **Weighted Ensemble Blending:** Checkpoints are blended proportionally to their validation $F_{0.5}$ score.
- **Tightly Calibrated Threshold Clamping (`[0.48, 0.60]`):** Centered directly around the record-setting 0.580 region discovered in Version 27.
- Complete documentation suite: `RELEASE.md`, `CHANGELOG.md`, `REPRODUCIBILITY.md`, and updated root `README.md`.
- `requirements.txt` for clean reproducibility across Linux and Windows environments.

### Fixed
- **Restored Spatial CV Split to 80/20:** Corrected regression observed in v25.0 (v30) caused by 75/25 split shifting 380 rows of training data from Fragment 3 into holdout.
- Ensured full compatibility with Kaggle 540-minute timeout (~415 min runtime on Tesla T4).

---

## [25.0.0] - 2026-10-22 (Kaggle Version 30)
### Changed
- Attempted `TRAIN_VAL_SPLIT = 0.75` (diagnosed as cause of signal dilution in validation holdout).
- Expanded threshold search clamp to `[0.40, 0.70]`.
- Lowered checkpoint eligibility to `epoch >= 6`.
### Results
- Public Score: `0.481275` | Private Score: `0.479732` (Runtime: 414 min).

---

## [24.1.0] - 2026-10-21 (Kaggle Version 29)
### Changed
- Experimented with `DEPTH = 20` depth anchor slices (increased epoch training time from ~13.5 min to ~21.4 min).
- Diverse checkpoint policy requiring minimum epoch gap of $\ge 2$.
### Results
- Public Score: `0.497277` | Private Score: `0.478691` (Runtime: 492 min).
- Finding: `DEPTH = 16` is faster and produces better generalization within fixed time constraints.

---

## [23.0.0] - 2026-10-20 (Kaggle Version 27) 🏆 **ALL-TIME RECORD**
### Added
- **Recall-Boost Loss Engine:** Adjusted Tversky to $\alpha=0.4, \beta=0.6$, increased Dice to $0.25$, decreased Focal to $0.05$, and removed artificial precision pressure.
- **Harmonized 4x TTA Calibration:** Search for ensemble threshold using the exact same 4x TTA used during test inference.
- Reduced morphological `MIN_COMPONENT_SIZE` from 8 to 4 to preserve letter terminals and diacritics.
### Results
- **Public Score: `0.508254` | Private Score: `0.486430` (Competition Best)**.
- Optimal calibrated threshold: `0.580`.

---

## [22.5.0] - 2026-10-19 (Kaggle Version 26)
### Added
- Speed Champion pipeline with fast validation (`VAL_STRIDE_EPOCH = 256`, ~1 min per epoch).
- Solved Kaggle timeout and worker crash issues (`NUM_WORKERS = 0`, pin_memory with AMP).
- 2D Hann window blending for seamless patch stitching.
### Results
- Public Score: `0.464323` | Private Score: `0.479973` (First version where Private > Public, confirming generalizability).

---

## [21.0.0 - 22.1.0] - 2026-08-25
### Added
- Initial 3D Residual Depth Encoder + SE Attention 2D Decoder architecture (`SUBITInkV22`).
- 4-valued Belnap logic boundary analysis (Yang/Yin output maps).
- Spatial split formulation on Fragment 3.

---

## [1.0.0 - 2.0.0] - 2026-06-26
### Added
- Early dual-head prototype and multi-depth Belnap combiner.
- Verification on Fragment 1 real volume data.
