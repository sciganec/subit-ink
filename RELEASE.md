# SUBIT-Ink v26.0 — Grandmaster-LateConverge 🚀

**Release Tag:** `v26.0`  
**Target:** Vesuvius Challenge — Ink Detection & Topological Signal Recovery  
**Author:** @Slavick  
**Theoretical Basis:** SUBIT-∞ v6.13 (§0.3 Principle of Non-collapse & §9 Tree∞ space)  

---

## 🌟 Highlights & Major Improvements

`SUBIT-Ink v26.0` consolidates the architectural and empirical findings across 30+ full Kaggle training iterations on the Vesuvius Challenge dataset (3D X-ray surface CT scans of carbonized Herculaneum papyri).

- **Proven High Generalization:** Version 27 established an all-time record of **0.5083 Public / 0.4864 Private score** on the official competition benchmark, beating the public-to-private collapse of standard binary models.
- **Late-Epoch Converged Ensemble (`epoch >= 16`):** Prevents under-converged early weights from polluting the ensemble. Only models with stabilized training loss ($\le 0.64$) and refined boundary sharpness participate.
- **Validation-Weighted Softmax Blending:** Checkpoints are blended weighted by their holdout $F_{0.5}$ validation score, giving optimal influence to peak representations.
- **Harmonized 4x TTA Calibration:** Both validation holdout threshold search and final test inference run identical 4x Test-Time Augmentation (flips $H \times V$) with 2D Hann window blending, eliminating threshold mismatch.
- **Strict 80/20 Spatial Split Restored:** Train on Frag 1, Frag 2, and the top 80% of Frag 3; holdout on the bottom 20% (1,522 rows), maximizing ink training signal while retaining faithful validation feedback.
- **Recall-Boost Loss Engine:** Custom 4-component loss balancing Tversky ($\alpha=0.4, \beta=0.6$), Soft Dice ($w=0.25$), Focal loss ($w=0.05, \gamma=2.0$), and 3-level Deep Supervision ($w=0.10$).

---

## 📊 Benchmark Progression

| Release / Kaggle Version | Public Score | **Private Score** | Threshold | Top Checkpoints | Key Feature |
|---|---|---|---|---|---|
| **v20 Baseline** | 0.5094 | 0.4367 | 0.550 | Ep12, Ep15 | Severe overfitting (-0.072 drop) |
| **v26 (v22.5)** | 0.4643 | 0.4799 | 0.585 | Ep16, Ep19, Ep18 | First spatial 80/20 CV split |
| **v27 (v23.0)** 🏆 | **0.5083** | **0.4864** | **0.580** | **Ep18, Ep19, Ep20** | **Recall-Boost + 4x TTA Calibration** |
| **v29 (v24.1)** | 0.4973 | 0.4787 | 0.505 | Ep14, Ep18, Ep20 | DEPTH=20 experiment |
| **v30 (v25.0)** | 0.4813 | 0.4797 | 0.655 | Ep21, Ep14, Ep16 | 75/25 split diagnosis |
| **v26.0 (Current)** | *Target 0.52+* | *Target 0.49+ – 0.51+* | [0.48, 0.60] | Ep18, Ep20, Ep21 | **Late-Converge + Weighted Ensemble** |

---

## 🛠️ Architecture Overview

```
3D Surface Volume (B, 1, 16, 256, 256)
  │
  ├──► Conv3DBlock (1 -> 32)   ──► DepthCollapse (Z=16 -> 1) ──► FusionBlock (32 -> 48)  ──► Skip 1
  ├──► Downsample (1, 2, 2)
  ├──► Conv3DBlock (64 -> 64)  ──► DepthCollapse (Z=16 -> 1) ──► FusionBlock (64 -> 80)  ──► Skip 2
  ├──► Downsample (1, 2, 2)
  ├──► Conv3DBlock (96 -> 96)  ──► DepthCollapse (Z=16 -> 1) ──► FusionBlock (96 -> 112) ──► Skip 3
  ├──► Downsample (1, 2, 2)
  └──► Conv3DBlock (128 -> 128)──► DepthCollapse (Z=16 -> 1) ──► FusionBlock (128 -> 144) (Bottleneck)
                                                                       │
  ┌────────────────────────────────────────────────────────────────────┘
  ▼
DecoderBlock 3 (144 + 112 -> 112) ──► Deep Supervision Aux 3 (1x1 Conv)
  ▼
DecoderBlock 2 (112 + 80 -> 80)   ──► Deep Supervision Aux 2 (1x1 Conv)
  ▼
DecoderBlock 1 (80 + 48 -> 48)
  ▼
Refinement Head (48 -> 48 -> 32) ──► Deep Supervision Aux 1 (1x1 Conv)
  ▼
Final Head (32 -> 1) ──► Logits (B, 1, 256, 256)
```

- **Parameters:** `3,120,710` (compact, high throughput, zero VRAM overflow on 16GB Tesla T4).
- **Inference Runtime:** ~415 minutes for 22 full training epochs + 4x TTA validation + 4x TTA test inference (well below Kaggle's 540-minute timeout).

---

## 📦 What's Included in This Release

1. `ink_detection_kaggle_v26.py` — Self-contained, single-file runnable Kaggle submission script (v26.0).
2. `requirements.txt` — Full environment specifications.
3. `REPRODUCIBILITY.md` — Step-by-step instructions to train and generate `submission.csv`.
4. `CHANGELOG.md` — Full chronological log of all iterations from v1 to v26.
5. `notebooks/` — Milestone notebooks including record-holder `SUBIT_Ink_v23.0.ipynb` (v27, Private 0.4864).
6. `surface_detection/` — Neural PDE attractor surface segmentation (Dice 0.997).
7. `belnap_combiner/` — Four-valued Belnap logic evidence combiner.
8. `docs/Herculaneum-Subit-Atlas-Real.html` — Interactive visual atlas of Herculaneum papyrus segmentation.

---

## 🚀 Quick Execution (Kaggle)

1. Create a new notebook on [Kaggle](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/code).
2. Attach dataset `vesuvius-challenge-ink-detection`.
3. Enable **GPU T4** and **Internet Off**.
4. Paste the entire content of `ink_detection_kaggle_v26.py`.
5. Run the notebook. Output `submission.csv` will be produced upon completion.
