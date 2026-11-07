# SUBIT-Ink v28.0 — WarpInfinity 🌟

**Release Tag:** `v28.0`  
**Target:** Vesuvius Challenge — Ink Detection & Topological Signal Recovery  
**Author:** @Slavick  
**Theoretical Basis:** SUBIT-WARP v0.4 (W0-W24 Geometric Curvature) & Belnap Boundary Regularization  
**Key Milestones:** **Public Score: 0.512781 (All-Time Record)** | **Private Score: 0.544153** (Kaggle Version 34)

---

## 🌟 Highlights & Breakthroughs

`SUBIT-Ink v28.0` sets a new all-time high on the Kaggle Public Leaderboard (**`0.512781`**) while confirming the reproducibility of the high-generalization regime with a **`0.544153` Private Score** on blinded Herculaneum papyri.

- **Geometric Warped Depth Attention (`WarpedDepthCollapse`):** Implements a learnable warp curvature prior $W(z) = \text{bias} - |\alpha| z^2$ along the $Z$-depth axis. Concentrates attention on the physical papyrus surface while suppressing attenuation noise from neighboring wraps.
- **Belnap $B$-State Boundary Regularization:** Added spatial L1-gradient loss penalizing blurred letter contours, directly optimizing precision for the $F_{0.5}$ metric.
- **Consecutive 0.54+ Generalization:** Following v27.0's $0.546201$, v28.0's $0.544153$ confirms that cross-fragment robustness is a structural property of the architecture.
- **Ensemble Convergence:** Epoch 18 ($w=0.336$), Epoch 19 ($w=0.333$), Epoch 20 ($w=0.330$) achieving holdout $F_{0.5} = 0.6924$ at $t = 0.590$.
- **5,000 Stratified Patches:** Enhanced training sample density with full $4\times$ TTA inference completed in $445.3$ minutes.

---

## 📊 Complete Benchmark Evolution

| Run | Architecture Version | Public Score | **Private Score** | Threshold | Ink Ratio (b) | Key Feature |
|---|---|---|---|---|---|---|
| **v20 Baseline** | Standard 3D U-Net | 0.5094 | **0.4367** | 0.550 | — | Overfitting baseline |
| **v26** | v22.5-SpeedChampion | 0.4643 | **0.4799** | 0.585 | — | Spatial 80/20 CV |
| **v27** | v23.0-InfinityChampion | 0.5083 | **0.4864** | 0.580 | ~6.87% | Tversky + 4x TTA |
| **v29** | v24.1-Grandmaster | 0.4973 | **0.4787** | 0.505 | — | Depth=20 test |
| **v30** | v25.0-RecallMaster | 0.4813 | **0.4797** | 0.655 | — | 75/25 split |
| **v31** | v24.0-DiverseEnsemble | 0.4272 | **0.4717** | 0.580 | 9.92% | Early epoch gap bug |
| **v32** | v26.0-GrandmasterLateConverge | 0.4745 | **0.4876** | 0.480 (clamped) | 7.05% | Late-epoch filter |
| **v33** | v27.0-OmegaAttained | 0.5008 | **0.546201 (RECORD)** | 0.620 | 6.93% | Unclamped [0.35, 0.65] |
| **v34** 🌟 | **v28.0-WarpInfinity** | **0.512781 (RECORD)** | **0.544153** | **0.590** | **7.12%** | **WarpedDepth + Belnap Edge** |

---

## 📦 What's Included

1. `ink_detection_kaggle_v28.py` — Flagship single-file submission pipeline (v28.0).
2. `ink_detection_kaggle_v27.py` — Prior milestone code (v27.0).
3. `docs/Warped_Depth_Attention_and_Belnap.md` — Formal engineering paper detailing SUBIT-WARP depth attention.
4. `docs/SUBIT_INK_Academic_Paper_EN.md` & `docs/SUBIT_INK_Academic_Paper_UA.md` — Theoretical overview papers.
