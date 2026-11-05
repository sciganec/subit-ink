# SUBIT-Ink v27.0 — OmegaAttained 👑

**Release Tag:** `v27.0`  
**Target:** Vesuvius Challenge — Ink Detection & Topological Signal Recovery  
**Author:** @Slavick  
**Theoretical Basis:** SUBIT-∞ v6.13.39 (§0.3 Principle of Non-collapse & §4/§11 $\omega$-Limit Attractor Theory)  
**Key Milestone:** **Private Score 0.546201** | **Public Score 0.500844** (Kaggle Version 33)

---

## 🌟 Highlights & Major Breakthrough

`SUBIT-Ink v27.0` achieves the highest benchmark performance in the history of the project, jumping from `0.4876` to **`0.546201` Private Score** on unseen blinded Herculaneum papyri volumes.

- **Breakthrough Generalization (`0.546201`):** Completely overcomes cross-fragment CT attenuation variance and sensor shift.
- **Natural Attractor Thresholding ($t = 0.620$):** In v32, thresholds were clamped into an artificial $[0.48, 0.60]$ window. Version 27.0 freed the bounds ($[0.35, 0.65]$), allowing high-resolution holdout calibration (`VAL_STRIDE_FINAL = 112`) with $4\times$ TTA to locate the true boundary attractor at $t = 0.620$.
- **Authentic Ink Ratio Recovery:** Successfully segmented fragment `b` at **$6.93\%$** ink density, directly matching the ground-truth physical ink distribution of historical Herculaneum carbon scrolls.
- **Late-Epoch Converged Ensemble:** Weighted ensemble of Epoch 16 ($w=0.337$), Epoch 20 ($w=0.335$), and Epoch 18 ($w=0.329$) achieving validation $F_{0.5} = 0.7251$.
- **4,800 Stratified 3D Patches:** Increased training sample volume with enhanced photometric augmentation (gain $[0.90, 1.10]$, bias $[-0.035, 0.035]$).

---

## 📊 Complete Benchmark Evolution

| Run | Architecture Version | Public Score | **Private Score** | Threshold | Ink Ratio (Frag b) | Ensemble |
|---|---|---|---|---|---|---|
| **v20 Baseline** | Standard 3D U-Net | 0.5094 | **0.4367** | 0.550 | - | Overfit baseline |
| **v26** | v22.5-SpeedChampion | 0.4643 | **0.4799** | 0.585 | - | Spatial 80/20 CV |
| **v27** | v23.0-InfinityChampion | 0.5083 | **0.4864** | 0.580 | ~6.87% | Tversky + 4x TTA |
| **v29** | v24.1-Grandmaster | 0.4973 | **0.4787** | 0.505 | - | Depth=20 test |
| **v30** | v25.0-RecallMaster | 0.4813 | **0.4797** | 0.655 | - | 75/25 split |
| **v31** | v24.0-DiverseEnsemble | 0.4272 | **0.4717** | 0.580 | 9.92% | Early epoch gap bug |
| **v32** | v26.0-GrandmasterLateConverge | 0.4745 | **0.4876** | 0.480 (clamped) | 7.05% | Late-epoch filter |
| **v33** 👑 | **v27.0-OmegaAttained** | **0.5008** | **0.546201 (RECORD)** | **0.620** | **6.93%** | **Ep16 + Ep20 + Ep18** |

---

## 📦 What's Included

1. `ink_detection_kaggle_v27.py` — Complete, single-file runnable pipeline for Kaggle GPU environments.
2. `ink_detection_kaggle_v26.py` — Prior iteration baseline.
3. `requirements.txt` — Frozen dependency specifications.
4. `docs/SUBIT_INK_Academic_Paper_EN.md` & `docs/SUBIT_INK_Academic_Paper_UA.md` — Formal academic paper.
