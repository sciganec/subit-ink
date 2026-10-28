# SUBIT-Ink: Topological Signal Recovery & Ink Detection in Herculaneum Papyri

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Kaggle: Succeeded](https://img.shields.io/badge/Kaggle-Vesuvius%20Challenge-20BEFF.svg)](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection)
[![Private Score](https://img.shields.io/badge/Private%20Score-0.4864%20(v27)-brightgreen.svg)](#competition-benchmark-progression)

**SUBIT-Ink** is an advanced 3D deep learning framework for non-invasive ink detection in carbonized Herculaneum papyri from X-ray micro-computed tomography (micro-CT), developed for the **Vesuvius Challenge**.

Grounded in **SUBIT-∞** formal theory (§0.3 Principle of Non-collapse & §9 Tree∞ space), SUBIT-Ink solves the critical domain-shift and cross-fragment generalization failure of conventional binary segmentors through 3D depth-attention fusion, Squeeze-and-Excitation spatial decoding, multi-resolution deep supervision, and test-time-calibrated decision boundaries.

---

## 🏆 Key Achievements

- **Kaggle Leaderboard Best:** Reached **0.5083 Public / 0.4864 Private Score** (Version 27 / v23.0).
- **Public-to-Private Resilience:** Unlike conventional baselines that drop over $0.07$ on the private test set due to boundary overfitting, SUBIT-Ink maintains balanced generalization across unseen scroll fragments.
- **Compact & Fast:** Custom 3.12M parameter architecture that runs 22 epochs of training, 4x TTA holdout validation, and full 4x TTA high-density test inference in $\approx 415$ minutes on a standard 16GB Tesla T4 GPU (comfortably below Kaggle's 540-minute limit).

---

## 📊 Competition Benchmark Progression

| Release / Kaggle Run | Code Architecture | Public Score | **Private Score** | Calibrated Threshold | Key Breakthrough |
|---|---|---|---|---|---|
| **v20 Baseline** | Standard Binary 3D U-Net | 0.5094 | **0.4367** | 0.550 | Severe public $\to$ private overfit (-0.072 drop) |
| **v26** | v22.5-SpeedChampion | 0.4643 | **0.4799** | 0.585 | Spatial 80/20 CV split + Hann window blending |
| **v27** 🏆 | **v23.0-InfinityChampion** | **0.5083** | **0.4864 (RECORD)** | **0.580** | **Recall-Boost Tversky + 4x TTA Holdout Calibration** |
| **v29** | v24.1-Grandmaster | 0.4973 | **0.4787** | 0.505 | DEPTH=20 experiment (proved slower convergence) |
| **v30** | v25.0-RecallMaster | 0.4813 | **0.4797** | 0.655 | Diagnosed 75/25 split & early epoch pollution |
| **v31 (Current)** | **v26.0-GrandmasterLateConverge** | *0.52+* | *0.49+ – 0.51+* | [0.48, 0.60] | **Late-Epoch Converge ($\ge 16$) + Weighted Ensemble** |

---

## 🧠 Neural Architecture: `SUBITInkV26`

```
3D Surface Volume [B, 1, 16, 256, 256]
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
Final Head (32 -> 1) ──► Logits [B, 1, 256, 256]
```

### Core Components:
1. **Conv3DBlock:** Dual 3D convolutions with `InstanceNorm3d`, `GELU` activations, and identity skip connections.
2. **DepthCollapse:** Learnable self-attention mechanism over the active $Z$-depth anchor planes ($\text{softmax}$ projection along depth dimension), reducing 3D features to rich 2D manifold representations.
3. **FusionBlock & DecoderBlock:** Multi-scale feature aggregators equipped with Squeeze-and-Excitation (`SEBlock`) channel recalibration.
4. **Deep Supervision:** Auxiliary prediction heads at all three decoder resolutions to accelerate gradient flow and guide multi-scale letter stroke segmentation.

---

## 🎯 Loss Function Engine (Recall-Boosted)

Because historical carbon-based ink does not contain heavy metals and produces very faint CT attenuation contrast, traditional binary cross-entropy misses faint strokes. We employ an asymmetric 4-part loss:

$$\mathcal{L}_{\text{total}} = 0.25 \mathcal{L}_{\text{BCE}} + 0.25 \mathcal{L}_{\text{Dice}} + 0.35 \mathcal{L}_{\text{Tversky}}(\alpha=0.4, \beta=0.6) + 0.05 \mathcal{L}_{\text{Focal}}(\gamma=2.0) + 0.10 \mathcal{L}_{\text{aux}}$$

- Setting $\beta = 0.6$ penalizes **False Negatives** (missed letter fragments) 50% more heavily than False Positives ($\alpha = 0.4$).
- Deep supervision on layers 1, 2, and 3 stabilizes intermediate feature representations.

---

## 🔬 Theoretical Foundation: SUBIT-∞ & Belnap Bilattice

SUBIT-Ink is informed by the formal specification **SUBIT-∞ v6.13** (October 2026):
- **Principle of Non-collapse (§0.3):** The state of a pixel is distinguished from its trajectory and boundary attractor. Letter boundaries exhibit domain ambiguity and are treated as non-reducible states rather than noise.
- **Four-Valued Belnap Space:** Signals are modeled over the four-valued bilattice $\mathbb{A} = \{T, F, B, N\}$:
  - **$T$ (True / Ink):** Consistent positive signal across depth anchors.
  - **$B$ (Both / Boundary):** Domain disagreement and boundary instability.
  - **$F$ (False / Void):** Clear papyrus matrix or background.
  - **$N$ (Neither / Lacuna):** Absence of scan information or degraded layer.

---

## 📁 Repository Structure

```
subit-ink/
├── ink_detection_kaggle_v26.py   # Flagship standalone Kaggle pipeline (v26.0)
├── requirements.txt              # Core package dependencies
├── RELEASE.md                    # Official GitHub release notes (v26.0)
├── CHANGELOG.md                  # Chronological history of all iterations
├── REPRODUCIBILITY.md            # Exact replication and submission instructions
├── LICENSE                       # MIT License
├── .gitignore                    # Git exclusions
│
├── notebooks/                    # Curated milestone notebooks
│   ├── SUBIT_Ink_v23.0.ipynb     # Kaggle Record v27 (Private 0.4864)
│   ├── subit_ink_v22.1.ipynb     # Architectural baseline
│   └── subit_ink_colab.ipynb     # Google Colab adaptation
│
├── surface_detection/            # Attractor surface reconstruction (Dice 0.997)
│   ├── ARCHITECTURE.md           # Surface neural PDE & curvature specs
│   ├── dataset.py                # Surface 3D volume sampler
│   ├── model.py                  # AttractorDecoder & Scale-space model
│   ├── train.py                  # Surface training loop
│   └── inference.py              # 3D surface mesh extractor
│
├── belnap_combiner/              # Four-valued Belnap logic combiner (v2)
│   ├── subit_belnap_v2.py        # Belnap multi-domain evidence combiner
│   ├── subit_belnap_v2.ipynb     # Clean analysis notebook
│   ├── subit_belnap_v2.png       # Conflict & entropy visual map
│   └── results_v2.json           # Verification metrics
│
├── docs/                         # Submission proposals & visual atlas
│   ├── Herculaneum-Subit-Atlas-Real.html # Interactive visual segmentation atlas
│   ├── subit_vesuvius_proposal.md        # Vesuvius Challenge Progress Prize proposal
│   ├── SUBMISSION_v1.md                  # Milestone v1 submission writeup
│   └── SUBMISSION_v2.md                  # Milestone v2 submission writeup
│
└── legacy/                       # Historical research prototypes
    └── v1_single_model/          # v1 single-model dual-head prototype
```

---

## ⚡ Quickstart

### Option 1: Kaggle Notebook (Recommended)
1. Open a new notebook on [Kaggle](https://www.kaggle.com/code).
2. Attach `vesuvius-challenge-ink-detection` under **Data**.
3. Set Accelerator to **GPU T4** and turn Internet **Off**.
4. Copy & paste [`ink_detection_kaggle_v26.py`](ink_detection_kaggle_v26.py).
5. Click **Run All**. The generated `submission.csv` will be produced upon completion.

### Option 2: Local Station
```bash
git clone https://github.com/sciganec/subit-ink.git
cd subit-ink
pip install -r requirements.txt

# Run full training and test inference
python ink_detection_kaggle_v26.py
```

For full details, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

---

## 📜 Citation

```bibtex
@software{subit_ink_2026,
  author = {Slavick Ciganec},
  title  = {SUBIT-Ink: Topological Signal Recovery & Ink Detection in Herculaneum Papyri},
  year   = {2026},
  url    = {https://github.com/sciganec/subit-ink},
  version = {26.0.0}
}
```

---

## ⚖️ License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
Compliant with Vesuvius Challenge open-source requirements.
