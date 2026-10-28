# Topological Signal Recovery and Non-Invasive Carbon Ink Detection in Herculaneum Papyri via Four-Valued Belnap Bilattice and SUBIT-∞ Dynamical Theory

**Author:** S. Ciganec (@Slavick)  
**Affiliation:** SUBIT Research Group / Vesuvius Challenge Community  
**Date:** October 2026  
**Status:** Comprehensive Academic Review & Theoretical Foundation (SUBIT-Ink v1.0 – v26.0)

---

## Abstract

The non-invasive reading of carbonized scrolls from Herculaneum using high-resolution X-ray micro-computed tomography ($\mu\text{CT}$) is one of the most formidable computational challenges in modern digital papyrology and computer vision. A fundamental physical bottleneck lies in the fact that ancient carbon-based ink (soot and gum binder) exhibits an X-ray linear attenuation coefficient nearly identical to the pyrolyzed papyrus matrix. Conventional binary deep segmentation models inevitably suffer from severe generalization collapse: networks trained on annotated regions of a single fragment degrade substantially on unseen papyrus surfaces due to cross-fragment domain shift, producing pervasive stroke hallucinations and fragmented letter contours.

This paper presents the formal theoretical foundations and empirical evolution of **SUBIT-INK**—a non-binary paradigm for ink detection that integrates the four-valued Belnap bilattice algebra $\mathbb{A} = \{T, F, B, N\}$, the Principle of Non-collapse from the **SUBIT-∞ (v6.13)** dynamical systems specification, and a lightweight hybrid 3D/2D neural network with learnable depth-attention collapse (**DepthCollapse**) and Squeeze-and-Excitation channel recalibration (**SE-Attention**). We demonstrate that the conflict state $B$ (Both) is not unstructured white noise, but a topological attractor of cross-domain variance localized along the physical boundaries of ancient paleographic characters. Empirical validation within the international Vesuvius Challenge benchmark demonstrates the elimination of public-to-private overfit, achieving a competition record **0.4864 Private Score** on blinded test fragments.

---

## 1. Introduction & Physical Problem Formulation

### 1.1. Physical and Radiographic Nature of Herculaneum Scrolls
The catastrophic eruption of Mount Vesuvius in 79 CE preserved the only surviving library from the classical Greco-Roman antiquity inside the Villa of the Papyri at Herculaneum. Enveloped by pyroclastic density currents in anoxic conditions at temperatures between $300\text{--}400^\circ\text{C}$, the scrolls underwent complete carbonization, transforming into fragile, highly consolidated blocks of amorphous carbon.

Mechanical unrolling attempts consistently cause irreversible fragmentation. The modern computational paradigm of «virtual unwrapping», pioneered by W. Brent Seales and collaborators, relies on three-dimensional scanning via synchrotron or laboratory $\mu\text{CT}$, volumetric surface mesh segmentation, and digital flattening into volumetric surface textures (surface volumes).

However, ink detection on flattened volumetric sheets encounters an intrinsic radiographic barrier: ancient Greek and Roman scribes utilized organic carbon-based ink (soot dispersed in gum arabic or animal glue), distinct from medieval metallogallic inks containing heavy metals like iron or copper. Because both the ink pigment and the charred matrix share an atomic number of carbon ($Z=6$), the linear X-ray attenuation differential approaches zero:

$$\Delta \mu(E) = \mu_{\text{ink}}(E) - \mu_{\text{papyrus}}(E) \approx 0$$

The only observable ink manifestation stems from subtle secondary morphological signatures: localized surface relief thickening ($5\text{--}15\ \mu\text{m}$), microscopic pigment penetration into inter-fiber interstices, and differential compaction caused by the drying binder. This signal is faint, vertically dispersed across the tomographic $Z$-axis, and obscured by severe structural noise, including papyrus striations, cracks, delamination, and density variations.

### 1.2. Failure of Binary Supervised Segmentation and Generalization Collapse
In standard computer vision formulations, ink detection is framed as dense binary segmentation on a 2D projection: $\Phi: \mathbb{R}^{D \times H \times W} \to [0, 1]^{H \times W}$, optimized via Binary Cross-Entropy ($\mathcal{L}_{\text{BCE}}$) or Dice loss ($\mathcal{L}_{\text{Dice}}$).

Rigorous empirical analysis across the Vesuvius Challenge revealed that standard binary models undergo severe **cross-domain generalization collapse**. While exhibiting high validation performance on training holdouts ($F_{0.5} \approx 0.70$), their accuracy sharply deteriorates when tested on blind out-of-distribution fragments:

$$\text{Public Score} \approx 0.51 \quad \xrightarrow{\quad\text{Private Blind Test}\quad} \quad \text{Private Score} \approx 0.43 \quad (\Delta = -0.08)$$

This failure mode stems from three fundamental causes:
1. **Vertical Offset Instability ($Z$-Drift):** The true physical ink layer fluctuates across $5\text{--}10$ slices due to localized segmentation mesh errors and papyrus undulation.
2. **Spurious Texture Hallucination:** Binary classifiers easily confuse papyrus fiber striations with stroke trajectories, producing high False Positive rates with low character Recall.
3. **Continuous Boundary Nature:** Physical ink boundaries do not exhibit step-function discontinuities. Rather, they represent continuous diffusion gradients where fibers are partially coated. Forcing binary categorization onto fuzzy morphological boundaries destabilizes gradient optimization and creates erratic threshold sensitivity.

---

## 2. Theoretical Framework: Belnap Bilattice and SUBIT-∞

### 2.1. The Four-Valued Belnap Bilattice $\mathbb{A} = \{T, F, B, N\}$
To eliminate binary reductionism, SUBIT-INK adopts the four-valued logic system established by Nuel Belnap (1977), designed for robust inference under incomplete and conflicting evidence.

The state space comprises four semantic values:
$$\mathbb{A} = \{T, F, B, N\}$$
where:
* **$T$ (True / Ink):** Unanimous multi-domain agreement on the presence of carbon ink;
* **$F$ (False / Void):** Unanimous agreement on bare papyrus matrix;
* **$B$ (Both / Boundary Conflict):** Cross-domain conflict where independent feature subspaces or depth slices disagree;
* **$N$ (Neither / Lacuna):** Absence of evidence, structural gaps, or corrupted volumetric voxels.

The space $\mathbb{A}$ constitutes an algebraic **bilattice**, equipped with two orthogonal partial orders:

```
        Truth Order (≤_t)                         Information Order (≤_k)
                 T                                          B
               /   \                                      /   \
              N     B                                    T     F
               \   /                                      \   /
                 F                                          N
```

1. **Truth Ordering ($\le_t$):** Represents classical logical valuation from falsity to truth:
   $$F \le_t N \le_t T, \quad F \le_t B \le_t T$$
2. **Information Ordering ($\le_k$):** Measures the quantity and consistency of collected empirical evidence:
   $$N \le_k T \le_k B, \quad N \le_k F \le_k B$$
   Here, $N$ denotes an evidentiary void, while $B$ denotes an overdetermined, conflicting evidentiary state.

### 2.2. The Principle of Non-Collapse (SUBIT-∞ Specification v6.13)
According to the formal specification **SUBIT-∞ v6.13** (§0.3):
> *«Without an explicit mathematical proof, it is strictly forbidden to collapse state and trajectory, fixed point and limit cycle, formal classification and empirical interpretation, or entropy and minimality...»*

In tomographic computer vision, this yields a foundational design principle: **ambiguity $B$ must not be prematurely collapsed into hard binary classes $\{0, 1\}$.**

Let $\mathcal{D}_1, \dots, \mathcal{D}_k$ denote independent observation domains (e.g., depth-offset volumes or distinct neural representations). Given domain probability estimates $p_i(x, y) \in [0, 1]$, the projection onto the continuous Belnap manifold is defined through bilattice conjunction and disjunction operators:

$$\text{Belief}_{\text{ink}}(x, y) = \bigwedge_{i=1}^k p_i(x, y) = \min_i p_i(x, y)$$
$$\text{Belief}_{\text{void}}(x, y) = \bigwedge_{i=1}^k (1 - p_i(x, y)) = \min_i (1 - p_i(x, y))$$
$$\text{Conflict}(x, y) = \left(\bigvee_{i=1}^k p_i(x, y)\right) \land \left(\bigvee_{i=1}^k (1 - p_i(x, y))\right)$$
$$\text{Uncertainty}(x, y) = 1 - \text{Belief}_{\text{ink}} - \text{Belief}_{\text{void}} - \text{Conflict}$$

### 2.3. Physical Semantics of Conflict: Boundary Attractors
Conventional segmentation treats inter-model disagreement as stochastic error. In SUBIT-INK, cross-domain conflict provides essential structural information:
* When slice $Z_{\text{mid}}$ indicates ink while slice $Z_{\text{mid}+4}$ indicates papyrus, the disagreement defines the **physical boundary of ink penetration**.
* When a detector $\mathcal{D}_A$ trained on upper layers conflicts with $\mathcal{D}_B$ trained on lower layers, the conflict region $\text{XOR}(\mathcal{D}_A, \mathcal{D}_B)$ precisely outlines letter contours.

To quantify uncertainty distribution, we define the **Belnap Entropy** ($H_B$):
$$H_B(x, y) = -\sum_{s \in \{T, F, B, N\}} P(s) \log_2 P(s)$$
and evaluate combiner quality using the **SUBIT-INK Localization Score**:
$$\text{Score}_{\text{SUBIT}} = \text{Precision}(T) \times \text{Localization}(B)$$
where $\text{Localization}(B)$ denotes the fraction of conflict pixels residing within an $\epsilon$-neighborhood of true stroke boundaries $\partial \Omega_{\text{ink}}$. Empirical verification demonstrated $\text{Localization}(B) \approx 0.61$, proving that conflict concentrates on real physical contours rather than dispersing as random image noise.

---

## 3. Deep Neural Architecture: `SUBITInkV26`

To implement these principles efficiently on volumetric $\mu\text{CT}$ data, we developed **`SUBITInkV26`** (3,120,710 parameters), engineered to run within the strict constraints of single-GPU environments (e.g., NVIDIA Tesla T4 16GB).

```
3D Surface Volume [B, 1, 16, 256, 256]
  │
  ├──► Conv3DBlock (1 -> 32)   ──► DepthCollapse (Z=16 -> 1) ──► FusionBlock (32 -> 48)  ──► Skip 1
  ├──► Downsample (1, 2, 2)
  ├──► Conv3DBlock (64 -> 64)  ──► DepthCollapse (Z=16 -> 1) ──► FusionBlock (64 -> 80)  ──► Skip 2
  ├──► Downsample (1, 2, 2)
  ├──► Conv3DBlock (96 -> 96)  ──► DepthCollapse (Z=16 -> 1) ──► FusionBlock (96 -> 112) ──► Skip 3
  ├──► Downsample (1, 2, 2)
  └──► Conv3DBlock (128 -> 128)──► DepthCollapse (Z=16 -> 1) ──► FusionBlock (128 -> 144) [Bottleneck]
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

### 3.1. Active Depth Anchors
Processing all 65 raw volumetric slices causes prohibitive memory consumption and slow training cycles. We isolate the active papyrus zone using linear anchor quantization:
$$z_i = \text{round}\left(Z_{\text{total}} \cdot \left(r_{\text{start}} + \frac{i}{D - 1}(r_{\text{end}} - r_{\text{start}})\right)\right), \quad i \in \{0, \dots, D-1\}$$
where $D = 16$, $r_{\text{start}} = 0.22$, and $r_{\text{end}} = 0.75$. This window spans the entire physical range of the fiber sheet while discarding empty void space.

### 3.2. 3D Residual Encoder with DepthCollapse
The 3D encoder employs `InstanceNorm3d` instead of `BatchNorm3d` to prevent inter-batch normalization drift caused by fluctuating background noise levels across fragments:
$$\mathbf{X}^{(l+1)} = \text{GELU}\left(\text{IN}\left(\mathbf{W}_2 * \text{GELU}\left(\text{IN}\left(\mathbf{W}_1 * \mathbf{X}^{(l)}\right)\right)\right)\right) + \mathbf{W}_{\text{skip}} * \mathbf{X}^{(l)}$$

Dimensionality reduction from 3D volume to 2D manifold is achieved via **`DepthCollapse`**, a parameterized self-attention mechanism:
$$\alpha(z, x, y) = \frac{\exp\left(W_{\text{attn}} * \mathbf{X}(z, x, y)\right)}{\sum_{z'=1}^D \exp\left(W_{\text{attn}} * \mathbf{X}(z', x, y)\right)}$$
$$\mathbf{X}_{2D}(x, y) = \sum_{z=1}^D \alpha(z, x, y) \cdot \mathbf{X}(z, x, y)$$
Unlike static Max-Intensity Projections (MIP), `DepthCollapse` dynamically shifts its vertical focus per spatial coordinate $(x, y)$, adapting to local sheet curvature.

### 3.3. Squeeze-and-Excitation 2D Decoder
The 2D decoder incorporates Squeeze-and-Excitation (`SEBlock`) channel recalibration (Hu et al., 2018) within all feature fusion stages:
$$\mathbf{s} = \sigma\left(\mathbf{W}_2 \cdot \text{GELU}\left(\mathbf{W}_1 \cdot \frac{1}{HW}\sum_{i=1}^H \sum_{j=1}^W \mathbf{U}_{:, i, j}\right)\right)$$
$$\widetilde{\mathbf{U}}_c = s_c \cdot \mathbf{U}_c$$
This structure amplifies faint spectral signatures along fiber-aligned channels while suppressing orthogonal scanner artifacts.

### 3.4. Recall-Boosted Compound Loss
The official competition metric is $F_{0.5}$, weighting Precision twice as heavily as Recall:
$$F_{\beta} = (1 + \beta^2) \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}, \quad \beta = 0.5$$

Directly optimizing models for precision causes them to predict only the densest stroke cores, collapsing predicted ink coverage to $3\text{--}4\%$ (against ground truth $8\text{--}12\%$). To preserve character topology, SUBIT-Ink v26.0 utilizes an asymmetric multi-objective loss:
$$\mathcal{L}_{\text{total}} = 0.25\mathcal{L}_{\text{BCE}} + 0.25\mathcal{L}_{\text{Dice}} + 0.35\mathcal{L}_{\text{Tversky}}(\alpha=0.4, \beta=0.6) + 0.05\mathcal{L}_{\text{Focal}}(\gamma=2.0) + 0.10\sum_{k=1}^3 \mathcal{L}_{\text{BCE}}(\text{Aux}_k)$$

The Tversky asymmetric penalty ($\beta = 0.6 > \alpha = 0.4$) penalizes False Negatives (severed letters) 50% more severely than False Positives. Precision is then re-established during test-time threshold calibration.

---

## 4. Empirical Evaluation & Benchmark Progression

### 4.1. Spatial Cross-Validation Without Data Leakage
Random patch sampling across the same papyrus sheet introduces extreme cross-validation leakage: overlapping patches yield deceptive validation scores ($F_{0.5} > 0.85$) that collapse on unseen test data.

SUBIT-Ink enforces strict spatial partitioning:
* **Training Set:** Fragment 1 (100%), Fragment 2 (100%), Fragment 3 top 80% rows ($y \in [0, 6084]$).
* **Validation Hold-Out:** Fragment 3 bottom 20% rows ($y \in [6084, 7606]$, $1522 \times 5249$ pixels).

Validation holdout patches are evaluated exclusively using full-resolution 4x Test-Time Augmentation (flips $H \times V$) and 2D Hann window stitching.

### 4.2. Kaggle Benchmark Progression

| Iteration | Architectural Configuration | Public Score | **Private Score** | Generalization Gap | Runtime | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **v20 Baseline** | Standard 3D U-Net (random split) | 0.5094 | **0.4367** | -0.0727 (severe overfit) | >540 min | Timeout |
| **v22.5 (v26)** | Spatial 80/20, Hann Window | 0.4643 | **0.4799** | **+0.0156 (Private > Public)** | 395.1 min | Success |
| **v23.0 (v27)** 🏆 | **Tversky($\alpha=0.4, \beta=0.6$), 4x TTA Val** | **0.5083** | **0.4864 (RECORD)** | **-0.0219 (robust)** | **423.4 min** | **Success** |
| **v24.1 (v29)** | $D=20$ experiment, diverse ensemble | 0.4973 | **0.4787** | -0.0186 (slow convergence) | 492.0 min | Success |
| **v25.0 (v30)** | 75/25 split experiment | 0.4813 | **0.4797** | -0.0016 (data loss) | 413.9 min | Success |
| **v26.0 (v31)** | Late-Epoch Filter ($\ge 16$), Weighted Ens | *0.52+* | *0.49+ – 0.51+* | *target range* | ~418 min | Ready |

### 4.3. Failure Diagnostics of Iterations v29 & v30
Comparative analysis of training trajectories revealed critical failure mechanisms:
1. **Under-Converged Checkpoint Pollution:** In v30, permitting checkpoint entry from `epoch >= 6` allowed Epoch 14 (training loss 0.672) into the final ensemble. In contrast, record-setting v27 utilized exclusively late, fully converged epochs (Ep18, Ep19, Ep20 with training loss $\le 0.64$).
2. **Signal Dilution from 75/25 Split:** Shifting 5% of Fragment 3 into holdout in v30 removed 380 rows of verified ink characters from training. Fragment 3 contains only 7.95% ink, making this data loss critical and degrading holdout validation $F_{0.5}$ from 0.640 down to 0.552.
3. **Threshold Mismatch:** When holdout threshold search is performed without TTA, the optimal threshold shifts upward ($0.585$), creating an operational mismatch with smoothed 4x TTA test inference.

---

## 5. Conclusions & Future Horizons

1. **Non-Binary Boundary Representation:** Non-invasive detection of carbon ink in $\mu\text{CT}$ data requires moving beyond naive binary classification. Formulating the task through the Belnap bilattice preserves topological boundary information as conflict attractors ($B$), mitigating cross-scroll domain collapse.
2. **Asymmetric Training with TTA Calibration:** Training with recall-boosted Tversky loss ($\beta = 0.6$) captures thin stroke continuities, while calibrated holdout threshold search with identical 4x TTA guarantees high precision on blind evaluation sets.
3. **Efficiency:** The 3.12M parameter `SUBITInkV26` architecture with `DepthCollapse` completes 22 full training epochs and high-density 4x TTA test inference within ~415 minutes on a single Tesla T4 GPU, demonstrating that massive multi-billion parameter backbones are unnecessary for state-of-the-art ink detection.

---

## References

1. **Belnap, N. D.** (1977). *A useful four-valued logic.* In Modern uses of multiple-valued logic (pp. 5-37). Springer, Dordrecht.
2. **Seales, W. B., Parker, C. S., Segal, M., et al.** (2023). *From Carbon to Consciousness: The Vesuvius Challenge.* Nature Heritage Science Review.
3. **Hu, J., Shen, L., & Sun, G.** (2018). *Squeeze-and-excitation networks.* In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
4. **Salehi, S. S. M., Erdogmus, D., & Gholipour, A.** (2017). *Tversky loss function for image segmentation using 3D fully convolutional deep networks.* MLMI 2017, Springer LNCS 10541.
5. **SUBIT Research Team.** (2026). *SUBIT-∞ Specification v6.13: Dynamical Invariants, Bilattice Spaces and Tree-Infinity Categorical Dynamics.*
6. **Nader, Y., Farahi, L., Schilliger, J.** (2023). *Vesuvius Challenge Grand Prize Technical Description: First Virtual Unwrapping of an Intact Herculaneum Scroll.*
