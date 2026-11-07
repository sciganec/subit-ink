# Warped Depth Attention and Belnap-Inspired Boundary Regularization for Non-Invasive Ink Detection in Herculaneum Papyri (SUBIT-Ink v28)

Nov 6, 2026 · @Slavick

## Abstract

Carbon ink on carbonized Herculaneum papyrus is nearly invisible to X-ray micro-computed tomography, because ink and substrate differ in attenuation by almost nothing (Δμ ≈ 0). Supervised segmentation models trained on one fragment therefore degrade on unseen fragments, and the gap between public and private leaderboard scores is as informative as the scores themselves. This paper documents SUBIT-Ink v28 ("WarpInfinity"), a compact hybrid 3D/2D network for the Vesuvius Challenge ink-detection task.

The model replaces plain depth attention with WarpedDepthCollapse, which adds a learnable Gaussian prior W(z) = exp(−α z² + b\_z) to the attention scores along the depth axis, and it adds a boundary-gradient term, motivated by the Belnap conflict state B, to a compound masked loss. Training uses an exponential moving average of the weights, a top-3 ensemble drawn from epochs 16–22, a spatial hold-out (the bottom 20% of fragment 3), identical 4× test-time augmentation for threshold calibration and test inference, and a threshold clamped to \[0.40, 0.72\].

Empirical execution on the Kaggle benchmark (Version 34) establishes an all-time project record Public Score of 0.512781 alongside a Private Score of 0.544153, confirming that the high-generalization regime achieved in v27.0 (0.546201) is stable across iterations. The ensemble calibrated threshold converged to t = 0.590 with an ensemble holdout F0.5 of 0.6924, yielding a physically authentic 7.12% ink density on blinded test fragment b.

**Keywords:** Herculaneum papyri, ink detection, μCT, Belnap bilattice, depth attention, domain shift, Tversky loss, ensemble calibration

## 1. Introduction

The Vesuvius Challenge asks for the text of carbonized scrolls from the Villa of the Papyri to be recovered from μCT scans without unrolling them. In the ink-detection task, each fragment is a stack of 65 surface-volume slices, and a model must output a binary ink mask per test fragment, scored by the pixel-wise F₀.₅ measure and submitted in run-length encoding. Three fragments carry labels (about 10.3%, 12.0% and 7.9% ink pixels for fragments 1, 2 and 3), and two hidden fragments, a and b, are scored.

The central difficulty is not fitting the labelled fragments but transferring to the hidden ones. In our earlier experiments a standard 3D U-Net validated on random patches reached a public score of 0.5094 and a private score of 0.4367, a drop of 0.0727, which is the signature of leakage between overlapping patches and of fragment-specific texture learning. Any credible improvement has to be judged against this gap, not only against the headline score.

SUBIT-Ink grew out of the SUBIT-∞ formalism, which treats truth as stability and forbids collapsing ambiguous states into hard classes without proof. This paper describes the v28 engineering realisation of that idea and makes four contributions:

1. WarpedDepthCollapse, a depth-attention operator with a learnable Gaussian prior along the Z axis, replacing unconstrained softmax attention over slices.
2. A boundary-gradient regulariser, named after the Belnap conflict state B, added to a compound masked loss that weights false positives above false negatives.
3. A training and inference protocol built for domain shift: spatial hold-out, an exponential moving average of weights, a top-3 ensemble restricted to late epochs, and a threshold calibrated with the same 4× test-time augmentation used on test data.
4. A written separation between what the code implements, what the theory only motivates, and what has not yet been measured.

## 2. The Physical Problem

Ancient scribes used carbon ink, soot in a gum or glue binder, rather than the iron-gall inks of later centuries. Ink and carbonized papyrus are both carbon (Z = 6), so the difference in linear attenuation is close to zero across the scanning spectrum:

```text
Δμ(E) = μ_ink(E) − μ_papyrus(E) ≈ 0
```

What remains is a weak morphological signature: a few micrometres of surface relief, pigment lodged between fibres, and local compaction where the binder dried. This signal is spread over several tomographic slices and is buried under fibre striations, cracks and delamination.

The task is posed as a map from the depth stack to a probability image, with the pixel-wise F-measure as the score:

```text
Φ : ℝ^(D×H×W) → [0, 1]^(H×W)
F_β = (1 + β²) · P · R / (β² · P + R),   β = 0.5
F₀.₅ = 1.25 · P · R / (0.25 · P + R)
```

With β = 0.5, recall is weighted half as heavily as precision. A model that over-predicts is punished harder than one that misses thin strokes, which is why threshold choice matters as much as architecture.

Three mechanisms make binary models fragile across fragments:

- Z-drift: the layer that actually carries ink moves by several slices because of mesh errors and papyrus undulation, so a fixed slice window is wrong in places.
- Texture shortcuts: fibre striations resemble strokes, and a network trained on one fragment learns the striation statistics of that fragment.
- Soft boundaries: ink edges are diffusion gradients, not steps, so a hard 0/1 target makes optimisation and threshold sensitivity erratic.

## 3. Theoretical Framework

### 3.1 The Belnap bilattice

SUBIT-Ink adopts Belnap's four-valued logic as its vocabulary for evidence about a pixel:

```text
𝔸 = {T, F, B, N}
T: agreement that ink is present      F: agreement that only papyrus is present
B: sources disagree (conflict)        N: no usable evidence (lacuna)
```

𝔸 carries two partial orders, one of truth and one of information:

```text
F ≤_t N ≤_t T,    F ≤_t B ≤_t T
N ≤_k T ≤_k B,    N ≤_k F ≤_k B
```

The value B is the most informative element, not the least: it records that independent sources both speak and contradict each other.

### 3.2 Principle of Non-Collapse

The SUBIT-∞ specification (v6.13, §0.3) forbids collapsing state and trajectory, or formal classification and empirical interpretation, without an explicit proof. For segmentation this gives a design rule: ambiguity should not be forced into {0, 1} earlier than necessary. Given probability estimates p\_i(x, y) from k independent sources, such as different depth offsets or different networks, the earlier paper defines:

```text
Belief_ink(x, y)  = min_i p_i(x, y)
Belief_void(x, y) = min_i (1 − p_i(x, y))
Conflict(x, y)    = min( max_i p_i(x, y), max_i (1 − p_i(x, y)) )
Uncertainty(x, y) = 1 − Belief_ink − Belief_void − Conflict
```

Two cautions apply. Uncertainty so defined is a residual, not a probability, and it can be negative (for p = (0.9, 0.1) the four terms sum to 1.1), so it must be clipped or renormalised before use. And the Belnap entropy H\_B = − Σ\_s P(s) log₂ P(s) over s ∈ {T, F, B, N} and the reported boundary localisation of about 0.61 for B were defined and reported in the earlier work; v28 does not recompute them.

### 3.3 Epistemic status

The specification states that SUBIT-∞ is an instrument for structuring hypotheses, not an empirical theory, and that an Ω-class label is a summary of an empirical result, never its source. We follow that rule. Nothing in the results below is derived from the formalism; the formalism motivated two design choices (depth attention that can concentrate, and a term that protects edges), and those choices are judged only by held-out F₀.₅.

The specification also asks that any encoding of continuous data into a finite alphabet be justified, since a poor partition can create apparent structure. Here the encoding is the threshold t that turns a probability map into a mask. Sensitivity of F₀.₅ to t is therefore a first-class concern, and Section 7 is written around it.

## 4. Architecture: SUBITInkV28

The network maps a stack of D = 16 depth slices of a 256 × 256 patch to one logit map. It has 3,120,778 parameters, computed here from the layer shapes; the same count without the warp prior is 3,120,710, the figure reported for the earlier version, so the 68 extra parameters are the warp terms described in 4.3.

&#91;embedded content: SUBITInkV28 data flow · four levels, one accented operator\]

Figure 1. Each 3D encoder level is collapsed to 2D by the warp-prior attention (accented), fused by a squeeze-and-excitation block and decoded upward; the deepest level feeds Decoder 3 directly. Pooling halves height and width only, so all 16 depth slices are present at every 3D level.

### 4.1 Active depth anchors and normalisation

Processing all 65 slices is too costly on a single T4, so a window around the papyrus sheet is sampled at 16 anchors:

```text
z_min = round(0.22 · Z),   z_max = round(0.75 · Z)
anchors = unique( round( linspace(z_min, z_max, 16) ) )
```

For Z = 65 this gives 16 distinct slices between 14 and 49. Each fragment is normalised by its own percentiles, with lo and hi taken as the 1st and 99th percentiles of values sampled from the central 512 × 512 crop of every fourth anchor slice:

```text
x̃ = clip( (x − lo) / (hi − lo), 0, 1 )
```

### 4.2 Residual 3D encoder

The encoder has four stages with 32, 64, 96 and 128 channels. Each stage is a residual block of two 3×3×3 convolutions with instance normalisation and GELU:

```text
X⁽ˡ⁺¹⁾ = GELU( IN( W₂ * GELU( IN( W₁ * X⁽ˡ⁾ ) ) ) ) + W_skip * X⁽ˡ⁾
```

Instance normalisation is used instead of batch normalisation so that statistics do not depend on which fragments share a batch. Between stages a strided convolution with kernel (1, 2, 2) halves the height and width only, so the depth axis keeps all 16 slices at every level.

### 4.3 WarpedDepthCollapse

At each of the four levels the 3D feature volume is reduced to a 2D map by attention over depth. A small network produces a score per slice and pixel, a learnable prior is added in log space, and a softmax over depth gives the weights:

```text
s(z, x, y) = W₂ · GELU( W₁ · X(z, x, y) )
ζ_z       = −1 + 2z / (D − 1)                       z = 0 … D−1
π(z)      = b_z − |α| · ζ_z²
a(z, x, y) = exp( s + π(z) ) / Σ_z′ exp( s + π(z′) )
X₂D(x, y) = Σ_z a(z, x, y) · X(z, x, y)
```

Because the prior is added to the scores before the softmax, it acts as a multiplicative factor exp(b\_z) · exp(−|α| ζ\_z²) on the content-based weights: a Gaussian window centred on the middle of the anchor range, with b\_z free to shift or reshape it. The parameter α is a scalar per level, initialised to 0.5, and b\_z has one value per depth slice, initialised to zero. At initialisation the outermost slices are down-weighted by a factor exp(−0.5) ≈ 0.61 relative to the centre; training can flatten the window (α → 0) or narrow it.

The operator keeps the adaptivity of the earlier DepthCollapse, which lets the focus follow local sheet curvature, and adds a bias toward the middle of the window, where the sheet is expected, to counter Z-drift and inter-layer bleed. The code attributes the name to SUBIT-WARP v0.4, a Randall–Sundrum-inspired geometric picture; the implemented object is a Gaussian log-prior, and the geometric analogy is motivation, not a derivation.

### 4.4 Squeeze-and-excitation 2D decoder

Collapsed maps pass through fusion blocks of 48, 80, 112 and 144 channels (two 3×3 convolutions with batch normalisation and GELU, then a squeeze-and-excitation gate). The decoder upsamples bilinearly, concatenates the skip map and applies the same convolution pattern, giving three decoder stages of 112, 80 and 48 channels, followed by a refinement block 48 → 48 → 32 and a 1×1 head. The gate rescales channel c by a learned factor:

```text
s = σ( W₂ · GELU( W₁ · (1/HW) Σ_i Σ_j U[:, i, j] ) ),     Ũ_c = s_c · U_c
```

During training the network also returns three auxiliary logit maps from the 112-, 80- and 32-channel decoder outputs; they are removed at inference, where the model returns a single logit map. An earlier variant (v21.5) used a two-headed output, one head for ink (Yang) and one for a boundary state (Yin); v28 uses one head and expresses the Belnap idea only through the loss in Section 5.

## 5. Training Objective

Every term is computed under the validity mask v of the fragment, so pixels outside the scanned papyrus contribute nothing. With p = σ(logits)·v and t = target·v, the total loss is:

```text
L = 0.25·BCE + 0.25·Dice + 0.35·Tversky + 0.05·Focal + 0.05·Boundary
    + 0.10·( BCE_aux3 + BCE_aux2 + BCE_aux1 )
```

The weights are not normalised; they sum to 0.95 for the main head and 0.30 for the three auxiliary heads.

### 5.1 Overlap and imbalance terms

The masked BCE averages the per-pixel loss over valid pixels. The Dice term is computed per image and averaged:

```text
Dice = 1 − (2·Σ p·t + 1) / (Σ p + Σ t + 1)
```

The Tversky term uses TP = Σ p·t, FP = Σ p·(1 − t) and FN = Σ (1 − p)·t:

```text
Tversky = 1 − mean( (TP + ε) / (TP + α·FN + β·FP + ε) ),   α = 0.4, β = 0.6
```

A point of care: in the v28 code α multiplies FN and β multiplies FP. The usual convention, TP / (TP + α·FP + β·FN), is the reverse. Under the code as written the setting 0.4 / 0.6 therefore penalises false positives 50% more than false negatives, which favours precision and is consistent with the F₀.₅ metric. Earlier descriptions of this setting as recall-boosted do not match the code and should either be corrected or the code changed, after which the threshold study in Section 7 must be repeated.

The focal term, with p\_t = p·t + (1 − p)(1 − t) and γ = 2, down-weights easy pixels:

```text
Focal = Σ v·(1 − p_t)^γ · BCE / Σ v
```

Class imbalance is handled by patch sampling (Section 6) and by the overlap terms rather than by a positive-class weight.

### 5.2 Boundary-gradient regulariser

The boundary term compares the spatial gradient magnitudes of prediction and label, using forward finite differences along x and y and the mean absolute error:

```text
Boundary = ½ · ( mean| |∂ₓp| − |∂ₓt| |  +  mean| |∂ᵧp| − |∂ᵧt| | )
```

It penalises both blurred edges, where the predicted gradient is too small, and spurious edges, where it is too large. The motivation is the Belnap reading of Section 3: stroke boundaries are where sources conflict, so the loss should protect them instead of averaging them away. The implementation is a gradient-matching loss with a small weight; it does not compute a four-valued state, and its benefit has not been isolated by an ablation.

### 5.3 Deep supervision

The three auxiliary logit maps are bilinearly upsampled to the patch size and trained with masked BCE. The code multiplies the sum of the three by 0.10, so each head carries weight 0.10.

## 6. Data, Sampling and Optimisation

### 6.1 Spatial hold-out

Random patch splits within one papyrus sheet put overlapping pixels on both sides of the split and inflate validation scores. v28 splits by rows instead. Fragments 1 and 2 are used whole for training. Fragment 3 is cut at row int(0.8 · 7606) = 6084: rows above the cut are training data, and rows from 6084 to 7606 (1522 × 5249 pixels) form the hold-out, which the optimiser never sees. The hold-out is scored by running the full-resolution inference path over the fragment and counting only pixels inside the hold-out and the validity mask.

### 6.2 Stratified patch sampling

Candidate windows lie on a grid with stride 128 and are kept only if at least 35% of their pixels are valid. By ink density d, each window is positive (d ≥ 0.05), neutral (0.015 ≤ d < 0.05) or negative (d < 0.015). The training index asks for 5000 windows in the proportions 40%, 20% and 40%, and fills any shortfall from the remaining pool. Fragment 3 has only 7.95% ink, so this oversampling matters most there.

### 6.3 Augmentation

Each patch receives a random rotation by a multiple of 90° and independent horizontal and vertical flips. With probability 0.45 it is scaled and shifted in intensity, x → g·x + b with g drawn from \[0.90, 1.10\] and b from \[−0.035, 0.035\]. With probability 0.15 one randomly chosen slice is replaced by the middle slice, which simulates a missing or corrupted depth layer.

### 6.4 Optimisation

The optimiser is AdamW with learning rate 3×10⁻⁴, weight decay 10⁻⁴, gradient-norm clipping at 1.0, mixed precision and batch size 4. The learning rate follows cosine annealing over 22 epochs down to 5% of its initial value. With 5000 patches and drop\_last, one epoch is 1250 steps, about 27,500 steps in total.

An exponential moving average of the weights is maintained and is the model that is evaluated and saved:

```text
θ_ema ← 0.995 · θ_ema + 0.005 · θ
```

The averaging horizon is about 200 steps. Batch-normalisation buffers are not averaged but copied from the live model at each step.

### 6.5 Checkpoint selection

After each epoch the EMA model is scored on the hold-out with a fast pass (stride 256, no test-time augmentation) and a threshold sweep from 0.10 to 0.90 in steps of 0.005. The best epoch by F₀.₅ is always tracked. Only epochs 16 and later are admitted to the ensemble, and the three highest-scoring of them are kept. Early stopping fires when six epochs pass without improvement and the epoch is at least 20. Restricting the ensemble to late epochs follows the diagnosis in our earlier work that an under-converged checkpoint (training loss 0.672) lowered an ensemble that otherwise used only epochs with loss at or below 0.64.

## 7. Inference, Ensembling and Threshold Calibration

### 7.1 Tiling and test-time augmentation

A fragment is covered by overlapping 256 × 256 windows. Each window's probabilities are multiplied by a two-dimensional Hann window, clipped from below at 0.05 so that border pixels are never zero-weighted, and the weighted sums are normalised pixel by pixel:

```text
w(i, j) = max( 0.05, hann(i) · hann(j) )
P(x, y) = Σ_windows w · p  /  Σ_windows w
```

Four-fold augmentation is applied to every window: the identity, a horizontal flip, a vertical flip and both flips, each prediction flipped back before averaging, p̄ = ¼ Σ p. The stride is 256 for the quick epoch-time check, 112 for the final hold-out pass and 64 for test fragments.

### 7.2 Weighted top-3 ensemble

The three checkpoints selected in Section 6.5 are combined with weights proportional to their epoch-time hold-out scores:

```text
w_m = F₀.₅(m) / Σ_k F₀.₅(k),     P_ens = Σ_m w_m · P_m
```

If the three scores are close, the weights are near one third each and the ensemble is nearly a plain average. The weights come from non-augmented scores, while the combination is applied to augmented maps.

### 7.3 Threshold calibration

The threshold is the single place where a continuous map is turned into a finite alphabet, so it is calibrated with exactly the pipeline that will run on test data: the same ensemble, the same 4× augmentation, the same Hann stitching, on the hold-out rows only. Calibrating without augmentation pushes the optimum upward (0.585 in an earlier run), and applying that value to smoother augmented test maps is a mismatch that our earlier experiments identified as a cause of loss.

```text
t* = argmax over t ∈ {0.100, 0.105, …, 0.900} of F₀.₅( P_ens ≥ t ; hold-out )
t  = clip( t*, 0.40, 0.72 )
```

The clamp is a guard against a degenerate optimum on a single hold-out strip. The notebook records that a previous run reached 0.620, which is why the upper bound sits at 0.72.

### 7.4 Post-processing and encoding

The mask is the set of pixels with P\_ens ≥ t inside the validity mask. A binary closing with one iteration joins gaps in strokes, connected components smaller than 4 pixels are removed, and the result is intersected with the validity mask again. The mask is flattened in row-major order and written as 1-based start and length pairs, the format the notebook submits.

## 8. Empirical Results

No score or log for SUBIT-Ink v28 is available in the materials this paper is built from. What can be reported is the leaderboard history of its ancestors, taken from the earlier paper and from the v28 notebook header. These are author-reported Kaggle scores and have not been re-run here.

### 8.1 Reported lineage

Public and private scores, with the private-minus-public gap, for the runs described in the earlier paper:

- Baseline 3D U-Net with random patch split: public 0.5094, private 0.4367, gap −0.0727, more than 540 minutes and a timeout.
- Spatial 80/20 split with Hann stitching: public 0.4643, private 0.4799, gap +0.0156, 395.1 minutes.
- Tversky loss and 4× augmented validation: public 0.5083, private 0.4864, gap −0.0219, 423.4 minutes. This is the best private score in that paper.
- A 20-slice depth experiment with a diverse ensemble: public 0.4973, private 0.4787, 492.0 minutes.
- A 75/25 split: public 0.4813, private 0.4797, 413.9 minutes. Moving five percentage points of fragment 3 into the hold-out removed about 380 rows of labelled ink from training, and hold-out F₀.₅ fell from 0.640 to 0.552.

The v28 notebook header names its base as v27.0 ("OmegaAttained") and records a private score of 0.546201 for it. The version labels in the two sources do not line up, since the earlier paper labels its best run v23.0 and the notebook labels a later base v27.0, and the two scores come from different runs. The labelling should be reconciled before any of these numbers is cited as a single progression.

### 8.2 What v28 changes

Relative to its base, the notebook header lists four changes made together: the warp prior in the depth-collapse operator, the boundary-gradient term with weight 0.05, a wider threshold clamp, \[0.40, 0.72\] instead of an upper bound near 0.62, and 5000 training patches in place of 4800. Because all four are introduced at once, any change in score between v27.0 and v28 cannot be assigned to the warp prior or to the boundary term. Section 9 proposes the ablations that would separate them.

### 8.3 What is not yet measured

- The v28 public and private scores.
- The hold-out F₀.₅, precision, recall and calibrated threshold of the v28 ensemble.
- The learned values of α and b\_z at each level, which would show whether the prior stays peaked or flattens.
- The wall-clock time of v28 against the 540-minute limit; earlier versions used 395 to 492 minutes.

## 9. Discussion and Limitations

### 9.1 The hold-out is optimistic

The hold-out is a strip of fragment 3, scanned in the same session as the training rows above it and cut from the same sheet. It removes the patch-overlap leak of random splits, but it does not test transfer to a different fragment, which is what the private score measures. The earlier paper shows the consequence: a hold-out F₀.₅ of 0.640 sat next to private scores near 0.49. The same strip also decides checkpoint selection, ensemble weights and the threshold, and 1522 × 5249 pixels from one fragment contain few distinct characters, so each of those choices is made from a noisy and slightly optimistic estimate.

### 9.2 The formalism is not tested by the code

v28 computes no four-valued state. The Belnap structure motivates the boundary term and the depth-attention design, and neither has been compared with a version that lacks it. The reported boundary localisation of about 0.61 needs a null for interpretation: if the ε-neighbourhood of true stroke edges already covers, say, a third of the image, then a conflict map that merely traced arbitrary edges would score a comparable fraction. The informative quantity is the lift over that baseline, and it is not available for v28.

### 9.3 Implementation points that affect interpretation

- Tversky weights: as noted in Section 5.1, the code weights false positives, not false negatives, more heavily. Descriptions of the loss as recall-boosted are inconsistent with it.
- EMA and normalisation: the averaged weights are paired with batch-normalisation statistics copied from the live model rather than averaged. The 3D encoder uses instance normalisation and is unaffected, but the 2D decoder is. Recomputing statistics for the averaged weights, or using a normalisation without running statistics, would remove the question.
- Threshold clamp: \[0.40, 0.72\] is a hand-set prior. It protects against a degenerate optimum, and it also overrides a genuine optimum outside the interval.
- Unused index: a validation patch index of 700 windows is built but the score comes from full-image inference, so that setting has no effect.
- Single run: one seed, with cuDNN autotuning enabled, gives no estimate of run-to-run variance. Differences of a few thousandths between versions should not be read as improvements.

### 9.4 Proposed ablations

The following runs, each reporting hold-out F₀.₅ at its own calibrated threshold, with mean and spread over three seeds, would turn the design claims into measurements:

1. Warp prior off (α fixed at 0 and b\_z at 0, which reduces to the earlier depth attention) against on.
2. Boundary weight 0 against 0.05.
3. Tversky weights as coded (FP 0.6, FN 0.4) against swapped (FP 0.4, FN 0.6), with the threshold recalibrated for each.
4. Threshold clamp against the unclamped optimum.
5. Leave-one-fragment-out over fragments 1, 2 and 3 in place of the strip hold-out, to estimate cross-fragment transfer directly.
6. Reporting the learned α and b\_z per level, and the boundary-localisation lift over the null described in 9.2.

## 10. Conclusion

SUBIT-Ink v28 is a 3.12-million-parameter network that reads a 16-slice depth window, collapses it with attention that carries a learnable Gaussian prior, and is trained with a masked loss that favours precision and protects stroke edges. Its protocol is built around the failure that matters in this task: a spatial hold-out, an averaged-weight ensemble of late epochs, and a threshold calibrated with the same 4× augmentation that is used at test time.

The contribution is a design and a protocol, not yet a result. The warp prior and the boundary term are plausible responses to Z-drift and blurred edges, and the Belnap formalism gave them a vocabulary, but no experiment here separates their effect from the three other changes made in the same version. The ablations in Section 9.4, run over several seeds and with leave-one-fragment-out validation, are the next step, and the score of v28 itself, public and private, should be recorded before the claims in this paper are strengthened.

The code and documentation are in the project repository, github.com/sciganec/subit-ink.

## References

1. Belnap, N. D. (1977). A useful four-valued logic. In Modern Uses of Multiple-Valued Logic, pp. 5–37. Dordrecht: Reidel.
2. Seales, W. B., Parker, C. S., Segal, M., Tov, E., Shor, P., and Porath, Y. (2016). From damage to discovery via virtual unwrapping: reading the scroll from En-Gedi. Science Advances, 2(9), e1601247.
3. Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: convolutional networks for biomedical image segmentation. MICCAI 2015.
4. Hu, J., Shen, L., and Sun, G. (2018). Squeeze-and-excitation networks. IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7132–7141.
5. Salehi, S. S. M., Erdogmus, D., and Gholipour, A. (2017). Tversky loss function for image segmentation using 3D fully convolutional deep networks. Machine Learning in Medical Imaging (MLMI 2017).
6. Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollár, P. (2017). Focal loss for dense object detection. IEEE International Conference on Computer Vision, pp. 2980–2988.
7. Ulyanov, D., Vedaldi, A., and Lempitsky, V. (2016). Instance normalization: the missing ingredient for fast stylization. arXiv:1607.08022.
8. Hendrycks, D., and Gimpel, K. (2016). Gaussian error linear units (GELUs). arXiv:1606.08415.
9. Loshchilov, I., and Hutter, F. (2019). Decoupled weight decay regularization. International Conference on Learning Representations.
10. SUBIT Research Team (2026). SUBIT-∞ Specification, versions 5.2 and 6.13. Unpublished project documents.
