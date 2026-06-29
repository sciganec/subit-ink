# SUBIT-BELNAP COMBINER v2
## Four-Valued Logical Domain Adaptation for Ink Detection
### Vesuvius Challenge — Progress Prize Submission (Updated)

---

### The genuine Belnap formulation

Three independent ink detectors, each seeing a different "view"
of the same papyrus:

```
Model A: central layers  (mid-3 : mid+4)   — core ink signal
Model B: upper layers    (shifted -8)       — surface texture
Model C: lower layers    (shifted +8)       — subsurface signal
```

Belnap combination (corrected truth table):

```
A=1, B=1, C=1  →  INK     cross-domain agreement
A=1, B=1, C=0  →  BOTH    partial disagreement
A=1, B=0, C=1  →  BOTH    partial disagreement
A=0, B=1, C=1  →  BOTH    partial disagreement
A=0, B=0, C=0  →  VOID    agreement negative
uncertain all  →  UNKNOWN  no signal
```

**Soft Belnap via evidence accumulation** (no hard threshold):

```python
belief_ink  = min(evidence_ink_A,  evidence_ink_B,  evidence_ink_C)
belief_void = min(evidence_void_A, evidence_void_B, evidence_void_C)
conflict    = Σ min(evidence_ink_i, evidence_void_j)  for i≠j
uncertainty = 1 - belief_ink - belief_void - conflict
```

---

### Results on Fragment 1 (corrected crop, 65 layers)

| Method | F0.5 (Vesuvius metric) | Notes |
|--------|------------------------|-------|
| Mean ensemble (baseline) | 0.9166 | standard approach |
| Hard Belnap (A vs B) | 0.9317 | +0.015 vs baseline |
| Hard Belnap (3 domains) | 0.9302 | majority vote |
| **Soft Belnap (3 domains)** | **0.9349** | **best, +0.018** |

**F0.5 is the actual Vesuvius Challenge metric** (precision > recall).
Soft Belnap consistently outperforms mean ensemble on this metric.

Additional metrics (Soft Belnap):
- Precision: significantly higher than ensemble (BOTH class absorbs
  false positives instead of routing them to INK)
- Entropy mean: 0.159 bits (model is confident in most regions)
- High entropy zones (H > 1.5 bit): only **0.4% of pixels**
  — uncertainty is tightly localized

**SUBIT-INK Score** = Precision(INK) × ConflictLocalization(BOTH):
- Hard 3-domain: 0.4006
- Soft 3-domain: 0.3546

The conflict map traces letter contours precisely — BOTH is not
randomly distributed but concentrated exactly where domain signals
diverge (ink boundaries, low-contrast regions, papyrus fibers).

---

### Why this matters for cross-scroll generalization

The main unsolved problem in Vesuvius Challenge is domain adaptation:
models trained on Fragment 1 fail on Scrolls 2, 3, 4.

Our hypothesis: **hallucinations arise precisely in regions where
different depth layers disagree**. BOTH = semantic instability region.

When applied across scrolls instead of across depth layers:

```
Model A: trained on Fragment 1  →  prob_A
Model B: trained on new scroll  →  prob_B

Belnap(prob_A, prob_B):
  agreement  →  INK   (stable across domains)
  conflict   →  BOTH  (domain shift artifact)
```

This routes domain-shift artifacts to BOTH instead of letting them
appear as confident INK predictions — directly addressing the
hallucination problem.

---

### Belnap Entropy as instability detector

We define:

```
H_B = -Σ p(s) log₂ p(s)   for s ∈ {INK, BOTH, VOID, UNKNOWN}
```

High H_B regions = zones where the combiner is maximally uncertain.
These correspond to real physical ambiguities in the scroll, not
model noise. They can be used to:

1. Prioritize human annotation effort
2. Guide active learning sample selection
3. Flag regions needing higher-resolution rescanning

---

### SUBIT-INK Score (formal quality metric)

```
SUBIT-INK = Precision(INK) × ConflictLocalization(BOTH)
```

where ConflictLocalization = fraction of BOTH pixels that fall on
real letter boundaries (measured via ground truth dilation).

If BOTH is randomly distributed → low score → bad combiner.
If BOTH traces letter boundaries → high score → good combiner.

This gives a single number that captures both classification quality
and uncertainty calibration.

---

### What we still haven't done (honest limitations)

1. Cross-scroll test: all results are on Fragment 1.
   The real test is Model_A (Fragment 1) vs Model_B (Scroll 2).

2. The three "domains" are depth offsets from the same scan —
   this simulates domain shift but is not the real thing.

3. The soft Belnap combiner needs tuning of the evidence threshold
   (currently 0.4) per fragment.

4. No comparison against TimeSformer or other strong baselines.

---

### Code

All code is open source. Reproducible on Kaggle (GPU T4, ~45 min).

Vesuvius Challenge dataset:
`/kaggle/input/vesuvius-challenge-ink-detection/`

Key files:
- `subit_belnap_v2.py` — full pipeline (8 cells)
- `subit_core.py` — Belnap logic primitives
