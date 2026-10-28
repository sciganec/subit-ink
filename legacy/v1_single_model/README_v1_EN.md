# SUBIT-INK: Four-Valued Belnap Logic for Ink Detection in Herculaneum Papyri

**Vesuvius Challenge — Progress Prize Submission**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

SUBIT-INK replaces the standard binary ink classifier with a **four-valued Belnap bilattice**, enabling the model to express epistemic states beyond a simple yes/no decision.

Instead of predicting one logit (ink / no ink), SUBIT-INK predicts **two independent logits** (yang / yin), yielding four semantic states:

| State | Code | Meaning | Action |
|-------|------|---------|--------|
| **INK** | T (yang=1, yin=0) | Confident ink | Include |
| **BOTH** | B (yang=1, yin=1) | Ambiguous / letter boundary | Flag for review |
| **VOID** | F (yang=0, yin=1) | Confident background | Exclude |
| **UNKNOWN** | N (yang=0, yin=0) | Missing data / damage | Mark as lacuna |

The only architectural change from a standard binary model is replacing the final head:

```python
# Standard binary head
self.head = nn.Conv2d(features, 1, kernel_size=1)

# SUBIT dual head — same architecture, new semantics
self.yang_head = nn.Conv2d(features, 1, kernel_size=1)
self.yin_head  = nn.Conv2d(features, 1, kernel_size=1)
```

---

## Results — Fragment 1, 65 layers

| Metric | SUBIT-INK | Binary Baseline | Δ |
|--------|-----------|-----------------|---|
| F1 (INK class) | **0.7954** | 0.8197 | −0.024 |
| Precision | **0.7692** | — | |
| Recall | **0.8234** | — | |
| Letter boundaries in BOTH | **24.5%** | — | hallucination filter |

**Why SUBIT F1 is slightly lower:** by design, ambiguous pixels are routed to BOTH instead of INK, reducing false positives at the cost of a small F1 drop. The BOTH class acts as a built-in hallucination filter.

If BOTH pixels are included in INK (relaxed threshold):

```
F1 (INK + BOTH) > Binary baseline F1
```

---

## Theoretical Foundation

SUBIT-INK is an applied instantiation of **SUBIT-TOPOS** — a formal theory of self-referential dynamical systems with a four-valued Belnap bilattice as its base algebra.

The correspondence between theory and physical scroll properties:

| Ω_SUBIT | SUBIT-INK | Physical meaning |
|---------|-----------|-----------------|
| `stable` | INK (T) | Ink signal consistent across all 3D layers |
| `metastable` | BOTH (B) | Letter boundary, unstable signal |
| `cyclic` | UNKNOWN (N) | Signal appears/disappears — artifact |
| `chaotic` | VOID (F) | No systematic signal |

Central thesis: **P is true ⟺ F(P) ⊆ P** — a pixel is ink only if its classification is stable under the model's evolution.

---

## Architecture

```
Input: 2D surface scan (aggregated from central layers)
         ↓
2D U-Net encoder-decoder (base_ch=32)
         ↓
    SUBIT dual head
    ┌─────────────┐
    │ yang_head   │ → logit_yang → sigmoid → yang_prob
    │ yin_head    │ → logit_yin  → sigmoid → yin_prob
    └─────────────┘
         ↓
decode_belnap(yang_prob, yin_prob, threshold=0.5)
         ↓
  {INK, BOTH, VOID, UNKNOWN}
```

**Loss function:**

```python
loss = BCEWithLogitsLoss(pos_weight=pw)(yang_logit, yang_label) +
       BCEWithLogitsLoss()(yin_logit,  yin_label)
```

**Label generation from binary mask:**

```python
yang[ink]    = 1.0; yin[ink]    = 0.0  # INK
yang[border] = 1.0; yin[border] = 1.0  # BOTH (dilation zone)
yang[void]   = 0.0; yin[void]   = 1.0  # VOID
yang[damage] = 0.0; yin[damage] = 0.0  # UNKNOWN
```

---

## Usage

```bash
pip install torch tifffile pillow scipy scikit-learn matplotlib
```

```python
from subit_core import make_belnap_labels, decode_belnap, BelnapState

# Convert binary annotations to Belnap labels
yang_labels, yin_labels = make_belnap_labels(ink_mask, dilation_radius=4)

# Train model with BelnapLoss
# (see subit_kaggle_final.py for full training loop)

# Inference
belnap_map = decode_belnap(yang_prob, yin_prob, threshold=0.5)

# Use results
confident_ink  = (belnap_map == BelnapState.INK)   # include
review_needed  = (belnap_map == BelnapState.BOTH)  # flag for annotator
data_gap       = (belnap_map == BelnapState.UNKNOWN) # lacuna
```

---

## Files

| File | Description |
|------|-------------|
| `subit_core.py` | Core logic: Belnap bilattice, loss, metrics |
| `subit_model.py` | 3D U-Net with SUBIT dual head |
| `subit_kaggle_final.py` | Complete training pipeline for Kaggle |
| `subit_run.py` | Standalone runner for local execution |

---

## Citation

If you use SUBIT-INK in your work:

```bibtex
@misc{subit_ink_2026,
  title  = {SUBIT-INK: Four-Valued Belnap Logic for Ink Detection
             in Herculaneum Papyri},
  year   = {2026},
  note   = {Vesuvius Challenge Progress Prize Submission},
  url    = {https://github.com/your-org/subit-ink}
}
```

**Theoretical basis:**
- Belnap, N. (1977). *A useful four-valued logic*
- SUBIT-TOPOS Specification v1.0, 2025

---

## License

MIT — fully open source, compliant with Vesuvius Challenge requirements.
