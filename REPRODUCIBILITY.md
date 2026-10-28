# Reproducibility Guide — SUBIT-Ink v26.0

This document provides exact steps to reproduce the training and submission artifacts of **SUBIT-Ink v26.0** for the [Vesuvius Challenge — Ink Detection](https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection).

---

## 1. Environment & Hardware Specifications

- **Operating System:** Ubuntu 20.04 / Linux x86_64 or Windows 10/11
- **GPU:** NVIDIA Tesla T4 (16 GB VRAM) or RTX 3080/3090/4090
- **CUDA:** 11.8 or 12.1+
- **Python:** $\ge$ 3.10
- **Total Pipeline Runtime:** $\approx 415 - 425$ minutes on a single Tesla T4

---

## 2. Dependencies Installation

To install all dependencies locally:

```bash
git clone https://github.com/sciganec/subit-ink.git
cd subit-ink
pip install -r requirements.txt
```

---

## 3. Kaggle Run (Recommended & Verified)

1. Open [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook**.
3. Under **Notebook Options** (right sidebar):
   - **Accelerator:** `GPU T4 x1` (or `GPU P100`)
   - **Language:** `Python`
   - **Persistence:** `No persistence`
   - **Environment:** `Always use latest environment`
   - **Internet:** `Off` (compatible with competition rules)
4. Under **Data**:
   - Click **+ Add Data**
   - Search for competition dataset: `vesuvius-challenge-ink-detection`
   - Attach it so the path is `/kaggle/input/competitions/vesuvius-challenge-ink-detection` (or `/kaggle/input/vesuvius-challenge-ink-detection`).
5. In the notebook editor:
   - Copy the entire code from [`ink_detection_kaggle_v26.py`](ink_detection_kaggle_v26.py).
   - Paste it into the first cell.
6. Click **Save Version** $\to$ **Save & Run All (Commit)**.
7. Wait for the run to complete (~7 hours). Upon completion, `submission.csv` will be generated in `/kaggle/working/submission.csv` and ready for submission.

---

## 4. Local Execution

If running on a local workstation with downloaded competition data:

```bash
# Organize your data directory as:
# ./data/train/1, ./data/train/2, ./data/train/3
# ./data/test/a, ./data/test/b

python ink_detection_kaggle_v26.py
```

The script automatically auto-detects standard candidate paths:
- `/kaggle/input/competitions/vesuvius-challenge-ink-detection`
- `/kaggle/input/vesuvius-challenge-ink-detection`
- `C:/Users/sciga/vesuvius`
- `./`

---

## 5. Key Deterministic Settings

The pipeline fixes random seeds across Python, NumPy, and PyTorch:
```python
SEED = 2026
```
Due to GPU non-deterministic floating-point atomic additions in `torch.nn.functional.interpolate` and cuDNN benchmarking, slight floating-point differences ($\pm 0.001$) across different GPU architectures (e.g., Turing vs Ampere vs Ada Lovelace) are normal and expected.

---

## 6. Output Artifacts

- `submission.csv`: Two columns (`Id`, `Predicted`), where `Predicted` contains Run-Length Encoded (RLE) bitmasks for test fragments `a` and `b`.
- Live logs with timestamping `[HH:MM:SS]` tracking loss, batch progress, validation $F_{0.5}$, Precision, Recall, and calibrated optimal threshold.
