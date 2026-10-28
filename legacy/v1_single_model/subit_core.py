"""
SUBIT-INK: Чотиризначна логіка Белнапа для детекції чорнила
============================================================
Базова алгебра: білаттис {T, F, B, N} з теорії SUBIT-TOPOS

Стани:
  INK     (T, yang=1 yin=0): впевнено чорнило
  BOTH    (B, yang=1 yin=1): суперечливий сигнал / межа букви
  VOID    (F, yang=0 yin=1): впевнено не чорнило
  UNKNOWN (N, yang=0 yin=0): відсутній сигнал / пошкодження
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


# ─── Білаттис Белнапа ────────────────────────────────────────────────────────

class BelnapState(IntEnum):
    UNKNOWN = 0   # N: yang=0, yin=0  — немає даних
    VOID    = 1   # F: yang=0, yin=1  — впевнено не чорнило
    INK     = 2   # T: yang=1, yin=0  — впевнено чорнило
    BOTH    = 3   # B: yang=1, yin=1  — суперечливий сигнал


BELNAP_COLORS = {
    BelnapState.INK:     (0,   200, 255),   # блакитний — чорнило
    BelnapState.BOTH:    (255, 165, 0),     # помаранчевий — перевір мене
    BelnapState.VOID:    (20,  20,  20),    # темний — пустота
    BelnapState.UNKNOWN: (128, 0,   128),   # фіолетовий — невідомо
}

BELNAP_LABELS = {
    BelnapState.INK:     "INK (T)",
    BelnapState.BOTH:    "BOTH (B) — перевір",
    BelnapState.VOID:    "VOID (F)",
    BelnapState.UNKNOWN: "UNKNOWN (N)",
}


def decode_belnap(yang: torch.Tensor, yin: torch.Tensor,
                  threshold: float = 0.5) -> torch.Tensor:
    """
    Перетворює два бінарних логіти в стан Белнапа.
    yang, yin: float тензори [0,1] після sigmoid
    Повертає: тензор int з BelnapState
    """
    y = (yang > threshold).long()
    n = (yin  > threshold).long()
    # Кодування: yang-bit * 2 + yin-bit
    # 0b10 = 2 = INK, 0b11 = 3 = BOTH, 0b01 = 1 = VOID, 0b00 = 0 = UNKNOWN
    return y * 2 + n


def binary_to_belnap_labels(binary_mask: np.ndarray,
                             dilation_radius: int = 3) -> tuple:
    """
    Перетворює бінарну маску (0/1) у пару yang/yin міток для навчання.

    Правила:
      INK-піксель (1)    → yang=1, yin=0
      VOID-піксель (0)   → yang=0, yin=1
      Межа (dilation)    → yang=1, yin=1  (BOTH — зона невизначеності)
      Порожнеча (маска)  → yang=0, yin=0  (UNKNOWN — немає даних)

    dilation_radius: ширина зони BOTH навколо букв (пікселі)
    """
    from scipy.ndimage import binary_dilation
    import numpy as np

    ink = binary_mask.astype(bool)

    # Зона навколо букв — природна зона BOTH
    border = binary_dilation(ink, iterations=dilation_radius) & ~ink

    yang = np.zeros_like(binary_mask, dtype=np.float32)
    yin  = np.zeros_like(binary_mask, dtype=np.float32)

    # INK: yang=1, yin=0
    yang[ink] = 1.0
    yin[ink]  = 0.0

    # BOTH (межа): yang=1, yin=1
    yang[border] = 1.0
    yin[border]  = 1.0

    # VOID (все інше, де є дані): yang=0, yin=1
    no_data = ~ink & ~border
    yang[no_data] = 0.0
    yin[no_data]  = 1.0

    return yang, yin


# ─── Функції втрат ───────────────────────────────────────────────────────────

class BelnapLoss(nn.Module):
    """
    Двокомпонентна втрата для навчання класифікатора Белнапа.
    Незалежне навчання yang і yin вимірів.
    """
    def __init__(self, yang_weight: float = 1.0, yin_weight: float = 1.0,
                 pos_weight_yang: float = 2.0):
        super().__init__()
        self.yang_w = yang_weight
        self.yin_w  = yin_weight
        # Позитивний клас частіше рідкісний — компенсуємо
        self.bce_yang = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_yang])
        )
        self.bce_yin = nn.BCEWithLogitsLoss()

    def forward(self, logits_yang: torch.Tensor, logits_yin: torch.Tensor,
                target_yang: torch.Tensor, target_yin: torch.Tensor):
        loss_yang = self.bce_yang(logits_yang, target_yang)
        loss_yin  = self.bce_yin(logits_yin,  target_yin)
        return self.yang_w * loss_yang + self.yin_w * loss_yin, {
            'loss_yang': loss_yang.item(),
            'loss_yin':  loss_yin.item(),
        }


# ─── Метрики ─────────────────────────────────────────────────────────────────

@dataclass
class BelnapMetrics:
    """Метрики для чотиризначного класифікатора."""
    # Порівняння з бінарним baseline: INK vs (VOID+UNKNOWN+BOTH)
    precision_ink: float
    recall_ink: float
    f1_ink: float

    # Розподіл станів
    pct_ink: float
    pct_both: float
    pct_void: float
    pct_unknown: float

    # Якість "галюцинаційного фільтру": BOTH відловлює межі
    border_in_both: float  # % пікселів меж букв, що потрапили в BOTH

    def __str__(self):
        return (
            f"\n{'='*50}\n"
            f"  SUBIT-INK Метрики\n"
            f"{'='*50}\n"
            f"  F1  (INK vs решта):  {self.f1_ink:.4f}\n"
            f"  Precision:           {self.precision_ink:.4f}\n"
            f"  Recall:              {self.recall_ink:.4f}\n"
            f"{'─'*50}\n"
            f"  Розподіл станів:\n"
            f"    INK     (T): {self.pct_ink:.1f}%\n"
            f"    BOTH    (B): {self.pct_both:.1f}%   ← галюцинаційний фільтр\n"
            f"    VOID    (F): {self.pct_void:.1f}%\n"
            f"    UNKNOWN (N): {self.pct_unknown:.1f}%\n"
            f"{'─'*50}\n"
            f"  Межі букв у BOTH:    {self.border_in_both:.1f}%\n"
            f"{'='*50}\n"
        )


def compute_belnap_metrics(pred_states: np.ndarray,
                            gt_binary: np.ndarray,
                            dilation_radius: int = 3) -> BelnapMetrics:
    """
    Обчислює метрики порівнянно з бінарним GT.
    pred_states: масив BelnapState
    gt_binary: оригінальна бінарна маска (0/1)
    """
    from scipy.ndimage import binary_dilation

    total = pred_states.size

    # INK-предикція: лише стан INK (не BOTH)
    pred_ink  = (pred_states == BelnapState.INK)
    true_ink  = gt_binary.astype(bool)
    border    = binary_dilation(true_ink, iterations=dilation_radius) & ~true_ink

    tp = (pred_ink &  true_ink).sum()
    fp = (pred_ink & ~true_ink).sum()
    fn = (~pred_ink & true_ink).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    # Розподіл
    pct = lambda s: 100.0 * (pred_states == s).sum() / total

    # Частка меж, що потрапила в BOTH
    both_pixels  = (pred_states == BelnapState.BOTH)
    border_total = border.sum()
    border_in_both = 100.0 * (both_pixels & border).sum() / (border_total + 1e-8)

    return BelnapMetrics(
        precision_ink=float(precision),
        recall_ink=float(recall),
        f1_ink=float(f1),
        pct_ink=pct(BelnapState.INK),
        pct_both=pct(BelnapState.BOTH),
        pct_void=pct(BelnapState.VOID),
        pct_unknown=pct(BelnapState.UNKNOWN),
        border_in_both=float(border_in_both),
    )
