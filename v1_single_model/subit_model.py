"""
SUBIT-INK Model
===============
3D CNN з двоголовим виходом (yang / yin) замість стандартного бінарного.

Архітектура: lightweight 3D U-Net з SUBIT-головою.
Сумісна з форматом Vesuvius Challenge surface volumes
(65 шарів × H × W, dtype=uint16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from subit_core import decode_belnap, BelnapState


# ─── Базові блоки ────────────────────────────────────────────────────────────

class Conv3dBN(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class DoubleConv3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            Conv3dBN(in_ch, out_ch),
            Conv3dBN(out_ch, out_ch),
        )
    def forward(self, x): return self.block(x)


class Down3d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool3d((1, 2, 2))  # пулінг лише по H, W (не по шарах)
        self.conv = DoubleConv3d(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))


class Up3d(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear',
                                  align_corners=False)
        self.conv = DoubleConv3d(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── SUBIT-голова ────────────────────────────────────────────────────────────

class SUBITHead(nn.Module):
    """
    Двоголовий вихід замість стандартного бінарного.
    Повертає незалежні логіти yang і yin.

    Yang-вимір: "чи є позитивний сигнал чорнила?"
    Yin-вимір:  "чи є негативний сигнал (впевнена пустота)?"
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.yang_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.yin_head  = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W] — після collapse по шарах
        return self.yang_head(x), self.yin_head(x)


# ─── Повна модель ────────────────────────────────────────────────────────────

class SUBITInkDetector(nn.Module):
    """
    Lightweight 3D U-Net із SUBIT-головою.

    Input:  [B, 1, D, H, W]  — D=65 шарів surface volume
    Output: yang [B, 1, H, W], yin [B, 1, H, W]  — логіти

    Порівняй зі стандартною моделлю:
      Standard: [B, 1, H, W]  (один логіт, BCE loss)
      SUBIT:    [B, 2, H, W]  (два логіти, BelnapLoss)

    Зміна в коді: лише заміна останнього шару + loss function.
    """
    def __init__(self, in_depth: int = 65, base_ch: int = 16):
        super().__init__()

        # ── Енкодер ──
        self.enc1 = DoubleConv3d(1, base_ch)        # [B, 16, D, H, W]
        self.enc2 = Down3d(base_ch, base_ch * 2)    # [B, 32, D, H/2, W/2]
        self.enc3 = Down3d(base_ch * 2, base_ch * 4) # [B, 64, D, H/4, W/4]

        # ── Bottleneck ──
        self.bottleneck = DoubleConv3d(base_ch * 4, base_ch * 8)

        # ── Декодер ──
        self.dec3 = Up3d(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = Up3d(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = Up3d(base_ch * 2, base_ch,     base_ch)

        # ── Collapse по глибині (шарах) → 2D ──
        # Замість MaxPool — увагова агрегація по шарах
        self.depth_attn = nn.Sequential(
            nn.Conv3d(base_ch, 1, kernel_size=1),
            nn.Softmax(dim=2),   # softmax по D-вимірю
        )

        # ── SUBIT-голова (замість стандартного Conv2d(ch, 1)) ──
        self.subit_head = SUBITHead(base_ch)

    def collapse_depth(self, x: torch.Tensor) -> torch.Tensor:
        """
        Агрегує 3D тензор [B, C, D, H, W] → 2D [B, C, H, W]
        за допомогою увагових ваг по шарах.
        Це краще за MaxPool, бо враховує, що чорнило може бути
        на різній глибині в різних ділянках.
        """
        attn = self.depth_attn(x)    # [B, 1, D, H, W]
        x = (x * attn).sum(dim=2)    # [B, C, H, W]
        return x

    def forward(self, x: torch.Tensor):
        """
        x: [B, 1, D, H, W]
        Повертає: (logit_yang, logit_yin), обидва [B, 1, H, W]
        """
        # Енкодер
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(Down3d(e3.shape[1], e3.shape[1]).to(x.device)(e3)
                             if False else e3)  # без додаткового down

        # Декодер із skip connections
        d3 = self.dec3(b,  e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Collapse: 3D → 2D
        out_2d = self.collapse_depth(d1)

        # SUBIT dual head
        yang_logit, yin_logit = self.subit_head(out_2d)

        return yang_logit, yin_logit

    @torch.no_grad()
    def predict_belnap(self, x: torch.Tensor,
                       threshold: float = 0.5) -> torch.Tensor:
        """
        Повертає карту станів Белнапа [B, H, W] типу int.
        Використовується для inference та візуалізації.
        """
        yang_logit, yin_logit = self.forward(x)
        yang_prob = torch.sigmoid(yang_logit).squeeze(1)
        yin_prob  = torch.sigmoid(yin_logit).squeeze(1)
        return decode_belnap(yang_prob, yin_prob, threshold)


# ─── Стандартна бінарна модель (для порівняння) ──────────────────────────────

class BinaryInkDetector(nn.Module):
    """
    Ідентична архітектура, але зі стандартним бінарним виходом.
    Використовується як baseline для порівняння з SUBIT.
    """
    def __init__(self, base_ch: int = 16):
        super().__init__()
        self.enc1 = DoubleConv3d(1, base_ch)
        self.enc2 = Down3d(base_ch, base_ch * 2)
        self.enc3 = Down3d(base_ch * 2, base_ch * 4)
        self.bottleneck = DoubleConv3d(base_ch * 4, base_ch * 8)
        self.dec3 = Up3d(base_ch * 8, base_ch * 4, base_ch * 4)
        self.dec2 = Up3d(base_ch * 4, base_ch * 2, base_ch * 2)
        self.dec1 = Up3d(base_ch * 2, base_ch, base_ch)
        self.depth_attn = nn.Sequential(
            nn.Conv3d(base_ch, 1, kernel_size=1),
            nn.Softmax(dim=2),
        )
        # Стандартний бінарний вихід — ЄДИНА відмінність від SUBITInkDetector
        self.binary_head = nn.Conv2d(base_ch, 1, kernel_size=1)

    def collapse_depth(self, x):
        attn = self.depth_attn(x)
        return (x * attn).sum(dim=2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = e3
        d3 = self.dec3(b, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        out_2d = self.collapse_depth(d1)
        return self.binary_head(out_2d)


# ─── Статистика параметрів ───────────────────────────────────────────────────

def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Всього: {total:,} | Тренованих: {trainable:,}"


if __name__ == "__main__":
    print("─" * 50)
    print("SUBIT-INK Model — перевірка архітектури")
    print("─" * 50)

    B, D, H, W = 2, 16, 64, 64  # зменшено для CPU-тесту
    x = torch.randn(B, 1, D, H, W)

    # SUBIT модель
    subit = SUBITInkDetector(base_ch=16)
    yang, yin = subit(x)
    print(f"\nSUBIT-INK:")
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Yang:   {tuple(yang.shape)}")
    print(f"  Yin:    {tuple(yin.shape)}")
    print(f"  Params: {count_params(subit)}")

    # Decode to Belnap states
    states = subit.predict_belnap(x)
    unique, counts = torch.unique(states, return_counts=True)
    print(f"\n  Розподіл станів (random init):")
    for s, c in zip(unique.tolist(), counts.tolist()):
        name = BelnapState(s).name
        print(f"    {name}: {c} пікселів")

    print("\n✓ Архітектура валідна")
