#!/usr/bin/env python3
"""
SUBIT-INK: Запуск на реальних даних Vesuvius Challenge
=======================================================

Використання:
    python subit_run.py --fragment_dir ./train/1/surface_volume/ --inklabels ./train/1/inklabels.png

Опціонально:
    --output    папка для результатів (default: ./subit_results)
    --layers    кількість шарів для завантаження (default: 65, можна 20 для швидкості)
    --epochs    кількість епох навчання (default: 30)
    --patch     розмір патча (default: 64)
    --device    cpu або cuda (default: авто)

Приклад швидкого тесту (5 хв на CPU):
    python subit_run.py --fragment_dir ./train/1/surface_volume/ --inklabels ./train/1/inklabels.png --layers 10 --epochs 15
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ─── Перевірка залежностей ────────────────────────────────────────────────────

def check_deps():
    missing = []
    for pkg, imp in [('tifffile','tifffile'), ('PIL','PIL'), ('scipy','scipy'),
                     ('sklearn','sklearn'), ('matplotlib','matplotlib')]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"Встанови залежності: pip install {' '.join(missing)}")
        sys.exit(1)

check_deps()

import tifffile
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Білаттис Белнапа ────────────────────────────────────────────────────────

BELNAP_COLORS = {
    2: (0,   200, 255),   # INK     — блакитний
    3: (255, 165,   0),   # BOTH    — помаранчевий
    1: ( 20,  20,  20),   # VOID    — темний
    0: (128,   0, 128),   # UNKNOWN — фіолетовий
}
BELNAP_LABELS = {
    2: "INK (T) — впевнено чорнило",
    3: "BOTH (B) — межа / перевір",
    1: "VOID (F) — не чорнило",
    0: "UNKNOWN (N) — немає даних",
}

def decode_belnap(yang: np.ndarray, yin: np.ndarray, thr=0.5) -> np.ndarray:
    return ((yang > thr).astype(int) * 2 + (yin > thr).astype(int)).astype(np.int32)

def make_belnap_labels(ink: np.ndarray, dilation_radius=4):
    """Бінарна маска → мітки yang/yin."""
    ink_b = ink.astype(bool)
    border = binary_dilation(ink_b, iterations=dilation_radius) & ~ink_b
    yang = np.zeros_like(ink, dtype=np.float32)
    yin  = np.zeros_like(ink, dtype=np.float32)
    yang[ink_b]  = 1.0; yin[ink_b]   = 0.0   # INK
    yang[border] = 1.0; yin[border]  = 1.0   # BOTH
    yang[~ink_b & ~border] = 0.0              # VOID
    yin[~ink_b & ~border]  = 1.0
    return yang, yin

# ─── Датасет ─────────────────────────────────────────────────────────────────

class VesuviusDataset(Dataset):
    def __init__(self, volume, ink, yang, yin, patch_size, n_patches, mode='subit', v_min=None, v_max=None):
        self.volume = volume  # [D, H, W] (uint16)
        self.ink = ink; self.yang = yang; self.yin = yin
        self.ps = patch_size; self.hp = patch_size // 2
        self.mode = mode
        self.v_min = v_min if v_min is not None else float(volume.min())
        self.v_max = v_max if v_max is not None else float(volume.max())
        D, H, W = volume.shape
        # Стратифікований семплінг: 60% з ink-регіонів, 40% — фон
        rng = np.random.default_rng(42)
        iy, ix = np.where(ink > 0); vy, vx = np.where(ink == 0)
        n_ink = int(n_patches * 0.6); n_void = n_patches - n_ink
        hp = self.hp
        def safe_coords(y_arr, x_arr, n):
            idx = rng.integers(0, len(y_arr), n * 3)
            coords = [(y_arr[i], x_arr[i]) for i in idx
                      if hp <= y_arr[i] < H-hp and hp <= x_arr[i] < W-hp]
            return coords[:n]
        self.coords = safe_coords(iy, ix, n_ink) + safe_coords(vy, vx, n_void)

    def __len__(self): return len(self.coords)

    def __getitem__(self, idx):
        py, px = self.coords[idx]
        hp = self.hp
        vol_patch = self.volume[:, py-hp:py+hp, px-hp:px+hp].astype(np.float32)
        vol_patch = (vol_patch - self.v_min) / (self.v_max - self.v_min + 1e-8)
        t = torch.from_numpy(vol_patch).unsqueeze(0)  # [1, D, H, W]
        if self.mode == 'subit':
            yang = torch.from_numpy(self.yang[py-hp:py+hp, px-hp:px+hp]).unsqueeze(0)
            yin  = torch.from_numpy(self.yin[py-hp:py+hp,  px-hp:px+hp]).unsqueeze(0)
            return t, yang, yin
        else:
            ink = torch.from_numpy(self.ink[py-hp:py+hp, px-hp:px+hp]).unsqueeze(0)
            return t, ink

# ─── Модель: 3D U-Net з SUBIT dual head ──────────────────────────────────────

class DoubleConv3d(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv3d(i,o,3,padding=1,bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True),
            nn.Conv3d(o,o,3,padding=1,bias=False), nn.BatchNorm3d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)

class SUBITNet(nn.Module):
    """
    3D U-Net із SUBIT dual head.
    Input:  [B, 1, D, H, W]
    Output: yang [B,1,H,W], yin [B,1,H,W]  (логіти)
    """
    def __init__(self, base=16, dual=True):
        super().__init__()
        self.e1 = DoubleConv3d(1, base)
        self.e2 = nn.Sequential(nn.MaxPool3d((1,2,2)), DoubleConv3d(base, base*2))
        self.e3 = nn.Sequential(nn.MaxPool3d((1,2,2)), DoubleConv3d(base*2, base*4))
        self.bot = nn.Sequential(nn.MaxPool3d((1,2,2)), DoubleConv3d(base*4, base*8))
        self.u3 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        self.d3 = DoubleConv3d(base*8 + base*4, base*4)
        self.u2 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        self.d2 = DoubleConv3d(base*4 + base*2, base*2)
        self.u1 = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=False)
        self.d1 = DoubleConv3d(base*2 + base, base)
        # Depth collapse з увагою
        self.attn = nn.Sequential(nn.Conv3d(base,1,1), nn.Softmax(dim=2))
        self.dual = dual
        if dual:
            self.yang_head = nn.Conv2d(base, 1, 1)
            self.yin_head  = nn.Conv2d(base, 1, 1)
        else:
            self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b  = self.bot(e3)
        d3 = self.d3(torch.cat([self.u3(b),  e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        # Collapse D → 2D
        attn = self.attn(d1)          # [B,1,D,H,W]
        out  = (d1 * attn).sum(dim=2) # [B,C,H,W]
        if self.dual:
            return self.yang_head(out), self.yin_head(out)
        return self.head(out)

# ─── Тренування ──────────────────────────────────────────────────────────────

def train_model(model, dataset, epochs, device, dual=True, pos_weight=5.0):
    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                        num_workers=0, pin_memory=False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    pw = torch.tensor([pos_weight]).to(device)
    model.to(device).train()
    history = []
    t0 = time.time()
    for ep in range(epochs):
        total = 0; nb = 0
        for batch in loader:
            opt.zero_grad()
            if dual:
                vol, yang, yin = [b.float().to(device) for b in batch]
                yl, nl = model(vol)
                loss = (nn.BCEWithLogitsLoss(pos_weight=pw)(yl, yang) +
                        nn.BCEWithLogitsLoss()(nl, yin))
            else:
                vol, ink = [b.float().to(device) for b in batch]
                loss = nn.BCEWithLogitsLoss(pos_weight=pw)(model(vol), ink)
            loss.backward(); opt.step(); total += loss.item(); nb += 1
        scheduler.step()
        avg = total / max(nb, 1); history.append(avg)
        elapsed = time.time() - t0
        eta = elapsed / (ep+1) * (epochs - ep - 1)
        print(f"  Epoch {ep+1:3d}/{epochs}  loss={avg:.4f}  "
              f"пройшло={elapsed/60:.1f}хв  залишилось≈{eta/60:.1f}хв")
    return history

# ─── Інференс ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def full_inference(model, volume, patch_size, device, dual=True, step=None, v_min=None, v_max=None):
    """Sliding window inference на повному зображенні."""
    D, H, W = volume.shape
    hp = patch_size // 2
    step = step or patch_size // 2
    if v_min is None or v_max is None:
        v_min, v_max = float(volume.min()), float(volume.max())
    yang_map = np.zeros((H, W), dtype=np.float32)
    yin_map  = np.zeros((H, W), dtype=np.float32)
    cnt      = np.zeros((H, W), dtype=np.float32)
    model.eval()
    total_tiles = len(range(hp, H-hp, step)) * len(range(hp, W-hp, step))
    done = 0
    for py in range(hp, H-hp, step):
        for px in range(hp, W-hp, step):
            vp = volume[:, py-hp:py+hp, px-hp:px+hp]
            if vp.shape[1:] != (patch_size, patch_size): continue
            vp_f = vp.astype(np.float32)
            vp_f = (vp_f - v_min) / (v_max - v_min + 1e-8)
            t = torch.from_numpy(vp_f).unsqueeze(0).unsqueeze(0).to(device)
            if dual:
                yl, nl = model(t)
                yang_map[py-hp:py+hp, px-hp:px+hp] += torch.sigmoid(yl).squeeze().cpu().numpy()
                yin_map[py-hp:py+hp,  px-hp:px+hp] += torch.sigmoid(nl).squeeze().cpu().numpy()
            else:
                yang_map[py-hp:py+hp, px-hp:px+hp] += torch.sigmoid(model(t)).squeeze().cpu().numpy()
            cnt[py-hp:py+hp, px-hp:px+hp] += 1
            done += 1
            if done % 500 == 0:
                print(f"    Інференс: {done}/{total_tiles} тайлів ({done/total_tiles*100:.0f}%)")
    cnt = np.maximum(cnt, 1)
    if dual:
        return decode_belnap(yang_map / cnt, yin_map / cnt)
    return (yang_map / cnt > 0.5).astype(np.int32)

# ─── Метрики ─────────────────────────────────────────────────────────────────

def compute_metrics(belnap_pred, gt_binary, dilation_radius=4):
    ink_pred = (belnap_pred == 2)
    both_pred = (belnap_pred == 3)
    border = binary_dilation(gt_binary.astype(bool), iterations=dilation_radius) & ~gt_binary.astype(bool)
    total = gt_binary.size
    f1   = f1_score(gt_binary.flatten().astype(int), ink_pred.flatten().astype(int), zero_division=0)
    prec = precision_score(gt_binary.flatten().astype(int), ink_pred.flatten().astype(int), zero_division=0)
    rec  = recall_score(gt_binary.flatten().astype(int), ink_pred.flatten().astype(int), zero_division=0)
    border_in_both = 100.0 * (both_pred & border).sum() / (border.sum() + 1e-8)
    return {
        'f1': f1, 'precision': prec, 'recall': rec,
        'pct_ink':  100.0 * (belnap_pred == 2).sum() / total,
        'pct_both': 100.0 * (belnap_pred == 3).sum() / total,
        'pct_void': 100.0 * (belnap_pred == 1).sum() / total,
        'pct_unk':  100.0 * (belnap_pred == 0).sum() / total,
        'border_in_both': border_in_both,
    }

# ─── Візуалізація ────────────────────────────────────────────────────────────

def make_visualization(tif_mid, ink_gt, belnap_pred, binary_pred,
                        loss_subit, loss_binary, metrics_subit, f1_binary,
                        out_path):
    ink_only = (belnap_pred == 2).astype(float)
    rgb = np.zeros((*ink_gt.shape, 3), dtype=np.uint8)
    for s, c in BELNAP_COLORS.items():
        rgb[belnap_pred == s] = c

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    fig.patch.set_facecolor('#080808')
    for ax in axes.flat:
        ax.set_facecolor('#0d0d0d')
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a2a')

    def show(ax, img, title, cmap='gray'):
        ax.imshow(img, cmap=cmap, interpolation='nearest')
        ax.set_title(title, color='#ddd', fontsize=9, pad=5, fontweight='bold')
        ax.axis('off')

    show(axes[0,0], tif_mid, 'Реальний скан сувою\n(середній шар)', 'bone')
    show(axes[0,1], ink_gt,  'Ground Truth\n(офіційні анотації Vesuvius)', 'gray')

    axes[0,2].imshow(rgb, interpolation='nearest')
    axes[0,2].set_title('SUBIT-INK: 4 стани Белнапа', color='#ddd', fontsize=9, pad=5, fontweight='bold')
    axes[0,2].axis('off')
    patches = [mpatches.Patch(color=np.array(c)/255, label=BELNAP_LABELS[s])
               for s, c in BELNAP_COLORS.items()]
    axes[0,2].legend(handles=patches, loc='lower right', fontsize=7,
                     facecolor='#1a1a1a', edgecolor='#555', labelcolor='white')

    show(axes[0,3], ink_only, 'SUBIT: тільки INK (T)\n(впевнене чорнило)', 'Blues')
    show(axes[1,0], binary_pred, 'Binary Baseline\n(стандартний підхід)', 'gray')

    diff = binary_pred.astype(int) - ink_only.astype(int)
    axes[1,1].imshow(diff, cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
    axes[1,1].set_title('Різниця: Binary − SUBIT\nчервоний=лише Binary, синій=лише SUBIT',
                         color='#ddd', fontsize=8, pad=5)
    axes[1,1].axis('off')

    ax = axes[1,2]; ax.set_facecolor('#111')
    ax.plot(loss_subit,  color='#00c8ff', lw=2, label='SUBIT-INK (Belnap)')
    ax.plot(loss_binary, color='#ff6b35', lw=2, label='Binary (BCE)', linestyle='--')
    ax.grid(color='#2a2a2a', lw=0.5)
    ax.set_title('Криві навчання', color='#ddd', fontsize=9, fontweight='bold')
    ax.set_xlabel('Epoch', color='#888'); ax.set_ylabel('Loss', color='#888')
    ax.tick_params(colors='#666')
    ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444', labelcolor='white')

    ax = axes[1,3]; ax.axis('off')
    m = metrics_subit
    delta = m['f1'] - f1_binary
    sign = '+' if delta >= 0 else ''
    txt = (f"  РЕЗУЛЬТАТИ — Vesuvius Challenge\n"
           f"  {'─'*32}\n"
           f"  SUBIT F1:       {m['f1']:.4f}\n"
           f"  Precision:      {m['precision']:.4f}\n"
           f"  Recall:         {m['recall']:.4f}\n\n"
           f"  Binary F1:      {f1_binary:.4f}\n"
           f"  Δ F1:           {sign}{delta:.4f}\n\n"
           f"  Розподіл станів:\n"
           f"   INK  (T): {m['pct_ink']:.1f}%\n"
           f"   BOTH (B): {m['pct_both']:.1f}%\n"
           f"   VOID (F): {m['pct_void']:.1f}%\n"
           f"   UNK  (N): {m['pct_unk']:.1f}%\n\n"
           f"  Межі букв у BOTH:\n"
           f"  {m['border_in_both']:.1f}% ← галюц. фільтр")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, color='#e0e0e0',
            fontsize=9, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#151515',
                      edgecolor='#00c8ff', lw=1.5))

    plt.suptitle('SUBIT-INK × Vesuvius Challenge — Четиризначна логіка Белнапа',
                 color='white', fontsize=13, fontweight='bold', y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#080808')
    plt.close()
    print(f"  ✓ Візуалізація: {out_path}")

# ─── Головна функція ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='SUBIT-INK: Vesuvius Challenge')
    parser.add_argument('--fragment_dir', required=True, help='Папка з .tif шарами')
    parser.add_argument('--inklabels',    required=True, help='Файл inklabels.png')
    parser.add_argument('--output',   default='./subit_results', help='Папка для результатів')
    parser.add_argument('--layers',   type=int, default=65,  help='Кількість шарів (default: 65)')
    parser.add_argument('--epochs',   type=int, default=30,  help='Епох навчання (default: 30)')
    parser.add_argument('--patch',    type=int, default=64,  help='Розмір патча (default: 64)')
    parser.add_argument('--patches',  type=int, default=2000, help='Кількість патчів (default: 2000)')
    parser.add_argument('--base_ch',  type=int, default=16,  help='Базові канали моделі (default: 16)')
    parser.add_argument('--device',   default='auto', help='cpu / cuda / auto')
    args = parser.parse_args()

    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"\n{'='*55}")
    print(f"  SUBIT-INK — Vesuvius Challenge")
    print(f"  Device: {device.upper()}")
    print(f"{'='*55}\n")

    # ── 1. Завантаження даних ──
    print("[1/5] Завантаження даних...")
    frag_dir = Path(args.fragment_dir)
    tif_files = sorted(frag_dir.glob('*.tif'))[:args.layers]
    if not tif_files:
        print(f"Помилка: не знайдено .tif файлів у {frag_dir}")
        sys.exit(1)
    print(f"  Шарів: {len(tif_files)} з {frag_dir}")

    layers = []
    for i, f in enumerate(tif_files):
        layer = tifffile.imread(str(f))
        layers.append(layer)
        if (i+1) % 10 == 0: print(f"  Завантажено {i+1}/{len(tif_files)} шарів...")
    volume = np.stack(layers)  # [D, H, W]

    # Нормалізація
    v_min, v_max = float(volume.min()), float(volume.max())
    print(f"  Volume (uint16): {volume.shape}, min={v_min}, max={v_max}")

    # Маска чорнила
    ink_full = np.array(Image.open(args.inklabels).convert('L'))
    ink_full = (ink_full > 128).astype(np.float32)

    # Вирівнюємо розміри
    D, H_vol, W_vol = volume.shape
    H_ink, W_ink = ink_full.shape
    H = min(H_vol, H_ink); W = min(W_vol, W_ink)
    volume   = volume[:, :H, :W]
    ink_full = ink_full[:H, :W]
    print(f"  Ink mask: {ink_full.shape}, покриття={ink_full.mean()*100:.1f}%")

    # ── 2. Мітки Белнапа ──
    print("\n[2/5] Підготовка міток Белнапа...")
    yang, yin = make_belnap_labels(ink_full, dilation_radius=4)
    ratio = (1 - ink_full).sum() / (ink_full.sum() + 1e-8)
    print(f"  Positive weight: {ratio:.1f}")

    # ── 3. Датасети ──
    print("\n[3/5] Підготовка датасетів...")
    ds_subit  = VesuviusDataset(volume, ink_full, yang, yin,
                                 args.patch, args.patches, mode='subit', v_min=v_min, v_max=v_max)
    ds_binary = VesuviusDataset(volume, ink_full, yang, yin,
                                 args.patch, args.patches, mode='binary', v_min=v_min, v_max=v_max)
    print(f"  Патчів: {len(ds_subit)}")

    # ── 4. Тренування ──
    print("\n[4/5] Тренування...")
    print("  --- SUBIT-INK (Belnap Loss) ---")
    subit_model  = SUBITNet(base=args.base_ch, dual=True)
    loss_subit   = train_model(subit_model, ds_subit, args.epochs, device,
                                dual=True, pos_weight=ratio)

    print("  --- Binary Baseline (BCE Loss) ---")
    binary_model = SUBITNet(base=args.base_ch, dual=False)
    loss_binary  = train_model(binary_model, ds_binary, args.epochs, device,
                                dual=False, pos_weight=ratio)

    # ── 5. Інференс і метрики ──
    print("\n[5/5] Інференс на повному зображенні...")
    print("  SUBIT-INK...")
    belnap_pred = full_inference(subit_model, volume, args.patch, device,
                                  dual=True, step=args.patch // 2, v_min=v_min, v_max=v_max)
    print("  Binary baseline...")
    binary_pred = full_inference(binary_model, volume, args.patch, device,
                                  dual=False, step=args.patch // 2, v_min=v_min, v_max=v_max)

    # Метрики
    metrics = compute_metrics(belnap_pred, ink_full)
    f1b = f1_score(ink_full.flatten().astype(int),
                   binary_pred.flatten().astype(int), zero_division=0)

    # Друк результатів
    print(f"\n{'='*55}")
    print(f"  РЕЗУЛЬТАТИ")
    print(f"{'='*55}")
    print(f"  SUBIT-INK F1:   {metrics['f1']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  Binary F1:      {f1b:.4f}")
    delta = metrics['f1'] - f1b
    print(f"  Δ F1:           {'+' if delta>=0 else ''}{delta:.4f}")
    print(f"{'─'*55}")
    print(f"  Розподіл станів:")
    print(f"    INK  (T): {metrics['pct_ink']:.1f}%")
    print(f"    BOTH (B): {metrics['pct_both']:.1f}%   ← галюцинаційний фільтр")
    print(f"    VOID (F): {metrics['pct_void']:.1f}%")
    print(f"    UNK  (N): {metrics['pct_unk']:.1f}%")
    print(f"  Межі букв у BOTH: {metrics['border_in_both']:.1f}%")
    print(f"{'='*55}\n")

    # Збереження результатів
    np.save(out_dir / 'belnap_prediction.npy', belnap_pred)
    np.save(out_dir / 'binary_prediction.npy', binary_pred)
    print(f"  ✓ Передбачення збережено: {out_dir}/belnap_prediction.npy")

    # Візуалізація — середній шар
    mid = D // 2
    tif_mid_normalized = volume[mid].astype(np.float32)
    tif_mid_normalized = (tif_mid_normalized - v_min) / (v_max - v_min + 1e-8)
    make_visualization(
        tif_mid=tif_mid_normalized,
        ink_gt=ink_full,
        belnap_pred=belnap_pred,
        binary_pred=binary_pred,
        loss_subit=loss_subit,
        loss_binary=loss_binary,
        metrics_subit=metrics,
        f1_binary=f1b,
        out_path=str(out_dir / 'subit_ink_result.png')
    )

    print(f"\n✓ Все готово! Результати у папці: {out_dir}/")
    print(f"  subit_ink_result.png     — візуалізація")
    print(f"  belnap_prediction.npy    — карта Белнапа (0=UNK,1=VOID,2=INK,3=BOTH)")
    print(f"  binary_prediction.npy    — бінарний baseline")

if __name__ == '__main__':
    main()
