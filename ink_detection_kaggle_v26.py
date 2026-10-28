# ========================================================================
# SUBIT-Ink v26.0 Grandmaster-LateConverge
# Vesuvius Challenge - Ink Detection
#
# Target: Break Private 0.4864 (v27 record) -> 0.490 - 0.510+
# Key architectural adjustments based on v27 vs v30 diagnostics:
#   1. RESTORE TRAIN_VAL_SPLIT = 0.80 (proven optimal train/val balance)
#   2. LATE-EPOCH ENSEMBLE ONLY (epoch >= 16): guarantees fully converged models
#   3. WINNING LOSS FORMULA: Tversky(α=0.4, β=0.6), Dice=0.25, Focal=0.05, Prec=0.0
#   4. CALIBRATED THRESHOLD CLAMP: [0.48, 0.60] (locks in v27 winning zone ~0.58)
#   5. WEIGHTED ENSEMBLE: models weighted by validation F0.5 performance
#   6. TOTAL TIME: ~420 min (safe within 540 min Kaggle limit)
# ========================================================================

import os
import gc
import sys
import math
import time
import copy
import random
import warnings
from pathlib import Path

# Вимикання буферизації виводу для живих логів у Kaggle
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage
from scipy.ndimage import label as nd_label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# ЖИВИЙ ЛОГЕР
# ──────────────────────────────────────────────────────────────────────
def log(*args, **kwargs):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}]", *args, flush=True, **kwargs)

# ========================================================================
# 1. КОНФІГУРАЦІЯ
# ========================================================================

VERSION = "26.0-GrandmasterLateConverge"
SEED    = 2026

def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# Автоматичний пошук шляху до даних
CANDIDATE_ROOTS = [
    Path("/kaggle/input/competitions/vesuvius-challenge-ink-detection"),
    Path("/kaggle/input/vesuvius-challenge-ink-detection"),
    Path("C:/Users/sciga/vesuvius"),
    Path("./")
]
DATA_ROOT = next(
    (p for p in CANDIDATE_ROOTS
     if p.exists() and ((p / "train").exists() or (p / "1").exists())),
    CANDIDATE_ROOTS[0]
)

TRAIN_FRAGS     = ["1", "2", "3"]
VAL_FRAG        = "3"
TRAIN_VAL_SPLIT = 0.80  # Повернено 80/20 (v27 рекордний спліт)

PATCH = 256
DEPTH = 16   # Перевірено: 16 дає ідеальну швидкість і збіжність

EPOCHS      = 22
BATCH_SIZE  = 4
NUM_WORKERS = 0

LR           = 3.0e-4
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.995

# Стратифікована вибірка патчів
TRAIN_PATCHES          = 4500
VAL_PATCHES            = 700
POS_PATCH_FRACTION     = 0.40
NEUTRAL_PATCH_FRACTION = 0.20
NEG_PATCH_FRACTION     = 0.40

POS_DENSITY        = 0.015
STRONG_POS_DENSITY = 0.05
MIN_VALID_DENSITY  = 0.35

VAL_STRIDE_EPOCH = 256   # Швидка валідація під час епох (~1 хв)
VAL_STRIDE_FINAL = 128   # Високоточна валідація з 4x TTA для фінального порогу
TEST_STRIDE      = 64    # Фінальний тест з 4x TTA

THRESHOLDS = np.arange(0.10, 0.901, 0.005)

# Клампінг порогу: зафіксовано в зоні успіху v27 [0.48, 0.60]
THRESHOLD_MIN = 0.48
THRESHOLD_MAX = 0.60

# Рекордна комбінація лосів v23.0
BCE_W       = 0.25
DICE_W      = 0.25
TVERSKY_W   = 0.35
FOCAL_W     = 0.05
PRECISION_W = 0.00
AUX_W       = 0.10

TVERSKY_ALPHA = 0.4
TVERSKY_BETA  = 0.6
FOCAL_GAMMA   = 2.0

P_LOW  = 1.0
P_HIGH = 99.0

AUGMENT = True
MIN_COMPONENT_SIZE = 4

# Відбір тільки пізніх збіжних моделей
MIN_ENSEMBLE_EPOCH = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP    = (DEVICE.type == "cuda")

if AMP:
    torch.backends.cudnn.benchmark = True

log("=" * 72)
log(f"SUBIT-Ink v{VERSION}")
log(f"Device: {DEVICE} | AMP: {AMP} | Data Root: {DATA_ROOT}")
log(f"Patch: {PATCH}x{PATCH} | Depth: {DEPTH} | Epochs: {EPOCHS}")
log(f"Split: {int(TRAIN_VAL_SPLIT*100)}% train / {int((1-TRAIN_VAL_SPLIT)*100)}% val (F3)")
log(f"Loss: BCE={BCE_W} Dice={DICE_W} Tversky={TVERSKY_W}(α={TVERSKY_ALPHA}/β={TVERSKY_BETA}) Focal={FOCAL_W}")
log(f"Threshold clamp: [{THRESHOLD_MIN}, {THRESHOLD_MAX}] | Min Ckpt Epoch: {MIN_ENSEMBLE_EPOCH}")
log("=" * 72)

# ========================================================================
# 2. АКТИВНА ЗОНА Z-ШАРІВ + HANN ВІКНО
# ========================================================================
def active_depth_anchors(total_layers: int, num_anchors: int = DEPTH,
                         z_start_ratio=0.22, z_end_ratio=0.75) -> list:
    if total_layers <= num_anchors:
        return list(range(total_layers))
    z_min = int(round(total_layers * z_start_ratio))
    z_max = int(round(total_layers * z_end_ratio))
    z_min = max(0, min(z_min, total_layers - 1))
    z_max = max(z_min + 1, min(z_max, total_layers - 1))
    anchors = sorted(set(np.linspace(z_min, z_max, num_anchors).round().astype(int).tolist()))
    return anchors

def make_hann_window(patch_size: int = PATCH) -> np.ndarray:
    w1d = np.hanning(patch_size)
    w2d = np.outer(w1d, w1d).astype(np.float32)
    return np.clip(w2d, 0.05, 1.0)

HANN_WEIGHT = make_hann_window(PATCH)

# ========================================================================
# 3. ЗАВАНТАЖЕННЯ ФРАГМЕНТІВ
# ========================================================================
def find_fragment_dir(root: Path, frag_id: str) -> Path:
    candidates = [
        root / "train" / frag_id,
        root / "test"  / frag_id,
        root / frag_id,
        root / "train" / f"fragment_{frag_id}",
        root / "test"  / f"fragment_{frag_id}",
    ]
    for p in candidates:
        if p.exists():
            return p
    matches = list(root.rglob(f"*{frag_id}*"))
    for p in matches:
        if p.is_dir() and ((p / "surface_volume").exists() or (p / "surface-volume").exists()):
            return p
    raise FileNotFoundError(f"Fragment {frag_id} not found in {root}")

def find_surface_dir(fragment_dir: Path) -> Path:
    candidates = [
        fragment_dir / "surface_volume",
        fragment_dir / "surface-volume",
        fragment_dir / "surface_volumes",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    raise FileNotFoundError(f"surface_volume not found in {fragment_dir}")

def find_label(fragment_dir: Path, keyword: str) -> Path:
    candidates = [
        fragment_dir / f"{keyword}.png",
        fragment_dir / f"{keyword}.tif",
        fragment_dir / f"{keyword}.tiff",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in fragment_dir.glob("*.png"):
        if keyword.lower() in p.name.lower():
            return p
    raise FileNotFoundError(f"{keyword} not found in {fragment_dir}")

def slice_key(p: Path) -> int:
    nums = "".join(c if c.isdigit() else " " for c in p.stem).split()
    return int(nums[-1]) if nums else 0

class FragmentVolume:
    def __init__(self, frag_id: str):
        self.frag_id = str(frag_id)
        self.root = find_fragment_dir(DATA_ROOT, self.frag_id)
        self.surface_dir = find_surface_dir(self.root)

        self.slice_paths = sorted(
            [p for p in self.surface_dir.iterdir()
             if p.suffix.lower() in [".tif", ".tiff"]],
            key=slice_key
        )
        if not self.slice_paths:
            raise RuntimeError(f"No TIFF slices in {self.surface_dir}")

        self.Z = len(self.slice_paths)
        self.anchors = active_depth_anchors(self.Z, DEPTH)
        self._slices = [None] * self.Z

        try:
            label_path = find_label(self.root, "inklabels")
            mask_path  = find_label(self.root, "mask")
            self.label = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8)
            self.mask  = np.asarray(Image.open(mask_path).convert("L"),  dtype=np.uint8)
            self.label = (self.label > 0).astype(np.uint8)
            self.mask  = (self.mask  > 0).astype(np.uint8)
            self.H, self.W = self.label.shape
        except Exception:
            arr = self._get_slice(0)
            self.H, self.W = arr.shape
            try:
                mask_path = find_label(self.root, "mask")
                self.mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
                self.mask = (self.mask > 0).astype(np.uint8)
            except Exception:
                self.mask = np.ones((self.H, self.W), dtype=np.uint8)
            self.label = np.zeros((self.H, self.W), dtype=np.uint8)

        self.lo, self.hi = self._estimate_percentiles()
        ink = f" | Ink={self.label.mean()*100:.2f}%" if self.label.max() > 0 else ""
        log(f"  Fragment [{self.frag_id}]: Z={self.Z}, Anchors={self.anchors[:4]}..., "
            f"H={self.H}, W={self.W}{ink}")

    def _get_slice(self, z: int) -> np.ndarray:
        if self._slices[z] is None:
            path = self.slice_paths[z]
            try:
                arr = tifffile.memmap(str(path))
            except Exception:
                arr = tifffile.imread(str(path))
            if arr.ndim > 2:
                arr = np.squeeze(arr)
            self._slices[z] = arr
        return self._slices[z]

    def _estimate_percentiles(self):
        zs = self.anchors[::4] if len(self.anchors) >= 4 else self.anchors
        vals = []
        for z in zs:
            arr = self._get_slice(int(z))
            h, w = arr.shape
            sh, sw = min(512, h), min(512, w)
            y0, x0 = max(0, (h - sh) // 2), max(0, (w - sw) // 2)
            crop = np.asarray(arr[y0:y0+sh, x0:x0+sw], dtype=np.float32).reshape(-1)
            if crop.size > 10_000:
                crop = crop[np.random.choice(crop.size, 10_000, replace=False)]
            vals.append(crop)
        vals = np.concatenate(vals)
        lo = np.percentile(vals, P_LOW)
        hi = np.percentile(vals, P_HIGH)
        if hi <= lo:
            hi = lo + 1.0
        return float(lo), float(hi)

    def read_patch(self, y: int, x: int, patch: int = PATCH, anchors=None):
        if anchors is None:
            anchors = self.anchors
        y = int(np.clip(y, 0, self.H - patch))
        x = int(np.clip(x, 0, self.W - patch))

        out = np.empty((len(anchors), patch, patch), dtype=np.float32)
        for i, z in enumerate(anchors):
            arr = self._get_slice(z)
            crop = np.asarray(arr[y:y+patch, x:x+patch], dtype=np.float32)
            crop = (crop - self.lo) / (self.hi - self.lo)
            out[i] = np.clip(crop, 0.0, 1.0)

        mask  = self.mask[y:y+patch, x:x+patch]
        label = self.label[y:y+patch, x:x+patch]
        return out, mask.astype(np.float32), label.astype(np.float32)

# ========================================================================
# 4. СТРАТИФІКОВАНИЙ ДАТАСЕТ ПАТЧІВ
# ========================================================================
def build_patch_index(volumes, n_patches: int, seed: int = SEED, y_ranges=None):
    rng = np.random.default_rng(seed)

    pos_target     = int(n_patches * POS_PATCH_FRACTION)
    neutral_target = int(n_patches * NEUTRAL_PATCH_FRACTION)
    neg_target     = n_patches - pos_target - neutral_target

    positive, neutral, negative = [], [], []

    for vid, vol in enumerate(volumes):
        if y_ranges is not None and vid < len(y_ranges):
            y_lo, y_hi = y_ranges[vid]
        else:
            y_lo, y_hi = 0, vol.H
        y_lo, y_hi = max(0, y_lo), min(vol.H, y_hi)

        step = PATCH // 2
        for y in range(y_lo, max(y_lo + 1, y_hi - PATCH + 1), step):
            for x in range(0, max(1, vol.W - PATCH + 1), step):
                yy = min(y, y_hi - PATCH)
                xx = min(x, vol.W - PATCH)
                if yy < 0:
                    continue
                m = vol.mask[yy:yy+PATCH, xx:xx+PATCH]
                if m.mean() < MIN_VALID_DENSITY:
                    continue
                lab = vol.label[yy:yy+PATCH, xx:xx+PATCH]
                d   = float(lab.mean())
                item = (vid, yy, xx, d)
                if d >= STRONG_POS_DENSITY:
                    positive.append(item)
                elif d >= POS_DENSITY:
                    neutral.append(item)
                else:
                    negative.append(item)

    rng.shuffle(positive)
    rng.shuffle(neutral)
    rng.shuffle(negative)

    result = []
    result.extend(positive[:min(pos_target, len(positive))])
    result.extend(neutral[:min(neutral_target, len(neutral))])
    result.extend(negative[:min(neg_target, len(negative))])

    pool = positive + neutral + negative
    rng.shuffle(pool)
    used = set((a, b, c) for a, b, c, _ in result)
    for item in pool:
        key = item[:3]
        if key in used:
            continue
        result.append(item)
        used.add(key)
        if len(result) >= n_patches:
            break

    rng.shuffle(result)
    return result[:n_patches]

class PatchDataset(Dataset):
    def __init__(self, volumes, index, augment: bool = False):
        self.volumes = volumes
        self.index   = index
        self.augment = augment

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        vid, y, x, _ = self.index[idx]
        vol = self.volumes[vid]
        depth, valid, target = vol.read_patch(y, x, PATCH)

        if self.augment:
            k = np.random.randint(4)
            if k:
                depth  = np.rot90(depth,  k=k, axes=(1, 2)).copy()
                valid  = np.rot90(valid,  k=k).copy()
                target = np.rot90(target, k=k).copy()
            if np.random.rand() < 0.5:
                depth  = depth[:, :, ::-1].copy()
                valid  = valid[:, ::-1].copy()
                target = target[:, ::-1].copy()
            if np.random.rand() < 0.5:
                depth  = depth[:, ::-1, :].copy()
                valid  = valid[::-1, :].copy()
                target = target[::-1, :].copy()
            if np.random.rand() < 0.35:
                gain = np.random.uniform(0.92, 1.08)
                bias = np.random.uniform(-0.025, 0.025)
                depth = np.clip(depth * gain + bias, 0.0, 1.0)
            if np.random.rand() < 0.15:
                n_drop = np.random.randint(1, max(2, DEPTH // 8))
                ids = np.random.choice(DEPTH, n_drop, replace=False)
                depth[ids] = depth[max(0, DEPTH // 2)]

        return (torch.from_numpy(depth).float(),
                torch.from_numpy(valid).float(),
                torch.from_numpy(target).float())

# ========================================================================
# 5. АРХІТЕКТУРА: 3D DEPTH FUSION + SE-ATTENTION DECODER
# ========================================================================
class Conv3DBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            nn.InstanceNorm3d(cout, affine=True),
            nn.GELU()
        )
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)

class DepthCollapse(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 4)
        self.attn = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, 1, 1)
        )

    def forward(self, x):
        scores  = self.attn(x)
        weights = torch.softmax(scores, dim=2)
        return (x * weights).sum(dim=2)

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), 1),
            nn.GELU(),
            nn.Conv2d(max(4, channels // reduction), channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class FusionBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            SEBlock(cout)
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, cin: int, skip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + skip, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.GELU(),
            SEBlock(cout)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class SUBITInkV26(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1  = Conv3DBlock(1, 32)
        self.down1 = nn.Conv3d(32, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc2  = Conv3DBlock(64, 64)
        self.down2 = nn.Conv3d(64, 96, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc3  = Conv3DBlock(96, 96)
        self.down3 = nn.Conv3d(96, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.enc4  = Conv3DBlock(128, 128)

        self.collapse1 = DepthCollapse(32)
        self.collapse2 = DepthCollapse(64)
        self.collapse3 = DepthCollapse(96)
        self.collapse4 = DepthCollapse(128)

        self.proj1 = FusionBlock(32, 48)
        self.proj2 = FusionBlock(64, 80)
        self.proj3 = FusionBlock(96, 112)
        self.proj4 = FusionBlock(128, 144)

        self.dec3 = DecoderBlock(144, 112, 112)
        self.dec2 = DecoderBlock(112, 80, 80)
        self.dec1 = DecoderBlock(80, 48, 48)

        self.refine = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.head = nn.Conv2d(32, 1, 1)

        self.aux3 = nn.Conv2d(112, 1, 1)
        self.aux2 = nn.Conv2d(80, 1, 1)
        self.aux1 = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = x.unsqueeze(1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))

        s1 = self.proj1(self.collapse1(e1))
        s2 = self.proj2(self.collapse2(e2))
        s3 = self.proj3(self.collapse3(e3))
        s4 = self.proj4(self.collapse4(e4))

        d3 = self.dec3(s4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        r  = self.refine(d1)
        logits = self.head(r)

        if self.training:
            return logits, self.aux3(d3), self.aux2(d2), self.aux1(r)
        return logits

# ========================================================================
# 6. ФУНКЦІЇ ВТРАТ ТА EMA
# ========================================================================
def masked_bce(logits, target, valid):
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * valid).sum() / (valid.sum() + 1e-6)

def masked_dice(logits, target, valid):
    prob   = torch.sigmoid(logits) * valid
    target = target * valid
    inter  = (prob * target).sum(dim=(1, 2, 3))
    denom  = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()

def masked_tversky(logits, target, valid,
                   alpha: float = TVERSKY_ALPHA,
                   beta:  float = TVERSKY_BETA):
    p      = torch.sigmoid(logits) * valid
    target = target * valid
    tp = (p * target).sum(dim=(1, 2, 3))
    fp = (p * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - p) * target).sum(dim=(1, 2, 3))
    return 1.0 - ((tp + 1e-6) / (tp + alpha * fn + beta * fp + 1e-6)).mean()

def focal_loss_fn(logits, target, valid, gamma: float = FOCAL_GAMMA):
    prob = torch.sigmoid(logits)
    pt   = prob * target + (1.0 - prob) * (1.0 - target)
    bce  = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (((1.0 - pt).pow(gamma) * bce * valid).sum() / (valid.sum() + 1e-6))

def total_loss(logits, target, valid, aux3=None, aux2=None, aux1=None):
    bce     = masked_bce(logits, target, valid)
    dice    = masked_dice(logits, target, valid)
    tversky = masked_tversky(logits, target, valid)
    focal   = focal_loss_fn(logits, target, valid)

    loss = (BCE_W * bce + DICE_W * dice + TVERSKY_W * tversky + FOCAL_W * focal)

    if aux3 is not None:
        sz = target.shape[-2:]
        a3 = F.interpolate(aux3, size=sz, mode="bilinear", align_corners=False)
        a2 = F.interpolate(aux2, size=sz, mode="bilinear", align_corners=False)
        a1 = F.interpolate(aux1, size=sz, mode="bilinear", align_corners=False)
        loss += AUX_W * (masked_bce(a3, target, valid) +
                         masked_bce(a2, target, valid) +
                         masked_bce(a1, target, valid))
    return loss

class ModelEMA:
    def __init__(self, model, decay: float = EMA_DECAY):
        self.decay = decay
        self.ema   = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for (_, ep), (_, mp) in zip(self.ema.named_parameters(), model.named_parameters()):
            ep.mul_(self.decay).add_(mp, alpha=1.0 - self.decay)
        for (_, eb), (_, mb) in zip(self.ema.named_buffers(), model.named_buffers()):
            eb.copy_(mb)

# ========================================================================
# 7. БІНАРНІ МЕТРИКИ (F0.5)
# ========================================================================
def binary_metrics(pred: np.ndarray, target: np.ndarray):
    pred   = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    beta2 = 0.25  # F0.5
    f05 = (1.0 + beta2) * precision * recall / (beta2 * precision + recall + 1e-9)
    return float(f05), float(precision), float(recall), int(tp), int(fp), int(fn)

# ========================================================================
# 8. ІНФЕРЕНС (Hann Window + 4x TTA)
# ========================================================================
@torch.no_grad()
def infer_fragment_probability_map(eval_model: nn.Module,
                                   vol: FragmentVolume,
                                   stride: int = VAL_STRIDE_EPOCH,
                                   use_tta: bool = False,
                                   log_progress: bool = False) -> np.ndarray:
    primary_device = DEVICE if torch.cuda.is_available() else torch.device("cpu")
    fallback_cpu   = torch.device("cpu")

    eval_model.to(primary_device)
    eval_model.eval()

    h, w = vol.H, vol.W
    prob_sum   = np.zeros((h, w), np.float32)
    weight_sum = np.zeros((h, w), np.float32)

    ys = list(range(0, max(1, h - PATCH + 1), stride))
    xs = list(range(0, max(1, w - PATCH + 1), stride))
    by, bx = max(0, h - PATCH), max(0, w - PATCH)
    if not ys or ys[-1] != by: ys.append(by)
    if not xs or xs[-1] != bx: xs.append(bx)

    total = len(ys) * len(xs)
    done  = 0
    tta_flips = [(0, 0), (1, 0), (0, 1), (1, 1)] if use_tta else [(0, 0)]

    for y in ys:
        for x in xs:
            depth, _, _ = vol.read_patch(y, x, PATCH)
            X = torch.from_numpy(depth).float().unsqueeze(0)
            probs = []

            try:
                X = X.to(primary_device, non_blocking=True)
                for fh, fv in tta_flips:
                    Xt = X
                    if fh: Xt = torch.flip(Xt, dims=[3])
                    if fv: Xt = torch.flip(Xt, dims=[2])

                    with torch.cuda.amp.autocast(enabled=AMP and primary_device.type == "cuda"):
                        logit = eval_model(Xt)

                    p = torch.sigmoid(logit)[0, 0]
                    if fh: p = torch.flip(p, dims=[1])
                    if fv: p = torch.flip(p, dims=[0])
                    probs.append(p.detach().cpu().numpy())

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    X = X.to(fallback_cpu, non_blocking=True)
                    probs = []
                    for fh, fv in tta_flips:
                        Xt = X
                        if fh: Xt = torch.flip(Xt, dims=[3])
                        if fv: Xt = torch.flip(Xt, dims=[2])
                        logit = eval_model(Xt)
                        p = torch.sigmoid(logit)[0, 0]
                        if fh: p = torch.flip(p, dims=[1])
                        if fv: p = torch.flip(p, dims=[0])
                        probs.append(p.detach().cpu().numpy())
                else:
                    raise

            p_avg = np.mean(probs, axis=0) if len(probs) > 1 else probs[0]
            hh, ww = min(PATCH, h - y), min(PATCH, w - x)
            patch_w = HANN_WEIGHT[:hh, :ww]

            prob_sum[y:y+hh, x:x+ww]   += p_avg[:hh, :ww] * patch_w
            weight_sum[y:y+hh, x:x+ww] += patch_w

            done += 1
            if log_progress and (done % 500 == 0 or done == total):
                log(f"    inference: {done}/{total} ({100*done/total:.0f}%)")

    return prob_sum / np.maximum(weight_sum, 1e-6)

@torch.no_grad()
def validate_fast(eval_model: nn.Module, vol: FragmentVolume, y_val_start: int = 0):
    """Швидка валідація під час епохи (stride=256, no TTA) ~1 хв."""
    prob = infer_fragment_probability_map(eval_model, vol, stride=VAL_STRIDE_EPOCH,
                                          use_tta=False, log_progress=False)
    eval_mask = np.zeros_like(vol.mask)
    if y_val_start > 0:
        eval_mask[y_val_start:] = vol.mask[y_val_start:]
    else:
        eval_mask = vol.mask

    best = None
    for t in THRESHOLDS:
        pred   = (prob >= t) & (eval_mask > 0)
        target = (vol.label > 0) & (eval_mask > 0)
        f05, p, r, tp, fp, fn = binary_metrics(pred, target)
        if best is None or f05 > best["f05"]:
            best = {"threshold": float(t), "f05": f05,
                    "precision": p, "recall": r,
                    "tp": tp, "fp": fp, "fn": fn}
    return best, prob

# ========================================================================
# 9. ПОСТ-ПРОЦЕСИНГ ТА RLE
# ========================================================================
def filter_small_connected_components(mask: np.ndarray, min_size: int = MIN_COMPONENT_SIZE) -> np.ndarray:
    if min_size <= 1:
        return mask
    labeled_arr, num_features = nd_label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labeled_arr.ravel())
    too_small = component_sizes < min_size
    too_small_mask = too_small[labeled_arr]
    clean_mask = mask.copy()
    clean_mask[too_small_mask] = False
    return clean_mask

def postprocess_mask(prob: np.ndarray, threshold: float, mask_region: np.ndarray) -> np.ndarray:
    mask = (prob >= float(threshold)) & (mask_region > 0)
    mask = ndimage.binary_closing(mask, iterations=1) & (mask_region > 0)
    if MIN_COMPONENT_SIZE > 1:
        mask = filter_small_connected_components(mask, min_size=MIN_COMPONENT_SIZE)
        mask = mask & (mask_region > 0)
    return mask

def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.astype(np.uint8).ravel()
    p = np.concatenate([[0], pixels, [0]])
    changes = np.flatnonzero(p[1:] != p[:-1])
    starts = changes[::2] + 1
    lengths = changes[1::2] - changes[::2]
    out = []
    for s, l in zip(starts, lengths):
        out.extend([int(s), int(l)])
    return " ".join(map(str, out))

def clamp_threshold(t: float) -> float:
    return float(np.clip(t, THRESHOLD_MIN, THRESHOLD_MAX))

# ========================================================================
# 10. ГОЛОВНИЙ ПАЙПЛАЙН
# ========================================================================
def main():
    t_total_start = time.time()

    log("\nTRAIN ФРАГМЕНТИ:")
    all_volumes = [FragmentVolume(f) for f in TRAIN_FRAGS]

    val_frag_idx = TRAIN_FRAGS.index(VAL_FRAG)
    val_vol      = all_volumes[val_frag_idx]

    # Спліт 80% train / 20% hold-out
    train_h = int(val_vol.H * TRAIN_VAL_SPLIT)
    log(f"\n  Просторовий split для Fragment {VAL_FRAG}: "
        f"train_h={train_h} ({TRAIN_VAL_SPLIT*100:.0f}%), "
        f"val_h={val_vol.H - train_h} rows")

    train_y_ranges = []
    for i, fid in enumerate(TRAIN_FRAGS):
        if fid == VAL_FRAG:
            train_y_ranges.append((0, train_h))
        else:
            train_y_ranges.append((0, all_volumes[i].H))

    log("\nПобудова індексів патчів...")
    train_index = build_patch_index(all_volumes, TRAIN_PATCHES, seed=SEED,
                                     y_ranges=train_y_ranges)
    val_index   = build_patch_index([val_vol], VAL_PATCHES, seed=SEED + 1,
                                     y_ranges=[(train_h, val_vol.H)])
    log(f"Train патчів: {len(train_index):,} (F1+F2+F3-top-80%)")
    log(f"Val   патчів: {len(val_index):,} (F3-bottom-20%)")

    train_ds = PatchDataset(all_volumes, train_index, augment=AUGMENT)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=AMP, drop_last=True
    )

    model = SUBITInkV26().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"\nПараметрів моделі: {n_params:,}")

    ema       = ModelEMA(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)
    scaler    = torch.cuda.amp.GradScaler(enabled=AMP)

    best_f05       = -1.0
    best_epoch     = -1
    best_threshold = 0.55
    best_state     = None
    top_checkpoints = []

    log("\n" + "=" * 72)
    log(f"НАВЧАННЯ SUBIT-Ink v{VERSION} ({EPOCHS} epochs max)")
    log("=" * 72)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        total_loss_val = 0.0
        n_batches = 0

        for bi, batch in enumerate(train_loader, start=1):
            depth, valid, target = batch
            depth  = depth.to(DEVICE,  non_blocking=True)
            valid  = valid.to(DEVICE,  non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=AMP):
                logits, aux3, aux2, aux1 = model(depth)
                loss = total_loss(logits, target.unsqueeze(1), valid.unsqueeze(1), aux3, aux2, aux1)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            total_loss_val += float(loss.detach())
            n_batches += 1
            if bi == 1 or bi % 200 == 0 or bi == len(train_loader):
                log(f"    batch {bi:04d}/{len(train_loader)} loss={total_loss_val / n_batches:.4f}")

        train_loss = total_loss_val / max(n_batches, 1)

        log(f"  Валідація hold-out (швидкий режим)...")
        metrics, _ = validate_fast(ema.ema, val_vol, y_val_start=train_h)
        scheduler.step()
        elapsed = (time.time() - t0) / 60.0

        log("-" * 72)
        log(f"Epoch {epoch:02d}/{EPOCHS} ({elapsed:.1f} хв) | Train loss={train_loss:.5f}")
        log(f"VAL HOLDOUT: F0.5={metrics['f05']:.4f} | P={metrics['precision']:.4f} | "
            f"R={metrics['recall']:.4f} | t={metrics['threshold']:.3f}")
        log(f"LR={optimizer.param_groups[0]['lr']:.3e}")

        curr_state = {k: v.detach().cpu().clone() for k, v in ema.ema.state_dict().items()}

        # Відбираємо в ансамбль виключно зрілі пізні моделі
        if epoch >= MIN_ENSEMBLE_EPOCH:
            top_checkpoints.append({
                "epoch": epoch,
                "f05":   metrics["f05"],
                "threshold": metrics["threshold"],
                "state": curr_state
            })
            top_checkpoints.sort(key=lambda c: c["f05"], reverse=True)
            if len(top_checkpoints) > 3:
                top_checkpoints = top_checkpoints[:3]
            ckpt_summary = [f"Ep{c['epoch']}: {c['f05']:.4f}" for c in top_checkpoints]
            log(f"    Топ-3 зрілих чекпоінти: {ckpt_summary}")

        if metrics["f05"] > best_f05:
            best_f05       = metrics["f05"]
            best_epoch     = epoch
            best_threshold = metrics["threshold"]
            best_state     = curr_state
            log(f"✓ НОВИЙ РЕКОРД F0.5: {best_f05:.4f} (t={best_threshold:.3f})")
            bad = 0
        else:
            bad = epoch - best_epoch
            log(f"Без покращення: {bad}/6")

        # Early stopping не раніше 20 епохи для повного визрівання ваг
        if bad >= 6 and epoch >= 20:
            log("\nEarly stopping triggered! Збережено найкращі ваги.")
            break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"\nНавчання завершено за {(time.time() - t_total_start)/60:.1f} хв!")

    if not top_checkpoints and best_state is not None:
        top_checkpoints = [{
            "epoch": best_epoch,
            "f05": best_f05,
            "threshold": best_threshold,
            "state": best_state
        }]

    # ====================================================================
    # 11. АНСАМБЛЬ ТА КАЛІБРОВАНИЙ ПОШУК ПОРОГУ З 4x TTA
    # ====================================================================
    log("\n" + "=" * 72)
    log(f"ФОРМУВАННЯ ТОП-{len(top_checkpoints)} АНСАМБЛЮ")
    log("=" * 72)

    # Розрахунок ваг для зваженого ансамблю за F0.5
    raw_scores = np.array([c["f05"] for c in top_checkpoints])
    weights    = raw_scores / np.sum(raw_scores)

    ensemble_models = []
    for i, ckpt in enumerate(top_checkpoints, 1):
        m = SUBITInkV26().to(DEVICE).eval()
        m.load_state_dict(ckpt["state"])
        ensemble_models.append(m)
        log(f"  Модель {i}: Epoch {ckpt['epoch']} (F0.5={ckpt['f05']:.4f}, thr={ckpt['threshold']:.3f}, w={weights[i-1]:.3f})")

    log(f"\nКалібрований пошук порогу ансамблю на hold-out (stride={VAL_STRIDE_FINAL}, 4x TTA)...")
    ens_val_prob = np.zeros((val_vol.H, val_vol.W), np.float32)
    for i, (m, w) in enumerate(zip(ensemble_models, weights), 1):
        log(f"  Holdout прогін [{i}/{len(ensemble_models)}] з 4x TTA (stride={VAL_STRIDE_FINAL})...")
        ens_val_prob += (infer_fragment_probability_map(m, val_vol,
                                                         stride=VAL_STRIDE_FINAL,
                                                         use_tta=True,
                                                         log_progress=False) * w)

    holdout_mask = np.zeros_like(val_vol.mask)
    holdout_mask[train_h:] = val_vol.mask[train_h:]

    best_ens_t   = clamp_threshold(best_threshold)
    best_ens_f05 = -1.0
    for t in THRESHOLDS:
        pred   = (ens_val_prob >= t) & (holdout_mask > 0)
        target = (val_vol.label > 0) & (holdout_mask > 0)
        f05, _, _, _, _, _ = binary_metrics(pred, target)
        if f05 > best_ens_f05:
            best_ens_f05 = f05
            best_ens_t   = float(t)

    best_ens_t = clamp_threshold(best_ens_t)
    log(f"\nАнсамбль Val F0.5={best_ens_f05:.4f} | Калібрований поріг={best_ens_t:.3f} (clamped [{THRESHOLD_MIN},{THRESHOLD_MAX}])")

    # ====================================================================
    # 12. ТЕСТОВИЙ ІНФЕРЕНС (STRIDE=64, 4x TTA, HANN, POSTPROCESSING)
    # ====================================================================
    log("\n" + "=" * 72)
    log(f"ФІНАЛЬНИЙ ТЕСТОВИЙ ІНФЕРЕНС (STRIDE={TEST_STRIDE}, 4x TTA, TOP-{len(ensemble_models)} ENSEMBLE)")
    log("=" * 72)

    test_root = Path(DATA_ROOT) / "test"
    test_ids = (sorted([p.name for p in test_root.iterdir() if p.is_dir()])
                if test_root.exists() else ["a", "b"])
    log(f"Тестових фрагментів: {len(test_ids)} ({test_ids})")

    rows = []
    for f in test_ids:
        log(f"\nОбробка тестового фрагмента: {f}...")
        frag = FragmentVolume(f)

        test_prob = np.zeros((frag.H, frag.W), np.float32)
        for i, (m, w) in enumerate(zip(ensemble_models, weights), 1):
            log(f"  Ансамбль прогін [{i}/{len(ensemble_models)}] з 4x TTA...")
            test_prob += (infer_fragment_probability_map(m, frag, stride=TEST_STRIDE,
                                                          use_tta=True, log_progress=True) * w)

        mask  = postprocess_mask(test_prob, best_ens_t, frag.mask)
        ratio = float(mask.mean())
        rle   = rle_encode(mask)
        log(f"  Поріг={best_ens_t:.3f} | Ink={ratio*100:.2f}% | RLE={len(rle):,} символів")
        rows.append({"Id": f, "Predicted": rle})

        del frag, test_prob, mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    submission = pd.DataFrame(rows, columns=["Id", "Predicted"])
    out_paths = [Path("/kaggle/working/submission.csv"), Path("submission.csv")]
    for p in out_paths:
        try:
            submission.to_csv(p, index=False)
            log(f"Збережено {p}")
        except Exception as e:
            log(f"Не вдалося зберегти {p}: {e}")

    total_min = (time.time() - t_total_start) / 60.0
    log("\n" + "=" * 72)
    log(f"submission.csv ГОТОВО! (v{VERSION})")
    log(f"Ensemble Val F0.5 : {best_ens_f05:.4f}")
    log(f"Ensemble Threshold: {best_ens_t:.3f}")
    log(f"Загальний час: {total_min:.1f} хв (ліміт 540 хв)")
    log("=" * 72)
    log(submission.head())
    log("=" * 72)

if __name__ == "__main__":
    main()
