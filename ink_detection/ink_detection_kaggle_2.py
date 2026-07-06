import os
import glob
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple, List
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
CFG = {
    "base_path":    "/kaggle/input/competitions/vesuvius-challenge-ink-detection",
    "fragments":    ["train/1", "train/2a"],
    "test_frags":   ["test/a", "test/b"],
    "crop_size":    (256, 256),
    "stride_train": (128, 128),
    "z_start":      27,
    "z_end":        37,
    "in_channels":  10,
    "epochs":       30,
    "batch_size":   4,
    "lr":           3e-5,
    "pos_weight":   4.0,
    "threshold":    0.4,
    "tta":          True,
    "infer_stride": 64,
}


# ─────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────
class VesuviusInkDataset(Dataset):

    def __init__(
        self,
        fragment_paths: List[str],
        crop_size: Tuple[int, int] = (256, 256),
        stride: Tuple[int, int] = (128, 128),
        z_start: int = 27,
        z_end: int = 37,
        augment: bool = True,
        balance: bool = True,
    ):
        self.crop_size = crop_size
        self.augment   = augment
        self.fragments = []

        for frag_path in fragment_paths:
            vol_dir   = os.path.join(frag_path, "surface_volume")
            mask_path = os.path.join(frag_path, "inklabels.png")

            if not os.path.exists(vol_dir):
                print(f"  ⚠ Не знайдено: {vol_dir}")
                continue

            # --- Маска ---
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                mask = (mask > 0).astype(np.uint8)
            else:
                print(f"  ⚠ Маска не знайдена: {mask_path}")
                alt = (glob.glob(os.path.join(frag_path, "*ink*")) +
                       glob.glob(os.path.join(frag_path, "*label*")))
                if alt:
                    mask = np.array(Image.open(alt[0]))
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    mask = (mask > 0).astype(np.uint8)
                    print(f"    Альтернатива: {alt[0]}")
                else:
                    mask = None

            # --- Z-зрізи ---
            z_files = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))[z_start:z_end]
            if not z_files:
                print(f"  ⚠ Немає tif у {vol_dir}")
                continue

            name = os.path.basename(frag_path)
            print(f"  Кешування {name} ({len(z_files)} зрізів) як uint8...")

            slices = []
            for f in z_files:
                try:
                    img = np.array(Image.open(f))
                    if len(img.shape) == 3:
                        img = img[:, :, 0]
                    slices.append(img)
                except Exception as e:
                    print(f"    ⚠ {os.path.basename(f)}: {e}")
                    if slices:
                        slices.append(np.zeros_like(slices[0]))

            if not slices:
                continue

            volume = np.stack(slices, axis=0)
            if volume.dtype == np.uint16:
                volume = (volume >> 8).astype(np.uint8)
            else:
                volume = volume.astype(np.uint8)

            H, W = volume.shape[1], volume.shape[2]
            if mask is None:
                mask = np.zeros((H, W), dtype=np.uint8)

            ram_mb = volume.nbytes // 1024**2
            print(f"    ✓ {volume.shape} uint8, RAM={ram_mb}MB, "
                  f"ink={mask.mean():.4f} ({mask.sum():,} px)")

            self.fragments.append((volume, mask, H, W))

        if not self.fragments:
            raise ValueError("Жоден фрагмент не завантажено!")

        # --- Чанки ---
        ink_chunks, empty_chunks = [], []
        ch, cw = crop_size
        for fi, (_, mask, H, W) in enumerate(self.fragments):
            for y in range(0, H - ch + 1, stride[0]):
                for x in range(0, W - cw + 1, stride[1]):
                    entry = (fi, y, x)
                    if mask[y:y+ch, x:x+cw].sum() > 100:
                        ink_chunks.append(entry)
                    else:
                        empty_chunks.append(entry)

        if balance and ink_chunks:
            n = len(ink_chunks)
            self.chunks = ink_chunks + empty_chunks[:n]
        else:
            self.chunks = ink_chunks + empty_chunks

        print(f"\n✓ Датасет: {len(self.chunks)} чанків "
              f"({len(ink_chunks)} з чорнилом, {len(empty_chunks)} порожніх)")

    def _normalize(self, patch: np.ndarray) -> np.ndarray:
        p = patch.astype(np.float32) / 255.0
        for c in range(p.shape[0]):
            mn, mx = p[c].min(), p[c].max()
            if mx > mn:
                p[c] = (p[c] - mn) / (mx - mn)
        return p

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        fi, y, x = self.chunks[idx]
        volume, mask, H, W = self.fragments[fi]
        h, w = self.crop_size

        y2, x2 = min(y + h, H), min(x + w, W)
        ct   = np.zeros((volume.shape[0], h, w), dtype=np.uint8)
        msk  = np.zeros((h, w), dtype=np.int64)
        ct[:, :y2-y, :x2-x] = volume[:, y:y2, x:x2]
        msk[:y2-y, :x2-x]   = mask[y:y2, x:x2].astype(np.int64)

        ct = self._normalize(ct)

        if self.augment:
            if np.random.rand() > 0.5:
                ct  = np.flip(ct,  axis=1).copy()
                msk = np.flip(msk, axis=0).copy()
            if np.random.rand() > 0.5:
                ct  = np.flip(ct,  axis=2).copy()
                msk = np.flip(msk, axis=1).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                ct  = np.rot90(ct,  k, axes=(1, 2)).copy()
                msk = np.rot90(msk, k, axes=(0, 1)).copy()

        return torch.from_numpy(ct), torch.from_numpy(msk)


# ─────────────────────────────────────────────────────────
# 2. MODEL
# ─────────────────────────────────────────────────────────
class CurvatureField2D(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        size = int(6*sigma+1)|1
        coords = torch.arange(size).float() - size//2
        g = torch.exp(-coords**2/(2*sigma**2)); g = g/g.sum()
        k = (g[:,None]*g[None,:]).unsqueeze(0).unsqueeze(0)
        self.smooth = nn.Conv2d(1,1,size,padding=size//2,bias=False)
        self.smooth.weight = nn.Parameter(k, requires_grad=False)

    def _d2(self, x, dim):
        pad = (1,1,0,0) if dim==3 else (0,0,1,1)
        xp = F.pad(x, pad, mode='replicate')
        s = [slice(None)]*xp.dim()
        sm,sc,sp = list(s),list(s),list(s)
        sm[dim]=slice(0,-2); sc[dim]=slice(1,-1); sp[dim]=slice(2,None)
        return xp[tuple(sm)]-2*xp[tuple(sc)]+xp[tuple(sp)]

    def forward(self, x):
        xs = self.smooth(x.mean(dim=1,keepdim=True))
        hyy, hxx = self._d2(xs,2), self._d2(xs,3)
        l1 = torch.min(hxx.abs(), hyy.abs())
        l2 = torch.max(hxx.abs(), hyy.abs())
        n  = l2 + 1e-6
        return torch.cat([
            1-l1/n, l1/n,
            ((hxx<0)&(hyy<0)).float(),
            ((hxx>0)&(hyy>0)).float()
        ], dim=1)


class SUBITEncoder2D(nn.Module):
    SCALES = [1, 2, 4, 8]
    def __init__(self, embed_dim=32):
        super().__init__()
        self.fields = nn.ModuleList([CurvatureField2D(s) for s in self.SCALES])
        self.proj   = nn.Sequential(
            nn.Conv2d(16, embed_dim//2, 1), nn.GroupNorm(8, embed_dim//2), nn.GELU(),
            nn.Conv2d(embed_dim//2, embed_dim, 1))
    def forward(self, x):
        return self.proj(torch.cat([f(x) for f in self.fields], dim=1))


class Ftheta2D(nn.Module):
    def __init__(self, ch, steps=3):
        super().__init__()
        self.steps = steps
        self.diff  = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,groups=ch),
            nn.Conv2d(ch,ch,1), nn.GroupNorm(8,ch), nn.GELU(),
            nn.Conv2d(ch,2,1))
        self.dt   = nn.Parameter(torch.ones(1,ch,1,1)*0.1)
        self.proj = nn.Sequential(nn.Conv2d(ch,ch,3,padding=1), nn.GroupNorm(8,ch), nn.GELU())
        self.gate = nn.Sequential(nn.Conv2d(ch,1,1), nn.Sigmoid())

    def _step(self, x, D):
        Dxx, Dyy = D[:,0:1].abs(), D[:,1:2].abs()
        xp = nn.ReplicationPad2d(1)(x)
        d2y = xp[:,:,2:,1:-1]-2*x+xp[:,:,:-2,1:-1]
        d2x = xp[:,:,1:-1,2:]-2*x+xp[:,:,1:-1,:-2]
        return x + self.dt*(Dyy*d2y + Dxx*d2x)

    def forward(self, x):
        r=x; D=self.diff(x); g=self.gate(x); h=x
        for _ in range(self.steps): h=self._step(h,D)
        return r + g*(self.proj(h)-r)


class Block2D(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ic,oc,3,padding=1), nn.GroupNorm(min(8,oc),oc), nn.GELU(),
            nn.Conv2d(oc,oc,3,padding=1), nn.GroupNorm(min(8,oc),oc), nn.GELU())
        self.skip = nn.Conv2d(ic,oc,1) if ic!=oc else nn.Identity()
    def forward(self, x): return self.block(x)+self.skip(x)


class Attn2D(nn.Module):
    def __init__(self, fc, sc):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(fc+sc,fc,1), nn.GroupNorm(min(8,fc),fc), nn.GELU(),
            nn.Conv2d(fc,fc,1), nn.Sigmoid())
    def forward(self, f, s): return f*self.gate(torch.cat([f,s],dim=1))


class SUBITInkUNet(nn.Module):
    def __init__(self, in_ch=10, features=(32,64,128,256), edim=32):
        super().__init__()
        self.subit = SUBITEncoder2D(edim)
        self.inp   = nn.Conv2d(in_ch+edim, features[0], 1)

        self.encs  = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.eattn = nn.ModuleList()
        ch = features[0]
        for f in features:
            self.encs.append(Block2D(ch,f))
            self.pools.append(nn.MaxPool2d(2))
            self.eattn.append(Attn2D(f,edim))
            ch = f

        self.btn  = Block2D(features[-1], features[-1]*2)
        self.evol = Ftheta2D(features[-1]*2, steps=3)

        self.ups   = nn.ModuleList()
        self.decs  = nn.ModuleList()
        self.dattn = nn.ModuleList()
        ch = features[-1]*2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(ch,f,2,stride=2))
            self.decs.append(Block2D(f*2,f))
            self.dattn.append(Attn2D(f,edim))
            ch = f

        self.out = nn.Conv2d(features[0], features[0], 1)

    def forward(self, x):
        se = self.subit(x)
        ss = [se]
        for _ in range(len(self.encs)-1):
            ss.append(F.avg_pool2d(ss[-1], 2))

        h = self.inp(torch.cat([x, se], dim=1))
        skips = []
        for i,(enc,pool,attn) in enumerate(zip(self.encs,self.pools,self.eattn)):
            h = attn(enc(h), ss[i]); skips.append(h); h = pool(h)

        h = self.evol(self.btn(h))

        for up,dec,attn,skip in zip(self.ups,self.decs,self.dattn,reversed(skips)):
            h = up(h)
            if h.shape != skip.shape:
                h = F.interpolate(h, size=skip.shape[2:], mode='bilinear', align_corners=False)
            h = attn(dec(torch.cat([h,skip],dim=1)),
                     F.interpolate(se, size=h.shape[2:], mode='bilinear', align_corners=False))
        return self.out(h), se


class AttractorDecoder2D(nn.Module):
    def __init__(self, ch=32, steps=5):
        super().__init__()
        self.steps  = steps
        self.ftheta = Ftheta2D(ch, steps=1)
        self.omega  = nn.Sequential(
            nn.Conv2d(ch*2,ch,1), nn.GroupNorm(min(8,ch),ch), nn.GELU(),
            nn.Conv2d(ch,4,1), nn.Softmax(dim=1))
        self.head = nn.Conv2d(ch, 2, 1)

    def forward(self, x):
        h = x; delta = torch.zeros_like(x)
        for _ in range(self.steps):
            hn = self.ftheta(h); delta += (hn-h).abs(); h = hn
        omega = self.omega(torch.cat([h,delta],dim=1))
        w = omega[:,3:4]*1.5 + omega[:,2:3]*1.2 + omega[:,1:2]*0.8 + omega[:,0:1]*0.3
        return self.head(h*w), omega


# ─────────────────────────────────────────────────────────
# 3. LOSS
# ─────────────────────────────────────────────────────────
class OmegaLoss2D(nn.Module):
    def __init__(self, pos_weight=4.0, alpha=0.4, beta=0.3, gamma=0.2, delta=0.1):
        super().__init__()
        self.alpha=alpha; self.beta=beta; self.gamma=gamma; self.delta=delta
        self.register_buffer('w', torch.tensor([1.0, pos_weight]))

    def forward(self, logits, labels, omega, se):
        probs = F.softmax(logits, dim=1)
        ce    = F.cross_entropy(logits, labels, weight=self.w.to(logits.device))

        smooth = 1e-5; p,l = probs[:,1], labels.float()
        inter  = (p*l).sum(dim=(1,2))
        union  = p.sum(dim=(1,2)) + l.sum(dim=(1,2))
        dice   = (1-(2*inter+smooth)/(union+smooth)).mean()

        if se.shape[2:] != probs.shape[2:]:
            se = F.interpolate(se, size=probs.shape[2:], mode='bilinear', align_corners=False)
        curv = (torch.sigmoid(se[:,0]) * (1-probs[:,1])).mean()
        stab = (omega[:,0] * probs[:,1]).mean()
        om   = omega.mean(dim=(0,2,3))
        oreg = (om*(om+1e-8).log()).sum() * 0.01

        total = self.alpha*ce + (1-self.alpha)*dice + self.gamma*curv + self.beta*stab + self.delta*oreg
        return total, {"ce":ce.item(), "dice":dice.item(), "curv":curv.item(),
                       "stab":stab.item(), "total":total.item()}


# ─────────────────────────────────────────────────────────
# 4. PIPELINE
# ─────────────────────────────────────────────────────────
class InkPipeline:
    def __init__(self, in_ch=10, pos_weight=4.0, lr=3e-5, device='cuda'):
        self.device    = device
        self.model     = SUBITInkUNet(in_ch=in_ch).to(device)
        self.attractor = AttractorDecoder2D().to(device)
        self.loss_fn   = OmegaLoss2D(pos_weight=pos_weight)
        params = list(self.model.parameters()) + list(self.attractor.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    def _forward(self, ct, mask):
        x, y = ct.to(self.device), mask.to(self.device)
        feats, se   = self.model(x)
        logits, omega = self.attractor(feats)
        if logits.shape[2:] != y.shape[1:]:
            logits = F.interpolate(logits, size=y.shape[1:], mode='bilinear', align_corners=False)
            omega  = F.interpolate(omega,  size=y.shape[1:], mode='bilinear', align_corners=False)
        loss, comps = self.loss_fn(logits, y, omega, se)
        pred = (F.softmax(logits,dim=1)[:,1] > 0.5).long()
        p, t = pred.float().flatten(), y.float().flatten()
        comps["dice"] = ((2*(p*t).sum()+1e-5)/(p.sum()+t.sum()+1e-5)).item()
        return loss, comps

    def train_step(self, ct, mask):
        self.model.train(); self.attractor.train(); self.optimizer.zero_grad()
        loss, comps = self._forward(ct, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters())+list(self.attractor.parameters()), 0.5)
        self.optimizer.step()
        return comps

    def val_step(self, ct, mask):
        self.model.eval(); self.attractor.eval()
        with torch.no_grad():
            _, comps = self._forward(ct, mask)
        return comps

    def predict_tta(self, ct_np, threshold=0.4):
        self.model.eval(); self.attractor.eval()
        def pred(arr):
            x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feats,_ = self.model(x)
                logits,_ = self.attractor(feats)
                return F.softmax(logits,dim=1)[0,1].cpu().numpy()
        p  = pred(ct_np)
        p += pred(ct_np[:,::-1,:].copy())[::-1,:]
        p += pred(ct_np[:,:,::-1].copy())[:,::-1]
        p += pred(ct_np[:,::-1,::-1].copy())[::-1,::-1]
        return (p/4 > threshold).astype(np.uint8)

    def save(self, path):
        torch.save({'model':self.model.state_dict(),
                    'attractor':self.attractor.state_dict()}, path)
        print(f"  ✓ Збережено: {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.attractor.load_state_dict(ckpt['attractor'])


# ─────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────
def train_loop(pipeline, train_loader, val_loader, epochs, save_path):
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        pipeline.optimizer, T_max=epochs, eta_min=1e-6)
    best = 0.0

    for ep in range(epochs):
        tl, td = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Train {ep+1}/{epochs}")
        for ct, mask in pbar:
            m = pipeline.train_step(ct, mask)
            tl += m['total']; td += m['dice']
            pbar.set_postfix({'L':f"{m['total']:.3f}", 'D':f"{m['dice']:.3f}"})

        vl, vd = 0.0, 0.0
        for ct, mask in val_loader:
            m = pipeline.val_step(ct, mask)
            vl += m['total']; vd += m['dice']

        nt, nv = len(train_loader), len(val_loader)
        avg_vd = vd/nv
        print(f"E{ep+1:02d}: TL={tl/nt:.4f} TD={td/nt:.4f} | "
              f"VL={vl/nv:.4f} VD={avg_vd:.4f} | LR={sched.get_last_lr()[0]:.1e}")
        sched.step()

        if avg_vd > best:
            best = avg_vd
            pipeline.save(save_path)
            print(f"  ✅ Best Val Dice={best:.4f}")

    return best


# ─────────────────────────────────────────────────────────
# 6. INFERENCE + RLE SUBMISSION
# ─────────────────────────────────────────────────────────
def mask_to_rle(mask):
    pixels = mask.flatten(order='F')
    runs = []; start = None
    for i, v in enumerate(pixels):
        if v == 1 and start is None:
            start = i+1
        elif v == 0 and start is not None:
            runs += [start, i-start+1]; start = None
    if start is not None:
        runs += [start, len(pixels)-start+1]
    return ' '.join(map(str, runs)) if runs else ''


def run_inference(pipeline, cfg):
    out_dir = "/kaggle/working/predictions"
    os.makedirs(out_dir, exist_ok=True)
    z0, z1 = cfg['z_start'], cfg['z_end']
    ch, cw = cfg['crop_size']
    stride  = cfg['infer_stride']
    rows = []

    for frag_name in cfg['test_frags']:
        frag_path = os.path.join(cfg['base_path'], frag_name)
        vol_dir   = os.path.join(frag_path, "surface_volume")
        z_files   = sorted(glob.glob(os.path.join(vol_dir, "*.tif")))[z0:z1]
        if not z_files:
            print(f"  ⚠ Немає зрізів: {vol_dir}"); continue

        slices = []
        for f in z_files:
            try:
                img = np.array(Image.open(f))
                if len(img.shape) == 3: img = img[:,:,0]
                if img.dtype == np.uint16: img = (img>>8).astype(np.uint8)
                slices.append(img.astype(np.uint8))
            except:
                slices.append(np.zeros_like(slices[0]) if slices else np.zeros((1,1),np.uint8))

        volume = np.stack(slices, axis=0).astype(np.float32) / 255.0
        C, H, W = volume.shape
        prob_sum = np.zeros((H, W), np.float32)
        counts   = np.zeros((H, W), np.float32)

        frag_id = os.path.basename(frag_name)
        print(f"Inference {frag_id}: {H}×{W}...")

        for y in tqdm(range(0, H, stride), desc=f"  {frag_id}"):
            for x in range(0, W, stride):
                y2, x2 = min(y+ch, H), min(x+cw, W)
                patch = np.zeros((C, ch, cw), dtype=np.float32)
                patch[:, :y2-y, :x2-x] = volume[:, y:y2, x:x2]

                if cfg['tta']:
                    pred = pipeline.predict_tta(patch, cfg['threshold'])
                else:
                    xt = torch.from_numpy(patch).unsqueeze(0).to(pipeline.device)
                    with torch.no_grad():
                        feats,_ = pipeline.model(xt)
                        logits,_ = pipeline.attractor(feats)
                        prob = F.softmax(logits,dim=1)[0,1].cpu().numpy()
                    pred = (prob > cfg['threshold']).astype(np.uint8)

                prob_sum[y:y2, x:x2] += pred[:y2-y, :x2-x].astype(np.float32)
                counts[y:y2, x:x2]   += 1

        final = (prob_sum/(counts+1e-8) > 0.5).astype(np.uint8)
        Image.fromarray(final*255).save(os.path.join(out_dir, f"{frag_id}.png"))
        rows.append({'Id': frag_id, 'Predicted': mask_to_rle(final)})
        print(f"  ✓ {frag_id}: ink={final.mean():.4f}")

    pd.DataFrame(rows).to_csv("/kaggle/working/submission.csv", index=False)
    print("✅ submission.csv створено!")


# ─────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("🚀 SUBIT-Ink v4.0")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Пристрій: {device}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")

    frag_paths = [
        os.path.join(CFG['base_path'], f) for f in CFG['fragments']
        if os.path.exists(os.path.join(CFG['base_path'], f, "surface_volume"))
    ]
    print(f"Фрагментів: {len(frag_paths)}")
    if not frag_paths:
        raise RuntimeError(
            f"Фрагменти не знайдено. Перевір base_path: {CFG['base_path']}\n"
            f"Вміст /kaggle/input: {os.listdir('/kaggle/input')}"
        )

    # ── ДАТАСЕТ ──
    print("\n[1/3] Датасет...")
    full_ds = VesuviusInkDataset(
        frag_paths,
        crop_size=CFG['crop_size'],
        stride=CFG['stride_train'],
        z_start=CFG['z_start'],
        z_end=CFG['z_end'],
        augment=True,
        balance=True,
    )

    val_n   = max(1, int(len(full_ds)*0.2))
    train_n = len(full_ds) - val_n

    train_ds = copy.copy(full_ds); train_ds.chunks = full_ds.chunks[:train_n]; train_ds.augment = True
    val_ds   = copy.copy(full_ds); val_ds.chunks   = full_ds.chunks[train_n:]; val_ds.augment   = False

    train_loader = DataLoader(train_ds, batch_size=CFG['batch_size'], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG['batch_size'], shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f"Train={train_n} | Val={val_n}")

    # ── НАВЧАННЯ ──
    print("\n[2/3] Навчання...")
    pipeline = InkPipeline(in_ch=CFG['in_channels'], pos_weight=CFG['pos_weight'],
                           lr=CFG['lr'], device=device)
    print(f"Параметри: {sum(p.numel() for p in pipeline.model.parameters()):,}")

    save_path  = "/kaggle/working/subit_ink_best.pt"
    best_dice  = train_loop(pipeline, train_loader, val_loader, CFG['epochs'], save_path)

    # ── INFERENCE ──
    print("\n[3/3] Inference...")
    pipeline.load(save_path)
    run_inference(pipeline, CFG)

    print(f"\n🏁 Best Val Dice: {best_dice:.4f}")