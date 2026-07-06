import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def read_image(path):
    try:
        img = Image.open(path)
        return np.array(img)
    except Exception:
        return None

class VesuviusInkDataset(Dataset):
    def __init__(self, surface_volume_dir, ink_mask_path, crop_size=(256, 256), stride=(128, 128), augment=True, max_chunks=None):
        self.crop_size, self.stride, self.augment = crop_size, stride, augment
        
        # Шукаємо всі файли z-зрізів (зазвичай 00.tif ... 31.tif)
        self.ct_files = sorted(glob.glob(os.path.join(surface_volume_dir, "*.tif")))
        if len(self.ct_files) == 0:
            raise ValueError(f"Не знайдено z-зрізів у {surface_volume_dir}")
            
        self.ink_mask = read_image(ink_mask_path)
        if self.ink_mask is None:
            raise ValueError(f"Не вдалося прочитати маску чорнила: {ink_mask_path}")
        if len(self.ink_mask.shape) == 3:
            self.ink_mask = self.ink_mask[:, :, 0]
            
        self.H, self.W = self.ink_mask.shape
        self.D = len(self.ct_files) # Зазвичай 32
        
        self.chunks = []
        for y in range(0, self.H - crop_size[0] + 1, stride[0]):
            for x in range(0, self.W - crop_size[1] + 1, stride[1]):
                self.chunks.append((y, x))
                if max_chunks and len(self.chunks) >= max_chunks: break
            if max_chunks and len(self.chunks) >= max_chunks: break

    def __len__(self): return len(self.chunks)

    def safe_crop(self, img, y, x, h, w):
        if img is None: return np.zeros((h, w), dtype=np.float32)
        if len(img.shape) == 3: img = img[:, :, 0]
        H, W = img.shape
        patch = np.zeros((h, w), dtype=img.dtype)
        y1, y2 = max(0, y), min(H, y + h)
        x1, x2 = max(0, x), min(W, x + w)
        if y1 < y2 and x1 < x2:
            patch[y1-y:y2-y, x1-x:x2-x] = img[y1:y2, x1:x2]
        return patch

    def normalize_ct(self, vol):
        p1, p99 = np.percentile(vol, [1, 99])
        vol = np.clip(vol, p1, p99)
        vol = (vol - p1) / (p99 - p1 + 1e-8)
        return vol.astype(np.float32)

    def __getitem__(self, idx):
        y, x = self.chunks[idx]
        h, w = self.crop_size
        
        # Читаємо 32 зрізи і формуємо з них 32-канальний тензор (C, H, W)
        ct_slices = [self.safe_crop(read_image(f), y, x, h, w) for f in self.ct_files]
        ct_vol = np.stack(ct_slices, axis=0).astype(np.float32) # (32, H, W)
        ct_vol = self.normalize_ct(ct_vol)
        
        # Маска чорнила (1, H, W)
        mask = self.safe_crop(self.ink_mask, y, x, h, w)
        mask_vol = (mask > 0).astype(np.int64)
            
        if self.augment:
            if np.random.rand() > 0.5:
                ct_vol = np.flip(ct_vol, axis=1).copy()
                mask_vol = np.flip(mask_vol, axis=0).copy()
            if np.random.rand() > 0.5:
                ct_vol = np.flip(ct_vol, axis=2).copy()
                mask_vol = np.flip(mask_vol, axis=1).copy
                    
        return torch.from_numpy(ct_vol), torch.from_numpy(mask_vol)