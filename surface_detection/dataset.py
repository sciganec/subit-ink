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

class VesuviusDataset(Dataset):
    def __init__(self, ct_dir, mask_dir, chunk_size=(32, 64, 64), stride=(16, 32, 32), augment=True, max_chunks=None, chunks_override=None):
        self.chunk_size, self.stride, self.augment = chunk_size, stride, augment
        self.ct_files = sorted(glob.glob(os.path.join(ct_dir, "*.*")))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.*"))) if mask_dir else []
        
        if len(self.ct_files) == 0: raise ValueError(f"Не знайдено зображень у {ct_dir}")
        img = read_image(self.ct_files[0])
        if img is None: raise ValueError(f"Не вдалося прочитати перший файл: {self.ct_files[0]}")
        if len(img.shape) == 3: img = img[:, :, 0]
        self.H, self.W = img.shape
        self.D = len(self.ct_files)
        
        if chunks_override:
            self.chunks = chunks_override
        else:
            self.chunks = []
            for z in range(0, self.D - chunk_size[0] + 1, stride[0]):
                for y in range(0, self.H - chunk_size[1] + 1, stride[1]):
                    for x in range(0, self.W - chunk_size[2] + 1, stride[2]):
                        self.chunks.append((z, y, x))
                        if max_chunks and len(self.chunks) >= max_chunks: break
                    if max_chunks and len(self.chunks) >= max_chunks: break
                if max_chunks and len(self.chunks) >= max_chunks: break

    def __len__(self): return len(self.chunks)

    def safe_crop(self, img, y, x, h, w):
        if img is None: 
            return np.zeros((h, w), dtype=np.float32)
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
        z, y, x = self.chunks[idx]
        d, h, w = self.chunk_size
        
        ct_slices = [self.safe_crop(read_image(self.ct_files[z+i]), y, x, h, w) for i in range(d)]
        ct_vol = np.stack(ct_slices).astype(np.float32)
        ct_vol = self.normalize_ct(ct_vol)
        
        if self.mask_files:
            mask_slices = [self.safe_crop(read_image(self.mask_files[z+i]), y, x, h, w) for i in range(d)]
            mask_vol = np.stack(mask_slices).astype(np.int64)
            mask_vol[mask_vol > 0] = 1
        else:
            mask_vol = np.zeros_like(ct_vol, dtype=np.int64)
            
        if self.augment:
            for ax in [0, 1, 2]:
                if np.random.rand() > 0.5:
                    ct_vol = np.flip(ct_vol, axis=ax).copy()
                    mask_vol = np.flip(mask_vol, axis=ax).copy()
                    
        return torch.from_numpy(ct_vol).unsqueeze(0), torch.from_numpy(mask_vol)