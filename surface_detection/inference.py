import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import zipfile
from model import VesuviusPipelineV2
from dataset import read_image

def run_inference(data_dir, model_path):
    print("=" * 60)
    print("SUBIT-Vesuvius v2 — Inference & Submission.zip Generator")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipeline = VesuviusPipelineV2(device=device)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Модель не знайдена: {model_path}")
        
    ckpt = torch.load(model_path, map_location=device)
    pipeline.model.load_state_dict(ckpt['model_state'])
    pipeline.attractor.load_state_dict(ckpt['attractor_state'])
    pipeline.model.eval()
    pipeline.attractor.eval()

    test_dir = os.path.join(data_dir, "test_images")
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.*")))
    
    out_dir = "/kaggle/working/predictions" if os.path.exists("/kaggle/working") else "./predictions"
    os.makedirs(out_dir, exist_ok=True)

    CHUNK_D, CHUNK_H, CHUNK_W = 32, 128, 128
    results = {}

    print("Генерація масок (Sliding Window)...")
    with torch.no_grad():
        for f in tqdm(test_files, desc="Інференс"):
            img_id = os.path.basename(f).split('.')[0]
            img = read_image(f)
            if img is None: continue
            if len(img.shape) == 3: img = img[:, :, 0]
            
            H, W = img.shape
            final_mask = np.zeros((H, W), dtype=np.uint8)
            pad_H, pad_W = max(0, CHUNK_H - H), max(0, CHUNK_W - W)
            img_padded = np.pad(img, ((0, pad_H), (0, pad_W)), mode='constant')
            
            for y in range(0, img_padded.shape[0], CHUNK_H):
                for x in range(0, img_padded.shape[1], CHUNK_W):
                    crop_2d = img_padded[y:y+CHUNK_H, x:x+CHUNK_W]
                    ct_vol = np.stack([crop_2d]*CHUNK_D).astype(np.float32)
                    x_tensor = torch.from_numpy(ct_vol).unsqueeze(0).unsqueeze(0).to(device)
                    
                    features, _ = pipeline.model(x_tensor)
                    logits, _ = pipeline.attractor(features)
                    mask_3d = (F.softmax(logits, dim=1)[0, 1] > 0.5).cpu().numpy().astype(np.uint8)
                    mask_2d = mask_3d[16] # Беремо центральний зріз
                    
                    y_end, x_end = min(y + CHUNK_H, H), min(x + CHUNK_W, W)
                    final_mask[y:y_end, x:x_end] = mask_2d[:y_end-y, :x_end-x]
            
            mask_img = Image.fromarray(final_mask)
            mask_img.save(os.path.join(out_dir, f"{img_id}.tif"))
            results[img_id] = 1

    # Пакування у submission.zip
    print("\nПакування результатів у submission.zip...")
    out_root = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    zip_path = os.path.join(out_root, "submission.zip")
    
    # Видаляємо старий csv, якщо він існує, щоб Kaggle не плутався
    csv_path = os.path.join(out_root, "submission.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
                
    print(f"\n✅ Успіх! Файл {zip_path} створено. Можна подавати на конкурс!")

if __name__ == "__main__":
    KAGGLE_DIR = "/kaggle/input/competitions/vesuvius-challenge-surface-detection"
    DATA_DIR = KAGGLE_DIR if os.path.exists(KAGGLE_DIR) else "./data"
    
    OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    MODEL_PATH = os.path.join(OUT_DIR, "subit_vesuvius_v2_trained.pt")
    
    run_inference(DATA_DIR, MODEL_PATH)