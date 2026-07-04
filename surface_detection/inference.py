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

def normalize_ct(vol):
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip(vol, p1, p99)
    vol = (vol - p1) / (p99 - p1 + 1e-8)
    return vol.astype(np.float32)

def run_inference(data_dir, model_path):
    print("=" * 60)
    print("SUBIT-Vesuvius v2 — True 3D Inference & Submission")
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
    STRIDE_Z = 16 

    print(f"Знайдено тестових зрізів: {len(test_files)}. Запуск True 3D Sliding Window...")

    # Отримуємо H, W з першого файлу
    first_img = read_image(test_files[0])
    if len(first_img.shape) == 3: first_img = first_img[:, :, 0]
    H, W = first_img.shape
    D_total = len(test_files)

    # Акумулятори для усереднення перекриттів (XY та Z)
    total_probs = np.zeros((D_total, H, W), dtype=np.float32)
    counts = np.zeros((D_total, H, W), dtype=np.uint8)

    with torch.no_grad():
        for z_start in tqdm(range(0, D_total, STRIDE_Z), desc="Z-Chunks"):
            z_end = min(z_start + CHUNK_D, D_total)
            
            ct_slices = []
            for i in range(z_start, z_end):
                img = read_image(test_files[i])
                if img is None: img = np.zeros((H, W), dtype=np.float32)
                if len(img.shape) == 3: img = img[:, :, 0]
                ct_slices.append(img)
                
            while len(ct_slices) < CHUNK_D:
                ct_slices.append(np.zeros((H, W), dtype=np.float32))
                
            ct_vol_3d = np.stack(ct_slices, axis=0).astype(np.float32)
            ct_vol_3d = normalize_ct(ct_vol_3d)
            
            for y in range(0, H, CHUNK_H):
                for x in range(0, W, CHUNK_W):
                    y_end, x_end = min(y + CHUNK_H, H), min(x + CHUNK_W, W)
                    
                    chunk = np.zeros((CHUNK_D, CHUNK_H, CHUNK_W), dtype=np.float32)
                    chunk[:, :y_end-y, :x_end-x] = ct_vol_3d[:, y:y_end, x:x_end]
                    
                    x_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
                    
                    features, _ = pipeline.model(x_tensor)
                    logits, _ = pipeline.attractor(features)
                    probs = F.softmax(logits, dim=1)[0, 1].cpu().numpy() # (32, 128, 128)
                    
                    # Додаємо ймовірності в акумулятор
                    z_len = z_end - z_start
                    total_probs[z_start:z_end, y:y_end, x:x_end] += probs[:z_len, :y_end-y, :x_end-x]
                    counts[z_start:z_end, y:y_end, x:x_end] += 1

    # Усереднення та фінальна бінаризація
    print("Усереднення перекриттів та бінаризація...")
    avg_probs = total_probs / (counts + 1e-8)
    final_mask = (avg_probs > 0.5).astype(np.uint8) # ВИПРАВЛЕНО: маски 0 або 1

    print("Збереження масок...")
    for i in range(D_total):
        img_id = os.path.basename(test_files[i]).split('.')[0]
        out_path = os.path.join(out_dir, f"{img_id}.tif")
        mask_img = Image.fromarray(final_mask[i])
        mask_img.save(out_path)

    # Пакування у submission.zip
    print("\nПакування результатів у submission.zip...")
    out_root = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    zip_path = os.path.join(out_root, "submission.zip")
    
    csv_path = os.path.join(out_root, "submission.csv")
    if os.path.exists(csv_path): os.remove(csv_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(out_dir):
            for file in files:
                zipf.write(os.path.join(root, file), file)
                
    print(f"\n✅ Успіх! Файл {zip_path} створено.")

if __name__ == "__main__":
    KAGGLE_DIR = "/kaggle/input/competitions/vesuvius-challenge-surface-detection"
    DATA_DIR = KAGGLE_DIR if os.path.exists(KAGGLE_DIR) else "./data"
    
    OUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    MODEL_PATH = os.path.join(OUT_DIR, "subit_vesuvius_v2_trained.pt")
    
    run_inference(DATA_DIR, MODEL_PATH)