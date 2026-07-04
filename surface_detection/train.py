import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import VesuviusPipelineV2
from dataset import VesuviusDataset

def train(data_dir, save_path="subit_vesuvius_v2_trained.pt", epochs=5, max_chunks=500):
    print("=" * 60)
    print("SUBIT-Vesuvius v2 — Training Pipeline")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Пристрій: {device}")
    
    ct_dir = os.path.join(data_dir, "train_images")
    mask_dir = os.path.join(data_dir, "train_labels")
    
    if not os.path.exists(ct_dir):
        raise FileNotFoundError(f"Папку {ct_dir} не знайдено. Перевірте шлях до даних.")
    
    # Ініціалізація датасету (max_chunks=None для повноцінного тренування)
    dataset = VesuviusDataset(ct_dir, mask_dir, chunk_size=(32, 64, 64), stride=(32, 128, 128), augment=True, max_chunks=max_chunks)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    
    pipeline = VesuviusPipelineV2(device=device)
    
    print(f"Початок тренування: {epochs} епох, {len(dataset)} чанків...")
    for epoch in range(epochs):
        epoch_loss, epoch_dice, batches = 0.0, 0.0, 0
        pbar = tqdm(train_loader, desc=f"Епоха {epoch+1}/{epochs}")
        for ct_batch, mask_batch in pbar:
            metrics = pipeline.train_step_batch(ct_batch, mask_batch)
            epoch_loss += metrics['total']
            epoch_dice += metrics['dice_metric']
            batches += 1
            pbar.set_postfix({'Loss': f"{metrics['total']:.4f}", 'Dice': f"{metrics['dice_metric']:.4f}"})
            
        print(f"Епоха {epoch+1} -> Середній Loss: {epoch_loss/batches:.4f}, Середній Dice: {epoch_dice/batches:.4f}\n")

    torch.save({
        'model_state': pipeline.model.state_dict(),
        'attractor_state': pipeline.attractor.state_dict(),
    }, save_path)
    print(f"Тренування завершено! Ваги збережено у: {save_path}")

if __name__ == "__main__":
    # Автоматичне визначення шляхів для Kaggle / локального запуску
    KAGGLE_DIR = "/kaggle/input/competitions/vesuvius-challenge-surface-detection"
    DATA_DIR = KAGGLE_DIR if os.path.exists(KAGGLE_DIR) else "./data"
    
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    SAVE_PATH = os.path.join(OUTPUT_DIR, "subit_vesuvius_v2_trained.pt")
    
    # Для тесту встановлено max_chunks=500 та epochs=3. Для реального конкурсу видаліть обмеження.
    train(DATA_DIR, save_path=SAVE_PATH, epochs=3, max_chunks=500)