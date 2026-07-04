import os
import torch
import torch.nn.functional as F # ВИПРАВЛЕНО: додано імпорт
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import VesuviusPipelineV2
from dataset import VesuviusDataset

def train(data_dir, save_path="subit_vesuvius_v2_trained.pt", epochs=10, max_chunks=500):
    print("=" * 60)
    print("SUBIT-Vesuvius v2 — Advanced Training Pipeline")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Пристрій: {device}")
    
    ct_dir = os.path.join(data_dir, "train_images")
    mask_dir = os.path.join(data_dir, "train_labels")
    
    if not os.path.exists(ct_dir):
        raise FileNotFoundError(f"Папку {ct_dir} не знайдено.")
    
    # 1. Датасет та Валідаційний спліт
    full_dataset = VesuviusDataset(ct_dir, mask_dir, chunk_size=(32, 64, 64), stride=(32, 128, 128), augment=True, max_chunks=max_chunks)
    
    # Розбиваємо індекси чанків 80/20
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_chunks = full_dataset.chunks[:train_size]
    val_chunks = full_dataset.chunks[train_size:]
    
    # Створюємо окремі датасети: train з аугментацією, val без неї
    train_dataset = VesuviusDataset(ct_dir, mask_dir, augment=True, chunks_override=train_chunks)
    val_dataset = VesuviusDataset(ct_dir, mask_dir, augment=False, chunks_override=val_chunks)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2)
    
    pipeline = VesuviusPipelineV2(device=device)
    
    # 2. LR Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_dice = 0.0
    
    print(f"Початок тренування: {epochs} епох. Train: {train_size}, Val: {val_size}")
    for epoch in range(epochs):
        # --- ТРЕНУВАННЯ ---
        pipeline.model.train()
        pipeline.attractor.train()
        train_loss, train_dice = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}/{epochs}")
        for ct_batch, mask_batch in pbar:
            metrics = pipeline.train_step_batch(ct_batch, mask_batch)
            train_loss += metrics['total']
            train_dice += metrics['dice_metric']
            pbar.set_postfix({'Loss': f"{metrics['total']:.4f}", 'Dice': f"{metrics['dice_metric']:.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice / len(train_loader)
        
        # --- ВАЛІДАЦІЯ ---
        pipeline.model.eval()
        pipeline.attractor.eval()
        val_loss, val_dice = 0.0, 0.0
        vbar = tqdm(val_loader, desc=f"Val E{epoch+1}/{epochs}")
        with torch.no_grad():
            for ct_batch, mask_batch in vbar:
                x, y = ct_batch.to(device), mask_batch.to(device)
                features, subit_emb = pipeline.model(x)
                logits, omega = pipeline.attractor(features)
                if logits.shape[2:] != y.shape[1:]:
                    logits = F.interpolate(logits, size=y.shape[1:], mode='trilinear', align_corners=False)
                loss, _ = pipeline.loss_fn(logits, y, omega, subit_emb)
                pred = (F.softmax(logits, dim=1)[:, 1] > 0.5).long()
                p, t = pred.float().flatten(), y.float().flatten()
                dice = ((2*(p*t).sum()+1e-5) / (p.sum()+t.sum()+1e-5)).item()
                
                val_loss += loss.item()
                val_dice += dice
                vbar.set_postfix({'V-Loss': f"{loss.item():.4f}", 'V-Dice': f"{dice:.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        
        print(f"Епоха {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f} | Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}\n")
        
        # 3. Оновлення Scheduler
        scheduler.step(avg_val_loss)
        
        # 4. Збереження Best Model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save({
                'model_state': pipeline.model.state_dict(),
                'attractor_state': pipeline.attractor.state_dict(),
                'val_dice': best_val_dice,
            }, save_path)
            print(f"✅ Нова найкраща модель збережена! Val Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    KAGGLE_DIR = "/kaggle/input/competitions/vesuvius-challenge-surface-detection"
    DATA_DIR = KAGGLE_DIR if os.path.exists(KAGGLE_DIR) else "./data"
    
    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "."
    SAVE_PATH = os.path.join(OUTPUT_DIR, "subit_vesuvius_v2_trained.pt")
    
    train(DATA_DIR, save_path=SAVE_PATH, epochs=5, max_chunks=500)