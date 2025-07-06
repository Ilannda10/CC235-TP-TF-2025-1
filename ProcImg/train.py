import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import CrackDataset
from dice_loss import dice_loss
from tqdm import tqdm
import time
import segmentation_models_pytorch as smp

def combined_loss(pred, target):
    pos_weight = torch.tensor([50.0]).to(pred.device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(pred, target)
    dice = dice_loss(torch.sigmoid(pred), target)
    return bce + 2.0 * dice

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EPOCHS = 30
    BATCH_SIZE = 8
    LR = 1e-4

    train_ds = CrackDataset("dataset_split/train", mask_type="CRACK")
    test_ds = CrackDataset("dataset_split/test", mask_type="CRACK")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=0)

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        print(f"\n📚 Época {epoch+1}/{EPOCHS} - Entrenando...")
        for imgs, masks in tqdm(train_loader, desc="🛠️ Entrenando", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                preds = model(imgs)
                loss = combined_loss(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        duration = time.time() - start_time
        print(f"📉 Train Loss: {avg_train_loss:.4f} | ⏱️ {duration:.2f}s | 💾 Mem: {torch.cuda.max_memory_allocated() // 1024**2} MB")
        torch.cuda.reset_peak_memory_stats()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(test_loader, desc="🧪 Validando", leave=False):
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                with torch.amp.autocast(device_type='cuda'):
                    preds = model(imgs)
                    loss = combined_loss(preds, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(test_loader)
        print(f"📊 Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "unetpp_crack_best.pth")
            print("✅ Mejor modelo guardado.")

    torch.save(model.state_dict(), "unetpp_crack_last.pth")
    print("💾 Modelo final guardado.")

if __name__ == "__main__":
    main()
