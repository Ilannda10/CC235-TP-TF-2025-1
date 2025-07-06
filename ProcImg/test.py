# eval_unetpp_visual.py
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_loader import CrackDataset
from dice_loss import dice_score
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo UNet++ (mismo que en entrenamiento)
model = smp.UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights=None,  # ya está entrenado
    in_channels=3,
    classes=1,
).to(DEVICE)
model.load_state_dict(torch.load("unetpp_crack_best.pth", map_location=DEVICE))
model.eval()

# Dataset de prueba
dataset = CrackDataset("dataset_split/test", mask_type="CRACK")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Métricas
total_dice = 0
total_samples = 0
empty_preds = 0
empty_masks = 0

# Visualizar y evaluar
with torch.no_grad():
    for i, (img, mask) in enumerate(loader):
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        pred = model(img)
        pred = torch.sigmoid(pred)
        pred_bin = (pred > 0.5).float()

        # Métricas
        dice = dice_score(pred_bin, mask).item()
        total_dice += dice
        total_samples += 1
        if pred_bin.sum() == 0:
            empty_preds += 1
        if mask.sum() == 0:
            empty_masks += 1

        # Mostrar resultados
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = pred_bin.squeeze().cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Imagen {i+1} - Dice: {dice:.4f}")
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Imagen")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask_np, cmap='gray')
        plt.title("Máscara real")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_np, cmap='gray')
        plt.title("Predicción")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        # Si quieres mostrar solo las primeras 10
        if i >= 9:
            break

# Resultado final
print(" Evaluación completa:")
print(f" Dice Score promedio: {total_dice / total_samples:.4f}")
print(f" Imágenes con máscara vacía: {empty_masks}/{total_samples}")
print(f" Predicciones vacías: {empty_preds}/{total_samples}")
