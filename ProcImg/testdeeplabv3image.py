import torch
from torch.utils.data import DataLoader
from dataset_loader import CrackDataset
from dice_loss import dice_score
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=1,
).to(DEVICE)

# Cargar pesos entrenados
model.load_state_dict(torch.load("deeplabv3_crack_last.pth", map_location=DEVICE))
model.eval()

# Dataset
dataset = CrackDataset("dataset_split/test", mask_type="CRACK")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Métricas
total_dice = 0
total_pixels = 0
empty_preds = 0
empty_masks = 0

def show_prediction(img_tensor, mask_tensor, pred_tensor):
    img_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    mask_np = mask_tensor.squeeze().cpu().numpy()
    pred_np = pred_tensor.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_np)
    axs[0].set_title("Imagen original")
    axs[1].imshow(mask_np, cmap="gray")
    axs[1].set_title("Máscara real (CRACK)")
    axs[2].imshow(pred_np, cmap="gray")
    axs[2].set_title("Predicción del modelo")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

# Evaluación
with torch.no_grad():
    for i, (img, mask) in enumerate(loader):
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        pred = model(img)
        pred = torch.sigmoid(pred)
        pred_bin = (pred > 0.5).float()

        dice = dice_score(pred_bin, mask).item()
        total_dice += dice
        total_pixels += 1

        if pred_bin.sum() == 0:
            empty_preds += 1
        if mask.sum() == 0:
            empty_masks += 1

        # 🔍 Mostrar solo las primeras 5 imágenes
        if i < 5:
            show_prediction(img, mask, pred_bin)

# Resultados globales
print("📊 Evaluación del modelo DeepLabV3+:")
print(f"🔹 Dice Score promedio: {total_dice / total_pixels:.4f}")
print(f"🔹 Imágenes con máscara vacía: {empty_masks}/{total_pixels}")
print(f"🔹 Predicciones vacías: {empty_preds}/{total_pixels}")
