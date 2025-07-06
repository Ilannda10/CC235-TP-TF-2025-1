# eval_deeplabv3.py
import torch
from torch.utils.data import DataLoader
from dataset_loader import CrackDataset
from dice_loss import dice_score
import segmentation_models_pytorch as smp  # DeepLabV3+

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo DeepLabV3+
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights=None,  # No usar imagenet en evaluación
    in_channels=3,
    classes=1,
).to(DEVICE)

# Cargar pesos entrenados
model.load_state_dict(torch.load("deeplabv3_crack_last.pth", map_location=DEVICE))
model.eval()

# Dataset de prueba
dataset = CrackDataset("dataset_split/test", mask_type="CRACK")
loader = DataLoader(dataset, batch_size=1)

# Métricas acumuladas
total_dice = 0
total_pixels = 0
empty_preds = 0
empty_masks = 0

with torch.no_grad():
    for img, mask in loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)

        pred = model(img)
        pred = torch.sigmoid(pred)         # Activación sigmoide para mapas de probabilidad
        pred_bin = (pred > 0.5).float()    # Umbral para binarizar

        dice = dice_score(pred_bin, mask).item()
        total_dice += dice
        total_pixels += 1

        if pred_bin.sum() == 0:
            empty_preds += 1
        if mask.sum() == 0:
            empty_masks += 1

# Resultados
print("📊 Evaluación del modelo DeepLabV3+:")
print(f"🔹 Dice Score promedio: {total_dice / total_pixels:.4f}")
print(f"🔹 Imágenes con máscara vacía: {empty_masks}/{total_pixels}")
print(f"🔹 Predicciones vacías: {empty_preds}/{total_pixels}")
