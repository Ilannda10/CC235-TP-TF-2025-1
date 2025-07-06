import torch
import matplotlib.pyplot as plt
from dataset_loader import CrackDataset
from unet import UNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
model = UNet().to(DEVICE)
model.load_state_dict(torch.load("unet_crack_last.pth", map_location=DEVICE))
model.eval()

# Dataset
dataset = CrackDataset("dataset_split/test", mask_type="CRACK")
img, mask = dataset[0]

# Pasar por el modelo
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(DEVICE))
    pred_sigmoid = torch.sigmoid(pred)
    pred_bin = (pred_sigmoid > 0.5).float()

# Visualizar
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img.permute(1, 2, 0).numpy())
plt.title("Imagen original")

plt.subplot(1, 3, 2)
plt.imshow(mask.squeeze().numpy(), cmap="gray")
plt.title("Máscara real")

plt.subplot(1, 3, 3)
plt.imshow(pred_bin.squeeze().cpu().numpy(), cmap="gray")
plt.title("Predicción (binaria)")

plt.tight_layout()
plt.show()
