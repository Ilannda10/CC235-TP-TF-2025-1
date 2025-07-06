import matplotlib.pyplot as plt
from dataset_loader import CrackDataset
import torch
ds = CrackDataset("dataset_split/test", mask_type="CRACK")
img, mask = ds[0]

# Pasar a numpy para visualizar
img_np = img.permute(1, 2, 0).numpy()  # CHW → HWC
mask_np = mask.squeeze().numpy()      # 1xHxW → HxW

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title("Imagen RAW")

plt.subplot(1, 2, 2)
plt.imshow(mask_np, cmap='gray')
plt.title("Máscara CRACK")

plt.show()
_, mask = ds[0]
print(torch.unique(mask))
grieta_count = sum(torch.sum(mask) for _, mask in ds)
print(f"Total píxeles con grieta en test: {int(grieta_count.item())}")
