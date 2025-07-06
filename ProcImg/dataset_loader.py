import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class CrackDataset(Dataset):
    def __init__(self, root_dir, mask_type="CRACK", img_size=(640, 1024)):
        """
        root_dir: ruta base del dataset (carpetas con imágenes)
        mask_type: 'CRACK', 'LANE' o 'POTHOLE'
        img_size: tamaño fijo (alto, ancho)
        """
        self.root_dir = root_dir
        self.mask_type = mask_type
        self.img_size = img_size  # (H, W)
        self.samples = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        raw_path = self.find_image_by_suffix(folder, "RAW")
        mask_path = self.find_image_by_suffix(folder, self.mask_type)

        if raw_path is None or mask_path is None:
            raise FileNotFoundError(f" No se encontró RAW o {self.mask_type} en: {folder}")

        # Leer imágenes
        raw = cv2.imread(raw_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if raw is None or mask is None:
            raise ValueError(f" Error al leer imagen RAW o {self.mask_type} en: {folder}")

        # Redimensionar (a tamaño fijo compatible con U-Net)
        h, w = self.img_size
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0

        # Binarizar máscara
        _, mask = cv2.threshold(mask, 0.5, 1.0, cv2.THRESH_BINARY)

        # Preparar tensores
        raw = np.transpose(raw, (2, 0, 1))      # HWC -> CHW
        mask = np.expand_dims(mask, axis=0)     # HW -> 1xHxW

        return torch.tensor(raw, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def find_image_by_suffix(self, folder, suffix):
        """Busca una imagen dentro de la carpeta con el sufijo dado y una extensión válida."""
        valid_exts = [".png", ".jpg", ".jpeg"]
        for file in os.listdir(folder):
            if suffix.lower() in file.lower():
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_exts:
                    return os.path.join(folder, file)
        return None
