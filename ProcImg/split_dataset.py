# split_dataset.py
import os
import shutil
import random

def split_dataset(original_path='dataset', train_path='dataset_split/train', test_path='dataset_split/test', test_ratio=0.2):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    folders = [f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))]
    random.shuffle(folders)

    split_idx = int(len(folders) * (1 - test_ratio))
    train_folders = folders[:split_idx]
    test_folders = folders[split_idx:]

    for folder in train_folders:
        shutil.copytree(os.path.join(original_path, folder), os.path.join(train_path, folder))
    for folder in test_folders:
        shutil.copytree(os.path.join(original_path, folder), os.path.join(test_path, folder))

    print(f"✅ División completa: {len(train_folders)} train / {len(test_folders)} test")

if __name__ == "__main__":
    split_dataset()
