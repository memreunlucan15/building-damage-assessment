"""Veri indeksi, .mat okuyucu ve PyTorch Dataset (Faz 1: optik RGB)."""
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config


def build_index(data_dir: Path = config.DATA_DIR):
    """Klasorleri tarayip her benzersiz bina ornegi icin kayit listesi dondurur.

    Donen her oge: {"id", "label", "cls", "path"} (path = optik .mat dosyasi).
    """
    samples = []
    for cls in config.CLASSES:                      # ["intact", "damaged"]
        cls_dir = data_dir / cls
        for f in cls_dir.glob(f"*_{config.MODALITY}.mat"):
            sample_id = f.name.rsplit(f"_{config.MODALITY}.mat", 1)[0]
            samples.append({
                "id": sample_id,
                "label": config.CLASS_TO_IDX[cls],
                "cls": cls,
                "path": str(f),
            })
    samples.sort(key=lambda s: (s["label"], s["id"]))
    return samples


def load_optical_image(path: str) -> Image.Image:
    """Optik .mat (anahtar x3, (3,H,W) uint8) -> PIL RGB goruntu."""
    with h5py.File(path, "r") as f:
        arr = f[config.MAT_KEY][()]                 # (3, H, W) uint8
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))          # -> (H, W, 3)
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),        # ust-bakis goruntu -> dikey flip gecerli
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


class QuickQuakeDataset(Dataset):
    def __init__(self, samples, train: bool):
        self.samples = samples
        self.transform = get_transforms(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = load_optical_image(s["path"])
        x = self.transform(img)
        y = torch.tensor(s["label"], dtype=torch.long)
        return x, y


if __name__ == "__main__":
    idx = build_index()
    n_dmg = sum(1 for s in idx if s["label"] == 1)
    print(f"Toplam ornek: {len(idx)} | damaged: {n_dmg} | intact: {len(idx) - n_dmg}")
    ds = QuickQuakeDataset(idx[:4] + idx[-4:], train=True)
    x, y = ds[0]
    print("Ornek tensor sekli:", tuple(x.shape), "| dtype:", x.dtype, "| label:", int(y))
    print("min/max:", float(x.min()), float(x.max()))
