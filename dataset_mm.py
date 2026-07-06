"""Cok-modlu veri seti: opt / SAR (+ opsiyonel footprint maskesi).

Her ornek icin {modalite: tensor} sozlugu ve etiket dondurur. Augmentasyon
parametreleri (flip/rotasyon) bir ornegin tum modalitelerinde tutarli uygulanir.
"""
import os
import random

import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

import config
from dataset import load_optical_image


def _path(sample, suffix):
    """Ayni klasordeki kardes modalite dosyasi: <id>_opt.mat -> <id>_<suffix>.mat.

    sample["path"] uzerinden turetilir; boylece veri seti klasoru disindaki
    dosyalarla da (orn. predict.py) calisir.
    """
    return os.path.join(os.path.dirname(sample["path"]), f"{sample['id']}_{suffix}.mat")


def _load_sar(path):
    with h5py.File(path, "r") as f:
        arr = np.asarray(f[config.MAT_KEYS["SAR"]][()], dtype=np.float32)
    # Bazi SAR dosyalarinda (5/4029) NaN/inf piksel var -> gecerli medyan ile doldur.
    finite = np.isfinite(arr)
    if not finite.all():
        fill = np.median(arr[finite]) if finite.any() else 0.0
        arr = np.where(finite, arr, fill).astype(np.float32)
    if config.SAR_DB_NORM:
        # SAR-HUB'a uygun: dB degerlerini sabit aralikta kirp ve [0,1]'e olcekle.
        arr = np.clip(arr, config.SAR_DB_MIN, config.SAR_DB_MAX)
        arr = (arr - config.SAR_DB_MIN) / (config.SAR_DB_MAX - config.SAR_DB_MIN)
    else:
        arr = (arr - arr.mean()) / (arr.std() + 1e-6)    # per-image z-score
        arr = np.clip(arr, -6.0, 6.0)                    # speckle aykiri degerlerini kirp
    return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)   # (1,H,W)


def _load_mask(sample, modality):
    suffix = config.FOOTPRINT_SUFFIX[modality]
    key = config.MAT_KEYS[suffix]
    with h5py.File(_path(sample, suffix), "r") as f:
        m = np.asarray(f[key][()], dtype=np.float32)
    return torch.from_numpy(m).unsqueeze(0)              # (1,H,W) {0,1}


def _modality_tensor(modality, sample, train, p):
    if modality == "opt":
        t = TF.to_tensor(load_optical_image(_path(sample, "opt")))   # (3,H,W) [0,1]
    else:  # SAR
        t = _load_sar(_path(sample, "SAR")).repeat(3, 1, 1)          # (1->3,H,W) cogalt
    t = TF.resize(t, [config.IMG_SIZE, config.IMG_SIZE], antialias=True)

    if config.USE_FOOTPRINT:
        m = _load_mask(sample, modality)
        m = TF.resize(m, [config.IMG_SIZE, config.IMG_SIZE],
                      interpolation=InterpolationMode.NEAREST)
        t = torch.cat([t, m], dim=0)

    if train:
        if p["h"]:
            t = TF.hflip(t)
        if p["v"]:
            t = TF.vflip(t)
        t = TF.rotate(t, p["angle"])
        if modality == "opt":                                        # renk jitter sadece RGB
            t[:3] = TF.adjust_brightness(t[:3], p["bright"])
            t[:3] = TF.adjust_contrast(t[:3], p["contrast"])

    if modality == "opt":                                            # ImageNet norm sadece RGB
        t[:3] = TF.normalize(t[:3], config.IMAGENET_MEAN, config.IMAGENET_STD)
    return t


class MultiModalDataset(Dataset):
    def __init__(self, samples, train, modalities=None):
        self.samples = samples
        self.train = train
        self.modalities = modalities or config.MODALITIES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # Augment parametreleri yalnizca egitimde uretilir; eval'da random
        # tuketmemek determinizmi korur.
        p = None
        if self.train:
            p = {
                "h": random.random() < 0.5,
                "v": random.random() < 0.5,
                "angle": random.uniform(-20, 20),
                "bright": random.uniform(0.8, 1.2),
                "contrast": random.uniform(0.8, 1.2),
            }
        inputs = {m: _modality_tensor(m, s, self.train, p) for m in self.modalities}
        y = torch.tensor(s["label"], dtype=torch.long)
        return inputs, y


def in_channels(modality):
    return config.MODALITY_CHANNELS[modality] + (1 if config.USE_FOOTPRINT else 0)


if __name__ == "__main__":
    from dataset import build_index
    idx = build_index()
    ds = MultiModalDataset(idx[:2] + idx[-2:], train=True, modalities=["opt", "SAR"])
    inputs, y = ds[0]
    for m, t in inputs.items():
        print(f"{m}: {tuple(t.shape)} dtype {t.dtype} min {t.min():.2f} max {t.max():.2f}")
    print("label:", int(y), "| in_channels:", {m: in_channels(m) for m in ["opt", "SAR"]})
