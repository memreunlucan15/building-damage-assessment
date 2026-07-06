"""Ortak test fixture'lari: sentetik .mat dosyalari (veri seti gerekmez)."""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_opt_mat(path, h=32, w=32, seed=0):
    """Optik .mat: anahtar x3, (3,H,W) uint8."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f["x3"] = rng.integers(0, 256, size=(3, h, w), dtype=np.uint8)


def write_sar_mat(path, h=32, w=32, seed=0, with_nan=False):
    """SAR .mat: anahtar x1, (H,W) float dB (~24-38 araliginda)."""
    rng = np.random.default_rng(seed)
    arr = rng.uniform(24.0, 38.0, size=(h, w)).astype(np.float32)
    if with_nan:
        arr[0, 0] = np.nan
        arr[1, 1] = np.inf
    with h5py.File(path, "w") as f:
        f["x1"] = arr


@pytest.fixture
def synth_dataset(tmp_path):
    """intact/damaged klasorlerinde 3+2 sentetik ornek (opt + SAR)."""
    counts = {"intact": 3, "damaged": 2}
    for cls, n in counts.items():
        d = tmp_path / cls
        d.mkdir()
        for i in range(n):
            sid = f"{cls}{i:03d}"
            write_opt_mat(d / f"{sid}_opt.mat", seed=hash((cls, i)) % 2**32)
            write_sar_mat(d / f"{sid}_SAR.mat", seed=hash((cls, i, "s")) % 2**32)
    return tmp_path
