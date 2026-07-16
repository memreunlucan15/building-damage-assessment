"""dataset / dataset_mm: sentetik .mat dosyalariyla okuma, indeks ve tutarlilik."""
import numpy as np
import torch

import config
from dataset import build_index, load_optical_image
from dataset_mm import MultiModalDataset, _load_sar


def _index(synth_dataset):
    return build_index(synth_dataset)


def test_build_index_counts_and_sorting(synth_dataset):
    idx = _index(synth_dataset)
    assert len(idx) == 5
    assert sum(s["label"] for s in idx) == 2                  # 2 damaged
    labels = [s["label"] for s in idx]
    assert labels == sorted(labels)                           # once intact (0)


def test_load_optical_image_shape(synth_dataset):
    idx = _index(synth_dataset)
    img = load_optical_image(idx[0]["path"])
    assert img.mode == "RGB" and img.size == (32, 32)


def test_load_sar_db_norm_range(synth_dataset):
    idx = _index(synth_dataset)
    sar_path = idx[0]["path"].replace("_opt.mat", "_SAR.mat")
    t = _load_sar(sar_path)
    assert t.shape == (1, 32, 32)
    assert 0.0 <= t.min() and t.max() <= 1.0                  # dB min-max normalize


def test_load_sar_fills_nonfinite(tmp_path):
    # "tests.conftest" DEGIL: ultralytics site-packages'a kuresel "tests" paketi
    # kuruyor ve onu golgeliyor; conftest yerel sys.path uzerinden bulunur.
    from conftest import write_sar_mat
    p = tmp_path / "x_SAR.mat"
    write_sar_mat(p, with_nan=True)
    t = _load_sar(str(p))
    assert torch.isfinite(t).all()


def test_mm_dataset_eval_is_deterministic(synth_dataset):
    idx = _index(synth_dataset)
    ds = MultiModalDataset(idx, train=False, modalities=["opt", "SAR"])
    a, ya = ds[0]
    b, yb = ds[0]
    assert ya == yb
    for m in ("opt", "SAR"):
        assert torch.equal(a[m], b[m])                        # eval'da augment yok


def test_mm_dataset_shapes_and_channels(synth_dataset):
    idx = _index(synth_dataset)
    ds = MultiModalDataset(idx, train=True, modalities=["opt", "SAR"])
    inputs, y = ds[0]
    for m in ("opt", "SAR"):
        assert inputs[m].shape == (3, config.IMG_SIZE, config.IMG_SIZE)
    assert y.dtype == torch.long


def test_mm_dataset_sar_channels_identical(synth_dataset):
    # SAR tek kanal 3'e kopyalanir; augment sonrasi da kanallar esit kalmali.
    idx = _index(synth_dataset)
    ds = MultiModalDataset(idx, train=True, modalities=["SAR"])
    inputs, _ = ds[0]
    sar = inputs["SAR"]
    assert torch.equal(sar[0], sar[1]) and torch.equal(sar[1], sar[2])
