"""inference.py testleri: goruntu okuma, on isleme, tahmin ve Grad-CAM (agirlik gerektirmez)."""
import io

import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import config
import dataset
import inference
from conftest import write_opt_mat, write_sar_mat


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _rand_pil(w=48, h=48, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8), "RGB")


# ---------------- on isleme ----------------
def test_preprocess_matches_training_eval_path(tmp_path):
    path = tmp_path / "a_opt.mat"
    write_opt_mat(path)
    pil = dataset.load_optical_image(path)
    t = inference.preprocess(pil)
    expected = TF.normalize(
        TF.resize(TF.to_tensor(pil), [config.IMG_SIZE, config.IMG_SIZE], antialias=True),
        config.IMAGENET_MEAN, config.IMAGENET_STD)
    assert t.shape == (3, config.IMG_SIZE, config.IMG_SIZE)
    assert torch.allclose(t, expected)


# ---------------- load_image_any ----------------
def test_load_image_any_mat_matches_dataset_reader(tmp_path):
    path = tmp_path / "a_opt.mat"
    write_opt_mat(path, h=40, w=30)
    img = inference.load_image_any(path.read_bytes(), "a_opt.mat")
    ref = dataset.load_optical_image(path)
    assert img.size == ref.size
    assert np.array_equal(np.asarray(img), np.asarray(ref))


def test_load_image_any_png_and_rgba():
    rgba = Image.new("RGBA", (30, 20), (200, 10, 10, 128))
    img = inference.load_image_any(_png_bytes(rgba), "a.png")
    assert img.mode == "RGB"
    assert img.size == (30, 20)


def test_to_rgb_modes():
    for mode in ("L", "P", "CMYK"):
        src = Image.new(mode, (10, 10))
        assert inference._to_rgb(src).mode == "RGB"
    # 16-bit: degerler 255'i asar; persentil olcekleme kirpmadan donusturmeli
    arr16 = np.linspace(0, 40000, 100, dtype=np.uint16).reshape(10, 10)
    out = inference._to_rgb(Image.fromarray(arr16))
    a = np.asarray(out)
    assert out.mode == "RGB" and a.dtype == np.uint8
    assert a.max() > 200 and a.min() < 50      # tum aralik kullanilmis, hep-beyaz degil


def test_load_image_any_sar_mat_rejected(tmp_path):
    path = tmp_path / "a_SAR.mat"
    write_sar_mat(path)
    with pytest.raises(inference.UploadError, match="x3"):
        inference.load_image_any(path.read_bytes(), "a_SAR.mat")


def test_load_image_any_garbage():
    with pytest.raises(inference.UploadError):
        inference.load_image_any(b"bozuk veri", "a.png")
    with pytest.raises(inference.UploadError):
        inference.load_image_any(b"bozuk veri", "a.mat")
    with pytest.raises(inference.UploadError, match="Desteklenmeyen"):
        inference.load_image_any(b"x", "a.gif")


def test_load_image_any_downscales_huge():
    big = Image.new("RGB", (3000, 1500), (10, 20, 30))
    img = inference.load_image_any(_png_bytes(big), "big.png")
    assert max(img.size) == inference.PRE_RESIZE
    assert img.size[0] / img.size[1] == pytest.approx(2.0, abs=0.01)


def test_load_image_any_max_side_none_keeps_size():
    big = Image.new("RGB", (3000, 1500), (10, 20, 30))
    img = inference.load_image_any(_png_bytes(big), "big.png", max_side=None)
    assert img.size == (3000, 1500)                 # 40 MP altinda: kucultme yok


# ---------------- tahmin ----------------
class _ConstModel(torch.nn.Module):
    """Her girdiye ayni logit'i donduren sahte model (TTA/fold ortalamasini sinar)."""

    def forward(self, inputs):
        n = inputs["opt"].shape[0]
        return torch.tensor([[0.0, 1.0]], device=inputs["opt"].device).repeat(n, 1)


def test_predict_pils_constant_model(monkeypatch):
    fake = _ConstModel().to(config.DEVICE)
    monkeypatch.setattr(inference, "get_models", lambda: [fake, fake])
    probs = inference.predict_pils([_rand_pil(seed=i) for i in range(3)])
    expected = torch.softmax(torch.tensor([0.0, 1.0]), 0)[1].item()
    assert probs.shape == (3,)
    assert probs == pytest.approx([expected] * 3, abs=1e-6)


class _CountingModel(torch.nn.Module):
    """Cagri sayan sahte model: fold1-only ve tek-gorunum kanitlari icin."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        n = inputs["opt"].shape[0]
        return torch.tensor([[0.0, 1.0]], device=inputs["opt"].device).repeat(n, 1)


def test_predict_pils_fast_uses_first_fold_only(monkeypatch):
    m0, m1 = _CountingModel().to(config.DEVICE), _CountingModel().to(config.DEVICE)
    monkeypatch.setattr(inference, "get_models", lambda: [m0, m1])
    pils = [_rand_pil(seed=i) for i in range(5)]
    probs = inference.predict_pils_fast(pils, batch_size=2)
    expected = torch.softmax(torch.tensor([0.0, 1.0]), 0)[1].item()
    assert probs.shape == (5,)
    assert probs == pytest.approx([expected] * 5, abs=1e-6)
    assert m0.calls == 3          # 5 goruntu / batch 2 -> 3 parca, TTA'siz tek gorunum
    assert m1.calls == 0          # fold2'ye hic dokunulmaz
    assert inference.predict_pils_fast([]).shape == (0,)


def test_gradcam_pils_shape_and_range(monkeypatch):
    from model_mm import build_mm_model
    model = build_mm_model(["opt"], pretrained=False)   # indirme yok, CI-guvenli
    model.eval()
    monkeypatch.setattr(inference, "get_models", lambda: [model])
    for _ in range(2):                                   # iki kez: hook temizligi kaniti
        cams = inference.gradcam_pils([_rand_pil(64, 64)])
        assert cams[0].shape == (config.IMG_SIZE, config.IMG_SIZE)
        assert cams[0].dtype == np.float32
        assert 0.0 <= cams[0].min() and cams[0].max() <= 1.0


# ---------------- kodlama ----------------
def test_overlay_and_thumb():
    pil = _rand_pil(60, 60)
    png = inference.overlay_cam(pil, np.zeros((config.IMG_SIZE, config.IMG_SIZE)))
    assert png[:4] == b"\x89PNG"
    uri = inference.thumb_data_uri(pil)
    assert uri.startswith("data:image/jpeg;base64,")
    assert inference.png_data_uri(png).startswith("data:image/png;base64,")
