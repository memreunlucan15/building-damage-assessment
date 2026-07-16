"""Web arayuzu icin cikarim cekirdegi: goruntu okuma, on isleme, ensemble tahmin, Grad-CAM.

Dagitim modeli ensemble.load_ensemble() ile yuklenen 5 optik fold'dur; on isleme
dataset_mm.py'nin eval yoluyla, TTA engine.py'nin 3-goruntu semasiyla birebirdir.
Kullaniciya gosterilecek girdi hatalari UploadError (Turkce mesajli) olarak yukselir.
"""
import base64
import io
import threading

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from matplotlib import colormaps
from PIL import Image

import config
from ensemble import FOLD_DIR, load_ensemble
from evaluate import GradCAM

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".mat"}
MAX_PIXELS = 40_000_000   # 40 MP ustu reddedilir
PRE_RESIZE = 1024         # daha buyuk goruntulerde en uzun kenar bu boyuta indirilir

_LOCK = threading.Lock()  # tembel yukleme + tum torch cagrilari tek kilit altinda
_MODELS = None


class UploadError(ValueError):
    """Kullaniciya gosterilecek Turkce mesajli girdi hatasi."""


# ---------------- Model yukleme (tembel tekil) ----------------
def fold_count() -> int:
    """Fold checkpoint dosyasi sayisi (modelleri yuklemeden)."""
    return len(list(FOLD_DIR.glob("fold*.pt")))


def models_loaded() -> bool:
    return _MODELS is not None


def get_models():
    """Ensemble'i ilk cagrida yukler; sonraki cagrilar onbellekten doner."""
    global _MODELS
    if _MODELS is None:
        with _LOCK:
            if _MODELS is None:            # double-checked: kilidi bekleyen ikinci istek
                _MODELS = load_ensemble()  # FileNotFoundError yayilir (rota 503 dondurur)
    return _MODELS


# ---------------- Goruntu okuma ----------------
def _mat_to_pil(data: bytes) -> Image.Image:
    """Yuklenen _opt.mat baytlarini (anahtar x3) PIL RGB'ye cevirir."""
    try:
        f = h5py.File(io.BytesIO(data), "r")
    except OSError:
        raise UploadError("Geçersiz .mat dosyası: HDF5 (MATLAB v7.3) biçimi değil.")
    with f:
        if "x3" not in f:
            keys = ", ".join(sorted(f.keys())) or "boş"
            msg = f".mat içinde optik anahtar 'x3' yok (bulunanlar: {keys})."
            if "x1" in f:
                msg += " Bu bir SAR dosyası görünüyor; optik (*_opt.mat) dosyası yükleyin."
            raise UploadError(msg)
        arr = np.asarray(f["x3"][()])
    if arr.ndim == 3 and arr.shape[0] == 3:        # (3,H,W) -> (H,W,3), dataset.py ile ayni
        arr = arr.transpose(1, 2, 0)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise UploadError(f"Beklenmeyen optik veri şekli: {tuple(arr.shape)} (beklenen (3,H,W)).")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _to_rgb(img: Image.Image) -> Image.Image:
    """Her PIL modunu RGB'ye cevirir; 16/32-bit tek kanalda persentil olcekleme yapar
    (naif convert 255'e kirpip goruntuyu bembeyaz yapardi)."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        lo, hi = np.percentile(arr, [1, 99])
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max() or 1.0)
        arr = np.clip((arr - lo) / (hi - lo + 1e-8), 0, 1) * 255
        return Image.fromarray(arr.astype(np.uint8), "L").convert("RGB")
    if img.mode != "RGB":                          # RGBA/L/P/CMYK...; alfa atilir
        return img.convert("RGB")
    return img


def load_image_any(data: bytes, filename: str, max_side: int | None = PRE_RESIZE) -> Image.Image:
    """Yuklenen baytlari uzantiya gore acar (PNG/JPG/TIF veya _opt.mat) -> PIL RGB.

    max_side=None: on-kucultme atlanir (genis sahne analizi tam cozunurluk ister);
    40 MP ust siniri her durumda uygulanir.
    """
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXT:
        raise UploadError(f"Desteklenmeyen uzantı: {ext or '(yok)'}. "
                          "İzin verilenler: PNG, JPG, TIF, MAT.")
    if ext == ".mat":
        img = _mat_to_pil(data)
    else:
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:
            raise UploadError("Görüntü açılamadı (bozuk veya desteklenmeyen dosya).")
        img = _to_rgb(img)
    w, h = img.size
    if w * h > MAX_PIXELS:
        raise UploadError(f"Görüntü çok büyük ({w}x{h}). Üst sınır 40 megapiksel.")
    if max_side and max(w, h) > max_side:          # nihai 224 kucultme preprocess'te
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


# ---------------- On isleme + tahmin ----------------
def preprocess(pil: Image.Image) -> torch.Tensor:
    """Egitimdeki eval yolu ile birebir: to_tensor -> resize(224, antialias) -> normalize."""
    t = TF.to_tensor(pil)
    t = TF.resize(t, [config.IMG_SIZE, config.IMG_SIZE], antialias=True)
    return TF.normalize(t, config.IMAGENET_MEAN, config.IMAGENET_STD)


def predict_pils(pils: list) -> np.ndarray:
    """PIL listesi -> p(damaged) dizisi. 3-goruntu TTA x fold sayisi ortalamasi."""
    models = get_models()
    x = torch.stack([preprocess(p) for p in pils]).to(config.DEVICE)
    with _LOCK, torch.no_grad():
        views = [x, torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]  # engine.py ile ayni
        total = torch.zeros(x.size(0), device=config.DEVICE)
        for m in models:
            for v in views:
                total += F.softmax(m({"opt": v}), dim=1)[:, config.POSITIVE_IDX]
        probs = total / (len(models) * len(views))
    return probs.cpu().numpy()


def predict_pils_fast(pils: list, batch_size: int = 256) -> np.ndarray:
    """Genis sahne on eleme: yalnizca fold-1, TTA'siz, parcali toplu tahmin.

    Tam ensemble'a (predict_pils) gore ~15x hizli; dusuk olasilikli pencereleri
    elemek icin kullanilir, nihai karar tam ensemble ile verilir.
    """
    if not pils:
        return np.zeros(0, dtype=np.float32)
    model = get_models()[0]
    out = []
    for i in range(0, len(pils), batch_size):
        chunk = pils[i:i + batch_size]
        x = torch.stack([preprocess(p) for p in chunk]).to(config.DEVICE)
        with _LOCK, torch.no_grad():
            probs = F.softmax(model({"opt": x}), dim=1)[:, config.POSITIVE_IDX]
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


# ---------------- Grad-CAM ----------------
class _SingleBranch(torch.nn.Module):
    """MultiModalNet'i duz tensor girdili modele sarar (GradCAM dict bilmez)."""

    def __init__(self, mm):
        super().__init__()
        self.mm = mm

    def forward(self, x):
        return self.mm({"opt": x})


def gradcam_pils(pils: list) -> list:
    """Her goruntu icin fold'lar uzerinden ortalanmis (224,224) [0,1] isi haritasi.

    Gosterilen olasilik ensemble'a ait oldugundan aciklama da ensemble ortalamasidir.
    """
    models = get_models()
    x = torch.stack([preprocess(p) for p in pils]).to(config.DEVICE)
    sums = [np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.float64)
            for _ in pils]
    with _LOCK:
        for m in models:
            cam_obj = GradCAM(_SingleBranch(m), m.branches["opt"].layer4[-1])
            try:
                with torch.enable_grad():
                    for i in range(len(pils)):
                        sums[i] += cam_obj(x[i:i + 1], config.POSITIVE_IDX)
            finally:
                cam_obj.remove()                   # hook'lar paylasilan modelde kalmasin
    out = []
    for s in sums:
        s /= len(models)
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        out.append(s.astype(np.float32))
    return out


def overlay_cam(pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> bytes:
    """Isi haritasini 224'e kucultulmus gorselin uzerine bindirir -> PNG baytlari."""
    base = np.asarray(pil.resize((config.IMG_SIZE, config.IMG_SIZE)), dtype=np.float64)
    jet = colormaps["jet"](cam)[..., :3] * 255.0
    out = ((1 - alpha) * base + alpha * jet).round().clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, "PNG")
    return buf.getvalue()


# ---------------- Kodlama yardimcilari ----------------
def thumb_data_uri(pil: Image.Image, size: int = 160) -> str:
    """Kucuk JPEG onizleme -> data URI (tarayici .mat/TIF gosteremez, sunucu uretir)."""
    im = pil.copy()
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def png_data_uri(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
