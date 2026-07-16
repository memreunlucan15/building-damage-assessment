"""YOLO bina tespit sarmalayicisi (opsiyonel ultralytics bagimliligi).

ultralytics'e dokunan TEK dosya budur: paket kurulu degilse modul yine import
edilir, availability() sebep bildirir ve genis sahne analizi izgara moduyla
calismaya devam eder. Agirlik: HF keremberke/yolov8m-building-segmentation
(uydu goruntusu, tek sinif 'Building', box mAP@0.5 ~0.62).

CLI:
    python detector.py --download   # agirligi indir
    python detector.py --check      # yukle + dummy predict
"""
import threading

import numpy as np

import config

try:
    from ultralytics import YOLO          # opsiyonel bagimlilik
    _IMPORT_ERR = None
except Exception as e:                    # ImportError + gecisli hatalar
    YOLO = None
    _IMPORT_ERR = f"{type(e).__name__}: {e}"

WEIGHTS_DIR = config.ROOT / "weights" / "yolo_building"
WEIGHTS_PATH = WEIGHTS_DIR / "best.pt"
HF_REPO = "keremberke/yolov8m-building-segmentation"
HF_FILE = "best.pt"
DET_CONF = 0.25
DET_TILE_IOU = 0.45

_DET = None
_DET_LOCK = threading.Lock()
_LOAD_ERR = None                          # kalici yukleme hatasi (tekrar denenmez)


class DetectorUnavailable(RuntimeError):
    """Tespitci kullanilamiyor: Turkce, kullaniciya gosterilebilir mesaj."""


def importable() -> bool:
    return YOLO is not None


def weights_present() -> bool:
    return WEIGHTS_PATH.exists()


def availability():
    """(hazir_mi, sebep) — modeli YUKLEMEDEN hizli durum kontrolu."""
    if not importable():
        return False, ("YOLO bina tespiti için 'ultralytics' paketi kurulu değil "
                       "(pip install ultralytics). İzgara modunu kullanabilirsiniz.")
    if not weights_present():
        return False, ("YOLO ağırlık dosyası yok (weights/yolo_building/best.pt). "
                       "İndirmek için: python detector.py --download")
    if _LOAD_ERR is not None:
        return False, f"YOLO modeli yüklenemedi: {_LOAD_ERR}"
    return True, None


def ensure_weights(auto_download: bool = True) -> None:
    if weights_present():
        return
    if not auto_download:
        raise DetectorUnavailable("YOLO ağırlık dosyası bulunamadı "
                                  "(weights/yolo_building/best.pt).")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise DetectorUnavailable("Ağırlık indirmek için 'huggingface_hub' gerekli "
                                  "(pip install huggingface_hub).")
    print("[detector] yolo agirliklari indiriliyor (HF, ~52 MB)...")
    try:
        src = hf_hub_download(HF_REPO, HF_FILE)
    except Exception as e:
        raise DetectorUnavailable(f"YOLO ağırlığı indirilemedi: {type(e).__name__}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(src, WEIGHTS_PATH)
    print(f"[detector] indirildi: {WEIGHTS_PATH}")


def get_detector():
    """Tembel tekil YOLO modeli; basarisizlik kalici olarak isaretlenir."""
    global _DET, _LOAD_ERR
    if _DET is not None:
        return _DET
    ok, reason = availability()
    if not ok:
        raise DetectorUnavailable(reason)
    with _DET_LOCK:
        if _DET is None:
            try:
                _DET = YOLO(str(WEIGHTS_PATH))   # ultralytics 8.4.92 ile dogrulandi
            except Exception as e:
                _LOAD_ERR = f"{type(e).__name__}"
                raise DetectorUnavailable(f"YOLO modeli yüklenemedi: {type(e).__name__}")
    return _DET


def predict_tiles(tiles: list, conf: float = DET_CONF) -> list:
    """PIL tile listesi -> her tile icin (M,5) float32 [x1,y1,x2,y2,conf] dizisi.

    Girdi PIL RGB olmalidir (numpy BGR karisikligina girilmez); koordinatlar
    tile'in kendi pikselleridir, sahneye ofsetleme cagiranin isidir.
    """
    det = get_detector()
    with _DET_LOCK:
        results = det.predict(tiles, conf=conf, iou=DET_TILE_IOU,
                              imgsz=640, verbose=False, device=str(config.DEVICE))
    out = []
    for r in results:
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        cf = r.boxes.conf.cpu().numpy().astype(np.float32).reshape(-1, 1)
        out.append(np.hstack([xyxy, cf]) if len(xyxy) else np.zeros((0, 5), np.float32))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="YOLO bina tespitcisi yardimci araci")
    ap.add_argument("--download", action="store_true", help="agirligi HF'ten indir")
    ap.add_argument("--check", action="store_true", help="yukle + dummy predict")
    args = ap.parse_args()
    if args.download:
        ensure_weights(auto_download=True)
    if args.check:
        from PIL import Image
        det = get_detector()
        res = predict_tiles([Image.new("RGB", (640, 640), (120, 110, 95))])
        print(f"model OK | siniflar: {det.names} | dummy kutu: {res[0].shape[0]}")
    if not (args.download or args.check):
        ok, reason = availability()
        print(f"hazir: {ok}" + (f" | sebep: {reason}" if reason else ""))
