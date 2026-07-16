"""Genis sahne hasar haritalama motoru: tespit/izgara + iki asamali siniflandirma.

Iki mod:
  - "yolo": detector.py ile bina kutulari bulunur, her kutu chip'e cevrilip
    ensemble ile siniflandirilir. Tespitci opsiyoneldir (DetectorUnavailable).
  - "grid": sahne ortusen pencerelere bolunur, her pencere siniflandirilir;
    isi haritasi + esik ustu hucre kutulari uretilir. Bagimliliksiz, daima calisir.

Iki asamali hiz: once fold-1 TTA'siz hizli gecis, yalnizca p >= CANDIDATE_THR
adaylara tam 5-fold+TTA ensemble uygulanir.

Bu modul ultralytics import ETMEZ (yalnizca detector.py uzerinden dolayli).
"""
import base64
import codecs
import csv
import io
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

import config
import detector
import inference

# --- sabitler (tek gercek kaynagi) ---
TILE = 640                 # YOLO tile boyutu
TILE_OVERLAP = 0.20        # tile ortusme orani (stride = 512)
DET_TILE_BATCH = 8         # detector.predict_tiles cagri basina tile
YOLO_NMS_IOU = 0.45        # global kutu birlestirme
GRID_NMS_IOU = 0.30        # izgara hucre kutulari birlestirme
MIN_DET_SIDE = 16          # bundan kucuk kenarli tespitler atilir
MAX_BOXES = 600            # sahne basina kutu tavani (skora gore)
PAD_FRAC = 0.20            # chip cikarirken kutu cevresine pay
CHIP_MIN = 32              # chip minimum kenari
CANDIDATE_THR = 0.15       # hizli gecis aday esigi
FAST_BATCH = 256
FULL_BATCH = 128
WINDOW_MIN, WINDOW_DEFAULT, WINDOW_MAX, WINDOW_STEP = 64, 96, 192, 16
MAX_WINDOWS = 9000         # izgara pencere tavani (sure guvenligi)
MAX_SIDE = 8000            # sahne kenar tavani (40 MP siniri inference'ta)
DISPLAY_MAX = 1600         # tarayici gorunum JPEG'inin uzun kenari
STORE_MAX = 2              # bellekte tutulan sahne sayisi
HEAT_COLOR = (220, 38, 38)
HEAT_ALPHA_MAX = 220


# ---------------- geometri ----------------
def _starts(total: int, size: int, stride: int) -> list:
    """0'dan baslayan stride'li baslangiclar + son parca daima kenara yaslanir."""
    if total <= size:
        return [0]
    xs = list(range(0, total - size + 1, stride))
    if xs[-1] != total - size:
        xs.append(total - size)
    return xs


def tile_grid(w: int, h: int, tile: int = TILE, overlap: float = TILE_OVERLAP) -> list:
    """YOLO icin (x, y, x2, y2) tile listesi; kucuk goruntude tek parca."""
    if w <= tile and h <= tile:
        return [(0, 0, w, h)]
    stride = max(1, round(tile * (1 - overlap)))
    return [(x, y, min(x + tile, w), min(y + tile, h))
            for y in _starts(h, tile, stride)
            for x in _starts(w, tile, stride)]


def grid_windows(w: int, h: int, win: int):
    """Izgara pencereleri (satir-major) + (nx, ny). Isi haritasi reshape bu ikiliyle."""
    stride = max(1, win // 2)
    xs = _starts(w, min(win, w), stride)
    ys = _starts(h, min(win, h), stride)
    wins = [(x, y, min(x + win, w), min(y + win, h)) for y in ys for x in xs]
    return wins, len(xs), len(ys)


def check_grid(w: int, h: int, win: int) -> None:
    if not (WINDOW_MIN <= win <= WINDOW_MAX):
        raise inference.UploadError(
            f"Pencere boyutu {WINDOW_MIN}-{WINDOW_MAX} piksel arasında olmalı (verilen: {win}).")
    _, nx, ny = grid_windows(w, h, win)
    if nx * ny > MAX_WINDOWS:
        raise inference.UploadError(
            f"İzgara çok yoğun ({nx * ny} pencere > {MAX_WINDOWS}). "
            "Pencere boyutunu büyütün veya daha küçük bir sahne yükleyin.")


def chip_rect(box, w: int, h: int, pad_frac: float = PAD_FRAC,
              min_size: int = CHIP_MIN):
    """Kutu (x,y,bw,bh) -> siniflandirma chip'i (x1,y1,x2,y2).

    Egitim kesitleri binayi payli cerceveler: kutu %pad buyutulur, kareye
    tamamlanir, once ceviri ile sahneye sokulur, sigmiyorsa kirpilir.
    """
    x, y, bw, bh = box
    side = max(bw, bh) * (1 + 2 * pad_frac)
    side = max(side, min_size)
    cx, cy = x + bw / 2, y + bh / 2
    x1, y1 = round(cx - side / 2), round(cy - side / 2)
    x2, y2 = round(x1 + side), round(y1 + side)
    if x2 - x1 > w:                       # kare sahneden genis: kirp
        x1, x2 = 0, w
    else:                                  # ceviri ile sok
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if x2 > w:
            x1 -= x2 - w
            x2 = w
    if y2 - y1 > h:
        y1, y2 = 0, h
    else:
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if y2 > h:
            y1 -= y2 - h
            y2 = h
    return int(x1), int(y1), int(x2), int(y2)


def nms_keep(boxes_xywh: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """xywh kutulara torchvision NMS uygular, kalan indisleri dondurur."""
    if len(boxes_xywh) == 0:
        return np.zeros(0, dtype=np.int64)
    b = np.asarray(boxes_xywh, dtype=np.float32)
    xyxy = np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)
    keep = nms(torch.from_numpy(xyxy), torch.from_numpy(np.asarray(scores, np.float32)),
               iou_thr)
    return keep.numpy().astype(np.int64)


# ---------------- iki asamali siniflandirma ----------------
def two_stage_probs(scene_pil: Image.Image, rects: list,
                    candidate_thr: float = CANDIDATE_THR,
                    fast_batch: int = FAST_BATCH, full_batch: int = FULL_BATCH):
    """rects: (x1,y1,x2,y2) listesi -> (p dizisi, istatistik sozlugu).

    Crop'lar parca basina tembel uretilir (5000 pencereyi ayni anda tutmayiz).
    Aday olmayanlar fold-1 olasiliginda kalir (esik altinda gorsel onemi yok).
    """
    n = len(rects)
    p = np.zeros(n, dtype=np.float32)
    t0 = time.time()
    for i in range(0, n, fast_batch):
        chunk = rects[i:i + fast_batch]
        pils = [scene_pil.crop(r) for r in chunk]
        p[i:i + len(chunk)] = inference.predict_pils_fast(pils, batch_size=fast_batch)
        print(f"  [sahne] hizli gecis {min(i + fast_batch, n)}/{n}")
    fast_s = time.time() - t0

    idx = np.where(p >= candidate_thr)[0]
    t0 = time.time()
    for i in range(0, len(idx), full_batch):
        sel = idx[i:i + full_batch]
        pils = [scene_pil.crop(rects[j]) for j in sel]
        p[sel] = inference.predict_pils(pils)
        print(f"  [sahne] tam ensemble {min(i + full_batch, len(idx))}/{len(idx)}")
    full_s = time.time() - t0
    return p, {"fast_s": round(fast_s, 2), "full_s": round(full_s, 2),
               "n_windows": n, "n_refined": int(len(idx))}


# ---------------- isi haritasi ----------------
def heatmap_png(prob_grid: np.ndarray, disp_w: int, disp_h: int,
                p_lo: float = CANDIDATE_THR) -> bytes:
    """(ny,nx) olasilik izgarasi -> gorunum boyutunda yari saydam kirmizi PNG."""
    a = np.clip((prob_grid - p_lo) / (1 - p_lo), 0, 1)
    alpha = (a * HEAT_ALPHA_MAX).round().astype(np.uint8)
    alpha_img = Image.fromarray(alpha, mode="L").resize((disp_w, disp_h),
                                                        Image.BILINEAR)
    rgba = Image.new("RGBA", (disp_w, disp_h), HEAT_COLOR + (0,))
    rgba.putalpha(alpha_img)
    buf = io.BytesIO()
    rgba.save(buf, "PNG")
    return buf.getvalue()


# ---------------- sahne deposu ----------------
class SceneStore:
    """Bellek ici LRU sahne deposu (tek kullanicili demo icin yeterli)."""

    def __init__(self, max_scenes: int = STORE_MAX):
        self.max = max_scenes
        self._d = OrderedDict()
        self._lock = threading.Lock()

    def put(self, entry: dict) -> str:
        sid = uuid.uuid4().hex[:12]
        entry["id"] = sid
        with self._lock:
            self._d[sid] = entry
            while len(self._d) > self.max:
                self._d.popitem(last=False)
        return sid

    def get(self, sid: str):
        with self._lock:
            e = self._d.get(sid)
            if e is not None:
                self._d.move_to_end(sid)
            return e


STORE = SceneStore()


# ---------------- ana analiz ----------------
def _display_jpeg(pil: Image.Image):
    """Tarayici gorunumu icin kucultulmus JPEG data-URI + olcek."""
    w, h = pil.size
    scale = min(DISPLAY_MAX / max(w, h), 1.0)
    disp = pil if scale == 1.0 else pil.resize(
        (round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    disp.save(buf, "JPEG", quality=85)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    return uri, disp.size, scale


def analyze_scene(pil: Image.Image, mode: str, threshold: float, window: int) -> dict:
    """Sahneyi analiz eder, STORE'a kaydeder, API yanit sozlugunu dondurur."""
    if mode not in ("yolo", "grid"):
        raise inference.UploadError(f"Geçersiz mod: {mode!r} (yolo veya grid olmalı).")
    W, H = pil.size
    if max(W, H) > MAX_SIDE:
        raise inference.UploadError(
            f"Sahne çok büyük ({W}x{H}). Kenar üst sınırı {MAX_SIDE} piksel.")

    t_total = time.time()
    boxes = []
    heat_bytes = None
    detect_s = 0.0

    if mode == "yolo":
        t0 = time.time()
        tiles = tile_grid(W, H)
        all_xywh, all_conf = [], []
        for i in range(0, len(tiles), DET_TILE_BATCH):
            batch = tiles[i:i + DET_TILE_BATCH]
            results = detector.predict_tiles([pil.crop(t) for t in batch])
            for (tx, ty, _, _), det in zip(batch, results):
                for x1, y1, x2, y2, cf in det:
                    bw, bh = x2 - x1, y2 - y1
                    if bw < MIN_DET_SIDE or bh < MIN_DET_SIDE:
                        continue
                    all_xywh.append((tx + x1, ty + y1, bw, bh))
                    all_conf.append(cf)
            print(f"  [sahne] yolo tile {min(i + DET_TILE_BATCH, len(tiles))}/{len(tiles)}")
        detect_s = time.time() - t0
        if all_xywh:
            xywh = np.array(all_xywh, dtype=np.float32)
            conf = np.array(all_conf, dtype=np.float32)
            keep = nms_keep(xywh, conf, YOLO_NMS_IOU)
            if len(keep) > MAX_BOXES:                    # skora gore tavan
                keep = keep[np.argsort(-conf[keep])[:MAX_BOXES]]
            xywh, conf = xywh[keep], conf[keep]
            rects = [chip_rect(tuple(b), W, H) for b in xywh]
            probs, stats = two_stage_probs(pil, rects)
            for k, (b, cf, p) in enumerate(zip(xywh, conf, probs), start=1):
                boxes.append({"id": k, "x": int(b[0]), "y": int(b[1]),
                              "w": int(round(b[2])), "h": int(round(b[3])),
                              "p": float(p), "source": "yolo",
                              "det_conf": round(float(cf), 3)})
        else:
            stats = {"fast_s": 0.0, "full_s": 0.0, "n_windows": 0, "n_refined": 0}
    else:  # grid
        check_grid(W, H, window)
        wins, nx, ny = grid_windows(W, H, window)
        probs, stats = two_stage_probs(pil, wins)

    uri, (dw, dh), scale = _display_jpeg(pil)

    if mode == "grid":
        heat_bytes = heatmap_png(probs.reshape(ny, nx), dw, dh)
        cand = np.where(probs >= CANDIDATE_THR)[0]
        if len(cand):
            xywh = np.array([(wins[j][0], wins[j][1],
                              wins[j][2] - wins[j][0], wins[j][3] - wins[j][1])
                             for j in cand], dtype=np.float32)
            keep = nms_keep(xywh, probs[cand], GRID_NMS_IOU)
            if len(keep) > MAX_BOXES:
                keep = keep[np.argsort(-probs[cand][keep])[:MAX_BOXES]]
            for k, ki in enumerate(keep, start=1):
                j = int(cand[ki])
                x1, y1, x2, y2 = wins[j]
                boxes.append({"id": k, "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
                              "p": float(probs[j]), "source": "grid",
                              "det_conf": None})

    det_ok, det_err = detector.availability()
    entry = {
        "pil": pil, "width": W, "height": H, "mode": mode,
        "window": window if mode == "grid" else None,
        "display_scale": scale, "boxes": boxes,
        "heatmap_png": heat_bytes,
        "next_box_id": (max((b["id"] for b in boxes), default=0) + 1),
        "threshold": threshold, "created": time.time(),
    }
    sid = STORE.put(entry)

    return {
        "scene_id": sid, "width": W, "height": H, "mode": mode,
        "threshold": threshold, "window": entry["window"],
        "display": {"uri": uri, "scale": scale, "width": dw, "height": dh},
        "detector_available": det_ok, "detector_error": det_err,
        "buildings": [{**b, "p": round(b["p"], 6)} for b in boxes],
        "heatmap": ("data:image/png;base64," + base64.b64encode(heat_bytes).decode("ascii"))
                   if heat_bytes else None,
        "timings": {"detect_s": round(detect_s, 2),
                    "classify_s": round(stats["fast_s"] + stats["full_s"], 2),
                    "total_s": round(time.time() - t_total, 2),
                    "n_windows": stats["n_windows"],
                    "n_refined": stats["n_refined"]},
    }


# ---------------- takip islemleri ----------------
def classify_manual_box(entry: dict, x, y, w, h) -> dict:
    """Elle cizilen kutuyu dogrular, tam ensemble ile siniflandirir, depoya ekler."""
    try:
        x, y, w, h = int(x), int(y), int(w), int(h)
    except (TypeError, ValueError):
        raise inference.UploadError("Kutu koordinatları tamsayı olmalı.")
    W, H = entry["width"], entry["height"]
    if w < MIN_DET_SIDE or h < MIN_DET_SIDE:
        raise inference.UploadError(f"Kutu çok küçük (en az {MIN_DET_SIDE}x{MIN_DET_SIDE} piksel).")
    if x < 0 or y < 0 or x + w > W or y + h > H:
        raise inference.UploadError("Kutu sahne sınırlarının dışında.")
    rect = chip_rect((x, y, w, h), W, H)
    p = float(inference.predict_pils([entry["pil"].crop(rect)])[0])
    box = {"id": entry["next_box_id"], "x": x, "y": y, "w": w, "h": h,
           "p": round(p, 6), "source": "manual", "det_conf": None}
    entry["next_box_id"] += 1
    entry["boxes"].append(box)
    return box


def cam_for_box(entry: dict, box_id: int) -> bytes:
    """Kutunun chip'i icin 5-fold ortalamali Grad-CAM kaplama PNG'si."""
    box = next((b for b in entry["boxes"] if b["id"] == box_id), None)
    if box is None:
        raise KeyError(box_id)
    if box["source"] == "grid":                    # izgara: pencerenin kendisi
        rect = (box["x"], box["y"], box["x"] + box["w"], box["y"] + box["h"])
    else:                                          # siniflandirmayla ayni cerceve
        rect = chip_rect((box["x"], box["y"], box["w"], box["h"]),
                         entry["width"], entry["height"])
    chip = entry["pil"].crop(rect)
    cam = inference.gradcam_pils([chip])[0]
    return inference.overlay_cam(chip, cam)


def render_annotated(entry: dict, threshold: float, with_heatmap: bool = True) -> bytes:
    """Tam cozunurluk isaretli sahne PNG'si (hasar haritasi ciktisi)."""
    img = entry["pil"].convert("RGBA")
    W, H = img.size
    if with_heatmap and entry["heatmap_png"]:
        heat = Image.open(io.BytesIO(entry["heatmap_png"])).resize((W, H),
                                                                   Image.BILINEAR)
        img = Image.alpha_composite(img, heat)
    draw = ImageDraw.Draw(img)
    lw = max(2, round(max(W, H) / 800))
    try:
        font = ImageFont.truetype("arial.ttf", max(14, lw * 6))
    except Exception:
        font = ImageFont.load_default()
    for b in entry["boxes"]:
        damaged = b["p"] >= threshold
        if b["source"] == "grid" and not damaged:
            continue                                # izgarada esik alti gurultudur
        color = (220, 38, 38, 255) if damaged else (22, 163, 74, 255)
        x2, y2 = b["x"] + b["w"], b["y"] + b["h"]
        draw.rectangle([b["x"], b["y"], x2, y2], outline=color, width=lw)
        draw.text((b["x"] + lw + 1, b["y"] + lw), f"#{b['id']}",
                  fill=color, font=font)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "PNG")
    return buf.getvalue()


def csv_bytes(entry: dict, threshold: float) -> bytes:
    """Kutu listesi CSV'si (noktali virgul + BOM: Turkce Excel uyumu)."""
    src_tr = {"yolo": "yolo", "grid": "izgara", "manual": "manuel"}
    sio = io.StringIO(newline="")
    wcsv = csv.writer(sio, delimiter=";")
    wcsv.writerow(["id", "kaynak", "x", "y", "genislik", "yukseklik",
                   "p_hasarli", "tahmin", "esik"])
    for b in entry["boxes"]:
        wcsv.writerow([b["id"], src_tr[b["source"]], b["x"], b["y"], b["w"], b["h"],
                       f"{b['p']:.4f}",
                       "HASARLI" if b["p"] >= threshold else "SAGLAM",
                       f"{threshold:.2f}"])
    return codecs.BOM_UTF8 + sio.getvalue().encode("utf-8")
