"""Sentetik genis sahne ureteci: veri seti kesitlerini arkaplana yerlestirir.

Veri setinde gercek genis sahne yok; bu arac bilinen etiketli kesitlerden
"onlarca binali" bir sahne uretir (demo + nicel dogrulama). Cikti: PNG sahne
+ ground-truth CSV (id;cls;x;y;w;h).

Kullanim:
    python make_mosaic.py --n-buildings 40 --damaged-frac 0.3 --size 2400x1800 \
        --seed 42 --out outputs/mosaic/demo.png
"""
import argparse
import csv
import os

import numpy as np
from PIL import Image

import config
from dataset import build_index, load_optical_image

MARGIN = 8          # binalar arasi asgari bosluk (piksel)
MAX_TRIES = 60      # bina basina yerlestirme denemesi


def make_background(w: int, h: int, kind: str, rng) -> Image.Image:
    """Dusuk frekansli dokulu arkaplan (toprak/gri/gurultu)."""
    if kind == "noise":
        arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")
    base = np.array([122, 112, 96] if kind == "soil" else [128, 128, 128],
                    dtype=np.float32)
    low = rng.normal(0, 18, size=(24, 32, 3)).astype(np.float32)
    low_img = Image.fromarray(np.clip(low + base, 0, 255).astype(np.uint8), "RGB")
    bg = np.asarray(low_img.resize((w, h), Image.BILINEAR), dtype=np.float32)
    bg += rng.normal(0, 8, size=bg.shape)
    return Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8), "RGB")


def pick_chips(index, n: int, damaged_frac: float, rng) -> list:
    """Sinif oranina gore rastgele ornek secimi (tekrarsiz, gerekirse tekrarli)."""
    n_dmg = int(round(n * damaged_frac))
    n_int = n - n_dmg
    pools = {"damaged": [s for s in index if s["label"] == 1],
             "intact": [s for s in index if s["label"] == 0]}
    picked = []
    for cls, k in (("damaged", n_dmg), ("intact", n_int)):
        pool = pools[cls]
        if not pool:
            raise ValueError(f"Veri setinde {cls} ornegi yok.")
        replace = k > len(pool)
        idx = rng.choice(len(pool), size=k, replace=replace)
        picked += [pool[int(i)] for i in idx]
    rng.shuffle(picked)
    return picked


def _intersects(box, placed, margin: int) -> bool:
    x, y, w, h = box
    for px, py, pw, ph in placed:
        if (x - margin < px + pw and px - margin < x + w
                and y - margin < py + ph and py - margin < y + h):
            return True
    return False


def place_chips(bg: Image.Image, chips: list, rng,
                margin: int = MARGIN, max_tries: int = MAX_TRIES) -> list:
    """Kesitleri cakismasiz rastgele yerlestirir; sigmayan sahnede ValueError."""
    W, H = bg.size
    placed_rects = []
    rows = []
    for s in chips:
        chip = load_optical_image(s["path"])
        cw, ch = chip.size
        if cw + 2 * margin > W or ch + 2 * margin > H:
            continue                                  # sahneye sigmayan dev kesit: atla
        for _ in range(max_tries):
            x = int(rng.integers(margin, W - cw - margin + 1))
            y = int(rng.integers(margin, H - ch - margin + 1))
            if not _intersects((x, y, cw, ch), placed_rects, margin):
                break
        else:
            raise ValueError(
                f"Sahne cok kucuk: {len(placed_rects)} bina yerlesti, "
                f"{len(chips)} istendi. Sahneyi buyutun veya bina sayisini azaltin.")
        bg.paste(chip, (x, y))
        placed_rects.append((x, y, cw, ch))
        rows.append({"id": s["id"], "cls": s["cls"], "x": x, "y": y,
                     "w": cw, "h": ch})
    return rows


def write_truth_csv(path, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "cls", "x", "y", "w", "h"])
        for r in rows:
            w.writerow([r["id"], r["cls"], r["x"], r["y"], r["w"], r["h"]])


def build_mosaic(data_dir, n_buildings: int, damaged_frac: float,
                 size: tuple, bg_kind: str, seed: int,
                 out_png, out_csv) -> dict:
    """Testlerin ve CLI'nin ortak giris noktasi."""
    from pathlib import Path
    rng = np.random.default_rng(seed)
    index = build_index(Path(data_dir))              # path'ler mutlak saklanir
    bg = make_background(size[0], size[1], bg_kind, rng)
    chips = pick_chips(index, n_buildings, damaged_frac, rng)
    rows = place_chips(bg, chips, rng)
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    bg.save(out_png, "PNG")
    write_truth_csv(out_csv, rows)
    n_dmg = sum(1 for r in rows if r["cls"] == "damaged")
    return {"png": str(out_png), "csv": str(out_csv), "n": len(rows),
            "damaged": n_dmg, "intact": len(rows) - n_dmg,
            "size": size}


def main():
    ap = argparse.ArgumentParser(description="Sentetik genis sahne ureteci")
    ap.add_argument("--n-buildings", type=int, default=40)
    ap.add_argument("--damaged-frac", type=float, default=0.3)
    ap.add_argument("--size", default="2400x1800", help="GENISLIKxYUKSEKLIK")
    ap.add_argument("--bg", choices=["soil", "gray", "noise"], default="soil")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default=str(config.DATA_DIR))
    ap.add_argument("--out", default="outputs/mosaic/scene.png")
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    out_csv = os.path.splitext(args.out)[0] + "_truth.csv"
    info = build_mosaic(args.data_dir, args.n_buildings, args.damaged_frac,
                        (w, h), args.bg, args.seed, args.out, out_csv)
    print(f"sahne: {info['png']} ({w}x{h})")
    print(f"ground truth: {info['csv']}")
    print(f"bina: {info['n']} ({info['damaged']} hasarli, {info['intact']} saglam)")


if __name__ == "__main__":
    main()
