"""Mozaik dogrulama: genis sahne boru hatti vs ground-truth karsilastirmasi.

make_mosaic.py'nin urettigi sahne + truth CSV uzerinde scene.analyze_scene'i
(in-process, HTTP'siz) calistirir; tespit recall/precision ve eslesen kutularda
siniflandirma metriklerini raporlar.

Kullanim:
    python evaluate_scene.py outputs/mosaic/demo.png outputs/mosaic/demo_truth.csv \
        --mode grid --window 96
"""
import argparse
import csv
import json
import os

from PIL import Image

import scene
from ensemble import DEPLOY_THRESHOLD


def iou_xywh(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_boxes(pred: list, truth: list, iou_thr: float = 0.5) -> list:
    """IoU >= esik ciftlerini IoU azalan sirada acgozlu eslestirir.

    pred/truth: (x,y,w,h) erisimli sozluk listeleri. Donen: (pred_i, truth_j).
    Her kutu en fazla bir kez eslesir.
    """
    pairs = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            iou = iou_xywh((p["x"], p["y"], p["w"], p["h"]),
                           (t["x"], t["y"], t["w"], t["h"]))
            if iou >= iou_thr:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_p, used_t, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_p or j in used_t:
            continue
        used_p.add(i)
        used_t.add(j)
        out.append((i, j))
    return out


def read_truth(path) -> list:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    return [{"id": r["id"], "cls": r["cls"], "x": int(r["x"]), "y": int(r["y"]),
             "w": int(r["w"]), "h": int(r["h"])} for r in rows]


def evaluate(scene_png, truth_csv, mode: str, window: int,
             threshold: float, iou_thr: float) -> dict:
    truth = read_truth(truth_csv)
    pil = Image.open(scene_png).convert("RGB")
    result = scene.analyze_scene(pil, mode=mode, threshold=threshold, window=window)
    pred = result["buildings"]

    matches = match_boxes(pred, truth, iou_thr)
    det_recall = len(matches) / len(truth) if truth else 0.0
    det_precision = len(matches) / len(pred) if pred else 0.0

    # eslesenlerde siniflandirma (tahmin: p >= threshold -> damaged)
    cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}   # pozitif sinif = damaged
    for i, j in matches:
        pred_dmg = pred[i]["p"] >= threshold
        true_dmg = truth[j]["cls"] == "damaged"
        if pred_dmg and true_dmg:
            cm["tp"] += 1
        elif pred_dmg:
            cm["fp"] += 1
        elif true_dmg:
            cm["fn"] += 1
        else:
            cm["tn"] += 1
    n_match = max(1, len(matches))
    cls_acc = (cm["tp"] + cm["tn"]) / n_match
    dmg_rec = cm["tp"] / max(1, cm["tp"] + cm["fn"])
    dmg_prec = cm["tp"] / max(1, cm["tp"] + cm["fp"])

    return {"mode": mode, "window": window if mode == "grid" else None,
            "threshold": threshold, "iou_thr": iou_thr,
            "n_truth": len(truth), "n_pred": len(pred), "n_matched": len(matches),
            "det_recall": round(det_recall, 3), "det_precision": round(det_precision, 3),
            "cls_acc_on_matched": round(cls_acc, 3),
            "damaged_recall": round(dmg_rec, 3), "damaged_precision": round(dmg_prec, 3),
            "confusion": cm, "timings": result["timings"]}


def main():
    ap = argparse.ArgumentParser(description="Mozaik uzerinde boru hatti dogrulama")
    ap.add_argument("scene_png")
    ap.add_argument("truth_csv")
    ap.add_argument("--mode", choices=["grid", "yolo"], default="grid")
    ap.add_argument("--window", type=int, default=scene.WINDOW_DEFAULT)
    ap.add_argument("--threshold", type=float, default=DEPLOY_THRESHOLD)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    r = evaluate(args.scene_png, args.truth_csv, args.mode, args.window,
                 args.threshold, args.iou)
    print(f"\n=== {args.mode} modu degerlendirmesi ===")
    print(f"gercek bina: {r['n_truth']} | tahmin kutusu: {r['n_pred']} "
          f"| eslesen: {r['n_matched']} (IoU>={args.iou})")
    print(f"tespit    recall {r['det_recall']:.3f} | precision {r['det_precision']:.3f}")
    print(f"siniflandirma (eslesenlerde, esik {args.threshold:.2f}): "
          f"dogruluk {r['cls_acc_on_matched']:.3f} | hasarli recall {r['damaged_recall']:.3f} "
          f"| hasarli precision {r['damaged_precision']:.3f}")
    print(f"karisiklik: {r['confusion']}")
    print(f"sureler: {r['timings']}")
    out = os.path.join("outputs", "mosaic", f"eval_{args.mode}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    print(f"json: {out}")


if __name__ == "__main__":
    main()
