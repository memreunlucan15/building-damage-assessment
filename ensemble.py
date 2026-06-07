"""Dagitim (deployment) ensemble: 5 optik fold modelinin olasilik ortalamasi.

Yeni bir bina goruntusu icin nihai tahmin araci budur. Raporlanan performans
tahmini 5-fold CV ortalamasidir (outputs/cv_opt/cv_results.json -> summary);
ensemble burada ayrica olculmez (ayni veride sizinti olurdu).

Kullanim:
    from ensemble import load_ensemble, predict_probs_ensemble, DEPLOY_THRESHOLD
    models = load_ensemble()
    probs, labels = predict_probs_ensemble(models, samples)   # samples: build_index() ogeleri
"""
import json

import numpy as np
import torch

import config
from engine_mm import predict_probs
from model_mm import build_mm_model

MODALITIES = ["opt"]
FOLD_DIR = config.OUTPUT_DIR / "cv_opt"


def _deploy_threshold():
    try:
        d = json.load(open(FOLD_DIR / "cv_results.json"))
        return float(d["oof"]["threshold"])
    except Exception:
        return 0.5


DEPLOY_THRESHOLD = _deploy_threshold()


def load_ensemble():
    """5 fold modelini yukleyip degerlendirme moduna alir."""
    models = []
    for f in sorted(FOLD_DIR.glob("fold*.pt")):
        ckpt = torch.load(f, map_location=config.DEVICE)
        model = build_mm_model(MODALITIES, pretrained=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models.append(model)
    if not models:
        raise FileNotFoundError(f"Fold modeli bulunamadi: {FOLD_DIR}")
    return models


@torch.no_grad()
def predict_probs_ensemble(models, samples):
    """Her modelin (TTA'li) damaged olasiligini ortalar."""
    probs = None
    labels = None
    for m in models:
        p, y = predict_probs(m, samples, MODALITIES)
        probs = p if probs is None else probs + p
        labels = y
    return probs / len(models), labels


def predict_labels(models, samples, threshold=None):
    threshold = DEPLOY_THRESHOLD if threshold is None else threshold
    probs, labels = predict_probs_ensemble(models, samples)
    preds = (probs >= threshold).astype(int)
    return preds, probs, labels


if __name__ == "__main__":
    from dataset import build_index
    config.set_seed()
    models = load_ensemble()
    print(f"Yuklenen model sayisi: {len(models)} | dagitim esigi: {DEPLOY_THRESHOLD:.2f}")

    # Demo: kucuk dengeli bir ornek kume uzerinde ensemble'i calistir (sadece gosterim).
    idx = build_index()
    dmg = [s for s in idx if s["label"] == 1][:10]
    intact = [s for s in idx if s["label"] == 0][:10]
    demo = dmg + intact
    preds, probs, labels = predict_labels(models, demo)
    name = {0: "intact", 1: "damaged"}
    print("\n  id              gercek     tahmin     p(damaged)")
    for s, pr, pb, y in zip(demo, preds, probs, labels):
        mark = "OK " if pr == y else "X  "
        print(f"  {s['id']:14s} {name[int(y)]:9s}  {name[int(pr)]:9s}  {pb:.3f}  {mark}")
    acc = (preds == labels).mean()
    print(f"\n  (demo dogrulugu: {acc:.2f} - bu sadece gosterim, performans tahmini DEGIL)")
    print("  Raporlanan performans: 5-fold CV ortalamasi (outputs/cv_opt/cv_results.json)")
