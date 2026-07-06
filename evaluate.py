"""Degerlendirme: metrikler, confusion matrix, esik ayari, Grad-CAM."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             classification_report, confusion_matrix,
                             fbeta_score, precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import DataLoader

import config
from dataset import QuickQuakeDataset, get_transforms, load_optical_image
from model import build_model
from split import load_split


@torch.no_grad()
def get_probs(model, samples):
    """Her ornek icin damaged (sinif 1) olasiligini ve gercek etiketi dondurur."""
    ds = QuickQuakeDataset(samples, train=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(config.DEVICE)
        p = F.softmax(model(x), dim=1)[:, config.POSITIVE_IDX]
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def best_threshold(probs, labels, beta=config.THRESHOLD_BETA):
    """Val uzerinde damaged F-beta'yi (beta>1 -> recall odakli) maksimize eden esik."""
    best_t, best_fb = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        fb = fbeta_score(labels, (probs >= t).astype(int), beta=beta,
                         pos_label=config.POSITIVE_IDX, zero_division=0)
        if fb > best_fb:
            best_fb, best_t = fb, t
    return best_t, best_fb


def report_at(probs, labels, threshold, tag):
    preds = (probs >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, pos_label=config.POSITIVE_IDX, average="binary", zero_division=0)
    acc = (preds == labels).mean()
    print(f"\n=== {tag} (esik={threshold:.2f}) ===")
    print(f"accuracy {acc:.3f} | damaged precision {p:.3f} recall {r:.3f} F1 {f1:.3f}")
    print(classification_report(labels, preds, target_names=config.CLASSES, zero_division=0))
    return {"threshold": float(threshold), "accuracy": float(acc),
            "precision": float(p), "recall": float(r), "f1": float(f1)}


def plot_confusion(labels, preds, path):
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASSES)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (test)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Confusion matrix: {path}")


# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        self._handles = [
            target_layer.register_forward_hook(self._fwd),
            target_layer.register_full_backward_hook(self._bwd),
        ]

    def remove(self):
        """Hook'lari kaldirir; model baska yerde tekrar kullanilacaksa cagrilmali."""
        for h in self._handles:
            h.remove()

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def gradcam_grid(model, samples, probs, threshold, n=6):
    cam = GradCAM(model, model.layer4[-1])
    tf = get_transforms(train=False)
    # Once gercek damaged ornekleri sec (ilgi cekici olanlar)
    chosen = [s for s in samples if s["label"] == 1][:n]
    if len(chosen) < n:
        chosen += [s for s in samples if s["label"] == 0][:n - len(chosen)]
    cols = len(chosen)
    fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 5.2))
    if cols == 1:
        axes = axes.reshape(2, 1)
    for j, s in enumerate(chosen):
        img = load_optical_image(s["path"]).resize((config.IMG_SIZE, config.IMG_SIZE))
        x = tf(load_optical_image(s["path"])).unsqueeze(0).to(config.DEVICE)
        heat = cam(x, config.POSITIVE_IDX)
        with torch.no_grad():
            p = F.softmax(model(x), 1)[0, config.POSITIVE_IDX].item()
        pred = "damaged" if p >= threshold else "intact"
        axes[0, j].imshow(img); axes[0, j].axis("off")
        axes[0, j].set_title(f"gercek={s['cls']}", fontsize=9)
        axes[1, j].imshow(img); axes[1, j].imshow(heat, cmap="jet", alpha=0.45)
        axes[1, j].axis("off")
        axes[1, j].set_title(f"tahmin={pred} ({p:.2f})", fontsize=9)
    fig.suptitle("Grad-CAM (ust: girdi, alt: damaged isi haritasi)")
    fig.tight_layout()
    out = config.OUTPUT_DIR / "gradcam.png"
    fig.savefig(out, dpi=130)
    print(f"Grad-CAM: {out}")
    cam.remove()


def main():
    config.set_seed()
    parts = load_split()
    model = build_model(num_classes=2, pretrained=False)
    ckpt = torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Yuklendi: {config.BEST_MODEL_PATH} (epoch {ckpt['epoch']}, val F1 {ckpt['val_f1']:.3f})")

    # Esik val uzerinde ayarlanir, test'e uygulanir (veri sizintisi yok).
    beta = config.THRESHOLD_BETA
    val_probs, val_labels = get_probs(model, parts["val"])
    thr, val_fb = best_threshold(val_probs, val_labels, beta)
    print(f"Val uzerinde secilen esik: {thr:.2f} (val F{beta:g} {val_fb:.3f}, recall odakli)")

    test_probs, test_labels = get_probs(model, parts["test"])
    results = {
        "default_0.5": report_at(test_probs, test_labels, 0.5, "TEST @0.5"),
        "tuned": report_at(test_probs, test_labels, thr, f"TEST @tuned(F{beta:g})"),
        "roc_auc": float(roc_auc_score(test_labels, test_probs)),
        "pr_auc": float(average_precision_score(test_labels, test_probs)),
    }
    print(f"\nROC-AUC {results['roc_auc']:.3f} | PR-AUC {results['pr_auc']:.3f}")

    preds_tuned = (test_probs >= thr).astype(int)
    plot_confusion(test_labels, preds_tuned, config.OUTPUT_DIR / "confusion_matrix.png")
    gradcam_grid(model, parts["test"], test_probs, thr)

    with open(config.OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrikler: {config.OUTPUT_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
