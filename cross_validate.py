"""Stratified 5-fold capraz dogrulama: guvenilir ortalama+/-std metrikler,
out-of-fold (OOF) toplama, PR/ROC egrileri ve ensemble icin fold modelleri."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
from dataset import build_index
from engine import pick_threshold, predict_probs, train_model


def _metrics_at(probs, labels, t):
    preds = (probs >= t).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, pos_label=config.POSITIVE_IDX, average="binary", zero_division=0)
    return {
        "accuracy": float((preds == labels).mean()),
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "pr_auc": float(average_precision_score(labels, probs)),
        "threshold": float(t),
    }


def _mean_std(dicts, key):
    vals = np.array([d[key] for d in dicts])
    return float(vals.mean()), float(vals.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.CV_EPOCHS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    args = ap.parse_args()

    config.set_seed()
    samples = np.array(build_index(), dtype=object)
    labels = np.array([s["label"] for s in samples])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)

    oof_probs = np.zeros(len(samples))
    oof_filled = np.zeros(len(samples), dtype=bool)
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(samples, labels), 1):
        print(f"\n===== Fold {fold}/{args.folds} =====")
        tr_samples = list(samples[tr_idx])
        te_samples = list(samples[te_idx])
        # Egitim kismindan ic-validation ayir (early stopping + esik secimi icin).
        inner_tr, inner_val = train_test_split(
            tr_samples, test_size=config.INNER_VAL_FRACTION,
            stratify=[s["label"] for s in tr_samples], random_state=config.SEED)

        model, _, best = train_model(inner_tr, inner_val, epochs=args.epochs)
        torch.save({"model_state": model.state_dict(), "fold": fold},
                   config.CV_DIR / f"fold{fold}.pt")

        # Esik ic-validation uzerinde secilir (sizinti yok), fold testine uygulanir.
        val_probs, val_labels = predict_probs(model, inner_val)
        thr, feasible = pick_threshold(val_probs, val_labels)
        te_probs, te_labels = predict_probs(model, te_samples)
        oof_probs[te_idx] = te_probs
        oof_filled[te_idx] = True

        m = _metrics_at(te_probs, te_labels, thr)
        m["threshold_feasible"] = bool(feasible)
        fold_metrics.append(m)
        print(f"  -> fold test: recall {m['recall']:.3f} prec {m['precision']:.3f} "
              f"F1 {m['f1']:.3f} ROC-AUC {m['roc_auc']:.3f} "
              f"(esik {thr:.2f}{'' if feasible else ', tabana ulasilamadi'})")

    # ---- Fold-bazli ozet (ortalama +/- std) ----
    print("\n========== 5-FOLD CV OZET (ortalama +/- std) ==========")
    summary = {}
    for key in ["recall", "precision", "f1", "roc_auc", "pr_auc", "accuracy"]:
        mu, sd = _mean_std(fold_metrics, key)
        summary[key] = {"mean": mu, "std": sd}
        print(f"  {key:10s}: {mu:.3f} +/- {sd:.3f}")

    # ---- OOF birlesik degerlendirme (tum 169 hasarli teste girdi) ----
    assert oof_filled.all()
    oof_labels = labels.astype(int)
    oof_thr, oof_feasible = pick_threshold(oof_probs, oof_labels)
    oof_m = _metrics_at(oof_probs, oof_labels, oof_thr)
    print("\n========== OOF (tum veri, tek operasyon noktasi) ==========")
    print(f"  esik {oof_thr:.2f}{'' if oof_feasible else ' (taban saglanamadi)'} | "
          f"recall {oof_m['recall']:.3f} prec {oof_m['precision']:.3f} "
          f"F1 {oof_m['f1']:.3f} ROC-AUC {oof_m['roc_auc']:.3f} PR-AUC {oof_m['pr_auc']:.3f}")

    _plot_curves(oof_probs, oof_labels)
    _plot_oof_confusion(oof_probs, oof_labels, oof_thr)

    out = {"per_fold": fold_metrics, "summary": summary, "oof": oof_m,
           "config": {"backbone": config.BACKBONE, "loss": config.LOSS,
                      "gamma": config.FOCAL_GAMMA, "sampler": config.USE_SAMPLER,
                      "tta": config.USE_TTA, "precision_floor": config.PRECISION_FLOOR}}
    with open(config.OUTPUT_DIR / "cv_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSonuclar: {config.OUTPUT_DIR / 'cv_results.json'} | modeller: {config.CV_DIR}")


def _plot_curves(probs, labels):
    prec, rec, _ = precision_recall_curve(labels, probs, pos_label=config.POSITIVE_IDX)
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=config.POSITIVE_IDX)
    ap = average_precision_score(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(rec, prec); ax[0].axhline(labels.mean(), ls="--", c="gray",
                                         label=f"taban {labels.mean():.3f}")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
    ax[0].set_title(f"PR egrisi (OOF, AP={ap:.3f})"); ax[0].legend()
    ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], ls="--", c="gray")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[1].set_title(f"ROC egrisi (OOF, AUC={auc:.3f})")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "cv_pr_roc.png", dpi=130)
    print(f"PR/ROC egrileri: {config.OUTPUT_DIR / 'cv_pr_roc.png'}")


def _plot_oof_confusion(probs, labels, t):
    from sklearn.metrics import ConfusionMatrixDisplay
    preds = (probs >= t).astype(int)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=config.CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"OOF Confusion Matrix (esik {t:.2f})")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "cv_confusion_matrix.png", dpi=130)
    print(f"OOF confusion matrix: {config.OUTPUT_DIR / 'cv_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
