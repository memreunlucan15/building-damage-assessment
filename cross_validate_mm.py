"""Cok-modlu (opt / SAR / opt+SAR) 5-fold CV. Ayni altyapidan gecen modalite
karsilastirmasi. Cikti: outputs/cv_<tag>/ (sonuclar + PR/ROC + confusion)."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

import config
from cross_validate import _mean_std, _metrics_at   # saf yardimcilar
from dataset import build_index
from engine_mm import pick_threshold, predict_probs, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", type=str, default=",".join(config.MODALITIES),
                    help="virgulle: opt | SAR | opt,SAR")
    ap.add_argument("--tag", type=str, default=None, help="cikti klasoru eki")
    ap.add_argument("--epochs", type=int, default=config.CV_EPOCHS)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    args = ap.parse_args()

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    tag = args.tag or "_".join(modalities)
    out_dir = config.OUTPUT_DIR / f"cv_{tag}"
    out_dir.mkdir(exist_ok=True)
    print(f"Modaliteler: {modalities} | footprint: {config.USE_FOOTPRINT} | cikti: {out_dir}")

    config.set_seed()
    samples = np.array(build_index(), dtype=object)
    labels = np.array([s["label"] for s in samples])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)

    oof_probs = np.zeros(len(samples))
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(samples, labels), 1):
        print(f"\n===== [{tag}] Fold {fold}/{args.folds} =====")
        tr_samples = list(samples[tr_idx])
        inner_tr, inner_val = train_test_split(
            tr_samples, test_size=config.INNER_VAL_FRACTION,
            stratify=[s["label"] for s in tr_samples], random_state=config.SEED)

        model, _, _ = train_model(inner_tr, inner_val, modalities, epochs=args.epochs)
        torch.save({"model_state": model.state_dict(), "fold": fold,
                    "modalities": modalities}, out_dir / f"fold{fold}.pt")

        val_probs, val_labels = predict_probs(model, inner_val, modalities)
        thr, feasible = pick_threshold(val_probs, val_labels)
        te_probs, te_labels = predict_probs(model, list(samples[te_idx]), modalities)
        oof_probs[te_idx] = te_probs

        m = _metrics_at(te_probs, te_labels, thr)
        m["threshold_feasible"] = bool(feasible)
        fold_metrics.append(m)
        print(f"  -> fold test: recall {m['recall']:.3f} prec {m['precision']:.3f} "
              f"F1 {m['f1']:.3f} ROC-AUC {m['roc_auc']:.3f} (esik {thr:.2f})")

    print(f"\n===== [{tag}] 5-FOLD CV OZET (ortalama +/- std) =====")
    summary = {}
    for key in ["recall", "precision", "f1", "roc_auc", "pr_auc", "accuracy"]:
        mu, sd = _mean_std(fold_metrics, key)
        summary[key] = {"mean": mu, "std": sd}
        print(f"  {key:10s}: {mu:.3f} +/- {sd:.3f}")

    oof_labels = labels.astype(int)
    oof_thr, _ = pick_threshold(oof_probs, oof_labels)
    oof_m = _metrics_at(oof_probs, oof_labels, oof_thr)
    print(f"\n  [OOF] esik {oof_thr:.2f} | recall {oof_m['recall']:.3f} "
          f"prec {oof_m['precision']:.3f} F1 {oof_m['f1']:.3f} "
          f"ROC-AUC {oof_m['roc_auc']:.3f} PR-AUC {oof_m['pr_auc']:.3f}")

    _plot(oof_probs, oof_labels, oof_thr, tag, out_dir)
    out = {"tag": tag, "modalities": modalities, "footprint": config.USE_FOOTPRINT,
           "per_fold": fold_metrics, "summary": summary, "oof": oof_m}
    with open(out_dir / "cv_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSonuclar: {out_dir / 'cv_results.json'}")


def _plot(probs, labels, thr, tag, out_dir):
    prec, rec, _ = precision_recall_curve(labels, probs, pos_label=config.POSITIVE_IDX)
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=config.POSITIVE_IDX)
    ap = average_precision_score(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(rec, prec); ax[0].axhline(labels.mean(), ls="--", c="gray")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
    ax[0].set_title(f"[{tag}] PR (OOF, AP={ap:.3f})")
    ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], ls="--", c="gray")
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[1].set_title(f"[{tag}] ROC (OOF, AUC={auc:.3f})")
    fig.tight_layout(); fig.savefig(out_dir / "pr_roc.png", dpi=130); plt.close(fig)

    cm = confusion_matrix(labels, (probs >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=config.CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"[{tag}] OOF Confusion (esik {thr:.2f})")
    fig.tight_layout(); fig.savefig(out_dir / "confusion_matrix.png", dpi=130); plt.close(fig)
    print(f"Figurler: {out_dir / 'pr_roc.png'}, {out_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)   # canli takip icin satir-satir flush
    except Exception:
        pass
    main()
