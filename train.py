"""Egitim: sinif-agirlikli loss, augmentation, damaged-F1 uzerinde early stopping."""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (f1_score, fbeta_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader

import config
from dataset import QuickQuakeDataset
from losses import FocalLoss
from model import build_model
from split import load_split, make_split


def build_criterion(weights):
    if config.LOSS == "focal":
        return FocalLoss(alpha=weights, gamma=config.FOCAL_GAMMA)
    return nn.CrossEntropyLoss(weight=weights)


def class_weights(train_samples):
    """Ters frekans agirligi -> azinlik (damaged) sinifina daha cok agirlik."""
    labels = np.array([s["label"] for s in train_samples])
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=config.DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "f1": f1_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
        "fbeta": fbeta_score(labels, preds, beta=config.THRESHOLD_BETA,
                             pos_label=config.POSITIVE_IDX, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--fresh-split", action="store_true", help="Bolmeyi yeniden uret")
    args = ap.parse_args()

    config.set_seed()
    if args.fresh_split or not config.SPLIT_PATH.exists():
        make_split(save=True)
    parts = load_split()

    train_ds = QuickQuakeDataset(parts["train"], train=True)
    val_ds = QuickQuakeDataset(parts["val"], train=False)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)

    model = build_model(num_classes=2, pretrained=True)
    weights = class_weights(parts["train"])
    print(f"Sinif agirliklari [intact, damaged]: {weights.tolist()}")
    print(f"Loss: {config.LOSS} (gamma={config.FOCAL_GAMMA}) | "
          f"model secimi: val F{config.THRESHOLD_BETA:g}")
    criterion = build_criterion(weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                 weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3)

    history, best_score, patience = [], -1.0, 0
    beta = config.THRESHOLD_BETA
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer)
        va = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(va["fbeta"])
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"[{epoch:02d}/{args.epochs}] "
              f"tr_loss {tr['loss']:.3f} | va_loss {va['loss']:.3f} | "
              f"va_F{beta:g} {va['fbeta']:.3f} F1 {va['f1']:.3f} "
              f"recall {va['recall']:.3f} prec {va['precision']:.3f}")

        if va["fbeta"] > best_score:
            best_score, patience = va["fbeta"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_fbeta": best_score, "val_f1": va["f1"]}, config.BEST_MODEL_PATH)
            print(f"    -> en iyi model kaydedildi (val F{beta:g} {best_score:.3f})")
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                print(f"    -> early stopping (patience {config.EARLY_STOP_PATIENCE})")
                break

    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    _plot_curves(history)
    print(f"\nEn iyi val damaged-F{beta:g}: {best_score:.3f} | model: {config.BEST_MODEL_PATH}")


def _plot_curves(history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, [h["train"]["loss"] for h in history], label="train")
    ax[0].plot(epochs, [h["val"]["loss"] for h in history], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(epochs, [h["val"]["f1"] for h in history], label="F1", color="tab:green")
    ax[1].plot(epochs, [h["val"]["recall"] for h in history], label="recall", color="tab:orange")
    ax[1].set_title("Val damaged metrics"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "training_curves.png", dpi=130)
    print(f"Egitim egrileri: {config.OUTPUT_DIR / 'training_curves.png'}")


if __name__ == "__main__":
    main()
