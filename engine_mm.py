"""Cok-modlu egitim motoru: sozluk girdileri, dengeli ornekleme, iki asamali
fine-tune, TTA cikarimi. Esik secimi engine.pick_threshold ile paylasilir."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (f1_score, fbeta_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from dataset_mm import MultiModalDataset
from engine import pick_threshold  # noqa: F401  (disari yeniden ihrac)
from losses import FocalLoss
from model_mm import build_mm_model


def _to_device(inputs):
    return {m: t.to(config.DEVICE, non_blocking=True) for m, t in inputs.items()}


def make_loaders(train_samples, val_samples, modalities):
    train_ds = MultiModalDataset(train_samples, train=True, modalities=modalities)
    val_ds = MultiModalDataset(val_samples, train=False, modalities=modalities)
    if config.USE_SAMPLER:
        labels = np.array([s["label"] for s in train_samples])
        counts = np.bincount(labels, minlength=2)
        sample_w = (1.0 / np.maximum(counts, 1))[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                        num_samples=len(labels), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
                                  num_workers=config.NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


def _criterion(train_samples):
    if config.USE_SAMPLER:
        alpha = None
    else:
        labels = np.array([s["label"] for s in train_samples])
        counts = np.bincount(labels, minlength=2)
        w = counts.sum() / (2.0 * np.maximum(counts, 1))
        alpha = torch.tensor(w, dtype=torch.float32, device=config.DEVICE)
    if config.LOSS == "focal":
        return FocalLoss(alpha=alpha, gamma=config.FOCAL_GAMMA)
    return nn.CrossEntropyLoss(weight=alpha)


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, preds, labels = 0.0, [], []
    for inputs, y in loader:
        inputs = _to_device(inputs)
        y = y.to(config.DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(inputs)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    return {
        "loss": total_loss / len(loader.dataset),
        "fbeta": fbeta_score(labels, preds, beta=config.THRESHOLD_BETA,
                             pos_label=config.POSITIVE_IDX, zero_division=0),
        "f1": f1_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
    }


def train_model(train_samples, val_samples, modalities, epochs=None, verbose=True):
    epochs = epochs or config.CV_EPOCHS
    model = build_mm_model(modalities, pretrained=True)
    criterion = _criterion(train_samples)
    train_loader, val_loader = make_loaders(train_samples, val_samples, modalities)
    beta = config.THRESHOLD_BETA

    # Asama 1: kollari (govde) dondur, yalnizca siniflandiriciyi egit.
    if config.FREEZE_EPOCHS > 0:
        for p in model.branches.parameters():
            p.requires_grad = False
        opt = torch.optim.Adam(model.classifier.parameters(), lr=config.LR_HEAD)
        for e in range(1, config.FREEZE_EPOCHS + 1):
            tr = run_epoch(model, train_loader, criterion, opt)
            if verbose:
                print(f"   [freeze {e}/{config.FREEZE_EPOCHS}] tr_loss {tr['loss']:.3f}")
        for p in model.branches.parameters():
            p.requires_grad = True

    # Asama 2: tum agi dusuk LR ile fine-tune.
    opt = torch.optim.Adam(model.parameters(), lr=config.LR,
                           weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    best_score, best_state, patience, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        tr = run_epoch(model, train_loader, criterion, opt)
        va = run_epoch(model, val_loader, criterion, optimizer=None)
        sched.step(va["fbeta"])
        history.append({"epoch": epoch, "train": tr, "val": va})
        if verbose:
            print(f"   [{epoch:02d}/{epochs}] tr_loss {tr['loss']:.3f} | "
                  f"va_F{beta:g} {va['fbeta']:.3f} recall {va['recall']:.3f} "
                  f"prec {va['precision']:.3f}")
        if va["fbeta"] > best_score:
            best_score = va["fbeta"]
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, history, best_score


@torch.no_grad()
def predict_probs(model, samples, modalities, tta=None):
    tta = config.USE_TTA if tta is None else tta
    ds = MultiModalDataset(samples, train=False, modalities=modalities)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    model.eval()
    probs, labels = [], []
    for inputs, y in loader:
        inputs = _to_device(inputs)
        view_sets = [inputs]
        if tta:
            view_sets.append({m: torch.flip(t, [3]) for m, t in inputs.items()})
            view_sets.append({m: torch.flip(t, [2]) for m, t in inputs.items()})
        p = torch.stack([F.softmax(model(v), dim=1)[:, config.POSITIVE_IDX]
                         for v in view_sets]).mean(0)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)
