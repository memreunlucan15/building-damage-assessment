"""Ortak egitim motoru: tek epoch dongusu (tensor VE {modalite: tensor} girdiler),
dengeli ornekleme, iki asamali fine-tune, TTA cikarimi ve saglam (precision-tabanli)
esik secimi. engine_mm.py ve train.py buradaki yapi taslarini yeniden kullanir.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (f1_score, fbeta_score,
                             precision_recall_fscore_support, precision_score,
                             recall_score)
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from dataset import QuickQuakeDataset
from losses import FocalLoss
from model import build_model


# ---------------- Ortak yapi taslari ----------------
def to_device(x):
    """Tensor veya {modalite: tensor} sozlugunu cihaza tasir."""
    if isinstance(x, dict):
        return {m: t.to(config.DEVICE, non_blocking=True) for m, t in x.items()}
    return x.to(config.DEVICE, non_blocking=True)


def _flip(x, dim):
    """TTA icin yatay/dikey flip; sozluk girdilerde tum modalitelere uygulanir."""
    if isinstance(x, dict):
        return {m: torch.flip(t, [dim]) for m, t in x.items()}
    return torch.flip(x, [dim])


def make_sampler(labels):
    """Sinif frekansinin tersiyle agirlikli, dengeli batch ureten sampler."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2)
    sample_w = (1.0 / np.maximum(counts, 1))[labels]
    return WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                 num_samples=len(labels), replacement=True)


def make_loaders_from_datasets(train_ds, val_ds, train_labels):
    """Dataset ciftinden loader'lari kurar (USE_SAMPLER'a gore dengeli/karisik)."""
    if config.USE_SAMPLER:
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                  sampler=make_sampler(train_labels),
                                  num_workers=config.NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


def make_loaders(train_samples, val_samples):
    train_ds = QuickQuakeDataset(train_samples, train=True)
    val_ds = QuickQuakeDataset(val_samples, train=False)
    return make_loaders_from_datasets(train_ds, val_ds,
                                      [s["label"] for s in train_samples])


def make_criterion(train_samples):
    """Loss secimi. Dengeli ornekleme acikken loss agirligi (alpha) eklemek
    dengesizligi cifte sayar -> alpha=None. Sampler kapaliysa ters-frekans agirligi."""
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


def run_epoch(model, loader, criterion, optimizer=None, clip_grad=5.0):
    """Tek epoch egitim (optimizer verilirse) veya degerlendirme.

    Girdi tensor ya da {modalite: tensor} sozlugu olabilir; metrikler pozitif
    (damaged) sinif uzerinden hesaplanir.
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, y in loader:
        x = to_device(x)
        y = y.to(config.DEVICE)
        with torch.set_grad_enabled(train_mode):
            logits = model(x)
            loss = criterion(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
        total_loss += loss.item() * y.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / len(loader.dataset),
        "fbeta": fbeta_score(labels, preds, beta=config.THRESHOLD_BETA,
                             pos_label=config.POSITIVE_IDX, zero_division=0),
        "f1": f1_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
        "precision": precision_score(labels, preds, pos_label=config.POSITIVE_IDX, zero_division=0),
    }


def fit(model, train_loader, val_loader, criterion, head_params,
        set_backbone_trainable, epochs, verbose=True):
    """Iki asamali egitim recetesi (tek- ve cok-modlu modeller icin ortak).

    Asama 1: govde dondurulur, yalnizca siniflandirici basligi egitilir.
    Asama 2: tum ag dusuk LR ile fine-tune edilir; val F-beta uzerinde
    early stopping + en iyi durumun geri yuklenmesi.
    """
    beta = config.THRESHOLD_BETA

    if config.FREEZE_EPOCHS > 0:
        set_backbone_trainable(False)
        opt = torch.optim.Adam(head_params, lr=config.LR_HEAD)
        for e in range(1, config.FREEZE_EPOCHS + 1):
            tr = run_epoch(model, train_loader, criterion, opt)
            if verbose:
                print(f"   [freeze {e}/{config.FREEZE_EPOCHS}] tr_loss {tr['loss']:.3f}")
        set_backbone_trainable(True)

    opt = torch.optim.Adam(model.parameters(), lr=config.LR,
                           weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)

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
def predict_from_loader(model, loader, tta):
    """Loader uzerinden (istege bagli flip-TTA'li) damaged olasiliklari dondurur."""
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = to_device(x)
        views = [x]
        if tta:
            views += [_flip(x, 3), _flip(x, 2)]      # yatay + dikey flip
        p = torch.stack([F.softmax(model(v), dim=1)[:, config.POSITIVE_IDX]
                         for v in views]).mean(0)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


# ---------------- Tek-modalite (optik) API ----------------
def train_model(train_samples, val_samples, epochs=None, verbose=True):
    epochs = epochs or config.CV_EPOCHS
    model = build_model(num_classes=2, pretrained=True)
    criterion = make_criterion(train_samples)
    train_loader, val_loader = make_loaders(train_samples, val_samples)

    def set_backbone_trainable(flag):
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = flag

    return fit(model, train_loader, val_loader, criterion,
               list(model.fc.parameters()), set_backbone_trainable,
               epochs, verbose)


def predict_probs(model, samples, tta=None):
    tta = config.USE_TTA if tta is None else tta
    ds = QuickQuakeDataset(samples, train=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    return predict_from_loader(model, loader, tta)


# ---------------- Saglam esik secimi ----------------
def pick_threshold(probs, labels, precision_floor=None, beta=None):
    """precision >= floor saglayan esikler arasinda recall'i maksimize et.
    Hicbiri tabani saglamazsa F-beta'yi maksimize eden esige geri don."""
    precision_floor = config.PRECISION_FLOOR if precision_floor is None else precision_floor
    beta = config.THRESHOLD_BETA if beta is None else beta
    grid = np.linspace(0.05, 0.95, 19)
    feasible = []   # (recall, precision, t)
    fallback = []   # (fbeta, t)
    for t in grid:
        preds = (probs >= t).astype(int)
        p, r, _, _ = precision_recall_fscore_support(
            labels, preds, pos_label=config.POSITIVE_IDX, average="binary", zero_division=0)
        fb = fbeta_score(labels, preds, beta=beta,
                         pos_label=config.POSITIVE_IDX, zero_division=0)
        fallback.append((fb, t))
        if p >= precision_floor:
            feasible.append((r, p, t))
    if feasible:
        feasible.sort(key=lambda z: (z[0], z[1]))   # once recall, sonra precision
        return float(feasible[-1][2]), True
    fallback.sort()
    return float(fallback[-1][1]), False
