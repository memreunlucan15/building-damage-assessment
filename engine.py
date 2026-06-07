"""Yeniden kullanilabilir egitim motoru: dengeli ornekleme, iki asamali fine-tune,
TTA cikarimi ve saglam (precision-tabanli) esik secimi."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler

import config
from dataset import QuickQuakeDataset
from losses import FocalLoss
from model import build_model
from train import run_epoch


# ---------------- Veri yukleyiciler ----------------
def make_loaders(train_samples, val_samples):
    train_ds = QuickQuakeDataset(train_samples, train=True)
    val_ds = QuickQuakeDataset(val_samples, train=False)
    if config.USE_SAMPLER:
        labels = np.array([s["label"] for s in train_samples])
        counts = np.bincount(labels, minlength=2)
        class_w = 1.0 / np.maximum(counts, 1)
        sample_w = class_w[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_w, dtype=torch.double),
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
    # Dengeli ornekleme acikken loss agirligi (alpha) eklemek dengesizligi cifte
    # sayar -> alpha=None. Sampler kapaliysa ters-frekans agirligi kullan.
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


def _set_backbone_trainable(model, flag):
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = flag


# ---------------- Egitim ----------------
def train_model(train_samples, val_samples, epochs=None, verbose=True):
    epochs = epochs or config.CV_EPOCHS
    model = build_model(num_classes=2, pretrained=True)
    criterion = _criterion(train_samples)
    train_loader, val_loader = make_loaders(train_samples, val_samples)
    beta = config.THRESHOLD_BETA

    # --- Asama 1: govdeyi dondur, yalnizca siniflandirici basligini egit ---
    if config.FREEZE_EPOCHS > 0:
        _set_backbone_trainable(model, False)
        opt = torch.optim.Adam(model.fc.parameters(), lr=config.LR_HEAD)
        for e in range(1, config.FREEZE_EPOCHS + 1):
            tr = run_epoch(model, train_loader, criterion, opt)
            if verbose:
                print(f"   [freeze {e}/{config.FREEZE_EPOCHS}] tr_loss {tr['loss']:.3f}")
        _set_backbone_trainable(model, True)

    # --- Asama 2: tum agi dusuk LR ile fine-tune et ---
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


# ---------------- Cikarim (TTA) ----------------
@torch.no_grad()
def predict_probs(model, samples, tta=None):
    tta = config.USE_TTA if tta is None else tta
    ds = QuickQuakeDataset(samples, train=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(config.DEVICE)
        views = [x]
        if tta:
            views += [torch.flip(x, [3]), torch.flip(x, [2])]  # yatay + dikey flip
        p = torch.stack([F.softmax(model(v), dim=1)[:, config.POSITIVE_IDX]
                         for v in views]).mean(0)
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


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
