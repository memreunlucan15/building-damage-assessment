"""Cok-modlu egitim motoru: engine.py'deki ortak yapi taslarini (run_epoch, fit,
dengeli ornekleme, TTA, esik secimi) MultiModalDataset/MultiModalNet ile kullanir."""
from torch.utils.data import DataLoader

import config
from dataset_mm import MultiModalDataset
from engine import (fit, make_criterion, make_loaders_from_datasets,  # noqa: F401
                    pick_threshold, predict_from_loader, run_epoch)
from model_mm import build_mm_model


def make_loaders(train_samples, val_samples, modalities):
    train_ds = MultiModalDataset(train_samples, train=True, modalities=modalities)
    val_ds = MultiModalDataset(val_samples, train=False, modalities=modalities)
    return make_loaders_from_datasets(train_ds, val_ds,
                                      [s["label"] for s in train_samples])


def train_model(train_samples, val_samples, modalities, epochs=None, verbose=True):
    epochs = epochs or config.CV_EPOCHS
    model = build_mm_model(modalities, pretrained=True)
    criterion = make_criterion(train_samples)
    train_loader, val_loader = make_loaders(train_samples, val_samples, modalities)

    def set_backbone_trainable(flag):
        for p in model.branches.parameters():
            p.requires_grad = flag

    return fit(model, train_loader, val_loader, criterion,
               list(model.classifier.parameters()), set_backbone_trainable,
               epochs, verbose)


def predict_probs(model, samples, modalities, tta=None):
    tta = config.USE_TTA if tta is None else tta
    ds = MultiModalDataset(samples, train=False, modalities=modalities)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS)
    return predict_from_loader(model, loader, tta)
