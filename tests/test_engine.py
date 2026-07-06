"""engine: run_epoch (tensor + dict girdiler), dengeli sampler, TTA tahmini."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import config
from engine import make_sampler, predict_from_loader, run_epoch


class _TensorData(Dataset):
    def __init__(self, n=24):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 3, 8, 8, generator=g)
        self.y = torch.randint(0, 2, (n,), generator=g)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class _DictData(_TensorData):
    def __getitem__(self, i):
        return {"opt": self.x[i]}, self.y[i]


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 2)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["opt"]
        return self.fc(x.flatten(1))


def _model():
    return _TinyNet().to(config.DEVICE)


def test_run_epoch_eval_returns_metrics():
    loader = DataLoader(_TensorData(), batch_size=8)
    out = run_epoch(_model(), loader, nn.CrossEntropyLoss(), optimizer=None)
    assert {"loss", "fbeta", "f1", "recall", "precision"} <= set(out)
    assert np.isfinite(out["loss"])


def test_run_epoch_train_decreases_loss():
    model = _model()
    loader = DataLoader(_TensorData(64), batch_size=16)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = run_epoch(model, loader, nn.CrossEntropyLoss(), opt)
    for _ in range(10):
        last = run_epoch(model, loader, nn.CrossEntropyLoss(), opt)
    assert last["loss"] < first["loss"]


def test_run_epoch_accepts_dict_inputs():
    loader = DataLoader(_DictData(), batch_size=8)
    out = run_epoch(_model(), loader, nn.CrossEntropyLoss(), optimizer=None)
    assert np.isfinite(out["loss"])


def test_make_sampler_balances_minority():
    labels = np.array([0] * 90 + [1] * 10)
    sampler = make_sampler(labels)
    torch.manual_seed(0)
    drawn = np.array([labels[i] for i in sampler])
    # Dengeli orneklemede azinlik orani ~0.5 olmali (0.1 degil).
    assert 0.35 < drawn.mean() < 0.65


def test_predict_from_loader_tta_shapes():
    loader = DataLoader(_TensorData(), batch_size=8)
    probs, labels = predict_from_loader(_model(), loader, tta=True)
    assert probs.shape == labels.shape == (24,)
    assert (0.0 <= probs).all() and (probs <= 1.0).all()


def test_predict_from_loader_dict_tta():
    loader = DataLoader(_DictData(), batch_size=8)
    probs, _ = predict_from_loader(_model(), loader, tta=True)
    assert probs.shape == (24,)
