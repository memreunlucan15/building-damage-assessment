"""FocalLoss dogrulugu: gamma=0 -> CE'ye esdegerlik, odaklama davranisi, alpha."""
import torch
import torch.nn.functional as F

from losses import FocalLoss


def _toy_batch():
    torch.manual_seed(0)
    logits = torch.randn(16, 2)
    target = torch.randint(0, 2, (16,))
    return logits, target


def test_gamma_zero_equals_cross_entropy():
    logits, target = _toy_batch()
    fl = FocalLoss(alpha=None, gamma=0.0)(logits, target)
    ce = F.cross_entropy(logits, target)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_gamma_downweights_easy_examples():
    # Kolay ornek (dogru sinifa yuksek guven) -> focal katkisi CE'den kucuk olmali.
    logits = torch.tensor([[5.0, -5.0]])   # sinif 0'a cok emin
    target = torch.tensor([0])
    fl = FocalLoss(gamma=2.0)(logits, target)
    ce = F.cross_entropy(logits, target)
    assert fl < ce


def test_alpha_weights_scale_loss():
    logits = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = torch.tensor([0, 1])
    base = FocalLoss(gamma=0.0)(logits, target)
    # Sinif 1'e 3x agirlik -> ortalama loss artmali (esit logitlerde).
    alpha = torch.tensor([1.0, 3.0])
    weighted = FocalLoss(alpha=alpha, gamma=0.0)(logits, target)
    assert weighted > base


def test_loss_is_finite_and_positive():
    logits, target = _toy_batch()
    loss = FocalLoss(gamma=2.0)(logits, target)
    assert torch.isfinite(loss) and loss > 0
