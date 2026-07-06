"""model_mm: conv1 uyarlama, uyumlu katman yukleme, fuzyon forward'i."""
import torch
import torch.nn as nn

import config
from model_mm import MultiModalNet, _adapt_conv1, _load_compatible


def _conv3():
    conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.normal_(conv.weight)
    return conv


def test_adapt_conv1_identity_for_3ch():
    conv = _conv3()
    assert _adapt_conv1(conv, 3) is conv


def test_adapt_conv1_single_channel_uses_mean():
    conv = _conv3()
    new = _adapt_conv1(conv, 1)
    assert new.weight.shape == (64, 1, 7, 7)
    assert torch.allclose(new.weight.data[:, 0], conv.weight.data.mean(dim=1))


def test_adapt_conv1_extra_channel_preserves_rgb():
    conv = _conv3()
    new = _adapt_conv1(conv, 4)
    assert new.weight.shape == (64, 4, 7, 7)
    assert torch.allclose(new.weight.data[:, :3], conv.weight.data)   # RGB korunur
    assert torch.allclose(new.weight.data[:, 3], conv.weight.data.mean(dim=1))


def test_load_compatible_filters_mismatches():
    net = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
    donor = {"0.weight": torch.ones(8, 4), "0.bias": torch.ones(8),
             "1.weight": torch.ones(99, 8)}          # sekli uyusmayan katman
    n_loaded, _ = _load_compatible(net, donor)
    assert n_loaded == 2                              # 0.weight + 0.bias
    assert torch.allclose(net[0].weight.data, torch.ones(8, 4))
    assert not torch.allclose(net[1].weight.data, torch.ones(2, 8))


def test_fusion_forward_shapes():
    for mods in (["opt"], ["opt", "SAR"]):
        model = MultiModalNet(mods, pretrained=False).to(config.DEVICE)
        x = {m: torch.randn(2, 3, 64, 64, device=config.DEVICE) for m in mods}
        out = model(x)
        assert out.shape == (2, 2)
        # Fuzyonda siniflandirici girisi kol sayisi * 512 olmali.
        assert model.classifier[1].in_features == 512 * len(mods)
