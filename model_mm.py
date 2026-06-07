"""Cok-modlu fuzyon modeli: her modalite icin ayri ResNet kolu + ozellik birlestirme.

Tek modalite verilirse tek kollu (standart transfer learning) modele indirgenir.
conv1, modalitenin kanal sayisina uyarlanir (on-egitimli agirliklar korunur).
"""
import torch
import torch.nn as nn
from torchvision import models

import config
from dataset_mm import in_channels

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1, 512),
}


def _adapt_conv1(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """conv1'i in_ch kanala uyarlar; on-egitimli agirliklari korur/ortalar."""
    if in_ch == 3:
        return conv
    w = conv.weight.data                       # (64,3,7,7)
    mean_w = w.mean(dim=1, keepdim=True)        # (64,1,7,7)
    if in_ch < 3:
        new_w = mean_w.repeat(1, in_ch, 1, 1)
    else:
        new_w = torch.cat([w, mean_w.repeat(1, in_ch - 3, 1, 1)], dim=1)
    new_conv = nn.Conv2d(in_ch, conv.out_channels, kernel_size=conv.kernel_size,
                         stride=conv.stride, padding=conv.padding, bias=False)
    new_conv.weight.data = new_w
    return new_conv


def _load_compatible(net, state_dict):
    """state_dict'ten yalnizca ismi VE sekli uyan katmanlari yukler (fc atlanir)."""
    model_sd = net.state_dict()
    filtered = {k: v for k, v in state_dict.items()
                if k in model_sd and v.shape == model_sd[k].shape}
    net.load_state_dict(filtered, strict=False)
    return len(filtered), len(model_sd)


def _make_branch(modality, in_ch, pretrained):
    ctor, w, feat_dim = _BACKBONES[config.BACKBONE]
    use_sarhub = (modality == "SAR" and config.SAR_PRETRAINED == "sarhub"
                  and config.BACKBONE == "resnet18" and pretrained)
    if use_sarhub:
        net = ctor(weights=None)                # SAR-HUB ile dolduracagiz
    else:
        net = ctor(weights=w if pretrained else None)   # ImageNet
    net.fc = nn.Identity()                      # 512-boyutlu ozellik dondur
    net.conv1 = _adapt_conv1(net.conv1, in_ch)
    if use_sarhub:
        sd = torch.load(config.SARHUB_WEIGHTS, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        n, tot = _load_compatible(net, sd)
        print(f"      [SAR kolu] SAR-HUB ({config.SARHUB_WEIGHTS.name}) yuklendi: "
              f"{n}/{tot} katman")
    return net, feat_dim


class MultiModalNet(nn.Module):
    def __init__(self, modalities=None, num_classes=2, pretrained=True):
        super().__init__()
        self.modalities = modalities or config.MODALITIES
        self.branches = nn.ModuleDict()
        total = 0
        for m in self.modalities:
            branch, feat_dim = _make_branch(m, in_channels(m), pretrained)
            self.branches[m] = branch
            total += feat_dim
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(total, num_classes),
        )

    def forward(self, inputs: dict):
        feats = [self.branches[m](inputs[m]) for m in self.modalities]
        return self.classifier(torch.cat(feats, dim=1))


def build_mm_model(modalities=None, pretrained=True):
    return MultiModalNet(modalities, pretrained=pretrained).to(config.DEVICE)


if __name__ == "__main__":
    for mods in (["opt"], ["SAR"], ["opt", "SAR"]):
        m = build_mm_model(mods)
        dummy = {mod: torch.randn(2, in_channels(mod), config.IMG_SIZE,
                                  config.IMG_SIZE, device=config.DEVICE) for mod in mods}
        out = m(dummy)
        n = sum(p.numel() for p in m.parameters())
        print(f"{str(mods):18s} -> cikti {tuple(out.shape)} | parametre {n:,}")
