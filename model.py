"""ImageNet on-egitimli ResNet18 transfer learning modeli (2 sinif)."""
import torch.nn as nn
from torchvision import models

import config


_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
}


def build_model(num_classes: int = 2, pretrained: bool = True, backbone: str = None):
    backbone = backbone or config.BACKBONE
    ctor, w = _BACKBONES[backbone]
    model = ctor(weights=w if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # resnet ailesi -> .fc
    return model.to(config.DEVICE)


if __name__ == "__main__":
    import torch
    m = build_model()
    n_params = sum(p.numel() for p in m.parameters())
    dummy = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE, device=config.DEVICE)
    out = m(dummy)
    print("Model: ResNet18 | parametre:", f"{n_params:,}", "| cihaz:", config.DEVICE)
    print("Cikti sekli:", tuple(out.shape))
