"""Cok-sinifli Focal Loss: azinlik (damaged) sinifina ve zor orneklere odaklanir."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha : (num_classes,) sinif agirliklari (None ise hepsi esit).
    gamma : kolay orneklerin katkisini bastiran odak parametresi (0 -> agirlikli CE).
    """

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = alpha.gather(0, target) * loss
        return loss.mean()
