import torch
import torch.nn as nn
import torch.nn.functional as F

from . import custom_losses


class DeepSupervisionWrapper(nn.Module):

    def __init__(self, cfg, loss_func):
        super().__init__()
        self.loss_func = loss_func
        self.weights = torch.tensor(cfg.deep_supervision_weights).float()

    def forward(self, p, t):
        assert len(p) == len(self.weights), f"length of p is [{len(p)}] whereas # of weights is [{len(self.weights)}]"
        self.weights = self.weights.to(t.device)
        loss = torch.tensor(0.).to(t.device)
        for level_idx, level_p in enumerate(p):
            if level_p.shape[2:] != t.shape[2:]:
                loss += self.weights[level_idx] * self.loss_func(level_p, F.interpolate(t.float(), size=level_p.shape[2:], mode="nearest"))
            else:
                loss += self.weights[level_idx] * self.loss_func(level_p, t)
        return loss / self.weights.sum()


def get_loss(cfg):
    loss_func = getattr(custom_losses, cfg.loss)(**cfg.loss_params)

    if cfg.deep_supervision:
        print(f"Using deep supervision with loss function `{cfg.loss}` and weights `{cfg.deep_supervision_weights}` ...")
        loss_func = DeepSupervisionWrapper(cfg, loss_func=loss_func)

    return loss_func
