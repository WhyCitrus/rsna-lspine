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
        # Calculate original loss
        loss_dict = self.loss_func(p[0], t)
        # Save original loss for tracking
        for k, v in loss_dict.copy().items():
            loss_dict[k+"_orig"] = v
        # Multiply by weight
        loss_dict = {k: self.weights[0] * v if "orig" not in k else v for k, v in loss_dict.items()}
        for level_idx, level_p in enumerate(p[1:]):
            # Calculate losses for preceding levels
            tmp_loss_dict = self.loss_func(level_p, F.interpolate(t.float(), size=level_p.shape[2:], mode="nearest"))
            # Multiply by weight
            for k, v in tmp_loss_dict.items():
                loss_dict[k] += self.weights[level_idx + 1] * v
        # Scale by sum of weights
        for k, v in loss_dict.items():
            if "orig" in k:
                continue
            loss_dict[k] = v / self.weights.sum()
        return loss_dict


class DeepSupervisionWrapperWithCls(nn.Module):

    def __init__(self, cfg, loss_func):
        super().__init__()
        self.loss_func = loss_func
        self.weights = torch.tensor(cfg.deep_supervision_weights).float()

    def forward_seg(self, p, t):
        assert len(p) == len(self.weights), f"length of p is [{len(p)}] whereas # of weights is [{len(self.weights)}]"
        self.weights = self.weights.to(t.device)
        # Calculate original loss
        loss_dict = self.loss_func.forward_seg(p[0], t)
        # Save original loss for tracking
        for k, v in loss_dict.copy().items():
            loss_dict[k+"_orig"] = v
        # Multiply by weight
        loss_dict = {k: self.weights[0] * v if "orig" not in k else v for k, v in loss_dict.items()}
        for level_idx, level_p in enumerate(p[1:]):
            # Calculate losses for preceding levels
            tmp_loss_dict = self.loss_func.forward_seg(level_p, F.interpolate(t.float(), size=level_p.shape[2:], mode="nearest"))
            # Multiply by weight
            for k, v in tmp_loss_dict.items():
                loss_dict[k] += self.weights[level_idx + 1] * v
        # Scale by sum of weights
        for k, v in loss_dict.items():
            if "orig" in k:
                continue
            loss_dict[k] = v / self.weights.sum()
        return loss_dict

    def forward_cls(self, p, t):
        return self.loss_func.forward_cls(p, t)

    def forward(self, p_seg, t_seg, p_cls, t_cls):
        loss_dict = {}
        loss_dict.update(self.forward_seg(p_seg, t_seg))
        loss_dict.update(self.forward_cls(p_cls, t_cls))
        loss_dict["loss"] = loss_dict["seg_loss"] + loss_dict["cls_loss"]
        return loss_dict


def get_loss(cfg):
    loss_func = getattr(custom_losses, cfg.loss)(**cfg.loss_params)

    if cfg.deep_supervision:
        print(f"Using deep supervision with loss function `{cfg.loss}` and weights `{cfg.deep_supervision_weights}` ...")
        if cfg.model.startswith("unet_2d_cls"):
            loss_func = DeepSupervisionWrapperWithCls(cfg, loss_func=loss_func)
        else:
            loss_func = DeepSupervisionWrapper(cfg, loss_func=loss_func)

    return loss_func
