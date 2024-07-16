import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import DiceLoss
from torchvision.ops import sigmoid_focal_loss


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        return F.binary_cross_entropy_with_logits(p.float(), t.float())


class SampleWeightedLogLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t, w):
        return F.binary_cross_entropy_with_logits(p.float(), t.float(), weight=w.unsqueeze(1))


class SampleWeightedLogLossV2(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        w = torch.ones((len(p), 1))
        w[t[:, 1] == 1] = 2
        w[t[:, 2] == 1] = 4
        loss = (F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none") * w.float().to(p.device)).mean()
        return loss


def torch_log_loss(p, t):
    w = torch.ones((len(p), )).to(p.device)
    w[t[:, 1] == 1] = 2
    w[t[:, 2] == 1] = 4
    loss = -torch.xlogy(t.float(), p.sigmoid().float()).sum(1)
    loss = loss * w
    loss = loss / w.sum()
    return loss.sum()


class SampleWeightedLogLossV3(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        return torch_log_loss(p, t)


class SampleWeightedLogLossBilat(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        p, t = torch.cat([p[:, :3], p[:, 3:]], dim=0), torch.cat([t[:, :3], t[:, 3:]], dim=0)
        w = torch.ones((len(p), ))
        w[t[:, 1] == 1] = 2.0
        w[t[:, 2] == 1] = 4.0 
        w = w.to(p.device)
        return F.binary_cross_entropy_with_logits(p.float(), t.float(), weight=w.unsqueeze(1))


class ComboLevelsAndMaskedCoordsLoss(nn.Module):

    def forward(self, p_coords, p_levels, t_coords, t_levels, included_levels):
        levels_loss = F.binary_cross_entropy_with_logits(p_levels.float(), t_levels.float())
        coords_loss = F.l1_loss(p_coords.float().sigmoid(), t_coords.float(), reduction="none")
        coords_mean_loss = []
        for b_idx, inc in enumerate(included_levels):
            tmp_indices = torch.where(inc)[0]
            tmp_indices = torch.cat([tmp_indices, tmp_indices + 15])
            # does throw error but uncommon so ignore for now
            # assert t_coords[b_idx, tmp_indices].max() <= 1 
            coords_mean_loss.append(coords_loss[b_idx, tmp_indices].mean())
        coords_loss = torch.stack(coords_mean_loss).mean(0)
        return {"loss": levels_loss + 10 * coords_loss, "levels_loss": levels_loss, "coords_loss": coords_loss}


class SampleWeightedLogLossBilatV2(nn.BCEWithLogitsLoss):

    def forward(self, p, t, w):
        # p.shape = t.shape = (N, 6); first 3 right, second 3 left
        # w.shape = (N, 2); 1st weight rt, 2nd weight left
        rt_loss = F.binary_cross_entropy_with_logits(p[:, :3].float(), t[:, :3].float(), weight=w[:, 0].unsqueeze(1))
        lt_loss = F.binary_cross_entropy_with_logits(p[:, 3:].float(), t[:, 3:].float(), weight=w[:, 1].unsqueeze(1))
        return (rt_loss + lt_loss) / 2


class MaskedBCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t, mask):
        b, n, c = p.size()
        assert mask.size(0) == b and mask.size(1) == n and mask.ndim == 2
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none").view(b*n, -1)
        mask = mask.view(b*n)
        assert mask.size(0) == loss.size(0)
        loss = loss[~mask] # negate mask because True indicates padding token
        return loss.mean()

class CrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, p, t):
        if t.ndim == 2:
            t = t[:, 0]
        return F.cross_entropy(p.float(), t.long())


class L1Loss(nn.L1Loss):

    def forward(self, p, t):
        assert p.shape == t.shape, f"p.shape {p.shape} does not equal t.shape {t.shape}"
        return F.l1_loss(p.sigmoid().float(), t.float())


class L1TanhLoss(nn.L1Loss):

    def forward(self, p, t):
        assert p.shape == t.shape, f"p.shape {p.shape} does not equal t.shape {t.shape}"
        return F.l1_loss(p.tanh().float(), t.float())


class L1LossNoSigmoid(nn.L1Loss):

    def forward(self, p, t):
        assert p.shape == t.shape, f"p.shape {p.shape} does not equal t.shape {t.shape}"
        return F.l1_loss(p.float(), t.float())


class WeightedLogLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.wts = torch.tensor([1.0, 2.0, 4.0])

    def forward(self, p, t):
        # p.shape == t.shape == (N, C)
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none")
        loss = self.wts.to(p.device) * loss
        return loss.mean()


class WeightedLogLoss30(nn.Module):

    def __init__(self):
        super().__init__()
        self.wts = torch.tensor([1.0, 2.0, 4.0] * 10)

    def forward(self, p, t):
        # p.shape == t.shape == (N, C)
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none")
        loss = self.wts.to(p.device) * loss
        return loss.mean()


class WeightedLogLoss6(nn.Module):

    def __init__(self):
        super().__init__()
        self.wts = torch.tensor([1.0, 2.0, 4.0] * 2)

    def forward(self, p, t):
        # p.shape == t.shape == (N, C)
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none")
        loss = self.wts.to(p.device) * loss
        return loss.mean()


class WeightedLogLossWithArea(nn.Module):

    def __init__(self):
        super().__init__()
        self.wts = torch.tensor([1.0, 2.0, 4.0, 0.33, 0.33, 0.33])

    def forward(self, p, t):
        # p.shape == t.shape == (N, C)
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none")
        loss = self.wts.to(p.device) * loss
        return loss.mean()


class BCELoss_SegCls(nn.Module):

    def __init__(self, pos_weight=None):
        super().__init__()
        self.wts = torch.tensor([0.5, 0.5])
        self.pos_weight = torch.tensor(pos_weight)

    def forward(self, p_seg, p_cls, t_seg, t_cls):
        segloss = F.binary_cross_entropy_with_logits(p_seg.float(), t_seg.float(), pos_weight=self.pos_weight)
        clsloss = F.binary_cross_entropy_with_logits(p_cls.float(), t_cls.float())
        loss = self.wts[0] * segloss + self.wts[1] * clsloss
        return {"loss": loss, "seg_loss": segloss, "cls_loss": clsloss}


class FocalBCELoss_SegCls(nn.Module):

    def __init__(self):
        super().__init__()
        self.wts = torch.tensor([0.5, 0.5])

    def forward(self, p_seg, p_cls, t_seg, t_cls):
        segloss = sigmoid_focal_loss(p_seg.float(), t_seg.float(), reduction="mean")
        clsloss = F.binary_cross_entropy_with_logits(p_cls.float(), t_cls.float())
        loss = self.wts[0] * segloss + self.wts[1] * clsloss
        return loss
