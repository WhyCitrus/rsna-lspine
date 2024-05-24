import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm


class DiceScore(tm.Metric):
    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg 
        self.add_state("dice_scores", default=[], dist_reduce_fx=None)

    def update(self, p, t):
        # p.shape = (B, C, [Z], H, W)
        # t.shape = (B, C, [Z], H, W)
        assert p.shape[1] == t.shape[1]

        if self.cfg.activation_fn == "softmax":
            p = p.softmax(dim=1)
        else:
            p = p.sigmoid()

        # Just use threshold of 0.5 for now
        p, t = (p >= 0.5).long(), t.long()

        intersection = (p * t).reshape(p.shape[0], p.shape[1], -1).sum(-1)
        denominator = (p + t).reshape(p.shape[0], p.shape[1], -1).sum(-1)

        dice = (2 * intersection) / denominator
        # dice.shape = (B, C)

        self.dice_scores.append(dice)

    def compute(self):
        dice_scores = torch.cat(self.dice_scores, dim=0) # (N, C)
        metrics_dict = {}
        for c in range(dice_scores.shape[1]):
            # ignore NaN
            tmp_dice_scores = dice_scores[:, c].cpu().numpy()
            tmp_dice_scores = tmp_dice_scores[~np.isnan(tmp_dice_scores)]
            metrics_dict[f"dice{c}"] = np.mean(tmp_dice_scores)
        metrics_dict["dice_mean"] = np.mean([v for v in metrics_dict.values()])
        return metrics_dict


class DiceScoreOneHot(DiceScore):

    def update(self, p, t):
        # p.shape = (B, C, [Z], H, W)
        # t.shape = (B, [Z], H, W)
        assert p.ndim - 1 == t.ndim 
        # need to add 1 back
        t = F.one_hot(t + 1, num_classes=p.shape[1])[..., 1:].movedim(-1, 0)
        assert p.shape[1] == t.shape[1]
        if self.cfg.activation_fn == "softmax":
            p = p.softmax(dim=1)
        else:
            p = p.sigmoid()

        # Just use threshold of 0.5 for now
        p, t = (p >= 0.5).long(), t.long()

        intersection = (p * t).reshape(p.shape[0], p.shape[1], -1).sum(-1)
        denominator = (p + t).reshape(p.shape[0], p.shape[1], -1).sum(-1)

        dice = (2 * intersection) / denominator
        # dice.shape = (B, C)

        self.dice_scores.append(dice)


class DiceScoreStatsOnly(DiceScore):
    """
    Variant of DiceScore that only returns mean, median, min, max, 
    25th percentile, and 75th percentile, as well as the class with the
    lowest value.

    Designed for segmentation tasks with a large number of classes
    (e.g., TotalSegmentator), to minimize the number of metrics which
    are tracked during training.
    """
    def compute(self):
        dice_scores = torch.cat(self.dice_scores, dim=0) # (N, C)
        metrics_dict = {}
        dice_scores_over_classes = []
        for c in range(dice_scores.shape[1]):
            # ignore NaN
            tmp_dice_scores = dice_scores[:, c].cpu().numpy()
            tmp_dice_scores = tmp_dice_scores[~np.isnan(tmp_dice_scores)]
            dice_scores_over_classes.append(np.mean(tmp_dice_scores))
        metrics_dict["dice_mean"] = np.mean(dice_scores_over_classes)
        metrics_dict["dice_median"] = np.median(dice_scores_over_classes)
        metrics_dict["dice_min"] = np.min(dice_scores_over_classes)
        metrics_dict["dice_max"] = np.max(dice_scores_over_classes)
        metrics_dict["dice_25pct"] = np.percentile(dice_scores_over_classes, 25)
        metrics_dict["dice_75pct"] = np.percentile(dice_scores_over_classes, 75)
        metrics_dict["dice_min_class"] = np.argmin(dice_scores_over_classes)
        return metrics_dict


class DiceScoreStatsOnlyOneHot(DiceScoreStatsOnly):

    def update(self, p, t):
        # p.shape = (B, C, [Z], H, W)
        # t.shape = (B, [Z], H, W)
        assert p.ndim - 1 == t.ndim 
        p, t = F.interpolate(p, scale_factor=0.25, mode="nearest"), F.interpolate(t.unsqueeze(1).float(), scale_factor=0.25, mode="nearest").squeeze(1)

        if self.cfg.activation_fn == "softmax":
            p = p.softmax(dim=1)
        else:
            p = p.sigmoid()

        # Just use threshold of 0.5 for now
        p = (p >= 0.5).long()

        intersection = (p * t).reshape(p.shape[0], p.shape[1], -1).sum(-1)
        denominator = (p + t).reshape(p.shape[0], p.shape[1], -1).sum(-1)

        dice = (2 * intersection) / denominator

        # dice_per_class = []
        # for each_class in range(p.shape[1]):
        #     intersection = ((p[:, each_class] >= 0.5) * (t == each_class)).reshape(p.shape[0], -1).sum(-1)
        #     denominator = ((p[:, each_class] >= 0.5) + (t == each_class)).reshape(p.shape[0], -1).sum(-1)
        #     dice = (2 * intersection) / denominator
        #     dice_per_class.append(dice)

        # dice = torch.stack(dice_per_class, dim=1)
        # print(dice.shape)
        # dice.shape = (B, C)

        self.dice_scores.append(dice)

