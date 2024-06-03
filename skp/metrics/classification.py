import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm

from functools import partial
from sklearn import metrics as skm 


def _roc_auc_score(t, p):
    try:
        return torch.tensor(skm.roc_auc_score(t, p))
    except Exception as e:
        print(e)
        return torch.tensor(0.5)


def _average_precision_score(t, p):
    try:
        return torch.tensor(skm.average_precision_score(t, p))
    except Exception as e:
        print(e)
        return torch.tensor(0)


class _BaseMetric(tm.Metric):
    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.cfg = cfg 

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)

    def update(self, p, t):
        self.p.append(p)
        self.t.append(t)

    def compute(self):
        raise NotImplementedError


class _ScoreBased(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()  # (N,) or (N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy()  # (N,) or (N,C)
        if p.ndim == 1:
            # Binary classification
            return {f"{self.name}_mean": self.metric_func(t, p)}
        metrics_dict = {}
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            # If multiclass using CE loss, p.shape[1] = num_classes and t.shape[1] = 1
            tmp_gt = t == c if t.shape[1] != p.shape[1] else t[:, c]
            metrics_dict[f"{self.name}{c}"] = self.metric_func(tmp_gt, p[:, c])
        metrics_dict[f"{self.name}_mean"] = np.mean([v for v in metrics_dict.values()])
        return metrics_dict


class _ClassBased(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()  # (N,) or (N,C)
        t = torch.cat(self.t, dim=0).cpu().numpy()  # (N,) or (N,C)
        if p.ndim == 1:
            # Binary classification
            return {f"{self.name}": self.metric_func(t, p)}
        p = np.argmax(p, axis=1)
        return {f"{self.name}": self.metric_func(t, p)} 


class AUROC(_ScoreBased):

    name = "auc"
    def metric_func(self, t, p): return _roc_auc_score(t, p)


class AUROCFlatten(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu() # (N, seq_len, C)
        t = torch.cat(self.t, dim=0).cpu() # (N, seq_len, C)
        p, t = p.view(-1, p.shape[-1]).numpy(), t.view(-1, t.shape[-1]).numpy()
        metrics_dict = {}
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            # If multiclass using CE loss, p.shape[1] = num_classes and t.shape[1] = 1
            tmp_gt = t == c if t.shape[1] != p.shape[1] else t[:, c]
            metrics_dict[f"auc_{c}"] = _roc_auc_score(tmp_gt, p[:, c])
        metrics_dict[f"auc_mean"] = np.mean([v for v in metrics_dict.values()])
        return metrics_dict


class AVP(_ScoreBased):

    name = "avp"
    def metric_func(self, t, p): return _average_precision_score(t, p)


class Accuracy(_ClassBased):

    name = "accuracy"
    def metric_func(self, t, p): return skm.accuracy_score(t, p)


class Kappa(_ClassBased):

    name = "kappa"
    def metric_func(self, t, p): return skm.cohen_kappa_score(t, p)


class QWK(_ClassBased):
    name = "qwk"
    def metric_func(self, t, p): return skm.cohen_kappa_score(t, p, weights="quadratic")


class MAE(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        return {"mae": np.mean(np.abs(p - t))}


class MAESigmoid(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).sigmoid().cpu().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        return {"mae": np.mean(np.abs(p - t))}


class MAEScale100(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        return {"mae": np.mean(np.abs(p - t)) * 100}


class MAE_CE(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()
        p = np.argmax(p, axis=1) + 18 
        t = torch.cat(self.t, dim=0).cpu().numpy()[:, 0] + 18
        assert p.shape == t.shape, f"p.shape is {p.shape} while t.shape is {t.shape}"
        return {"mae": np.mean(np.abs(p - t))}


class CompetitionMetric(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu() # (N, 3)
        t = torch.cat(self.t, dim=0).cpu() # (N, 3)
        wts = torch.tensor([1.0, 2.0, 4.0])
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none")
        loss = wts * loss
        loss = loss.mean().numpy()
        return {"comp_loss": loss}


class CompetitionMetricIgnoreAreas(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu()[:, :3] # (N, 3)
        t = torch.cat(self.t, dim=0).cpu()[:, :3] # (N, 3)
        assert p.size(1) == t.size(1) == 3
        wts = torch.tensor([1.0, 2.0, 4.0])
        loss = F.binary_cross_entropy_with_logits(p.float(), t.float(), reduction="none")
        loss = wts * loss
        loss = np.mean(loss.numpy())
        return {"comp_loss": loss}
