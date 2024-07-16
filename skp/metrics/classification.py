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
        self.p.append(p.float())
        self.t.append(t.float())

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
            metrics_dict[f"auc{c}"] = _roc_auc_score(tmp_gt, p[:, c])
        metrics_dict[f"auc_mean"] = np.mean([v for v in metrics_dict.values()])
        return metrics_dict


class AUROCBilat(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu() # (N, 6)
        t = torch.cat(self.t, dim=0).cpu() # (N, 6)
        p = torch.cat([p[:, :3], p[:, 3:]], dim=0).numpy()
        t = torch.cat([t[:, :3], t[:, 3:]], dim=0).numpy()
        metrics_dict = {}
        for c in range(p.shape[1]):
            # Depends on whether it is multilabel or multiclass
            # If multiclass using CE loss, p.shape[1] = num_classes and t.shape[1] = 1
            tmp_gt = t == c if t.shape[1] != p.shape[1] else t[:, c]
            metrics_dict[f"auc{c}"] = _roc_auc_score(tmp_gt, p[:, c])
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


class MAE10(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        metrics_dict = {"mae": np.mean(np.abs(p - t))}
        for i in range(p.shape[1]):
            metrics_dict[f"mae{i:02d}"] = np.mean(np.abs(p[:, i] - t[:, i]))
        return metrics_dict


class MAEDistAndCoords(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu()
        p[:, 10:] = p[:, 10:].sigmoid()
        p = p.numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        metrics_dict = {"mae": np.mean(np.abs(p - t))}
        for i in range(p.shape[1]):
            metrics_dict[f"mae{i:02d}"] = np.mean(np.abs(p[:, i] - t[:, i]))
        metrics_dict["mae_dist"] = np.mean(np.abs(p[:, :10] - t[:, :10]))
        metrics_dict["mae_coords"] = np.mean(np.abs(p[:, 10:] - t[:, 10:]))
        return metrics_dict


class MAESigmoid(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).sigmoid().cpu().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        return {"mae": np.mean(np.abs(p - t))}


class MAETanh(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).tanh().cpu().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        metrics_dict = {"mae": np.mean(np.abs(p - t))}
        for i in range(p.shape[1]):
            metrics_dict[f"mae{i:02d}"] = np.mean(np.abs(p[:, i] - t[:, i]))
        return metrics_dict


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
        p = torch.cat(self.p, dim=0).cpu().sigmoid().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        return {"comp_loss": skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)}


class CompetitionMetricTorchAndNumpy(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().sigmoid()
        t = torch.cat(self.t, dim=0).cpu()
        w = torch.ones((len(p), 1))
        w[t[:, 1] == 1] = 2
        w[t[:, 2] == 1] = 4
        torch_loss = F.binary_cross_entropy(p, t, weight=w).item()
        p = p.numpy()
        t = t.numpy()
        w = w.numpy()
        numpy_loss = skm.log_loss(y_true=t, y_pred=p, sample_weight=w[:, 0])
        return {"comp_loss": torch_loss, "numpy_loss": numpy_loss}


class CompetitionMetricBilat(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().sigmoid()
        t = torch.cat(self.t, dim=0).cpu()
        p = torch.cat([p[:, :3], p[:, 3:]], dim=0).numpy()
        t = torch.cat([t[:, :3], t[:, 3:]], dim=0).numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        return {"comp_loss": skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)}


class SubarticularLevelsAndCoords(tm.Metric):
    
    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.cfg = cfg 

        self.add_state("p_coords", default=[], dist_reduce_fx=None)
        self.add_state("p_levels", default=[], dist_reduce_fx=None)
        self.add_state("t_coords", default=[], dist_reduce_fx=None)
        self.add_state("t_levels", default=[], dist_reduce_fx=None)
        self.add_state("included_levels", default=[], dist_reduce_fx=None)

    def update(self, p_coords, p_levels, t_coords, t_levels, included_levels):
        self.p_coords.append(p_coords)
        self.p_levels.append(p_levels)
        self.t_coords.append(t_coords)
        self.t_levels.append(t_levels)
        self.included_levels.append(included_levels)

    def compute(self):
        p_levels = torch.cat(self.p_levels, dim=0).cpu().numpy()
        t_levels = torch.cat(self.t_levels, dim=0).cpu().numpy()
        metrics_dict = {}
        for c in range(p_levels.shape[1]):
            metrics_dict[f"auc{c}"] = _roc_auc_score(t=t_levels[:, c], p=p_levels[:, c])
        metrics_dict["auc_mean"] = np.mean([v for v in metrics_dict.values()])
        p_coords = torch.cat(self.p_coords, dim=0).sigmoid().cpu().numpy()
        t_coords = torch.cat(self.t_coords, dim=0).cpu().numpy()
        coords_loss = np.abs(p_coords - t_coords)
        included_levels = torch.cat(self.included_levels, dim=0).cpu().numpy()
        coords_mean_loss = []
        for b_idx, inc in enumerate(included_levels):
            tmp_indices = np.where(inc)[0]
            tmp_indices = np.concatenate([tmp_indices, tmp_indices + 15])
            coords_mean_loss.append(coords_loss[b_idx, tmp_indices].mean())
        metrics_dict["coords_mae"] = np.mean(coords_mean_loss)
        return metrics_dict


