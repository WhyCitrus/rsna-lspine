import numpy as np
import pandas as pd
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


class CompetitionMetricTorch(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().sigmoid()
        p = p / (p.sum(1).unsqueeze(1) + 1e-10)
        t = torch.cat(self.t, dim=0).cpu()
        w = torch.ones((len(p), ))
        w[t[:, 1] == 1] = 2
        w[t[:, 2] == 1] = 4
        loss = -torch.xlogy(t.float(), p.float()).sum(1)
        loss = loss * w
        loss = loss / w.sum()
        return {"comp_loss_torch": loss.sum().item()}


class CompetitionMetric(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().sigmoid()
        p = p / (p.sum(1).unsqueeze(1) + 1e-10)
        p = p.numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        return {"comp_loss": skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)}


class CompetitionMetricWithSoftmaxTorch(_BaseMetric):

    @staticmethod
    def torch_log_loss_with_logits(logits, t, w=None):
        loss = (-t * F.log_softmax(logits, dim=1)).sum(1)
        if isinstance(w, torch.Tensor):
            loss = loss * w
            return loss.sum() / w.sum()
        else:
            return loss.mean()

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        wts = torch.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        return {"comp_loss_torch": self.torch_log_loss_with_logits(p.float(), t.float(), w=wts)}


class CompetitionMetricWithSoftmax(_BaseMetric):

    def compute(self):
        p = F.softmax(torch.cat(self.p, dim=0).cpu(), dim=1)
        p = p.numpy()
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


class CompetitionMetricPlusAUROCBilateralSoftmax(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu()
        t = torch.cat(self.t, dim=0).cpu()
        p = F.softmax(torch.cat([p[:, :3], p[:, 3:]], dim=0), dim=1).numpy()
        t = torch.cat([t[:, :3], t[:, 3:]], dim=0).numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {"comp_loss": skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)}
        for i in range(p.shape[1]):
            metrics_dict[f"auc{i}"] = _roc_auc_score(t=t[:, i], p=p[:, i])
        metrics_dict["auc_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc" in k])
        return metrics_dict


class CompetitionMetricPlusAUROCSigmoid(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0).cpu().sigmoid()
        t = torch.cat(self.t, dim=0).cpu()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {"comp_loss": skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)}
        for i in range(p.shape[1]):
            metrics_dict[f"auc{i}"] = _roc_auc_score(t=t[:, i], p=p[:, i])
        metrics_dict["auc_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc" in k])
        return metrics_dict


class CompetitionMetricPlusAUROCMultiAug(tm.Metric):

    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("unique_id", default=[], dist_reduce_fx=None)

    def update(self, p, t, unique_id):
        self.p.append(p)
        self.t.append(t)
        self.unique_id.append(unique_id)

    def compute(self):
        p = F.softmax(torch.cat(self.p, dim=0).cpu().float(), dim=1).numpy()
        t = torch.cat(self.t).cpu().numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {}
        metrics_dict["loss_crop"] = skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_crop{c}"] = _roc_auc_score(t=t[:, c], p=p[:, c])
        metrics_dict["auc_crop_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_crop" in k])
        # Take mean and median of crop augs
        tmp_df = pd.DataFrame({
            "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], 
            "t0": t[:, 0], "t1": t[:, 1], "t2": t[:, 2],  
            "wts": wts, 
            "unique_id": torch.cat(self.unique_id).cpu().numpy()
        })
        df_by_mean = tmp_df.groupby("unique_id").mean() 
        df_by_median = tmp_df.groupby("unique_id").median()
        metrics_dict["loss_mean"] = skm.log_loss(y_true=df_by_mean[["t0", "t1", "t2"]].values, 
                                                 y_pred=df_by_mean[["p0", "p1", "p2"]].values, 
                                                 sample_weight=df_by_mean.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_meanagg{c}"] = _roc_auc_score(t=df_by_mean[f"t{c}"].values, p=df_by_mean[f"p{c}"].values)
        metrics_dict["auc_meanagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_meanagg" in k])
        metrics_dict["loss_median"] = skm.log_loss(y_true=df_by_median[["t0", "t1", "t2"]].values, 
                                                   y_pred=df_by_median[["p0", "p1", "p2"]].values, 
                                                   sample_weight=df_by_median.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_medianagg{c}"] = _roc_auc_score(t=df_by_median[f"t{c}"].values, p=df_by_median[f"p{c}"].values)
        metrics_dict["auc_medianagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_medianagg" in k])
        return metrics_dict


class CompetitionMetricPlusAUROCMultiAugSigmoid(tm.Metric):

    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("unique_id", default=[], dist_reduce_fx=None)

    def update(self, p, t, unique_id):
        self.p.append(p)
        self.t.append(t)
        self.unique_id.append(unique_id)

    def compute(self):
        p = torch.cat(self.p).sigmoid().cpu().float().numpy()
        t = torch.cat(self.t).cpu().numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {}
        metrics_dict["loss_crop"] = skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_crop{c}"] = _roc_auc_score(t=t[:, c], p=p[:, c])
        metrics_dict["auc_crop_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_crop" in k])
        # Take mean and median of crop augs
        tmp_df = pd.DataFrame({
            "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], 
            "t0": t[:, 0], "t1": t[:, 1], "t2": t[:, 2],  
            "wts": wts, 
            "unique_id": torch.cat(self.unique_id).cpu().numpy()
        })
        df_by_mean = tmp_df.groupby("unique_id").mean() 
        df_by_median = tmp_df.groupby("unique_id").median()
        metrics_dict["loss_mean"] = skm.log_loss(y_true=df_by_mean[["t0", "t1", "t2"]].values, 
                                                 y_pred=df_by_mean[["p0", "p1", "p2"]].values, 
                                                 sample_weight=df_by_mean.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_meanagg{c}"] = _roc_auc_score(t=df_by_mean[f"t{c}"].values, p=df_by_mean[f"p{c}"].values)
        metrics_dict["auc_meanagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_meanagg" in k])
        metrics_dict["loss_median"] = skm.log_loss(y_true=df_by_median[["t0", "t1", "t2"]].values, 
                                                   y_pred=df_by_median[["p0", "p1", "p2"]].values, 
                                                   sample_weight=df_by_median.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_medianagg{c}"] = _roc_auc_score(t=df_by_median[f"t{c}"].values, p=df_by_median[f"p{c}"].values)
        metrics_dict["auc_medianagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_medianagg" in k])
        return metrics_dict


class CompetitionMetricPlusAUROCMultiAugSigmoidPseudo(tm.Metric):

    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("unique_id", default=[], dist_reduce_fx=None)

    def update(self, p, t, unique_id):
        self.p.append(p)
        self.t.append(t)
        self.unique_id.append(unique_id)

    def compute(self):
        p = torch.cat(self.p).sigmoid().cpu().float().numpy()
        t = torch.cat(self.t).cpu().numpy()[:, :3]
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {}
        metrics_dict["loss_crop"] = skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_crop{c}"] = _roc_auc_score(t=t[:, c], p=p[:, c])
        metrics_dict["auc_crop_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_crop" in k])
        # Take mean and median of crop augs
        tmp_df = pd.DataFrame({
            "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], 
            "t0": t[:, 0], "t1": t[:, 1], "t2": t[:, 2],  
            "wts": wts, 
            "unique_id": torch.cat(self.unique_id).cpu().numpy()
        })
        df_by_mean = tmp_df.groupby("unique_id").mean() 
        df_by_median = tmp_df.groupby("unique_id").median()
        metrics_dict["loss_mean"] = skm.log_loss(y_true=df_by_mean[["t0", "t1", "t2"]].values, 
                                                 y_pred=df_by_mean[["p0", "p1", "p2"]].values, 
                                                 sample_weight=df_by_mean.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_meanagg{c}"] = _roc_auc_score(t=df_by_mean[f"t{c}"].values, p=df_by_mean[f"p{c}"].values)
        metrics_dict["auc_meanagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_meanagg" in k])
        metrics_dict["loss_median"] = skm.log_loss(y_true=df_by_median[["t0", "t1", "t2"]].values, 
                                                   y_pred=df_by_median[["p0", "p1", "p2"]].values, 
                                                   sample_weight=df_by_median.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_medianagg{c}"] = _roc_auc_score(t=df_by_median[f"t{c}"].values, p=df_by_median[f"p{c}"].values)
        metrics_dict["auc_medianagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_medianagg" in k])
        return metrics_dict


class CompetitionMetricPlusAUROCMultiAugSigmoidSubarticularBilat(tm.Metric):

    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("unique_id", default=[], dist_reduce_fx=None)

    def update(self, p, t, unique_id):
        self.p.append(p)
        self.t.append(t)
        self.unique_id.append(unique_id)

    def compute(self):
        p = torch.cat(self.p)
        p = F.softmax(torch.cat([p[:, :3], p[:, 3:]], dim=0), dim=1).cpu().float().numpy()
        t = torch.cat(self.t)
        t = torch.cat([t[:, :3], t[:, 3:]], dim=0).cpu().numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {}
        metrics_dict["loss_crop"] = skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_crop{c}"] = _roc_auc_score(t=t[:, c], p=p[:, c])
        metrics_dict["auc_crop_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_crop" in k])
        unique_id = torch.cat(self.unique_id)
        unique_id = torch.cat([unique_id, unique_id + 1000000]).cpu().numpy()
        # Take mean and median of crop augs
        tmp_df = pd.DataFrame({
            "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], 
            "t0": t[:, 0], "t1": t[:, 1], "t2": t[:, 2],  
            "wts": wts, 
            "unique_id": unique_id
        })
        df_by_mean = tmp_df.groupby("unique_id").mean() 
        df_by_median = tmp_df.groupby("unique_id").median()
        metrics_dict["loss_mean"] = skm.log_loss(y_true=df_by_mean[["t0", "t1", "t2"]].values, 
                                                 y_pred=df_by_mean[["p0", "p1", "p2"]].values, 
                                                 sample_weight=df_by_mean.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_meanagg{c}"] = _roc_auc_score(t=df_by_mean[f"t{c}"].values, p=df_by_mean[f"p{c}"].values)
        metrics_dict["auc_meanagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_meanagg" in k])
        metrics_dict["loss_median"] = skm.log_loss(y_true=df_by_median[["t0", "t1", "t2"]].values, 
                                                   y_pred=df_by_median[["p0", "p1", "p2"]].values, 
                                                   sample_weight=df_by_median.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_medianagg{c}"] = _roc_auc_score(t=df_by_median[f"t{c}"].values, p=df_by_median[f"p{c}"].values)
        metrics_dict["auc_medianagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_medianagg" in k])
        return metrics_dict


class CompetitionMetricPlusAUROCMultiAugSigmoidSpinalSubarticular(tm.Metric):

    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("unique_id", default=[], dist_reduce_fx=None)

    def update(self, p, t, unique_id):
        self.p.append(p)
        self.t.append(t)
        self.unique_id.append(unique_id)

    def compute_subarticular(self):
        p = torch.cat(self.p)[:, :6]
        t = torch.cat(self.t)[:, :6]
        p = torch.cat([p[:, :3], p[:, 3:]], dim=0).sigmoid().cpu().float().numpy()
        t = torch.cat([t[:, :3], t[:, 3:]], dim=0).cpu().numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {}
        metrics_dict["loss_crop"] = skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_crop{c}"] = _roc_auc_score(t=t[:, c], p=p[:, c])
        metrics_dict["auc_crop_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_crop" in k])
        unique_id = torch.cat(self.unique_id)
        unique_id = torch.cat([unique_id, unique_id + 1000000]).cpu().numpy()
        # Take mean and median of crop augs
        tmp_df = pd.DataFrame({
            "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], 
            "t0": t[:, 0], "t1": t[:, 1], "t2": t[:, 2],  
            "wts": wts, 
            "unique_id": unique_id
        })
        df_by_mean = tmp_df.groupby("unique_id").mean() 
        df_by_median = tmp_df.groupby("unique_id").median()
        metrics_dict["loss_mean"] = skm.log_loss(y_true=df_by_mean[["t0", "t1", "t2"]].values, 
                                                 y_pred=df_by_mean[["p0", "p1", "p2"]].values, 
                                                 sample_weight=df_by_mean.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_meanagg{c}"] = _roc_auc_score(t=df_by_mean[f"t{c}"].values, p=df_by_mean[f"p{c}"].values)
        metrics_dict["auc_meanagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_meanagg" in k])
        metrics_dict["loss_median"] = skm.log_loss(y_true=df_by_median[["t0", "t1", "t2"]].values, 
                                                   y_pred=df_by_median[["p0", "p1", "p2"]].values, 
                                                   sample_weight=df_by_median.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_medianagg{c}"] = _roc_auc_score(t=df_by_median[f"t{c}"].values, p=df_by_median[f"p{c}"].values)
        metrics_dict["auc_medianagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_medianagg" in k])
        return {f"{k}_subart": v for k, v in metrics_dict.items()}

    def compute_spinal(self):
        p = torch.cat(self.p)[:, 6:].sigmoid().cpu().float().numpy()
        t = torch.cat(self.t)[:, 6:].cpu().numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {}
        metrics_dict["loss_crop"] = skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_crop{c}"] = _roc_auc_score(t=t[:, c], p=p[:, c])
        metrics_dict["auc_crop_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_crop" in k])
        # Take mean and median of crop augs
        tmp_df = pd.DataFrame({
            "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], 
            "t0": t[:, 0], "t1": t[:, 1], "t2": t[:, 2],  
            "wts": wts, 
            "unique_id": torch.cat(self.unique_id).cpu().numpy()
        })
        df_by_mean = tmp_df.groupby("unique_id").mean() 
        df_by_median = tmp_df.groupby("unique_id").median()
        metrics_dict["loss_mean"] = skm.log_loss(y_true=df_by_mean[["t0", "t1", "t2"]].values, 
                                                 y_pred=df_by_mean[["p0", "p1", "p2"]].values, 
                                                 sample_weight=df_by_mean.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_meanagg{c}"] = _roc_auc_score(t=df_by_mean[f"t{c}"].values, p=df_by_mean[f"p{c}"].values)
        metrics_dict["auc_meanagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_meanagg" in k])
        metrics_dict["loss_median"] = skm.log_loss(y_true=df_by_median[["t0", "t1", "t2"]].values, 
                                                   y_pred=df_by_median[["p0", "p1", "p2"]].values, 
                                                   sample_weight=df_by_median.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_medianagg{c}"] = _roc_auc_score(t=df_by_median[f"t{c}"].values, p=df_by_median[f"p{c}"].values)
        metrics_dict["auc_medianagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_medianagg" in k])
        return {f"{k}_spinal": v for k, v in metrics_dict.items()}

    def compute(self):
        subart_metrics = self.compute_subarticular()
        subart_metrics.update(self.compute_spinal())
        subart_metrics["loss_median"] = (subart_metrics["loss_median_subart"] + subart_metrics["loss_median_spinal"]) / 2
        return subart_metrics


class CompetitionMetricPlusAUROCMultiAugSigmoidAll(tm.Metric):

    def __init__(self, cfg, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("p", default=[], dist_reduce_fx=None)
        self.add_state("t", default=[], dist_reduce_fx=None)
        self.add_state("unique_id", default=[], dist_reduce_fx=None)

    def update(self, p, t, unique_id):
        self.p.append(p)
        self.t.append(t)
        self.unique_id.append(unique_id)

    def compute_one(self, p, t, unique_id, suffix):
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {}
        metrics_dict["loss_crop"] = skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_crop{c}"] = _roc_auc_score(t=t[:, c], p=p[:, c])
        metrics_dict["auc_crop_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_crop" in k])
        # Take mean and median of crop augs
        tmp_df = pd.DataFrame({
            "p0": p[:, 0], "p1": p[:, 1], "p2": p[:, 2], 
            "t0": t[:, 0], "t1": t[:, 1], "t2": t[:, 2],  
            "wts": wts, 
            "unique_id": unique_id
        })
        df_by_mean = tmp_df.groupby("unique_id").mean() 
        df_by_median = tmp_df.groupby("unique_id").median()
        metrics_dict["loss_mean"] = skm.log_loss(y_true=df_by_mean[["t0", "t1", "t2"]].values, 
                                                 y_pred=df_by_mean[["p0", "p1", "p2"]].values, 
                                                 sample_weight=df_by_mean.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_meanagg{c}"] = _roc_auc_score(t=df_by_mean[f"t{c}"].values, p=df_by_mean[f"p{c}"].values)
        metrics_dict["auc_meanagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_meanagg" in k])
        metrics_dict["loss_median"] = skm.log_loss(y_true=df_by_median[["t0", "t1", "t2"]].values, 
                                                   y_pred=df_by_median[["p0", "p1", "p2"]].values, 
                                                   sample_weight=df_by_median.wts.values)
        for c in range(p.shape[1]):
            metrics_dict[f"auc_medianagg{c}"] = _roc_auc_score(t=df_by_median[f"t{c}"].values, p=df_by_median[f"p{c}"].values)
        metrics_dict["auc_medianagg_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc_medianagg" in k])
        return {f"{k}_{suffix}": v for k, v in metrics_dict.items()}

    def compute(self):
        p = torch.cat(self.p, dim=0).sigmoid().cpu().numpy()
        t = torch.cat(self.t, dim=0).cpu().numpy()
        unique_id = torch.cat(self.unique_id).cpu().numpy()
        try:
            foramen_metrics = self.compute_one(p=p[unique_id < 100000, :3], t=t[unique_id < 100000, :3], unique_id=unique_id[unique_id < 100000], suffix="foramen")
            spinal_metrics = self.compute_one(p=p[((unique_id >= 100000) & (unique_id < 200000)), 3:6], 
                                              t=t[((unique_id >= 100000) & (unique_id < 200000)), 3:6], 
                                              unique_id=unique_id[((unique_id >= 100000) & (unique_id < 200000))], suffix="spinal")
            subart_metrics = self.compute_one(p=p[unique_id >= 200000, 6:], t=t[unique_id >= 200000, 6:], unique_id=unique_id[unique_id >= 200000], suffix="subart")
            subart_metrics.update(foramen_metrics)
            subart_metrics.update(spinal_metrics)
            subart_metrics["loss_median"] = (subart_metrics["loss_median_subart"] + subart_metrics["loss_median_spinal"] + subart_metrics["loss_median_foramen"]) / 3
        except ValueError as e:
            print(e)
            subart_metrics = self.compute_one(p[:, :3], t[:, :3], unique_id, suffix="")
            subart_metrics["loss_median"] = 0
        return subart_metrics


class CompetitionMetricPlusAUROCWholeSpinal(_BaseMetric):

    def compute(self):
        p = torch.cat(self.p, dim=0)[:, :15]
        t = torch.cat(self.t, dim=0)[:, :15]
        p = torch.cat([p[:, i:i+3] for i in range(0, 15, 3)], dim=0)
        t = torch.cat([t[:, i:i+3] for i in range(0, 15, 3)], dim=0)
        p = p.sigmoid().cpu().numpy()
        t = t.cpu().numpy()
        wts = np.ones((len(p), ))
        wts[t[:, 1] == 1] = 2
        wts[t[:, 2] == 1] = 4
        metrics_dict = {}
        metrics_dict["comp_loss"] = skm.log_loss(y_true=t, y_pred=p, sample_weight=wts)
        for c in range(p.shape[1]):
            metrics_dict[f"auc{c}"] = _roc_auc_score(t=t[:, c], p=p[:, c])
        metrics_dict["auc_mean"] = np.mean([v for k, v in metrics_dict.items() if "auc" in k])
        return metrics_dict


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


