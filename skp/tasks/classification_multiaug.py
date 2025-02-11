import numpy as np
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch

from collections import defaultdict
from neptune.utils import stringify_unsupported
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import build_dataloader


class Task(pl.LightningModule): 

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.val_loss = defaultdict(list)

    def set(self, name, attr):
        if name == "metrics":
            attr = nn.ModuleList(attr) 
        setattr(self, name, attr)
    
    def on_train_start(self): 
        for obj in ["model", "datasets", "optimizer", "scheduler", "metrics", "val_metric"]:
            assert hasattr(self, obj)

        self.logger.experiment["cfg"] = stringify_unsupported(self.cfg.__dict__)

    def mixup(self, batch):
        x, y =  batch["x"], batch["y"]
        assert x.dtype == torch.float, f"x.dtype is {x.dtype}, not float"
        assert y.dtype == torch.float, f"y.dtype is {y.dtype}, not float"
        batch_size = y.size(0)
        lamb = np.random.beta(self.cfg.mixup, self.cfg.mixup, batch_size)
        lamb = torch.from_numpy(lamb).to(x.device).float()
        permuted_indices = torch.randperm(batch_size)
        # also compute mixup of weights
        weights = torch.ones(batch_size).to(x.device)
        weights[y[:, 1] == 1] = 2
        weights[y[:, 2] == 1] = 4
        weights = lamb * weights + (1 - lamb) * weights[permuted_indices]
        if lamb.ndim < y.ndim:
            for _ in range(y.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)
        ymix = lamb * y + (1 - lamb) * y[permuted_indices]
        if lamb.ndim < x.ndim:
            for _ in range(x.ndim - lamb.ndim):
                lamb = lamb.unsqueeze(-1)
        assert lamb.ndim == x.ndim, f"lamb has {lamb.ndim} dims whereas x has {x.ndim} dims"
        xmix = lamb * x + (1 - lamb) * x[permuted_indices]
        batch["x"] = xmix
        batch["y"] = ymix
        batch["wts"] = weights
        return batch

    def training_step(self, batch, batch_idx):             
        if isinstance(self.cfg.mixup, float):
            batch = self.mixup(batch)
        out = self.model(batch, return_loss=True) 
        for k, v in out.items():
            if "loss" in k:
                self.log(k, v)
        return out["loss"]

    def validation_step(self, batch, batch_idx): 
        out = self.model(batch, return_loss=True) 
        for k, v in out.items():
            if "loss" in k:
                self.val_loss[k].append(v)
        for m in self.metrics:
            if self.cfg.model == "all_levels_net_2d":
                unique_ids = batch["unique_id"]
                unique_ids = [_[i] for i in range(len(unique_ids[0])) for _ in unique_ids]
                unique_ids = torch.tensor(unique_ids, device=unique_ids[0].device)
                y = batch["y"]
                y = y.reshape(len(y) * 5, -1)
                m.update(out.get("logits", None), y, unique_ids)
            elif self.cfg.model == "net_2d_all_slices_seq":
                logits = out["logits"]
                sz = len(logits)
                y = batch["y"].reshape(sz, -1)
                mask = out["mask"].reshape(sz)
                unique_ids = batch["unique_id"].reshape(sz)
                logits = logits[~mask]
                y = y[~mask]
                unique_ids = unique_ids[~mask]
                m.update(logits, y, unique_ids)
            else:
                m.update(out.get("logits", None), batch.get("y", None), batch.get("unique_id", None))
        return out["loss"]

    def on_validation_epoch_end(self, *args, **kwargs):
        metrics = {}
        for m in self.metrics:
            metrics.update(m.compute())
        for k, v in self.val_loss.items():
            metrics[k] = torch.stack(v).mean()
        self.val_loss = defaultdict(list)

        if isinstance(self.val_metric, list):
            metrics["val_metric"] = torch.sum(torch.stack([metrics[_vm.lower()].cpu() for _vm in self.val_metric]))
        else:
            metrics["val_metric"] = metrics[self.val_metric.lower()]

        for m in self.metrics: m.reset()

        if self.global_rank == 0:
            print("\n========")
            max_strlen = max([len(k) for k in metrics.keys()])
            for k,v in metrics.items(): 
                print(f"{k.ljust(max_strlen)} | {v.item() if isinstance(v, torch.Tensor) else v:.4f}")

        if self.trainer.state.stage != pl.trainer.states.RunningStage.SANITY_CHECKING: # don't log metrics during sanity check

            for k,v in metrics.items():
                self.logger.experiment[f"val/{k}"].append(v)

            self.log("val_metric", metrics["val_metric"], sync_dist=True)

    def configure_optimizers(self):
        lr_scheduler = {
            "scheduler": self.scheduler,
            "interval": self.cfg.scheduler_interval
        }
        if isinstance(self.scheduler, ReduceLROnPlateau): 
            lr_scheduler["monitor"] = self.val_metric
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": lr_scheduler
            }

    def train_dataloader(self):
        return build_dataloader(self.cfg, self.datasets[0], "train")

    def val_dataloader(self):
        return build_dataloader(self.cfg, self.datasets[1], "val")
