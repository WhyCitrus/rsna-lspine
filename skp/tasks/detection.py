import numpy as np
import os
import pytorch_lightning as pl
import torch.nn as nn
import torch

from neptune.utils import stringify_unsupported
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .utils import build_dataloader


class Task(pl.LightningModule): 

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def set(self, name, attr):
        if name == "metrics":
            attr = nn.ModuleList(attr) 
        setattr(self, name, attr)
    
    def on_train_start(self): 
        for obj in ["model", "datasets", "optimizer", "scheduler", "metrics", "val_metric"]:
            assert hasattr(self, obj)

        self.logger.experiment["cfg"] = stringify_unsupported(self.cfg.__dict__)

    def _apply_mixaug(self, X, y):
        return apply_mixaug(X, y, self.mixaug)

    def training_step(self, batch, batch_idx):             
        # if isinstance(self.mixaug, dict):
        #     X, y = self._apply_mixaug(X, y)
        out = self.model(batch) 
        out["loss"] = sum(loss for loss in out.values())  
        self.log("loss", out["loss"]) 
        self.log("cls_loss", out["classification"])
        self.log("bbox_loss", out["bbox_regression"])
        return out["loss"]

    def validation_step(self, batch, batch_idx): 
        out = self.model(batch) 
        for m in self.metrics:
            m.update(out, batch["targets"])
        return 0

    def on_validation_epoch_end(self, *args, **kwargs):
        metrics = {}
        for m in self.metrics:
            metrics.update(m.compute())

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
