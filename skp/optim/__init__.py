from torch import optim
from torch.optim import lr_scheduler

from .custom_schedulers import OneCycleLR


def get_optimizer(cfg, model):
    return getattr(optim, cfg.optimizer)(params=model.parameters(), **cfg.optimizer_params)


def get_scheduler(cfg, optimizer):
    if cfg.scheduler == "CosineAnnealingLR":
        if cfg.sampler == "IterationBasedSampler":
            cfg.scheduler_params["T_max"] = (cfg.num_iterations_per_epoch * cfg.num_epochs)
        else:
            cfg.scheduler_params["T_max"] = (cfg.n_train * cfg.num_epochs) // cfg.batch_size // cfg.world_size

    if cfg.scheduler == "OneCycleLR":
        scheduler_obj = OneCycleLR
    else:
        scheduler_obj = getattr(lr_scheduler, cfg.scheduler)

    return scheduler_obj(optimizer=optimizer, **cfg.scheduler_params)
