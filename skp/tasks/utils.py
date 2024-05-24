import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from . import samplers as custom_samplers 


def build_dataloader(cfg, dataset, mode):

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    dataloader_params = {}
    dataloader_params["num_workers"] = cfg.num_workers
    dataloader_params["drop_last"] = mode == "train"
    dataloader_params["shuffle"] = mode == "train"
    dataloader_params["pin_memory"] = cfg.pin_memory or True
    dataloader_params["collate_fn"] = dataset.collate_fn

    if mode == "train":
        dataloader_params["batch_size"] = cfg.batch_size
    else:
        dataloader_params["batch_size"] = cfg.val_batch_size

    sampler = None
    if cfg.sampler and mode == "train":
        sampler = getattr(custom_samplers, cfg.sampler)(dataset=dataset, cfg=cfg)

    if sampler:
        dataloader_params["shuffle"] = False
        if cfg.args["strategy"] == "ddp":
            sampler = custom_samplers.DistributedSamplerWrapper(sampler)
        print(f"Using sampler {sampler} for training ...")
        dataloader_params["sampler"] = sampler
    elif cfg.args["strategy"] == "ddp":
        dataloader_params["shuffle"] = False
        dataloader_params["sampler"] = DistributedSampler(dataset, shuffle=mode == "train")

    loader = DataLoader(dataset,
        **dataloader_params,
        worker_init_fn=worker_init_fn)
    
    return loader
