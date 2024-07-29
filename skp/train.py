import argparse
import losses
import metrics
import neptune
import os
import pickle
import pytorch_lightning as pl
import sys
import torch
import uuid

from importlib import import_module
from losses import get_loss
from optim import get_optimizer, get_scheduler
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.plugins import TorchSyncBatchNorm
from timm.layers import convert_sync_batchnorm


class TimmSyncBatchNorm(TorchSyncBatchNorm):
    """
    Default SyncBN plugin for Lightning does not work with latest version of timm
    EfficientNets because it uses the native PyTorch `convert_sync_batchnorm` function.

    Use this plugin instead, which uses the timm helper and should work for non-timm
    models as well.
    """
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        return convert_sync_batchnorm(model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--sync_batchnorm", action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    return parser.parse_known_args()


def load_config(args, overwrite_args):

    cfg_file = args.__dict__.pop("config")
    cfg = import_module(f"configs.{cfg_file}").cfg

    cfg.args = args.__dict__

    cfg.config = cfg_file

    if len(overwrite_args) > 1:
        # assumes all overwrite args are prepended with '--''
        # overwrite_args will be a list following:
        # [arg_name, arg_value, arg_name, arg_value, ...]
        # here we turn it into a dict following: {arg_name: arg_value}
        overwrite_args = {k.replace("-", ""):v for k, v in zip(overwrite_args[::2], overwrite_args[1::2])}

    # some config parameters, such as loss_params or optimizer_params
    # are dictionaries
    # to overwrite these params via CLI, we use dot concatenation to specify
    # params within these dictionaries i.e., optimizer_params.lr 
    for key in overwrite_args:

        # require that you can only overwrite an existing config parameter
        # split by . to deal with subparams
        if key.split(".")[0] in cfg.__dict__:

            if len(key.split(".")) == 1:
                # no subparam, just overwrite
                print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {overwrite_args[key]}')
            else:
                # subparam, need to identify dict param and key-value pair
                param_dict, param_key = key.split(".")
                print(f'overwriting cfg.{key}: {cfg.__dict__[param_dict][param_key]} -> {overwrite_args[key]}')

            cfg_type = type(cfg.__dict__[key.split(".")[0]])
            # check if param is a dict
            if cfg_type == dict:
                assert len(key.split(".")) > 1
                param_dict, param_key = key.split(".")
                param_type = type(cfg.__dict__[param_dict][param_key])
                if param_type == bool:
                    # note that because we are not using argparse to add arguments
                    # we cannot have `store_true` args, so if it is a boolean, we need
                    # to specify True after the arg flag in command line
                    cfg.__dict__[param_dict][param_key] = overwrite_args[key] == "True"
                elif param_type == type(None):
                    cfg.__dict__[param_dict][param_key] = overwrite_args[key]
                else:
                    cfg.__dict__[param_dict][param_key] = param_type(overwrite_args[key])
            else:
                if cfg_type == bool:
                    cfg.__dict__[key] = overwrite_args[key] == 'True'
                elif cfg_type == type(None):
                    cfg.__dict__[key] = overwrite_args[key]
                else:
                    cfg.__dict__[key] = cfg_type(overwrite_args[key])
        else:
            raise Exception(f"{key} is not specified in config")
    if isinstance(cfg.load_pretrained_backbone, list):
        cfg.load_pretrained_backbone = cfg.load_pretrained_backbone[cfg.fold]
    
    return cfg, args


def symlink_best_model_path(trainer):
    wd = os.getcwd()
    best_model_path = None
    for callback in trainer.callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_model_path = callback.best_model_path
            break
    if best_model_path:
        save_checkpoint_path = os.path.dirname(best_model_path)
        os.chdir(save_checkpoint_path)
        if os.path.exists("best.ckpt"):
            _ = os.system(f"rm best.ckpt")
        _ = os.system(f"ln -s {best_model_path.split('/')[-1]} best.ckpt")
        os.chdir(wd)


def get_trainer(cfg, args):
    
    save_dir = "."
    run_id = uuid.uuid4().hex[:8]
    cfg.run_id = run_id
    while os.path.exists(save_dir):
        save_dir = os.path.join(cfg.save_dir, cfg.config, cfg.run_id, f"fold{cfg.fold}")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    cfg.save_dir = save_dir 

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            # Set dirpath explicitly to save checkpoints in the desired folder
            # This is so that we can keep the desired directory structure and format locally
            dirpath=os.path.join(save_dir, "checkpoints"),
            monitor="val_metric",
            filename="{epoch:03d}-{val_metric:.4f}",
            save_last=True,
            save_weights_only=True,
            mode=cfg.val_track,
            save_top_k=getattr(cfg, "save_top_k") or 1,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    if cfg.early_stopping:
        print(">> Using early stopping ...")
        early_stopping = pl.callbacks.EarlyStopping(
            patience=cfg.early_stopping_patience,
            monitor="val_metric",
            min_delta=cfg.early_stopping_min_delta,
            verbose=getattr(cfg, "early_stopping_verbose") or False,
            mode=cfg.val_track,
        )
        callbacks.append(early_stopping)

    if cfg.args["strategy"] == "ddp": 
        strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
        plugins = [TimmSyncBatchNorm()]
    else:
        strategy = cfg.args["strategy"]
        plugins = None

    neptune_logger = NeptuneLogger(project=cfg.project, 
                                   source_files=[f"configs/{cfg.config}.py", f"models/{cfg.model}.py", f"datasets/{cfg.dataset}.py"], 
                                   mode=cfg.neptune_mode,
                                   log_model_checkpoints=False)
    
    args_dict = args.__dict__

    trainer = pl.Trainer(
        **args_dict,
        max_epochs=cfg.num_epochs,
        callbacks=callbacks,
        plugins=plugins,
        logger=neptune_logger,
        # easier to handle custom samplers if below is False
        # see tasks/samplers.py
        # just use native torch DistributedSampler as default
        use_distributed_sampler=False,
        accumulate_grad_batches=cfg.accumulate_grad_batches or 1,
        profiler="simple",
    )

    return trainer, cfg


def main():

    # uses parse_known_args() to separate into specified args 
    # in parse_args and unknown args which will be exclusively used for 
    # overwriting config parameters
    args, overwrite_args = parse_args()

    cfg, args = load_config(args, overwrite_args)
    cfg.world_size = args.num_nodes * (args.devices if args.devices else 1)

    print("\nENVIRONMENT\n")
    print(f"  Python {sys.version}\n")
    print(f"  torch.__version__              = {torch.__version__}")
    print(f"  torch.version.cuda             = {torch.version.cuda}")
    print(f"  torch.backends.cudnn.version() = {torch.backends.cudnn.version()}\n")
    print(f"  pytorch_lightning.__version__  = {pl.__version__}\n")
    print(f"  world_size={cfg.world_size}, num_nodes={args.num_nodes}, num_gpus={args.devices if args.devices else 1}")
    print("\n")
    
    model = import_module(f"models.{cfg.model}").Net(cfg)
    if not getattr(model, "has_loss", False):
        loss = get_loss(cfg)
        model.set_criterion(loss)
    ds_class = import_module(f"datasets.{cfg.dataset}").Dataset
    train_dataset = ds_class(cfg, "train")
    val_dataset = ds_class(cfg, "val")
    cfg.n_train, cfg.n_val = len(train_dataset), len(val_dataset)
    print(f"TRAIN : N={cfg.n_train}")
    print(f"VAL   : N={cfg.n_val}\n")
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    task = import_module(f"tasks.{cfg.task}").Task(cfg) 

    task.set("model", model)
    task.set("datasets", [train_dataset, val_dataset])
    task.set("optimizer", optimizer)
    task.set("scheduler", scheduler)
    task.set("metrics", [getattr(metrics, m)(cfg) for m in cfg.metrics])
    task.set("val_metric", cfg.val_metric)

    trainer, cfg = get_trainer(cfg, args)
    trainer.fit(task)
    symlink_best_model_path(trainer)

    # Can think about a solution not using pickle at a later point
    # Mainly to save a copy of the config if it was modified by command line
    # arguments, since the original config would not be correct in that case
    # Although the parameters should be correct in Neptune
    with open(os.path.join(cfg.save_dir, "config.pkl"), "wb") as f:
        pickle.dump(cfg, f)

    if cfg.neptune_mode == "offline":
        # Avoid multiple uploads in case using server which would potentially flag
        # as suspicious activity
        st = os.system(f"neptune sync --project {cfg.project}")


if __name__ == "__main__":
    main()
