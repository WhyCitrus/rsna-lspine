import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/sandbox"

cfg.task = "classification"

cfg.model = "basic_transformer"
cfg.num_classes = 1
cfg.transformer_d_model = 256
cfg.transformer_dim_feedforward = 1024
cfg.transformer_num_layers = 4
cfg.transformer_nhead = 16
cfg.transformer_dropout = 0.5
cfg.transformer_activation = "gelu"

cfg.fold = 0 
cfg.dataset = "sequence_cls"
cfg.data_dir = "/mnt/stor/datasets/spr-head-ct-age-prediction/train_features/foldx"
cfg.annotations_file = "/mnt/stor/datasets/spr-head-ct-age-prediction/train_features_kfold.csv"
cfg.inputs = "filename"
cfg.targets = ["Age"]
cfg.resample_or_pad = "resample"
cfg.resample_or_truncate = "resample"
cfg.seq_len = 128
cfg.num_workers = 28
cfg.pin_memory = True

cfg.loss = "L1Loss"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.num_epochs = 20
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 1e-3}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["MAE"]
cfg.val_metric = "mae"
cfg.val_track = "min"