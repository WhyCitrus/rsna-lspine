import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "sequence_seq"
cfg.transformer_d_model = 256
cfg.transformer_nhead = 16
cfg.transformer_dim_feedforward = cfg.transformer_d_model * 4
cfg.transformer_dropout = 0.1
cfg.transformer_activation = "gelu"
cfg.transformer_num_layers = 1
cfg.num_classes = 6
cfg.max_seq_len = 192

cfg.fold = 0 
cfg.dataset = "sequence_seq"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_axial_subarticular_slice_identifier_features_with_level/foldx"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_axial_subarticular_slice_identifier_features_with_level_kfold.csv"
cfg.inputs = "filepath"
cfg.targets = "labelpath"
cfg.num_workers = 14
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 100

cfg.loss = "MaskedBCEWithLogitsLoss"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.num_epochs = 3
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["AUROCFlatten"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"
