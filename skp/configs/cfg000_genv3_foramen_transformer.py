import cv2
import monai.transforms as T

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "crop_transformer_cls"
cfg.transformer_d_model = 256
cfg.transformer_nhead = 16
cfg.transformer_dim_feedforward = cfg.transformer_d_model * 4
cfg.transformer_dropout = 0.5
cfg.transformer_activation = "gelu"
cfg.transformer_num_layers = 1
cfg.num_classes = 3

cfg.fold = 0 
cfg.dataset = "crop_features"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_foramen_cropaug_features/foldx/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_foramen_crop_features.csv"
cfg.input = "filepath"
cfg.targets = ["normal_mild", "moderate", "severe"]
cfg.num_workers = 14
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 100

cfg.loss = "WeightedLogLossWithLogits"
cfg.loss_params = {}

cfg.batch_size = 512
cfg.num_epochs = 5
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = 1970
cfg.metrics = ["CompetitionMetricWithSoftmax", "CompetitionMetricWithSoftmaxTorch", "AUROC"]
cfg.val_metric = "comp_loss"
cfg.val_track = "min"