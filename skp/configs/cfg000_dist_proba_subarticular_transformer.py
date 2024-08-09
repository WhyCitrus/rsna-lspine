import cv2
import monai.transforms as T

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "dist_transformer_v2"
cfg.position_duplication_factor = 16
cfg.feature_dim = 1536 + cfg.position_duplication_factor
cfg.transformer_d_model = 256
cfg.transformer_nhead = 16
cfg.transformer_dim_feedforward = cfg.transformer_d_model * 4
cfg.transformer_dropout = 0.5
cfg.transformer_activation = "gelu"
cfg.transformer_num_layers = 4
cfg.num_classes = 20

cfg.fold = 0 
cfg.dataset = "dist_position_features"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_subarticular_dist_coord_features/mixnet_m/foldx/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_subarticular_dist_coord_features_v3.csv"
cfg.num_workers = 14
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000
cfg.max_seq_len = 72

cfg.loss = "MaskedSmoothL1LossBCESubarticular"
cfg.loss_params = {}

cfg.batch_size = 128
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = 1
cfg.metrics = ["Dummy"]
cfg.val_metric = "loss"
cfg.val_track = "min"