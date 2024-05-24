import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/sandbox"

cfg.task = "classification"

cfg.model = "sequence_cls"
cfg.num_classes = 4
cfg.embedding_dim = 256
cfg.hidden_dim = cfg.embedding_dim * 2
cfg.n_layers = 4
cfg.n_heads = 16
cfg.dropout = 0.2
cfg.attention_dropout = 0.1
cfg.output_attentions = False
cfg.activation = "gelu"
cfg.output_hidden_states = False
cfg.chunk_size_feed_forward = 0

cfg.fold = 0 
cfg.dataset = "sequence_cls"
cfg.data_dir = "data_media/pos_slice_features_v2/foldx"
cfg.annotations_file = "data_media/train_seq_kfold_df.csv"
cfg.inputs = "filename"
cfg.targets = ["label"]
cfg.resample_or_pad = "pad"
cfg.resample_or_truncate = "resample"
cfg.seq_len = 64
cfg.num_workers = 4
cfg.pin_memory = True

cfg.loss = "CrossEntropyLoss"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.num_epochs = 20
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 1e-5}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["Accuracy", "AUROC"]
cfg.val_metric = "accuracy"
cfg.val_track = "max"