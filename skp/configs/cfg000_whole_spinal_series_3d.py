import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "net_x3d"
cfg.backbone = "x3d_l"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.z_strides = [2, 1, 1, 1, 1]
cfg.dropout = 0.5
cfg.num_classes = 30
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "whole_spinal_series_3d"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs_3ch/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_spinal_3d_whole_series.csv"
cfg.inputs = "series_folder"
cfg.targets = [
    "l1_l2_normal_mild", "l1_l2_moderate", "l1_l2_severe",
    "l2_l3_normal_mild", "l2_l3_moderate", "l2_l3_severe",
    "l3_l4_normal_mild", "l3_l4_moderate", "l3_l4_severe",
    "l4_l5_normal_mild", "l4_l5_moderate", "l4_l5_severe",
    "l5_s1_normal_mild", "l5_s1_moderate", "l5_s1_severe",
    "l1_l2_slice_rescaled", "l1_l2_x_rescaled", "l1_l2_y_rescaled",
    "l2_l3_slice_rescaled", "l2_l3_x_rescaled", "l2_l3_y_rescaled",
    "l3_l4_slice_rescaled", "l3_l4_x_rescaled", "l3_l4_y_rescaled",
    "l4_l5_slice_rescaled", "l4_l5_x_rescaled", "l4_l5_y_rescaled",
    "l5_s1_slice_rescaled", "l5_s1_x_rescaled", "l5_s1_y_rescaled"
]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 2
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.loss = "WeightedLogLossWholeSpinalSeriesPlusCoords"
cfg.loss_params = {}

cfg.batch_size = 12
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["CompetitionMetricPlusAUROCWholeSpinal"]
cfg.val_metric = "comp_loss"
cfg.val_track = "min"

cfg.image_height = 512
cfg.image_width = 512
cfg.image_z = 12

cfg.train_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.image_z)})

cfg.val_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.image_z)})
