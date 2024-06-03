import cv2
import monai.transforms as T

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
cfg.z_strides = [1, 1, 1, 1, 1]
cfg.dropout = 0.5
cfg.num_classes = 6
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_lspine_3d_png"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_foramina_crops_3d/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_foramina_crops_3d.csv"
cfg.inputs = "series_folder"
cfg.targets = ["normal_mild_rt", "moderate_rt", "severe_rt", "normal_mild_lt", "moderate_lt", "severe_lt"]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 14
cfg.pin_memory = True
# cfg.sampler = "IterationBasedSampler"
# cfg.num_iterations_per_epoch = 1000

cfg.loss = "WeightedLogLoss6"
cfg.loss_params = {}

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["AUROC"]
cfg.val_metric = "loss"
cfg.val_track = "min"

cfg.image_height = 64
cfg.image_width = 64
cfg.image_z = 24

cfg.train_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
    T.SomeOf([
      T.RandAffined(prob=1, rotate_range=(0, 30, 30), keys="image", padding_mode="zeros"),
      T.RandAffined(prob=1, scale_range=(0, 0.2, 0.2), keys="image", padding_mode="zeros"),
      T.RandAdjustContrastd(prob=1, gamma=(1.2, 2.0), keys="image"),
      T.RandGaussianSmoothd(prob=1, keys="image"),
      T.RandGaussianNoised(prob=1, std=0.2, keys="image")
    ], num_transforms=3, replace=False)
])

cfg.val_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
])
