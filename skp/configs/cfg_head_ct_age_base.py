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
cfg.num_input_channels = 3
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.reduce_feat_dim = 256
cfg.z_strides = [1, 1, 1, 1, 1]
cfg.dropout = 0.5
cfg.num_classes = 30
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_3d"
cfg.data_dir = "/mnt/stor/datasets/spr-head-ct-age-prediction/"
cfg.annotations_file = cfg.data_dir + "train_kfold.csv"
cfg.inputs = "filename"
cfg.targets = ["Age"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 2
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 100

cfg.loss = "L1Loss"
cfg.loss_params = {}

cfg.batch_size = 8
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["MAE"]
cfg.val_metric = "mae"
cfg.val_track = "min"

cfg.image_height = 384
cfg.image_width = 384
cfg.image_z = 24

cfg.train_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
    T.CenterSpatialCropd(roi_size=(cfg.image_z, int(0.875 * cfg.image_height), int(0.875 * cfg.image_height)), keys="image"),
    T.RandAxisFlipd(prob=0.5, keys="image"),
    # T.RandRotate90d(prob=0.5, spatial_axes=(1, 2), keys="image"),
    # T.SomeOf([
    # 	T.RandAffined(prob=1, rotate_range=(0, 30, 30), keys="image", padding_mode="zeros"),
    # 	T.RandAffined(prob=1, scale_range=(0, 0.2, 0.2), keys="image", padding_mode="zeros"),
    # 	T.RandAdjustContrastd(prob=1, gamma=(1.2, 2.0), keys="image"),
    # 	T.RandGaussianSmoothd(prob=1, keys="image"),
    # 	T.RandGaussianNoised(prob=1, std=0.2, keys="image")
    # ], num_transforms=3, replace=False)
])

cfg.val_transforms = T.Compose([
	T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
	T.CenterSpatialCropd(roi_size=(cfg.image_z, int(0.875 * cfg.image_height), int(0.875 * cfg.image_height)), keys="image")
])
