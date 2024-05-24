import cv2
import monai.transforms as T

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/sandbox"

cfg.task = "classification"

cfg.model = "net_x3d"
cfg.backbone = "x3d_s"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.reduce_feat_dim = 256
cfg.z_strides = [2, 2, 2, 2, 2]
cfg.dropout = 0.5
cfg.num_classes = 4

cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_3d"
cfg.data_dir = "data_media/"
cfg.annotations_file = "data_media/train_sub_kfold.csv"
cfg.inputs = "filename"
cfg.targets = ["label"]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 1
cfg.pin_memory = True

cfg.loss = "CrossEntropyLoss"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["Accuracy"]
cfg.val_metric = "accuracy"
cfg.val_track = "max"

cfg.image_height = 256
cfg.image_width = 256
cfg.image_z = 64

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
