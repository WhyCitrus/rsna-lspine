import cv2
import monai.transforms as T

from .base import Config


cfg = Config()
cfg.neptune_mode = "offline"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/sandbox"

cfg.task = "segmentation"

cfg.model = "x3d_unet"
cfg.backbone = "x3d_s"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.num_classes = 6
cfg.z_strides = [2, 2, 2, 2]
cfg.decoder_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False

cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_3d"
cfg.data_dir = "data_ich/studies/"
cfg.annotations_file = "data_ich/series_kfold.csv"
cfg.inputs = "filename"
cfg.targets = ["any", "epidural", "subdural", "subarachnoid", "intraparenchymal", "intraventricular"]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 1
cfg.pin_memory = True

cfg.loss = "BCEWithLogitsLoss"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["AUROC"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

cfg.roi_x = 128
cfg.roi_y = 128
cfg.roi_z = 128

cfg.train_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z), keys="image"),
    T.CenterSpatialCropd(roi_size=(cfg.roi_x, int(0.875 * cfg.roi_y), int(0.875 * cfg.roi_z)), keys="image"),
    T.RandAxisFlipd(prob=0.5, keys="image"),
    # T.RandRotate90d(prob=0.5, spatial_axes=(1, 2), keys="image"),
    # T.SomeOf([
    #   T.RandAffined(prob=1, rotate_range=(0, 30, 30), keys="image", padding_mode="zeros"),
    #   T.RandAffined(prob=1, scale_range=(0, 0.2, 0.2), keys="image", padding_mode="zeros"),
    #   T.RandAdjustContrastd(prob=1, gamma=(1.2, 2.0), keys="image"),
    #   T.RandGaussianSmoothd(prob=1, keys="image"),
    #   T.RandGaussianNoised(prob=1, std=0.2, keys="image")
    # ], num_transforms=3, replace=False)
])

cfg.val_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z), keys="image"),
    T.CenterSpatialCropd(roi_size=(cfg.roi_x, int(0.875 * cfg.roi_y), int(0.875 * cfg.roi_z)), keys="image")
])
