import cv2
import monai.transforms as T

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/totalsegmentator"

cfg.task = "segmentation"

cfg.model = "x3d_unet"
cfg.backbone = "x3d_l"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.subset_segmentations = list(range(92, 117))
cfg.num_classes = len(cfg.subset_segmentations)
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}
cfg.z_strides = [2, 2, 2, 2, 2]
cfg.decoder_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.infer_sw_batch_size = 1
cfg.infer_overlap = 0.2

cfg.fold = 0 
cfg.dataset = "totalsegmentator"
cfg.data_dir = "/mnt/stor/datasets/totalsegmentator/pngs-v201/"
cfg.annotations_file = "/mnt/stor/datasets/totalsegmentator/train_pngs_kfold.csv"
cfg.inputs = "image_folder"
cfg.targets = "segmentation_folder"
cfg.num_workers = 2
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000
cfg.dataset_ignore_errors = True
cfg.randomly_drop_axial_slices = 0

cfg.loss = "BCEWithLogitsLoss"
cfg.deep_supervision = True
cfg.deep_supervision_weights = [1., 0.5, 0.25]
cfg.loss_params = {}

cfg.batch_size = 8
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 1e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = 1
cfg.metrics = ["DiceScoreStatsOnly"]
cfg.val_metric = "dice_mean"
cfg.val_track = "max"

cfg.roi_x = 96
cfg.roi_y = 128
cfg.roi_z = 128

cfg.train_transforms = T.Compose([
    T.RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.75, max_zoom=1.5, keep_size=False),
    T.SpatialPadd(keys=["image", "label"], spatial_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z)),
    T.RandSpatialCropd(
        keys=["image", "label"], roi_size=[cfg.roi_x, cfg.roi_y, cfg.roi_z], random_size=False
    ),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    T.RandScaleIntensityd(keys="image", prob=1.0, factors=0.1),
    T.RandAdjustContrastd(keys="image", prob=0.5, gamma=(1.1, 1.5)),
    T.ToTensord(keys=["image", "label"]),
])

cfg.val_transforms = T.Compose([
    T.SpatialPadd(keys=["image", "label"], spatial_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z)),
    T.ToTensord(keys=["image", "label"]),
])
