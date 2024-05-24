import cv2
import monai.transforms as T

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/brats"

cfg.task = "segmentation"

cfg.model = "x3d_unet"
cfg.backbone = "x3d_l"
cfg.pretrained = True
cfg.num_input_channels = 4
cfg.num_classes = 3
cfg.z_strides = [2, 2, 2, 2, 2]
cfg.decoder_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False
cfg.infer_sw_batch_size = 4
cfg.infer_overlap = 0.5

cfg.fold = 0 
cfg.dataset = "brats"
cfg.data_dir = "/mnt/stor/datasets/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/"
cfg.annotations_file = "/mnt/stor/datasets/brats/train_kfold.csv"
cfg.inputs = "study_folder"
cfg.targets = ["study_folder"]
cfg.convert_labels_to_brats_subregions = True
cfg.num_workers = 2
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000
cfg.dataset_ignore_errors = True

cfg.loss = "DiceLoss"
cfg.deep_supervision = True
cfg.deep_supervision_weights = [1., 0.5, 0.25]
cfg.loss_params = {"to_onehot_y": False, "sigmoid": True}

cfg.batch_size = 8
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 1e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = 1
cfg.metrics = ["DiceScore"]
cfg.val_metric = "dice_mean"
cfg.val_track = "max"

cfg.roi_x = 128
cfg.roi_y = 128
cfg.roi_z = 128

cfg.train_transforms = T.Compose([
    T.CropForegroundd(
        keys=["image", "label"], source_key="image", k_divisible=[cfg.roi_x, cfg.roi_y, cfg.roi_z],
    ),
    T.RandSpatialCropd(
        keys=["image", "label"], roi_size=[cfg.roi_x, cfg.roi_y, cfg.roi_z], random_size=False
    ),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    T.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    T.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
    T.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    T.ToTensord(keys=["image", "label"]),
])

cfg.val_transforms = T.Compose([
    T.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    T.ToTensord(keys=["image", "label"]),
])
