import cv2
import monai.transforms as T

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/sandbox"

cfg.task = "segmentation"

cfg.model = "swin_unetr"
cfg.load_pretrained_model = "/home/neurolab/ianpan/fold0_f48_ep300_4gpu_dice0_8854/model.pt"
cfg.num_classes = 2

cfg.fold = 0 
cfg.dataset = "idh_nifti"
cfg.data_dir = "/mnt/IDH_Project/Segmented_Images_Reviewed/"
cfg.annotations_file = "/home/neurolab/ianpan/train_segmented_images_reviewed_kfold.csv"
cfg.inputs = "study_folder"
cfg.targets = ["study_folder"]
cfg.num_workers = 4
cfg.pin_memory = True

cfg.loss = "DiceLoss"
cfg.loss_params = {"to_onehot_y": False, "sigmoid": True}

cfg.batch_size = 2
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
