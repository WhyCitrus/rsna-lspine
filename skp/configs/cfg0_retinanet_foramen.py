import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "detection"

cfg.model = "retinanet_torchvision"
cfg.pretrained_backbone = True
cfg.num_classes = 5

cfg.fold = 0 
cfg.dataset = "foramen_bboxes"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_foramen_bboxes_smaller.pkl"
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 14
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["mAP_Simple", "CenterDiffAbs"]
cfg.val_metric = "avg_diff"
cfg.val_track = "min"

# Avoid changing image dimensions via command line args
# if using these vars later (e.g., in crop transforms)
cfg.min_size = 1024
cfg.max_size = 1024

cfg.train_transforms = A.Compose([
    A.Resize(cfg.min_size, cfg.max_size, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.RandomRotate90(p=0.5),
    A.SomeOf([
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
        A.RandomGamma(p=1),
        A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0, p=1),
        A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2, p=1),
        # A.CoarseDropout(min_height=0.05, max_height=0.2, min_width=0.05, max_width=0.2,
        #                 min_holes=2, max_holes=8, fill_value=0, p=1),

    ], n=3, p=0.95, replace=False)
], bbox_params=A.BboxParams(format="pascal_voc", min_area=256, min_visibility=0.5, label_fields=["class_labels"]))

cfg.val_transforms = A.Compose([A.Resize(cfg.min_size, cfg.max_size, p=1)],
    bbox_params=A.BboxParams(format="pascal_voc", min_area=256, min_visibility=0.5, label_fields=["class_labels"]))
