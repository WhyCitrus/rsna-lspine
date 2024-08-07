import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "net_2d"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.dropout = 0.5
cfg.num_classes = 30

cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_2d"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs_3ch/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_foramen_dist_coord_proba_sampling_weight.csv"
cfg.inputs = "pngfile"
levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
cfg.targets = [f"rt_{_}_no_rescale" for _ in levels] + [f"lt_{_}_no_rescale" for _ in levels]
cfg.targets += [f"rt_{_}_proba_score" for _ in levels] + [f"lt_{_}_proba_score" for _ in levels]
cfg.targets += [f"{_}_foramen_coord_x" for _ in levels] + [f"{_}_foramen_coord_y" for _ in levels]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 10
cfg.channel_reverse = False
cfg.pin_memory = True
cfg.sampler = "WeightedSampler"
cfg.backbone_img_size = False 

cfg.loss = "L1DistCoordBCE"
cfg.loss_params = {}

cfg.batch_size = 16
cfg.num_epochs = 25
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["Dummy"]
cfg.val_metric = "loss"
cfg.val_track = "min"

# Avoid changing image dimensions via command line args
# if using these vars later (e.g., in crop transforms)
cfg.image_height = 512
cfg.image_width = 512

cfg.train_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.Transpose(p=0.5),
    # A.RandomRotate90(p=0.5),
    A.SomeOf([
        # A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1),
        # A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
        # A.GridDistortion(p=1),
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
        A.RandomGamma(p=1),
        A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0, p=1),
        A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2, p=1),
        # A.CoarseDropout(min_height=0.05, max_height=0.2, min_width=0.05, max_width=0.2,
        #                 min_holes=2, max_holes=8, fill_value=0, p=1),

    ], n=3, p=0.95, replace=False)
])

cfg.val_transforms = A.Compose([A.Resize(cfg.image_height, cfg.image_width, p=1)])
