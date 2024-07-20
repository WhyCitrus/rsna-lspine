import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "segmentation_2d_cls"

cfg.model = "unet_2d_cls"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.seg_dropout = 0.5
cfg.cls_dropout = 0.5
cfg.seg_num_classes = 5
cfg.cls_num_classes = 15
cfg.decoder_channels = [256, 128, 64, 32, 16]
cfg.decoder_n_blocks = 5
cfg.decoder_norm_layer = "bn"
cfg.decoder_attention_type = None
cfg.decoder_center_block = False

cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "foramen_dist_coord_seg"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs_3ch/"
cfg.seg_dir = "/home/ian/projects/rsna-lspine/data/train_spinal_segmentation_maps_smaller/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_spinal_distance_each_level_with_spinal_coords_and_ignore.csv"
cfg.inputs = "pngfile"
levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
cfg.targets = [f"{_}_no_rescale" for _ in levels]
cfg.targets += [f"{_}_spinal_coord_x" for _ in levels]
cfg.targets += [f"{_}_spinal_coord_y" for _ in levels]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 10
cfg.pin_memory = True

cfg.loss = "L1LossDistCoordSegSpinalV3"
cfg.loss_params = {"loss_weights": [1., 100., 1.], "seg_pos_weight": 10000}
cfg.deep_supervision = False

cfg.batch_size = 16
cfg.num_epochs = 10
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
cfg.image_height = 256
cfg.image_width = 256

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
