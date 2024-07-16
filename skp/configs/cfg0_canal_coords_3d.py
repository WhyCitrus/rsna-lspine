import cv2
import albumentations as A

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
cfg.z_strides = [2, 1, 1, 1, 1]
cfg.dropout = 0.5
cfg.num_classes = 15
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_lspine_3d_png_albumentations"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_to_test_predicting_sagittal_canal_coords_kfold_v2.csv"
cfg.inputs = "series_folder"
cfg.targets = [
    "canal_l1_l2_x", "canal_l2_l3_x", "canal_l3_l4_x", "canal_l4_l5_x", "canal_l5_s1_x",
    "canal_l1_l2_y", "canal_l2_l3_y", "canal_l3_l4_y", "canal_l4_l5_y", "canal_l5_s1_y",
    "canal_l1_l2_z", "canal_l2_l3_z", "canal_l3_l4_z", "canal_l4_l5_z", "canal_l5_s1_z"
]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 2
cfg.reverse_dim0 = False
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 500

cfg.loss = "L1Loss"
cfg.loss_params = {}

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["MAESigmoid"]
cfg.val_metric = "mae"
cfg.val_track = "min"

cfg.image_height = 320
cfg.image_width = 320
cfg.image_z = 24

cfg.train_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.Transpose(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.SomeOf([
    #     A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1),
    #     A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
    #     A.GridDistortion(p=1),
    #     A.GaussianBlur(p=1),
    #     A.GaussNoise(p=1),
    #     A.RandomGamma(p=1),
    #     A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0, p=1),
    #     A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2, p=1),
    #     # A.CoarseDropout(min_height=0.05, max_height=0.2, min_width=0.05, max_width=0.2,
    #     #                 min_holes=2, max_holes=8, fill_value=0, p=1),

    # ], n=3, p=0.95, replace=False)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.image_z)})

cfg.val_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.image_z)})

