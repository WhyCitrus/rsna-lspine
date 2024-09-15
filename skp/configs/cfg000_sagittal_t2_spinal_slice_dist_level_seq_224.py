import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "axial_t2_stack"
cfg.backbone = "tf_efficientnetv2_m"
cfg.pretrained = True
cfg.load_pretrained_backbone = None
cfg.num_input_channels = 3
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.dropout = 0.1
cfg.num_classes = 10
cfg.backbone_img_size = False
cfg.transformer_nhead = 16
cfg.expansion_factor = 1
cfg.transformer_dropout = 0.1
cfg.transformer_activation = "gelu"
cfg.transformer_num_layers = 2
cfg.freeze_backbone = False

cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "sagittal_t2_stack"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs_3ch/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_sagittal_t2_stack_kfold.csv"
cfg.inputs = "filepath"
levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
cfg.targets = levels + [f"{lvl}_dist" for lvl in levels]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 10
cfg.channel_reverse = False
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 100

cfg.loss = "LevelSpinalDistSeq"
cfg.loss_params = {}

cfg.batch_size = 8
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = 1
cfg.metrics = ["AUROCDistSeq"]
cfg.val_metric = "auc_mean"
cfg.val_track = "max"

# Avoid changing image dimensions via command line args
# if using these vars later (e.g., in crop transforms)
cfg.image_height = 224
cfg.image_width = 224
cfg.max_num_images = 20

cfg.train_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1),
    A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.Transpose(p=0.5),
    # A.RandomRotate90(p=0.5),
    A.SomeOf([
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
        # A.GridDistortion(p=1),
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
        A.RandomGamma(p=1),
        A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0, p=1),
        A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2, p=1),
        # A.CoarseDropout(min_height=0.05, max_height=0.2, min_width=0.05, max_width=0.2,
        #                 min_holes=2, max_holes=8, fill_value=0, p=1),

    ], n=3, p=0.95, replace=False)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.max_num_images)})

cfg.val_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1),
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.max_num_images)})
