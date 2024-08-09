import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "net_2d"
cfg.backbone = "tf_efficientnetv2_m"
cfg.pretrained = True
cfg.num_input_channels = 5
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.dropout = 0.5
cfg.num_classes = 15
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_2dc_png_albumentations"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_spinal_cropped_whole_slices/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_spinal_cropped_whole_slices_2d.csv"
cfg.inputs = "series_folder"
cfg.targets = [
    "l1_l2_normal_mild", "l1_l2_moderate", "l1_l2_severe",
    "l2_l3_normal_mild", "l2_l3_moderate", "l2_l3_severe",
    "l3_l4_normal_mild", "l3_l4_moderate", "l3_l4_severe",
    "l4_l5_normal_mild", "l4_l5_moderate", "l4_l5_severe",
    "l5_s1_normal_mild", "l5_s1_moderate", "l5_s1_severe",
]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 2
cfg.pin_memory = True
# cfg.sampler = "IterationBasedSampler"
# cfg.num_iterations_per_epoch = 1000
cfg.reverse_dim0 = True

cfg.loss = "SampleWeightedWholeSpinalBCE"
cfg.loss_params = {}

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["CompetitionMetricPlusAUROCWholeSpinal"]
cfg.val_metric = "comp_loss"
cfg.val_track = "min"

cfg.image_height = 512
cfg.image_width = 128
cfg.image_z = 5

cfg.train_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1),
    A.HorizontalFlip(p=0.5),
    # A.SomeOf([
    #     A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1),
    #     A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
    #     A.GridDistortion(p=1),
    #     A.GaussianBlur(p=1),
    #     A.GaussNoise(p=1),
    #     A.RandomGamma(p=1),
    #     A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0, p=1),
    #     A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2, p=1),
    # ], n=3, p=0.95, replace=False)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.image_z)})

cfg.val_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.image_z)})
