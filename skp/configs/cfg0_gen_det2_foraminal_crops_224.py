import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "net_2d"
cfg.backbone = "coatnet_2_rw_224.sw_in12k_ft_in1k"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.reduce_feat_dim = 256
cfg.dropout = 0.5
cfg.num_classes = 3

cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_2d_sample_weights"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_generated_crops_using_detection2/foramina/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_gen_det2_foraminal_crops_kfold.csv"
cfg.inputs = "filepath"
cfg.targets = ["normal_mild", "moderate", "severe"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 14
cfg.pin_memory = True
cfg.channel_reverse = True
# cfg.sampler = "IterationBasedSampler"
# cfg.num_iterations_per_epoch = 10000

cfg.loss = "SampleWeightedLogLoss"
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
cfg.val_metric = "loss"
cfg.val_track = "min"

# Avoid changing image dimensions via command line args
# if using these vars later (e.g., in crop transforms)
cfg.image_height = 224
cfg.image_width = 224

cfg.train_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.RandomRotate90(p=0.5),
    A.SomeOf([
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.2, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.ShiftScaleRotate(shift_limit=0.00, scale_limit=0.0, rotate_limit=30, border_mode=cv2.BORDER_CONSTANT, p=1),
        A.GridDistortion(p=1),
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
