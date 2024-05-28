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
cfg.num_input_channels = 1
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.reduce_feat_dim = 256
cfg.dropout = 0.5
cfg.num_classes = 10

cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_2d"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_predict_which_sagittal_slice_foramina_canal_2d.csv"
cfg.inputs = "filepath"
cfg.targets = [
    "l1_l2_foramen_rel_target", "l2_l3_foramen_rel_target", "l3_l4_foramen_rel_target", "l4_l5_foramen_rel_target", "l5_s1_foramen_rel_target",
    "l1_l2_canal_rel_target", "l2_l3_canal_rel_target", "l3_l4_canal_rel_target", "l4_l5_canal_rel_target", "l5_s1_canal_rel_target"
]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 14
cfg.pin_memory = True
# cfg.sampler = "IterationBasedSampler"
# cfg.num_iterations_per_epoch = 10000

cfg.loss = "L1Loss"
cfg.loss_params = {}

cfg.batch_size = 32
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

# Avoid changing image dimensions via command line args
# if using these vars later (e.g., in crop transforms)
cfg.image_height = 256
cfg.image_width = 256

cfg.train_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1),
    A.SomeOf([
        A.GaussNoise(p=1),
        A.RandomGamma(p=1),
        A.RandomBrightnessContrast(contrast_limit=0.2, brightness_limit=0.0, p=1),
        A.RandomBrightnessContrast(contrast_limit=0.0, brightness_limit=0.2, p=1),

    ], n=3, p=0.5, replace=False)
])

cfg.val_transforms = A.Compose([A.Resize(cfg.image_height, cfg.image_width, p=1)])
