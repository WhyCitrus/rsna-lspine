import cv2
import monai.transforms as T

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
cfg.num_classes = 30
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_lspine_3d_png"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_grade_foramina_whole_series.csv"
cfg.inputs = "series_folder"
cfg.targets = [
    'lt_l1_l2_normal_mild', 'lt_l1_l2_mod', 'lt_l1_l2_severe', 
    'lt_l2_l3_normal_mild', 'lt_l2_l3_mod', 'lt_l2_l3_severe', 
    'lt_l3_l4_normal_mild', 'lt_l3_l4_mod', 'lt_l3_l4_severe', 
    'lt_l4_l5_normal_mild', 'lt_l4_l5_mod', 'lt_l4_l5_severe', 
    'lt_l5_s1_normal_mild', 'lt_l5_s1_mod', 'lt_l5_s1_severe', 
    'rt_l1_l2_normal_mild', 'rt_l1_l2_mod', 'rt_l1_l2_severe', 
    'rt_l2_l3_normal_mild', 'rt_l2_l3_mod', 'rt_l2_l3_severe', 
    'rt_l3_l4_normal_mild', 'rt_l3_l4_mod', 'rt_l3_l4_severe', 
    'rt_l4_l5_normal_mild', 'rt_l4_l5_mod', 'rt_l4_l5_severe', 
    'rt_l5_s1_normal_mild', 'rt_l5_s1_mod', 'rt_l5_s1_severe'
]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 2
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.loss = "WeightedLogLoss30"
cfg.loss_params = {}

cfg.batch_size = 8
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

cfg.image_height = 512
cfg.image_width = 512
cfg.image_z = 20

cfg.train_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
    T.SomeOf([
      T.RandAffined(prob=1, rotate_range=(0, 30, 30), keys="image", padding_mode="zeros"),
      T.RandAffined(prob=1, scale_range=(0, 0.2, 0.2), keys="image", padding_mode="zeros"),
      T.RandAdjustContrastd(prob=1, gamma=(1.2, 2.0), keys="image"),
      T.RandGaussianSmoothd(prob=1, keys="image"),
      T.RandGaussianNoised(prob=1, std=0.2, keys="image")
    ], num_transforms=3, replace=False)
])

cfg.val_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
])
