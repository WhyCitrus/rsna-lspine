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
cfg.reduce_feat_dim = 256
cfg.z_strides = [2, 1, 1, 1, 1]
cfg.dropout = 0.5
cfg.num_classes = 30
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_lspine_3d_png"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_to_test_predicting_sagittal_foramina_coords_kfold.csv"
cfg.inputs = "series_folder"
cfg.targets = [
    "lt_foramen_l1_l2_x", "lt_foramen_l2_l3_x", "lt_foramen_l3_l4_x", "lt_foramen_l4_l5_x", "lt_foramen_l5_s1_x",
    "rt_foramen_l1_l2_x", "rt_foramen_l2_l3_x", "rt_foramen_l3_l4_x", "rt_foramen_l4_l5_x", "rt_foramen_l5_s1_x",
    "lt_foramen_l1_l2_y", "lt_foramen_l2_l3_y", "lt_foramen_l3_l4_y", "lt_foramen_l4_l5_y", "lt_foramen_l5_s1_y",
    "rt_foramen_l1_l2_y", "rt_foramen_l2_l3_y", "rt_foramen_l3_l4_y", "rt_foramen_l4_l5_y", "rt_foramen_l5_s1_y",
    "lt_foramen_l1_l2_z", "lt_foramen_l2_l3_z", "lt_foramen_l3_l4_z", "lt_foramen_l4_l5_z", "lt_foramen_l5_s1_z",
    "rt_foramen_l1_l2_z", "rt_foramen_l2_l3_z", "rt_foramen_l3_l4_z", "rt_foramen_l4_l5_z", "rt_foramen_l5_s1_z"
]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 2
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.loss = "L1Loss"
cfg.loss_params = {}

cfg.batch_size = 12
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

cfg.image_height = 384
cfg.image_width = 384
cfg.image_z = 24

cfg.train_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
])

cfg.val_transforms = T.Compose([
	T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
])
