import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification_subarticular"

cfg.model = "net_x3d_subarticular"
cfg.backbone = "x3d_l"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.z_strides = [2, 1, 1, 1, 1]
cfg.dropout = 0.2
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "predict_subarticular_levels_and_coords_3d_with_flips"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_pngs_v2/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_subarticular_levels_and_coords_3d.pkl"
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 2
cfg.pin_memory = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 1000

cfg.loss = "ComboLevelsAndMaskedCoordsLoss"
cfg.loss_params = {}

cfg.batch_size = 12
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["SubarticularLevelsAndCoords"]
cfg.val_metric = "loss"
cfg.val_track = "min"

cfg.image_height = 384
cfg.image_width = 384
cfg.image_z = 28

cfg.train_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.image_z)})

cfg.val_transforms = A.Compose([
    A.Resize(cfg.image_height, cfg.image_width, p=1)
], additional_targets={f"image{idx}": "image" for idx in range(1, cfg.image_z)})

