import cv2
import monai.transforms as T

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification"

cfg.model = "net_csn_r101"
cfg.pretrained = True
cfg.num_input_channels = 1
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.dropout = 0.5
cfg.num_classes = 6
cfg.normalization = "-1_1"
cfg.normalization_params = {"min": 0, "max": 255}

cfg.fold = 0 
cfg.dataset = "simple_3d_bilateral_sample_weights"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_cropped_foramina_blocks/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_foramina_blocks_kfold.csv"
cfg.inputs = "series_folder"
cfg.targets = ["rt_normal_mild", "rt_moderate", "rt_severe", "lt_normal_mild", "lt_moderate", "lt_severe"]
cfg.cv2_load_flag = cv2.IMREAD_GRAYSCALE
cfg.num_workers = 14
cfg.pin_memory = True
cfg.reverse_dim0 = True
cfg.flip_lr = True
cfg.flip_ud = True
cfg.foramina = True
# cfg.sampler = "IterationBasedSampler"
# cfg.num_iterations_per_epoch = 10000

cfg.loss = "SampleWeightedLogLossBilatV2"
cfg.loss_params = {}

cfg.batch_size = 16
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["AUROCBilat"]
cfg.val_metric = "loss"
cfg.val_track = "min"

# Avoid changing image dimensions via command line args
# if using these vars later (e.g., in crop transforms)
cfg.image_height = 64
cfg.image_width = 64
cfg.image_z = 17

cfg.train_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
])

cfg.val_transforms = T.Compose([
    T.Resized(spatial_size=(cfg.image_z, cfg.image_height, cfg.image_width), keys="image"),
])
