import albumentations as A
import cv2

from .base import Config


cfg = Config()
cfg.neptune_mode = "async"

cfg.save_dir = "experiments/"
cfg.project = "gradientecho/rsna-lspine"

cfg.task = "classification_multiaug"

cfg.model = "net_2d_all_slices_seq"
cfg.backbone = "tf_efficientnetv2_s"
cfg.pretrained = True
cfg.num_input_channels = 3
cfg.pool = "gem"
cfg.pool_params = dict(p=3)
cfg.dropout = 0.5
cfg.num_classes = 4
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
cfg.dataset = "crop_all_slices_seq"
cfg.data_dir = "/home/ian/projects/rsna-lspine/data/train_crops_gt_coords_all_slices/foramina/"
cfg.annotations_file = "/home/ian/projects/rsna-lspine/data/train_crops_gt_coords_all_slices_foramina_kfold_half.csv"
cfg.inputs = "filepath"
cfg.targets = ["normal_mild", "moderate", "severe", "valid_slice"]
cfg.cv2_load_flag = cv2.IMREAD_COLOR
cfg.num_workers = 14
cfg.pin_memory = True
cfg.channel_reverse = True
cfg.sampler = "IterationBasedSampler"
cfg.num_iterations_per_epoch = 100
cfg.backbone_img_size = False
cfg.convert_to_3d = False

cfg.loss = "SampledWeightedBCEWithValidSliceWithMaskV2"
cfg.loss_params = {}

cfg.batch_size = 32
cfg.num_epochs = 10
cfg.optimizer = "AdamW"
cfg.optimizer_params = {"lr": 3e-4}

cfg.scheduler = "CosineAnnealingLR"
cfg.scheduler_params = {"eta_min": 0}
cfg.scheduler_interval = "step"

cfg.val_batch_size = cfg.batch_size * 2
cfg.metrics = ["CompetitionMetricPlusAUROCMultiAugSigmoidValidSlice"]
cfg.val_metric = "loss_mean"
cfg.val_track = "min"

# Avoid changing image dimensions via command line args
# if using these vars later (e.g., in crop transforms)
cfg.image_height = 64
cfg.image_width = 64
cfg.max_num_images = 10

additional_targets = additional_targets={f"image{idx}": "image" for idx in range(1, 50)}

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
], additional_targets=additional_targets, is_check_shapes=False)

cfg.val_transforms = A.Compose([A.Resize(cfg.image_height, cfg.image_width, p=1)], additional_targets=additional_targets, is_check_shapes=False)
