import torch.nn as nn

from functools import partial
from torchvision.models.mobilenet import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import _mobilenet_extractor
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, _default_anchorgen
from torchvision.ops.feature_pyramid_network import LastLevelP6P7


class Net(nn.Module):

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		weights = MobileNet_V3_Large_Weights.DEFAULT if cfg.pretrained_backbone else None
		backbone = mobilenet_v3_large(weights=weights)
		backbone = _mobilenet_extractor(backbone, fpn=True, trainable_layers=6, returned_layers=[2, 3, 5], extra_blocks=LastLevelP6P7(960, 256))

		if isinstance(cfg.anchor_sizes, tuple):
			assert isinstance(cfg.aspect_ratios, tuple), f"aspect_ratios must be specified in config if anchor_sizes is specified"
			anchor_generator = AnchorGenerator(cfg.anchor_sizes, cfg.aspect_ratios)
		else:
			anchor_generator = _default_anchorgen()

		head = RetinaNetHead(
			backbone.out_channels,
			anchor_generator.num_anchors_per_location()[0],
			cfg.num_classes,
			norm_layer=partial(nn.GroupNorm, 32),
		)
		head.regression_head._loss_type = "giou"

		self.retinanet = RetinaNet(backbone, cfg.num_classes, anchor_generator=anchor_generator, head=head, min_size=cfg.min_size, max_size=cfg.max_size)
		self.has_loss = True 

	def forward(self, batch):
		images = batch["images"]
		targets = batch.get("targets", None)
		return self.retinanet(images=images, targets=targets)
