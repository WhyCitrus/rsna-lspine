import torch.nn as nn

from functools import partial
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNetHead


class Net(nn.Module):

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg


		self.retinanet = retinanet_resnet50_fpn_v2(num_classes=cfg.num_classes, 
			                                       min_size=cfg.min_size, 
			                                       max_size=cfg.max_size,
			                                       weights_backbone="DEFAULT" if cfg.pretrained_backbone else None)
		if isinstance(cfg.anchor_sizes, tuple):
			assert isinstance(cfg.aspect_ratios, tuple), f"aspect_ratios must be specified in config if anchor_sizes is specified"
			anchor_generator = AnchorGenerator(cfg.anchor_sizes, cfg.aspect_ratios)
			head = RetinaNetHead(
				self.retinanet.backbone.out_channels,
				anchor_generator.num_anchors_per_location()[0],
				cfg.num_classes,
				norm_layer=partial(nn.GroupNorm, 32),
			)
			head.regression_head._loss_type = "giou"
			self.retinanet.anchor_generator = anchor_generator
			self.retinanet.head = head
		self.has_loss = True 

	def forward(self, batch):
		images = batch["images"]
		targets = batch.get("targets", None)
		return self.retinanet(images=images, targets=targets)
