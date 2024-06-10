import torch.nn as nn

from collections import OrderedDict
from functools import partial
from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead, _default_anchorgen
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7, LastLevelMaxPool


class EfficientNetBackboneWithFPN(nn.Module):

    def __init__(self, backbone, out_channels, extra_blocks, norm_layer):
        super().__init__()

        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        self.return_layers = {3: "0", 5: "1", 6: "2"}
        self.body = backbone.features[:-1]
        in_channels_list = [self.body[idx][-1].out_channels for idx in self.return_layers]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=norm_layer,
        )
        self.out_channels = out_channels

    def forward(self, x):
        out = OrderedDict()
        for idx, extractor in enumerate(self.body):
            x = extractor(x)
            if idx in self.return_layers:
                out[self.return_layers[idx]] = x
        x = self.fpn(out)
        return x


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        weights = EfficientNet_V2_S_Weights if cfg.pretrained_backbone else None
        backbone = efficientnet_v2_s(weights=weights)
        backbone = EfficientNetBackboneWithFPN(backbone=backbone, out_channels=256, extra_blocks=LastLevelP6P7(256, 256), norm_layer=None)

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

        self.retinanet = RetinaNet(backbone, 
            cfg.num_classes, 
            anchor_generator=anchor_generator, 
            head=head, 
            min_size=cfg.min_size, 
            max_size=cfg.max_size,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5])
        self.has_loss = True 

    def forward(self, batch):
        images = batch["images"]
        targets = batch.get("targets", None)
        return self.retinanet(images=images, targets=targets)
