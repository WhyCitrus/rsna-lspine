import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model

from .pool_3d import SelectAdaptivePool3d
from .unet_3d import UnetDecoder


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = F.avg_pool3d(x.clamp(min=self.eps).pow(self.p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./self.p)
        return self.flatten(x)


class X3DEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Set up X3D backbone
        if not self.cfg.pretrained:
            from pytorchvideo.models import hub
            self.backbone = getattr(hub, self.cfg.backbone)(pretrained=False)
        else:
            self.backbone = torch.hub.load("facebookresearch/pytorchvideo", model=self.cfg.backbone, pretrained=True)

        del self.backbone.blocks[-1]

        for idx, z in enumerate(self.backbone.blocks):
            if idx == 0 and self.cfg.z_strides[idx] == 2:
                stem_layer = self.backbone.blocks[0].conv.conv_t
                w = stem_layer.weight
                w = w.repeat(1, 1, 3, 1, 1)
                in_channels, out_channels = stem_layer.in_channels, stem_layer.out_channels
                self.backbone.blocks[0].conv.conv_t = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
                with torch.no_grad():
                    self.backbone.blocks[0].conv.conv_t.weight.copy_(w)
            elif idx > 0:
                self.backbone.blocks[idx].res_blocks[0].branch1_conv.stride = (self.cfg.z_strides[idx], 2, 2)
                self.backbone.blocks[idx].res_blocks[0].branch2.conv_b.stride = (self.cfg.z_strides[idx], 2, 2)

    def forward(self, x):
        feature_maps = []
        for block in self.backbone.blocks:
            x = block(x)
            feature_maps.append(x)
        return feature_maps


class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, size, kernel_size=3, dropout=0):
        super().__init__()
        self.drop = nn.Dropout3d(p=dropout)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        if isinstance(size, (tuple, list)):
            self.up = nn.Upsample(size=size, mode="trilinear")
        else:
            self.up = nn.Identity()
        
    def forward(self, x):
        return self.up(self.conv(self.drop(x)))


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = X3DEncoder(self.cfg)
        self.change_num_input_channels()
        with torch.no_grad():
            out = self.encoder(torch.randn((1, self.cfg.num_input_channels, self.cfg.roi_x, self.cfg.roi_y, self.cfg.roi_z)))
            self.cfg.encoder_channels = [self.cfg.num_input_channels] + [o.shape[1] for o in out]
            del out

        self.decoder = UnetDecoder(self.cfg)
        self.segmentation_head = SegmentationHead(self.cfg.decoder_channels[-1], self.cfg.num_classes, 
                                                  size=(self.cfg.roi_x, self.cfg.roi_y, self.cfg.roi_z),
                                                  dropout=self.cfg.seg_dropout or 0)

        if self.cfg.deep_supervision:
            self.aux_segmentation_head1 = SegmentationHead(self.cfg.decoder_channels[-2], 
                                                           self.cfg.num_classes,
                                                           size=None)
            self.aux_segmentation_head2 = SegmentationHead(self.cfg.decoder_channels[-3], 
                                                           self.cfg.num_classes,
                                                           size=None)

        if self.cfg.load_pretrained_encoder:
            print(f"Loading pretrained encoder from {self.cfg.load_pretrained_encoder} ...")
            weights = torch.load(self.cfg.load_pretrained_encoder, map_location=lambda storage, loc: storage)["state_dict"]
            weights = {re.sub(r"^model\.", "", k): v for k,v in weights.items()}
            # Get backbone only
            weights = {re.sub(r"^backbone\.", "", k): v for k,v in weights.items() if "backbone." in k}
            self.encoder.backbone.load_state_dict(weights)

        if self.cfg.freeze_encoder:
            self.freeze_encoder()

        if self.cfg.load_pretrained_model:
            print(f"Loading pretrained model from {self.cfg.load_pretrained_model} ...")
            weights = torch.load(self.cfg.load_pretrained_model, map_location=lambda storage, loc: storage)["state_dict"]
            encoder_weights = {re.sub(r"^model\.encoder\.", "", k): v for k, v in weights.items() if "encoder." in k}
            decoder_weights = {re.sub(r"^model\.decoder\.", "", k): v for k, v in weights.items() if "decoder." in k}
            if not self.cfg.do_not_load_segmentation_heads:
                segmentation_head_weights = {re.sub(r"^model\.segmentation.head\.", "", k): v for k, v in weights.items() if bool(re.search(r"^model\.segmentation_head", k))}
                self.segmentation_head.load_state_dict(segmentation_head_weights)
                if self.cfg.deep_supervision:
                    aux_segmentation_head1_weights = {re.sub(r"^model\.aux_segmentation_head1\.", "", k): v for k, v in weights.items() if bool(re.search(r"^model\.aux_segmentation_head1", k))}
                    aux_segmentation_head2_weights = {re.sub(r"^model\.aux_segmentation_head2\.", "", k): v for k, v in weights.items() if bool(re.search(r"^model\.aux_segmentation_head2", k))}
                    self.aux_segmentation_head1.load_state_dict(aux_segmentation_head1_weights)
                    self.aux_segmentation_head2.load_state_dict(aux_segmentation_head2_weights)
            self.encoder.load_state_dict(encoder_weights)
            self.decoder.load_state_dict(decoder_weights)

    def normalize(self, x):
        if self.cfg.normalization == "-1_1":
            mini, maxi = self.cfg.normalization_params["min"], self.cfg.normalization_params["max"]
            x = x - mini
            x = x / (maxi - mini) 
            x = x - 0.5 
            x = x * 2.0
        elif self.cfg.normalization == "0_1":
            mini, maxi = self.cfg.normalization_params["min"], self.cfg.normalization_params["max"]
            x = x - mini
            x = x / (maxi - mini) 
        elif self.cfg.normalization == "mean_sd":
            mean, sd = self.cfg.normalization_params["mean"], self.cfg.normalization_params["sd"]
            x = (x - mean) / sd
        elif self.cfg.normalization == "per_channel_mean_sd":
            mean, sd = self.cfg.normalization_params["mean"], self.cfg.normalization_params["sd"]
            assert len(mean) == len(sd) == x.size(1)
            mean, sd = torch.tensor(mean).unsqueeze(0), torch.tensor(sd).unsqueeze(0)
            for i in range(x.ndim - 2):
                mean, sd = mean.unsqueeze(-1), sd.unsqueeze(-1)
            x = (x - mean) / sd
        return x 

    def forward(self, batch, return_loss=False, return_features=False):
        x = batch["x"]
        y = batch["y"] if "y" in batch else None

        if return_loss:
            assert isinstance(y, torch.Tensor)

        x = self.normalize(x) 
        feature_maps = [x] + self.encoder(x)
        decoder_output = self.decoder(*feature_maps)
        logits = self.segmentation_head(decoder_output[-1])

        out = {"logits": logits}

        if return_features:
            out["features"] = feature_maps 
        if return_loss: 
            if self.cfg.deep_supervision:
                level1 = self.aux_segmentation_head1(decoder_output[-2])
                level2 = self.aux_segmentation_head2(decoder_output[-3])
                loss = self.criterion([logits, level1, level2], y)
            else:
                loss = self.criterion(logits, y)
            out["loss"] = loss

        return out

    def get_pool_layer(self):
        assert self.cfg.pool in ["avg", "max", "fast", "avgmax", "catavgmax", "gem"], f"{layer_name} is not a valid pooling layer"
        if self.cfg.pool == "gem":
            return GeM(**self.cfg.pool_params) if hasattr(self.cfg, "pool_params") else GeM()
        else:
            return SelectAdaptivePool3d(pool_type=self.cfg.pool, flatten=True)

    def freeze_encoder(self):
        for param in self.encoder.parameters(): 
            self.param.requires_grad = False

    def change_num_input_channels(self):
        # Assumes original number of input channels in model is 3
        for i, m in enumerate(self.encoder.modules()):
          if isinstance(m, nn.Conv3d) and m.in_channels == 3:
            m.in_channels = self.cfg.num_input_channels
            # First, sum across channels
            W = m.weight.sum(1, keepdim=True)
            # Then, divide by number of channels
            W = W / self.cfg.num_input_channels
            # Then, repeat by number of channels
            size = [1] * W.ndim
            size[1] = self.cfg.num_input_channels
            W = W.repeat(size)
            m.weight = nn.Parameter(W)
            break

    def set_criterion(self, loss):
        self.criterion = loss
