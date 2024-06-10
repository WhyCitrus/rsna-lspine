import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from timm.models.layers import SelectAdaptivePool2d


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        norm_layer="bn",
    ):
        if norm_layer == "bn":
            NormLayer = nn.BatchNorm2d
        elif norm_layer == "gn":
            NormLayer = nn.GroupNorm
        else:
            raise Exception(f"`norm_layer` must be one of [`bn`, `gn`], got `{norm_layer}`")

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.ReLU(inplace=False)

        norm = NormLayer(out_channels)

        super(Conv2dReLU, self).__init__(conv, norm, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer="bn",
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        norm_layer="bn",
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_layer="bn",
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self, cfg
    ):
        super().__init__()
        
        self.cfg = cfg 

        if self.cfg.decoder_n_blocks != len(self.cfg.decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    self.cfg.decoder_n_blocks, len(self.cfg.decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = self.cfg.encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(self.cfg.decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = self.cfg.decoder_channels

        if self.cfg.decoder_center_block:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(norm_layer=self.cfg.decoder_norm_layer, attention_type=self.cfg.decoder_attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        output = [self.center(head)]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            output.append(decoder_block(output[-1], skip))

        return output


class SegmentationHead(nn.Module):

    def __init__(self, in_channels, out_channels, size, kernel_size=3, dropout=0):
        super().__init__()
        self.drop = nn.Dropout2d(p=dropout)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        if isinstance(size, (tuple, list)):
            self.up = nn.Upsample(size=size, mode="bilinear")
        else:
            self.up = nn.Identity()
        
    def forward(self, x):
        return self.up(self.conv(self.drop(x)))


class ClassificationHead(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=0):
        super().__init__()
        self.drop = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        return self.linear(self.drop(x))


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        return self.flatten(x)


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = create_model(self.cfg.backbone,
            pretrained=self.cfg.pretrained,
            num_classes=0,
            global_pool="",
            features_only=True,
            in_chans=self.cfg.num_input_channels)
        # self.change_num_input_channels() - use timm 
        with torch.no_grad():
            out = self.encoder(torch.randn((1, self.cfg.num_input_channels, self.cfg.image_height, self.cfg.image_width)))
            self.cfg.encoder_channels = [self.cfg.num_input_channels] + [o.shape[1] for o in out]
            del out

        self.decoder = UnetDecoder(self.cfg)
        self.segmentation_head = SegmentationHead(self.cfg.decoder_channels[-1], self.cfg.seg_num_classes, 
                                                  size=(self.cfg.image_height, self.cfg.image_width),
                                                  dropout=self.cfg.seg_dropout or 0)
        self.pooling = self.get_pool_layer()
        self.classification_head = ClassificationHead(self.cfg.encoder_channels[-1], self.cfg.cls_num_classes,
                                                      dropout=self.cfg.cls_dropout or 0)
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
                segmentation_head_weights = {re.sub(r"^model\.segmentation_head\.", "", k): v for k, v in weights.items() if bool(re.search(r"^model\.segmentation_head", k))}
                self.segmentation_head.load_state_dict(segmentation_head_weights)
                if self.cfg.deep_supervision:
                    aux_segmentation_head1_weights = {re.sub(r"^model\.aux_segmentation_head1\.", "", k): v for k, v in weights.items() if bool(re.search(r"^model\.aux_segmentation_head1", k))}
                    aux_segmentation_head2_weights = {re.sub(r"^model\.aux_segmentation_head2\.", "", k): v for k, v in weights.items() if bool(re.search(r"^model\.aux_segmentation_head2", k))}
                    self.aux_segmentation_head1.load_state_dict(aux_segmentation_head1_weights)
                    self.aux_segmentation_head2.load_state_dict(aux_segmentation_head2_weights)
            if not self.cfg.do_not_load_classification_head:
                classification_headweights = {re.sub(r"^model\.classification_head\.", "", k): v for k, v in weights.items() if bool(re.search(r"^model\.classification_head", k))}
                self.classification_head.load_state_dict(classification_head_weights)
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
        y_cls = batch["y_cls"] if "y_cls" in batch else None
        y_seg = batch["y_seg"] if "y_seg" in batch else None

        if return_loss:
            assert isinstance(y_cls, torch.Tensor)
            assert isinstance(y_seg, torch.Tensor)

        x = self.normalize(x) 
        feature_maps = [x] + self.encoder(x)
        features_cls = self.pooling(feature_maps[-1])
        decoder_output = self.decoder(*feature_maps)
        logits_seg = self.segmentation_head(decoder_output[-1])
        logits_cls = self.classification_head(features_cls)

        out = {"logits_seg": logits_seg, "logits_cls": logits_cls}

        if return_features:
            out["features"] = feature_maps, features_cls 
        if return_loss: 
            if self.cfg.deep_supervision:
                # TODO: figure out deep supervision with combined segmentation/classification loss
                level1 = self.aux_segmentation_head1(decoder_output[-2])
                level2 = self.aux_segmentation_head2(decoder_output[-3])
                loss = self.criterion([logits, level1, level2], y)
            else:
                loss = self.criterion(logits_seg, logits_cls, y_seg, y_cls)
                # loss is a dict with total loss, seg loss, and cls loss
            out.update(loss)

        return out

    def get_pool_layer(self):
        assert self.cfg.pool in ["avg", "max", "fast", "avgmax", "catavgmax", "gem"], f"{layer_name} is not a valid pooling layer"
        if self.cfg.pool == "gem":
            return GeM(**self.cfg.pool_params) if hasattr(self.cfg, "pool_params") else GeM()
        else:
            return SelectAdaptivePool2d(pool_type=self.cfg.pool, flatten=True)

    def freeze_encoder(self):
        for param in self.encoder.parameters(): 
            self.param.requires_grad = False

    def change_num_input_channels(self):
        # Assumes original number of input channels in model is 3
        for i, m in enumerate(self.encoder.modules()):
          if isinstance(m, nn.Conv2d) and m.in_channels == 3:
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
