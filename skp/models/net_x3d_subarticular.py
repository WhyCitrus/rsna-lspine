import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model

from .pool_3d import SelectAdaptivePool3d


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = F.avg_pool3d(x.clamp(min=self.eps).pow(self.p), (x.size(-3), x.size(-2), x.size(-1))).pow(1./self.p)
        return self.flatten(x)


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Set up X3D backbone
        if not self.cfg.pretrained:
            from pytorchvideo.models import hub
            self.backbone = getattr(hub, self.cfg.backbone)(pretrained=False)
        else:
            self.backbone = torch.hub.load("facebookresearch/pytorchvideo", model=self.cfg.backbone, pretrained=True)
        for idx, z in enumerate(self.cfg.z_strides):
            assert len(self.cfg.z_strides) == 5
            assert z in [1, 2], "Only z-strides of 1 or 2 are supported"
            if z == 2:
                if idx == 0:
                    stem_layer = self.backbone.blocks[0].conv.conv_t
                    w = stem_layer.weight
                    w = w.repeat(1, 1, 3, 1, 1)
                    in_channels, out_channels = stem_layer.in_channels, stem_layer.out_channels
                    self.backbone.blocks[0].conv.conv_t = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
                    with torch.no_grad():
                        self.backbone.blocks[0].conv.conv_t.weight.copy_(w)
                else:
                    self.backbone.blocks[idx].res_blocks[0].branch1_conv.stride = (2, 2, 2)
                    self.backbone.blocks[idx].res_blocks[0].branch2.conv_b.stride = (2, 2, 2)
        
        self.backbone.blocks[-1] = nn.Sequential(
            self.backbone.blocks[-1].pool.pre_conv,
            self.backbone.blocks[-1].pool.pre_norm,
            self.backbone.blocks[-1].pool.pre_act,
        )
        self.change_num_input_channels()

        self.dim_feats = self.backbone(torch.randn((2, self.cfg.num_input_channels, 32, 128, 128))).size(1)
        self.dim_feats = self.dim_feats * (2 if self.cfg.pool == "catavgmax" else 1)
        self.pooling = self.get_pool_layer()

        if isinstance(self.cfg.reduce_feat_dim, int):
            # Use 1D grouped convolution to reduce # of parameters
            groups = math.gcd(self.dim_feats, self.cfg.reduce_feat_dim)
            self.feat_reduce = nn.Conv1d(self.dim_feats, self.cfg.reduce_feat_dim, groups=groups, kernel_size=1,
                                         stride=1, bias=False)
            self.dim_feats = self.cfg.reduce_feat_dim

        self.dropout = nn.Dropout(p=self.cfg.dropout) 
        self.linear_levels = nn.Linear(self.dim_feats, 5) # levels
        self.linear_coords = nn.Linear(self.dim_feats, 30)

        if self.cfg.load_pretrained_backbone:
            print(f"Loading pretrained backbone from {self.cfg.load_pretrained_backbone} ...")
            weights = torch.load(self.cfg.load_pretrained_backbone, map_location=lambda storage, loc: storage)['state_dict']
            weights = {re.sub(r'^model.', '', k) : v for k,v in weights.items()}
            # Get feature_reduction, if present
            feat_reduce_weight = {re.sub(r"^feat_reduce.", "", k): v
                                  for k, v in weights.items() if "feat_reduce" in k}
            # Get backbone only
            weights = {re.sub(r'^backbone.', '', k) : v for k,v in weights.items() if 'backbone' in k}
            self.backbone.load_state_dict(weights)
            if len(feat_reduce_weight) > 0:
                self.feat_reduce.load_state_dict(feat_reduce_weight)

        if self.cfg.freeze_backbone:
            self.freeze_backbone()

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

        if return_loss:
            pass

        x = self.normalize(x) 
        
        features = self.pooling(self.backbone(x)) 

        if hasattr(self, "feat_reduce"):
            features = self.feat_reduce(features.unsqueeze(-1)).squeeze(-1) 

        if self.cfg.multisample_dropout:
            logits_levels = torch.mean(torch.stack([self.linear_levels(self.dropout(features)) for _ in range(5)]), dim=0)
            logits_coords = torch.mean(torch.stack([self.linear_coords(self.dropout(features)) for _ in range(5)]), dim=0)
        else:
            logits_levels = self.linear_levels(self.dropout(features))
            logits_coords = self.linear_coords(self.dropout(features))

        out = {"logits_levels": logits_levels, "logits_coords": logits_coords}
        if return_features:
            out["features"] = features 
        if return_loss: 
            loss = self.criterion(logits_coords, logits_levels, batch["coords"], batch["level_labels"], batch["included_levels"])
            out.update(loss)
            
        return out

    def get_pool_layer(self):
        assert self.cfg.pool in ["avg", "max", "fast", "avgmax", "catavgmax", "gem"], f"{layer_name} is not a valid pooling layer"
        if self.cfg.pool == "gem":
            return GeM(**self.cfg.pool_params) if hasattr(self.cfg, "pool_params") else GeM()
        else:
            return SelectAdaptivePool3d(pool_type=self.cfg.pool, flatten=True)

    def freeze_backbone(self):
        for param in self.backbone.parameters(): 
            param.requires_grad = False
        if hasattr(self, "feat_reduce"):
            for param in self.feat_reduce.parameters():
                param.requires_grad = False

    def change_num_input_channels(self):
        # Assumes original number of input channels in model is 3
        for i, m in enumerate(self.backbone.modules()):
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
