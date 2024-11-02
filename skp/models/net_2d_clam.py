import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
from timm.models.layers import SelectAdaptivePool2d


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.flatten = nn.Flatten(1)

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        return self.flatten(x)


class GatedAttentionModule(nn.Module):

    def __init__(self, emb_dim, attn_dim, num_classes, attn_dropout=0.25):
        super().__init__()

        # Attention
        self.attn_a = nn.Sequential(
            nn.Linear(emb_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(attn_dropout)
        )
        self.attn_b = nn.Sequential(
            nn.Linear(emb_dim, attn_dim),
            nn.Sigmoid(),
            nn.Dropout(attn_dropout)
        )
        self.attn = nn.Sequential(
            nn.Linear(attn_dim, num_classes),
        )

    def forward(self, x):
        a = self.attn_a(x)
        b = self.attn_b(x)
        A = a.mul(b)
        A = self.attn(A)
        A = torch.softmax(A, dim=1)
        h = A.transpose(2, 1).bmm(x)
        return h, A


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone_args = {
            "pretrained": self.cfg.pretrained,
            "num_classes": 0,
            "global_pool": "",
            "features_only": self.cfg.features_only,
            "in_chans": self.cfg.num_input_channels
        }
        if self.cfg.backbone_img_size:
            if "efficientvit" in self.cfg.backbone:
                backbone_args["img_size"] = self.cfg.image_height
            else:
                backbone_args["img_size"] = (self.cfg.image_height, self.cfg.image_width)
        self.backbone = create_model(self.cfg.backbone, 
            **backbone_args)
        self.dim_feats = self.backbone(torch.randn((2, self.cfg.num_input_channels, self.cfg.image_height, self.cfg.image_width))).size(-1 if "xcit" in self.cfg.backbone else 1)
        self.dim_feats = self.dim_feats * (2 if self.cfg.pool == "catavgmax" else 1)
        if self.cfg.pool != "none":
            self.pooling = self.get_pool_layer()

        if isinstance(self.cfg.reduce_feat_dim, int):
            # Use 1D grouped convolution to reduce # of parameters
            groups = math.gcd(self.dim_feats, self.cfg.reduce_feat_dim)
            self.feat_reduce = nn.Conv1d(self.dim_feats, self.cfg.reduce_feat_dim, groups=groups, kernel_size=1,
                                         stride=1, bias=False)
            self.dim_feats = self.cfg.reduce_feat_dim

        self.attn = GatedAttentionModule(self.dim_feats, self.dim_feats or self.cfg.attn_dim, self.cfg.num_classes, self.cfg.attn_dropout or 0.25)
        self.dropout = nn.Dropout(p=self.cfg.dropout) 
        self.linears = nn.ModuleList([nn.Linear(self.dim_feats, 1) for _ in range(self.cfg.num_classes)])

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

        self.backbone_frozen = False
        if self.cfg.freeze_backbone:
            self.freeze_backbone()
            self.backbone_frozen = True

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
        elif self.cfg.normalization == "none": 
            x = x
        return x 

    def forward(self, batch, return_loss=False, return_features=False):
        x = batch["x"]
        y = batch["y"] if "y" in batch else None

        if return_loss:
            assert isinstance(y, torch.Tensor)

        x = self.normalize(x) 
        # x.shape = (B, C, H, W)
        B, C, H, W = x.shape
        x = x.reshape(B*C, 1, H, W)

        if self.cfg.pool != "none":
            features = self.pooling(self.backbone(x)) 
        else:
            features = self.backbone(x).mean(1)
            
        features = features.reshape(B, C, -1)
        features, attn_weights = self.attn(features)

        if hasattr(self, "feat_reduce"):
            features = self.feat_reduce(features.unsqueeze(-1)).squeeze(-1) 

        logits = torch.cat([self.linears[c](features[:, c]) for c in range(self.cfg.num_classes)], dim=1)

        out = {"logits": logits}
        if return_features:
            out["features"] = features 
        if return_loss: 
            loss = self.criterion(logits, y, w=batch["wts"]) if "wts" in batch else self.criterion(logits, y)
            if isinstance(loss, dict):
                out.update(loss)
            else:
                out["loss"] = loss

        return out

    def get_pool_layer(self):
        assert self.cfg.pool in ["avg", "max", "fast", "avgmax", "catavgmax", "gem"], f"{layer_name} is not a valid pooling layer"
        if self.cfg.pool == "gem":
            return GeM(**self.cfg.pool_params) if hasattr(self.cfg, "pool_params") else GeM()
        else:
            return SelectAdaptivePool2d(pool_type=self.cfg.pool, flatten=True)

    def freeze_backbone(self):
        for param in self.backbone.parameters(): 
            param.requires_grad = False
        if hasattr(self, "feat_reduce"):
            for param in self.feat_reduce.parameters():
                param.requires_grad = False

    def set_criterion(self, loss):
        self.criterion = loss
