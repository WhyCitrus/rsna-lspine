import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        layer = nn.TransformerEncoderLayer(
                d_model=cfg.transformer_d_model,
                nhead=cfg.transformer_nhead,
                dim_feedforward=cfg.transformer_dim_feedforward or cfg.transformer_d_model * 4,
                dropout=cfg.transformer_dropout or 0.1,
                activation=cfg.transformer_activation or "gelu",
                batch_first=True
            )

        self.transformer = nn.TransformerEncoder(layer, 
            num_layers=cfg.transformer_num_layers)
        self.classifier = nn.Linear(cfg.transformer_d_model, cfg.num_classes)

    def forward(self, batch, return_loss=False, return_features=False):
        x = batch["x"]
        y = batch["y"] if "y" in batch else None
        # mask for padded tokens
        mask = batch["mask"] if "mask" in batch else None
        if return_loss:
            assert isinstance(y, torch.Tensor)

        features = self.transformer(x, src_key_padding_mask=mask)

        logits = self.classifier(features)
        
        out = {"logits": logits}
        if return_features:
            out["features"] = features 
        if return_loss: 
            loss = self.criterion(logits, y, mask)
            out["loss"] = loss

        return out

    def set_criterion(self, loss):
        self.criterion = loss
        