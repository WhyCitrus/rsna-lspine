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
        self.linear1 = nn.Linear(cfg.transformer_d_model, 10) # predict distance
        self.linear2 = nn.Linear(cfg.transformer_d_model, 20)

    def forward(self, batch, return_loss=False, return_features=False):
        x = batch["x"]
        y_dist = batch["y_dist"] if "y_dist" in batch else None
        y_coord = batch["y_coord"] if "y_coord" in batch else None
        # mask for padded tokens
        mask = batch["mask"] if "mask" in batch else None

        if return_loss:
            assert isinstance(y_dist, torch.Tensor) and isinstance(y_coord, torch.Tensor)

        features = self.transformer(x, src_key_padding_mask=mask)
        feature0 = features[:, 0]
        logits_dist = self.linear1(features)
        logits_coord = self.linear2(feature0)
        
        out = {"logits_dist": logits_dist, "logits_coord": logits_coord}
        if return_features:
            out["features"] = features 
        if return_loss: 
            loss = self.criterion(logits_dist, logits_coord, y_dist, y_coord, mask)
            if isinstance(loss, dict):
                out.update(loss)
            else:
                out["loss"] = loss

        return out

    def set_criterion(self, loss):
        self.criterion = loss
        