import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # position_layer = nn.TransformerEncoderLayer(
        #     d_model=16,
        #     nhead=4,
        #     dim_feedforward=64,
        #     dropout=0.0,
        #     activation="gelu",
        #     batch_first=True
        # )

        # self.position_encoder = nn.TransformerEncoder(position_layer, num_layers=1)
        
        self.cfg = cfg 

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.transformer_d_model,
            nhead=cfg.transformer_nhead,
            dim_feedforward=cfg.transformer_dim_feedforward or cfg.transformer_d_model * 4,
            dropout=cfg.transformer_dropout or 0.1,
            activation=cfg.transformer_activation or "gelu",
            batch_first=True
            )

        self.pre_linear = nn.Linear(cfg.feature_dim, cfg.transformer_d_model)

        self.transformer = nn.TransformerEncoder(layer, 
            num_layers=cfg.transformer_num_layers)

        self.linear = nn.Linear(cfg.transformer_d_model, cfg.num_classes)

    def forward(self, batch, return_loss=False, return_features=False):
        x = batch["x"]
        y = batch["y"] if "y" in batch else None
        mask = batch["mask"]
        features = x[:, :, :-1]
        positions = x[:, :, -1].unsqueeze(-1)
        positions = positions.repeat(1, 1, self.cfg.position_duplication_factor)

        if return_loss:
            assert isinstance(y, torch.Tensor)

        #positions = self.position_encoder(positions, src_key_padding_mask=mask)
        features = torch.cat([features, positions], dim=2)
        features = self.pre_linear(features)
        features = self.transformer(features, src_key_padding_mask=mask)
        logits = self.linear(features)

        out = {"logits": logits}
        if return_features:
            out["features"] = features 
        if return_loss: 
            loss = self.criterion(logits, y, mask)
            if isinstance(loss, dict):
                out.update(loss)
            else:
                out["loss"] = loss

        return out

    def set_criterion(self, loss):
        self.criterion = loss
        