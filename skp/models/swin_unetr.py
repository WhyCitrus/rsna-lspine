import torch
import torch.nn as nn

from monai.networks.nets import SwinUNETR


class Net(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


        if isinstance(self.cfg.load_pretrained_model, str):
            # Loading BraTS pretrained model weights 
            print(f"Loading pretrained model weights from {self.cfg.load_pretrained_model} ...")
            weights = torch.load(self.cfg.load_pretrained_model)["state_dict"]
            # We are predicting nonenhancing T2 signal abnormality and enhancement
            # Classes appear to be in order of: 1) tumor core, 2) whole tumor, 3) enhancing tumor
            # So using class 2 and 3 weights seems to make the most sense
            assert self.cfg.num_classes == 2
            weights["out.conv.conv.weight"] = weights["out.conv.conv.weight"][1:]
            weights["out.conv.conv.bias"] = weights["out.conv.conv.bias"][1:]

            self.model = SwinUNETR(img_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
                in_channels=4,
                out_channels=2,
                feature_size=48,
                use_checkpoint=cfg.use_checkpoint or False,
                )

            self.model.load_state_dict(weights)
        else:
            self.model = SwinUNETR(img_size=(cfg.roi_x, cfg.roi_y, cfg.roi_z),
                in_channels=cfg.in_channels,
                out_channels=cfg.out_channels,
                feature_size=cfg.feature_size or 48,
                use_checkpoint=cfg.use_checkpoint or False,
                )

    def forward(self, batch, return_loss=False, return_features=False):
        x = batch["x"]
        y = batch["y"] if "y" in batch else None

        if return_loss:
            assert isinstance(y, torch.Tensor)

        logits = self.model(x)

        out = {"logits": logits}
        if return_loss: 
            loss = self.criterion(logits, y)
            out["loss"] = loss

        return out

    def set_criterion(self, loss):
        self.criterion = loss
