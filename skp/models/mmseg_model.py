try:
	from mmengine import Config, ConfigDict
	from mmseg.models.builder import build_segmentor
except ModuleNotFoundError:
	print("note: `mmsegmentation` is not installed")

import torch.nn as nn


class Net(nn.Module):

	def __init__(self, cfg):
		super().__init__()

		self.cfg = cfg 
		mmcv_config = Config()
		mmcv_config.type = "EncoderDecoder"
		mmcv_config.pretrained = None 
		mmcv_config.backbone = dict(
				type="MixVisionTransformer",
				in_channels=3,
				embed_dims=32,
				num_stages=4,
				num_layers=[2, 2, 2, 2],
				num_heads=[1, 2, 5, 8],
				patch_sizes=[7, 3, 3, 3],
				sr_ratios=[8, 4, 2, 1],
				out_indices=[0, 1, 2, 3],
				mlp_ratio=4,
				qkv_bias=True,
				drop_rate=0.0,
				attn_drop_rate=0.0,
				drop_path_rate=0.1 
			)
		mmcv_config.backbone = ConfigDict(mmcv_config.backbone)
		mmcv_config.decode_head = dict(
				type="SegformerHead",
				in_channels=[32, 64, 160, 256],
				in_index=[0, 1, 2, 3],
				channels=256,
				dropout_ratio=0.1,
				num_classes=19,
				norm_cfg=dict(type="SyncBN", requires_grad=True),
				align_corners=False
			)
		mmcv_config.decode_head = ConfigDict(mmcv_config.decode_head)
		self.model = build_segmentor(mmcv_config)

	def forward(self, x):
		return self.model(x)
