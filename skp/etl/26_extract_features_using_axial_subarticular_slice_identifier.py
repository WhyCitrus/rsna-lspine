import albumentations as A
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pydicom
import sys
sys.path.insert(0, "../../skp")
import torch

from collections import defaultdict
from importlib import import_module
from tasks.utils import build_dataloader
from tqdm import tqdm


def load_model_fold_dict(checkpoint_dict, cfg):
    model_dict = {}
    cfg.pretrained = False
    for fold, checkpoint_path in checkpoint_dict.items():
        print(f"Loading weights from {checkpoint_path} ...")
        wts = torch.load(checkpoint_path)["state_dict"]
        wts = {k.replace("model.", ""): v for k, v in wts.items()}
        model = import_module(f"models.{cfg.model}").Net(cfg)
        model.load_state_dict(wts)
        model = model.eval().cuda()
        model_dict[fold] = model
    return model_dict


cfg_file = "cfg_identify_subarticular_slices"
cfg = import_module(f"configs.{cfg_file}").cfg
checkpoint_dict = {
    0: "../../skp/experiments/cfg_identify_subarticular_slices/fda7fcb2/fold0/checkpoints/last.ckpt",
    1: "../../skp/experiments/cfg_identify_subarticular_slices/47bb2f11/fold1/checkpoints/last.ckpt",
    2: "../../skp/experiments/cfg_identify_subarticular_slices/cb964484/fold2/checkpoints/last.ckpt",
    3: "../../skp/experiments/cfg_identify_subarticular_slices/9563e2ed/fold3/checkpoints/last.ckpt",
    4: "../../skp/experiments/cfg_identify_subarticular_slices/b7ea0fee/fold4/checkpoints/last.ckpt"
}

subarticular_slice_finder_model_2d = {
    "cfg": cfg,
    "models": load_model_fold_dict(checkpoint_dict, cfg)
}

df = pd.read_csv("../../data/train_axial_subarticular_slice_identifier_sequence.csv")

image_dir = "../../data/train_pngs/"
save_dir = "../../data/train_axial_subarticular_slice_identifier_features/"
for fold in range(5):
	os.makedirs(os.path.join(save_dir, f"fold{fold}"))

for series_id, series_df in df.groupby("series_id"):
	series_df = series_df.sort_values("instance_number", ascending=True)
	array = np.stack([cv2.imread(os.path.join(image_dir, str(row.study_id), str(row.series_id), f"IM{row.instance_number:06d}.png"), 0) for row in series_df.itertuples()])
	array = np.expand_dims(array, axis=-1)
	array = np.stack([subarticular_slice_finder_model_2d["cfg"].val_transforms(image=img)["image"] for img in array])
	array = array.transpose(0, 3, 1, 2)
	array = torch.from_numpy(array).cuda().float()
	feature_dict = {}
	with torch.inference_mode():
		for fold, fold_model in subarticular_slice_finder_model_2d["models"].items():
			feature_dict[fold] = fold_model({"x": array}, return_features=True)["features"].cpu().numpy()
	for fold, fold_features in feature_dict.items():
		np.save(os.path.join(save_dir, f"fold{fold}", f"{series_df.study_id.iloc[0]}-{series_id}-feature.npy"), fold_features)
		np.save(os.path.join(save_dir, f"fold{fold}", f"{series_df.study_id.iloc[0]}-{series_id}-label.npy"), series_df[["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]].values)


features = glob.glob(os.path.join(save_dir, "fold0", "*feature.npy"))

new_df = pd.DataFrame({"filepath": features})
new_df["filepath"] = new_df.filepath.apply(lambda x: os.path.basename(x))
new_df["labelpath"] = new_df.filepath.apply(lambda x: x.replace("feature", "label"))
new_df["study_id"] = new_df.filepath.apply(lambda x: x.split("-")[0]).astype("int")
new_df["series_id"] = new_df.filepath.apply(lambda x: x.split("-")[1].replace(".npy", "")).astype("int")
folds_df = pd.read_csv("../../data/folds_cv5.csv")
new_df = new_df.merge(folds_df, on="study_id")
new_df.to_csv("../../data/train_axial_subarticular_slice_identifier_features_kfold.csv", index=False)