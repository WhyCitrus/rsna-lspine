import cv2
import glob
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def crop_block_around_center(img, xc, yc, size_factor=0.15):
	h, w = size_factor * img.shape[1], size_factor * img.shape[2]
	x1, y1 = xc - w / 2, yc - h / 2
	x2, y2 = x1 + w, y1 + h
	x1, y1, x2, y2 = [int(_) for _ in [x1, y1, x2, y2]]
	return img[:, y1:y2, x1:x2]


coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
coords_df = coords_df.loc[coords_df.condition.apply(lambda x: "Foraminal" in x)]

image_dir = "../../data/train_pngs_3ch/"
save_dir = "../../data/train_foramen_blocks_gt/"

num_images = []
bad = []
for series_id, series_df in tqdm(coords_df.groupby("series_id"), total=len(coords_df.series_id.unique())):
	if len(series_df) != 10:
		bad.append(series_id)
		continue
	study_id = series_df.study_id.iloc[0]
	images = np.sort(glob.glob(os.path.join(image_dir, str(study_id), str(series_id), "*.png")))
	num_images.append(len(images))
	stack = np.stack([cv2.imread(_)[..., 1] for _ in images])
	for level, level_df in series_df.groupby("level"):
		mean_x, mean_y = level_df.x.mean(), level_df.y.mean()
		mean_x, mean_y = int(mean_x), int(mean_y)
		block = crop_block_around_center(stack, mean_x, mean_y, size_factor=0.15)
		tmp_save_dir = os.path.join(save_dir, f"{study_id}_{series_id}_{level.replace('/', '_')}")
		os.makedirs(tmp_save_dir, exist_ok=True)
		for idx, img_slice in enumerate(block):
			status = cv2.imwrite(os.path.join(tmp_save_dir, f"IM{idx:06d}.png"), img_slice)

np.percentile(num_images, [0, 5, 10, 15, 25, 50, 75, 85, 90, 95, 100])

train_df = pd.read_csv("../../data/train_narrow.csv")
train_df = train_df.loc[train_df.condition == "foraminal"]
lt = train_df.loc[train_df.laterality == "L"]
rt = train_df.loc[train_df.laterality == "R"]
train_df = lt.merge(rt, on=["study_id", "level"], suffixes=["_lt", "_rt"])
folds_df = pd.read_csv("../../data/folds_cv5.csv")
train_df = train_df.merge(folds_df, on="study_id")
description_df = pd.read_csv("../../data/train_series_descriptions.csv")
description_df = description_df.loc[description_df.series_description == "Sagittal T1"]
train_df = train_df.merge(description_df, on="study_id")
train_df["image_folder"] = train_df.study_id.astype("str") + "_" + train_df.series_id.astype("str") + "_" + train_df.level
train_df = train_df[~train_df.series_id.isin(bad)]
train_df = train_df[train_df.study_id.isin(coords_df.study_id.tolist())]
train_df.to_csv("../../data/train_foramen_blocks_gt.csv", index=False)
