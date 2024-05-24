import cv2
import glob
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/predicted_sagittal_t2_stir_canal_coords_oof.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")

num_slices_per_series = {}
for series_id, series_df in meta_df.groupby("series_id"):
    num_slices_per_series[series_id] = len(series_df)

df = df.merge(meta_df[["study_id", "series_id", "rows", "cols"]].drop_duplicates(), on=["study_id", "series_id"])
df["num_slices"] = df.series_id.map(num_slices_per_series)

targets = [
    "canal_l1_l2_x", "canal_l2_l3_x", "canal_l3_l4_x", "canal_l4_l5_x", "canal_l5_s1_x",
    "canal_l1_l2_y", "canal_l2_l3_y", "canal_l3_l4_y", "canal_l4_l5_y", "canal_l5_s1_y",
    "canal_l1_l2_z", "canal_l2_l3_z", "canal_l3_l4_z", "canal_l4_l5_z", "canal_l5_s1_z"
]
for targ in targets:
	if targ.endswith("_x"):
		df[f"{targ}_abs"] = df[targ] * df["cols"]
	elif targ.endswith("_y"):
		df[f"{targ}_abs"] = df[targ] * df["rows"]
	elif targ.endswith("_z"):
		df[f"{targ}_abs"] = df[targ] * df["num_slices"]

for col in df.columns:
	if col.endswith("_abs"):
		df[col] = df[col].round().astype("int")


save_dir = "../../data/train_sagittal_canal_crops_3ch/"
levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
for row_idx, row in tqdm(df.iterrows(), total=len(df)):
	images = np.sort(glob.glob(os.path.join("../../data/train_pngs/", str(row.study_id), str(row.series_id), "*.png")))
	tmp_save_dir = os.path.join(save_dir, str(row.study_id), str(row.series_id))
	os.makedirs(tmp_save_dir, exist_ok=True)
	for each_level in levels:
		instance = "canal_" + each_level + "_z_abs"
		ch1, ch2, ch3 = max(row[instance] - 1, 0), row[instance], min(row[instance] + 1, len(images) - 1)
		instance_img = np.stack([cv2.imread(images[ch1], 0), cv2.imread(images[ch2], 0), cv2.imread(images[ch3], 0)], axis=-1)
		h, w = int(0.15 * instance_img.shape[0]), int(0.15 * instance_img.shape[1])
		xc, yc = row["canal_" + each_level + "_x_abs"], row["canal_" + each_level + "_y_abs"]
		x1, x2 = xc - w // 2, xc + w // 2
		y1, y2 = yc - h // 2, yc + h // 2
		cropped_canal = instance_img[y1:y2, x1:x2]
		status = cv2.imwrite(os.path.join(tmp_save_dir, f"{each_level.upper()}.png"), cropped_canal)
