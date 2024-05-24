import cv2
import glob
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/axial_slices_based_on_sagittal_canal_coords.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
num_slices_per_series = {}
for series_id, series_df in meta_df.groupby("series_id"):
    num_slices_per_series[series_id] = len(series_df)

save_dir = "../../data/train_axial_canal_crops_3ch/"

for row_idx, row in tqdm(df.iterrows(), total=len(df)):
	tmp_save_dir = os.path.join(save_dir, str(row.study_id), str(row.series_id))
	os.makedirs(tmp_save_dir, exist_ok=True)
	tmp_series_df = meta_df.loc[meta_df.series_id == row.series_id]
	tmp_series_df = tmp_series_df.sort_values("instance_number", ascending=True).reset_index(drop=True)
	instance_numbers = tmp_series_df.instance_number.tolist()
	ch1 = instance_numbers[max(instance_numbers.index(row.instance_number) - 1, 0)]
	ch2 = row.instance_number
	ch3 = instance_numbers[min(instance_numbers.index(row.instance_number) + 1, len(tmp_series_df) - 1)]
	img_dir = os.path.join("../../data/train_pngs/", str(row.study_id), str(row.series_id))
	instance_img = np.stack([
		cv2.imread(os.path.join(img_dir, f"IM{ch1:06d}.png"), 0),
		cv2.imread(os.path.join(img_dir, f"IM{ch2:06d}.png"), 0),
		cv2.imread(os.path.join(img_dir, f"IM{ch3:06d}.png"), 0)
	], axis=-1)
	h, w = int(0.25 * instance_img.shape[0]), int(0.25 * instance_img.shape[1])
	xc, yc = row.canal_x, row.canal_y
	x1, x2 = xc - w // 2, xc + w // 2
	y1, y2 = yc - h // 2, yc + h // 2
	cropped_canal = instance_img[y1:y2, x1:x2]
	status = cv2.imwrite(os.path.join(tmp_save_dir, f"{row['level'].upper()}.png"), cropped_canal)
