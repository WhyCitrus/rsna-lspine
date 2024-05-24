import cv2
import glob
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/predicted_axial_subarticular_coords.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")
num_slices_per_series = {}
for series_id, series_df in meta_df.groupby("series_id"):
    num_slices_per_series[series_id] = len(series_df)

df = df.merge(meta_df[["study_id", "series_id", "rows", "cols"]].drop_duplicates(), on=["study_id", "series_id"])
df["num_slices"] = df.series_id.map(num_slices_per_series)

targets = [
    "lt_subarticular_x", "lt_subarticular_y", "rt_subarticular_x", "rt_subarticular_y"
]
for targ in targets:
	if targ.endswith("_x"):
		df[f"{targ}_abs"] = df[targ] * df["cols"]
	elif targ.endswith("_y"):
		df[f"{targ}_abs"] = df[targ] * df["rows"]

for col in df.columns:
	if col.endswith("_abs"):
		df[col] = df[col].round().astype("int")

save_dir = "../../data/train_axial_unilateral_subarticular_crops_3ch/"

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
	lt_x, lt_y, rt_x, rt_y = row.lt_subarticular_x_abs, row.lt_subarticular_y_abs, row.rt_subarticular_x_abs, row.rt_subarticular_y_abs
	h, w = (0.15 * instance_img.shape[0]), (0.15 * instance_img.shape[1])
	rt_x1 = rt_x - w // 2
	rt_x2 = rt_x1 + w
	rt_y1 = rt_y - h // 2
	rt_y2 = rt_y1 + h 
	lt_x1 = lt_x - w // 2
	lt_x2 = lt_x1 + w
	lt_y1 = lt_y - h // 2
	lt_y2 = lt_y1 + h 
	rt_x1, rt_y1, lt_x1, lt_y1 = max(0, rt_x1), max(0, rt_y1), max(0, lt_x1), max(0, lt_y1)
	max_h, max_w = instance_img.shape[:2]
	rt_x2, rt_y2, lt_x2, lt_y2 = min(max_w, rt_x2), min(max_h, rt_y2), min(max_w, lt_x2), min(max_h, lt_y2)
	rt_cropped_subarticular = instance_img[int(rt_y1):int(rt_y2), int(rt_x1):int(rt_x2)]
	lt_cropped_subarticular = instance_img[int(lt_y1):int(lt_y2), int(lt_x1):int(lt_x2)]
	status = cv2.imwrite(os.path.join(tmp_save_dir, f"RT_{row['level'].upper()}.png"), rt_cropped_subarticular)
	status = cv2.imwrite(os.path.join(tmp_save_dir, f"LT_{row['level'].upper()}.png"), lt_cropped_subarticular)
