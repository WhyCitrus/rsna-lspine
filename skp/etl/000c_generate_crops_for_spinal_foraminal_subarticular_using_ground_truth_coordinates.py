# Use ground truth coordinates to extract crops
# Can serve as a form of data augmentation when used in training along with
# Crops which are generated using the models to predict coordinates
import cv2
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


def get_condition(string):
	string = string.lower()
	for condition in ["spinal", "foraminal", "subarticular"]:
		if condition in string:
			return condition


def crop_square_around_center(img, xc, yc, size_factor=0.15):
	h, w = size_factor * img.shape[0], size_factor * img.shape[1]
	x1, y1 = xc - w / 2, yc - h / 2
	x2, y2 = x1 + w, y1 + h
	x1, y1, x2, y2 = [int(_) for _ in [x1, y1, x2, y2]]
	return img[y1:y2, x1:x2]


coords_df = pd.read_csv("../../data/train_label_coordinates.csv")
meta_df = pd.read_csv("../../data/dicom_metadata.csv")

image_dir = "../../data/train_pngs/"
save_dir = "../../data/train_crops_gt_coords/"

failed = []
for row in tqdm(coords_df.itertuples(), total=len(coords_df)):
	tmp_meta_df = meta_df[meta_df.series_id == row.series_id]
	sort_ascending = "Subarticular" not in row.condition
	tmp_meta_df = tmp_meta_df.sort_values("SliceLocation", ascending=sort_ascending).reset_index(drop=True)
	this_instance = row.instance_number
	series_instances = tmp_meta_df.instance_number.values 
	ch2 = np.where(series_instances == row.instance_number)[0][0]
	ch1 = max(0, ch2 - 1)
	ch3 = min(len(tmp_meta_df) - 1, ch2 + 1)
	instances_3ch = series_instances[[ch1, ch2, ch3]]
	images = np.stack([
		cv2.imread(os.path.join(image_dir, str(row.study_id), str(row.series_id), f"IM{_:06d}.png"), 0)
		for _ in instances_3ch
	], axis=-1)
	cropped_3ch = crop_square_around_center(images, row.x, row.y, size_factor=0.15)
	condition = get_condition(row.condition)
	laterality = "" if condition == "spinal" else row.condition[:1]
	level = f"{row.condition[:1]}_{row.level.replace('/', '_')}" if laterality != "" else row.level.replace('/', '_')
	savefile = os.path.join(save_dir, condition, f"{row.study_id}_{row.series_id}_{level}.png")
	os.makedirs(os.path.dirname(savefile), exist_ok=True)
	try:
		status = cv2.imwrite(savefile, cropped_3ch)
	except Exception as e:
		print(row.series_id, e)
		failed.append(row)
		