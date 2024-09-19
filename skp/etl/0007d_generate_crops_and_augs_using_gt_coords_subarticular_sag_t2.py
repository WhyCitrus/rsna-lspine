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


coords_df = pd.read_csv("../../data/sag_t2_mapped_subarticular_coords.csv")

description_df = pd.read_csv("../../data/train_series_descriptions.csv")
series_to_description = {row.series_id: row.series_description for row in description_df.itertuples()}

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(coords_df.series_id.tolist())]
instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient2" if series_to_description[series_id] == "Axial T2" else "ImagePositionPatient0", ascending=True)
	series_df["position_index"] = np.arange(len(series_df))
	instance_to_position_index_dict.update({f"{series_id}_{row.instance_number}": row.position_index for row in series_df.itertuples()})

# We are going also going to include instance numbers offset by +/- 1
coords_df1 = coords_df.copy()
coords_df1["instance_number"] -= 1
coords_df2 = coords_df.copy()
coords_df2["instance_number"] += 1

coords_df = pd.concat([coords_df, coords_df1, coords_df2])
coords_df["series_instance"] = coords_df.series_id.astype("str") + "_" + coords_df.instance_number.astype("str")
coords_df["position_index"] = coords_df.series_instance.map(instance_to_position_index_dict)

image_dir = "../../data/train_pngs_3ch/"
save_dir = "../../data/train_sag_t2_subarticular_crops_gt_with_augs/"

failed = []
pngfile_does_not_exist = []
for row in tqdm(coords_df.itertuples(), total=len(coords_df)):
	if np.isnan(row.position_index):
		continue
	pngfile = os.path.join(image_dir, f"{row.study_id}/{row.series_id}/IM{int(row.position_index):06d}_INST{row.instance_number:06d}.png")
	if not os.path.exists(pngfile):
		# There may be rare instances where the file does not exist for the ones we offset by +/- 1
		pngfile_does_not_exist.append(pngfile)
		continue
	img = cv2.imread(pngfile)
	offset_x, offset_y = int(0.0175 * img.shape[1]), int(0.0175 * img.shape[0])
	xs = [row.x, row.x - offset_x, row.x + offset_x]
	ys = [row.y, row.y - offset_y, row.y + offset_y]
	crops = []
	for xi in xs:
		for yi in ys:
			crops.append(crop_square_around_center(img, xi, yi, size_factor=0.15))
	level = f"{row.condition[:1]}_{row.level.replace('/', '_')}" if laterality != "" else row.level.replace('/', '_')
	save_files = [os.path.join(save_dir, f"{row.study_id}_{row.series_id}_{level}_INST{row.instance_number:06d}_{aug_idx:03d}.png") for aug_idx in range(len(crops))]
	os.makedirs(os.path.dirname(save_files[0]), exist_ok=True)
	for each_crop, each_save_file in zip(crops, save_files):
		try:
			sts = cv2.imwrite(each_save_file, each_crop)
		except Exception as e:
			print(row.series_id, e)
			failed.append(row.index)
