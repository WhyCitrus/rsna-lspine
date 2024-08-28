import cv2
import glob
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
coords_df = coords_df.loc[coords_df.condition.apply(lambda x: "Spinal" in x)]

description_df = pd.read_csv("../../data/train_series_descriptions.csv")
series_to_description = {row.series_id: row.series_description for row in description_df.itertuples()}

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(coords_df.series_id.tolist())]
instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient2" if series_to_description[series_id] == "Axial T2" else "ImagePositionPatient0", ascending=True)
	series_df["position_index"] = np.arange(len(series_df))
	instance_to_position_index_dict.update({f"{series_id}_{row.instance_number}": row.position_index for row in series_df.itertuples()})

coords_df["series_instance"] = coords_df.series_id.astype("str") + "_" + coords_df.instance_number.astype("str")
coords_df["position_index"] = coords_df.series_instance.map(instance_to_position_index_dict)

image_dir = "../../data/train_pngs_3ch/"
save_dir = "../../data/train_sag_t2_subarticular_crops_gt/"

for series_id, series_df in tqdm(coords_df.groupby("series_id"), total=len(coords_df.series_id.unique())):
	if len(series_df) != 5:
		continue
	study_id = series_df.study_id.iloc[0]
	images = np.sort(glob.glob(os.path.join(image_dir, str(study_id), str(series_id), "*.png")))
	for row in series_df.itertuples():
		spinal_image = os.path.join(image_dir, str(study_id), str(series_id), f"IM{row.position_index:06d}_INST{row.instance_number:06d}.png")
		spinal_image_index = list(images).index(spinal_image)
		# Take the 5 slices adjacent
		rt_subart = images[max(0, spinal_image_index-5):spinal_image_index]
		while len(rt_subart) < 5:
			spinal_image_index += 1
			rt_subart = images[max(0, spinal_image_index-5):spinal_image_index]
		spinal_image_index = list(images).index(spinal_image)
		lt_subart = images[spinal_image_index+1:min(len(images) + 1, spinal_image_index+6)]
		while len(lt_subart) < 5:
			spinal_image_index -= 1
			lt_subart = images[spinal_image_index+1:min(len(images) + 1, spinal_image_index+6)]
		rt_subart = np.stack([cv2.imread(_)[..., 1] for _ in rt_subart], axis=-1)
		lt_subart = np.stack([cv2.imread(_)[..., 1] for _ in lt_subart], axis=-1)
		rt_subart = crop_square_around_center(rt_subart, row.x, row.y, size_factor=0.15)
		lt_subart = crop_square_around_center(lt_subart, row.x, row.y, size_factor=0.15)
		if rt_subart.shape[0] * rt_subart.shape[1] > 0:
			tmp_save_dir = os.path.join(save_dir, f"{study_id}_{series_id}_R_{row.level.replace('/', '_')}")
			os.makedirs(tmp_save_dir, exist_ok=True)
			for each_img in range(rt_subart.shape[2]):
				sts = cv2.imwrite(os.path.join(tmp_save_dir, f"IM{each_img:06d}.png"), rt_subart[..., each_img])
		if lt_subart.shape[0] * lt_subart.shape[1] > 0:
			tmp_save_dir = os.path.join(save_dir, f"{study_id}_{series_id}_L_{row.level.replace('/', '_')}")
			os.makedirs(tmp_save_dir, exist_ok=True)
			for each_img in range(lt_subart.shape[2]):
				sts = cv2.imwrite(os.path.join(tmp_save_dir, f"IM{each_img:06d}.png"), lt_subart[..., each_img])
