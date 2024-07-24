import cv2
import numpy as np
import os
import pandas as pd

from tqdm import tqdm


df = pd.read_csv("../../data/train_label_coordinates.csv")
subart_df = df.loc[df.condition.apply(lambda x: "Subarticular" in x)]

meta_df = pd.read_csv("../../data/dicom_metadata.csv")
meta_df = meta_df.loc[meta_df.series_id.isin(subart_df.series_id.tolist())]
instance_to_position_index_dict = {}
for series_id, series_df in meta_df.groupby("series_id"):
	series_df = series_df.sort_values("ImagePositionPatient2", ascending=True)
	series_df["position_index"] = np.arange(len(series_df))
	instance_to_position_index_dict.update({f"{series_id}_{row.instance_number}": row.position_index for row in series_df.itertuples()})

meta_df["series_instance"] = meta_df.series_id.astype("str") + "_" + meta_df.instance_number.astype("str")
meta_df["position_index"] = meta_df.series_instance.map(instance_to_position_index_dict)

position_index_to_instance_dict = {f"{k.split('_')[0]}_{v}": int(k.split("_")[1]) for k, v in instance_to_position_index_dict.items()}

levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
col_names = [f"rt_{_}" for _ in levels] + [f"lt_{_}" for _ in levels]

side_to_pixel_value = {"Left": 175, "Right": 255}
save_dir = "../../data/train_subarticular_segmentation_maps_smaller/"
os.makedirs(save_dir, exist_ok=True)

subart_df["laterality"] = subart_df.condition.apply(lambda x: x.split()[0])
subart_df["series_instance"] = subart_df.series_id.astype("str") + "_" + subart_df.instance_number.astype("str")
subart_df["position_index"] = subart_df.series_instance.map(instance_to_position_index_dict)

for series_id, series_df in tqdm(subart_df.groupby("series_id"), total=len(subart_df.series_id.unique())):
	# skip if not full complement of right and left for each level
	if len(np.unique(series_df.condition + series_df.level)) != 10:
		continue
	# corresponding filepath of the image
	series_df["filepath"] = [f"{row.study_id}/{row.series_id}/IM{row.position_index:06d}_INST{row.instance_number:06d}.png" for row in series_df.itertuples()]
	series_meta = meta_df.loc[meta_df.series_id == series_id]
	# for some axial T2 series, some images are 640 x 640 while others are 608 x 608, within the same series
	# not sure why, but need to account for that here
	if len(series_meta.rows.unique()) != 1:
		series_meta["row_scale_factor"] = series_meta.rows.max() / series_meta.rows 
		series_df = series_df.merge(series_meta, on=["study_id", "series_id", "instance_number"])
		series_df["y"] = series_df.y * series_df.row_scale_factor
	if len(series_meta.cols.unique()) != 1:
		series_meta["col_scale_factor"] = series_meta.cols.max() / series_meta.cols 
		series_df = series_df.merge(series_meta, on=["study_id", "series_id", "instance_number"])
		series_df["x"] = series_df.x * series_df.col_scale_factor
	rows, cols = series_meta.rows.max(), series_meta.cols.max()
	tmp_mask = np.zeros((rows, cols, len(series_meta)))
	for position_index, position_df in series_df.groupby("position_index"):
		tmp_mask_inst = np.zeros((rows, cols))
		for pos_row_idx, pos_row in position_df.iterrows():
			tmp_mask_inst = cv2.ellipse(tmp_mask_inst.copy(), 
										(round(pos_row.x), round(pos_row.y)), 
										(int(0.0125 * cols), int(0.0125 * rows)), 
										0, 0, 360, 
										side_to_pixel_value[pos_row.laterality], -1)
			tmp_mask[:, :, pos_row.position_index] = tmp_mask_inst
	tmp_mask_copy = np.zeros_like(tmp_mask)
	for ch in range(1, tmp_mask.shape[2]):
		if ch == tmp_mask.shape[2] - 1:
			tmp_mask_copy[:, :, -1] = tmp_mask[:, :, -1] + tmp_mask[:, :, -2] * 0.5
		else:
			tmp_mask_copy[:, :, ch] = tmp_mask[:, :, ch] + 0.5 * tmp_mask[:, :, ch + 1] + 0.5 * tmp_mask[:, :, ch - 1]
	tmp_mask_copy = tmp_mask_copy.astype("uint8")
	# ch corresponds to position index
	for ch in range(tmp_mask_copy.shape[2]):
		if tmp_mask_copy[..., ch].sum() == 0:
			continue
		tmp_save_dir = os.path.join(save_dir, str(series_df.study_id.values[0]), str(series_id))
		os.makedirs(tmp_save_dir, exist_ok=True)
		tmp_filepath = f"IM{ch:06d}_INST{position_index_to_instance_dict[f'{series_id}_{ch}']:06d}.png"
		status = cv2.imwrite(os.path.join(tmp_save_dir, tmp_filepath), tmp_mask_copy[..., ch])
